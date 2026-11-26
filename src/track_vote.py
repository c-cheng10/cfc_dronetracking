"""Alarm only when k of a box's last n windows score as drone: how much does it cut false alarms?

python -m src.track_vote --out FILE

Uses Max's Phase 3 GRUs at rate 0.01 (runs/phase3-2026-11-24/lr0.01/), scores every
validation window, and groups windows into streams: one box followed over time,
keyed by (recording, FRED track id, box name). Windows are 0.5 s long and start
every 0.25 s, so "the last n windows" covers the last n x 0.25 s of a stream.
FRED's boxes are used, so this measures voting alone, with no finder and no tracker.

Claim fixed before any result was seen (2026-11-25): for the drop-trained GRU,
"3 of the last 4" cuts false alarm windows per background-box minute at least
5-fold against single windows, while recall on drone windows falls by less than
5 percentage points.

Reported per rule, mean of 5 seeds, each seed at its own threshold (2% of training
background windows pass, as in packets.py):
  recall           share of drone windows that raise an alarm
  fa_per_min       background windows that raise an alarm, per background-box minute
  episodes_per_min separate alarm episodes (runs of alarming windows) per background-box minute
  tracks_found     share of drone streams that alarm at least once
  first_alarm_s    median seconds from a drone stream's first window to its first alarm
"""

import argparse
import json
from collections import defaultdict
from functools import cache
from pathlib import Path

import numpy as np
import torch

from src import phase1_fft as phase1
from src.count_matched import both_aucs
from src.data.cnn_windows import load_splits
from src.data.event_packets import FEATURES, load_packets
from src.models.packet_models import PACKET_ARMS
from src.phase1_multiseq import tune_threshold
from src.train.cnn import SEEDS
from src.train.packets import predict, tensors

RUNS = Path("runs/phase3-2026-11-24/lr0.01")
ARMS = ("time_gru", "time_gru_aug")
RULES = ((1, 1), (2, 2), (2, 3), (3, 4), (3, 5), (4, 6))      # (k, n): k of the last n windows
HOP_US = int(phase1.HOP_S * 1e6)


def streams(rows):
    """{stream key: window indices sorted by start time}."""
    out = defaultdict(list)
    for i, r in enumerate(rows):
        out[(r["sequence"], r["track_id"], r["box_name"])].append(i)
    return {key: sorted(idx, key=lambda i: int(rows[i]["t_start_us"])) for key, idx in out.items()}


def vote(hits, starts, k, n):
    """Alarm at each window when at least k of the stream's windows in the last n hops were hits."""
    alarm = np.zeros(len(hits), bool)
    for i in range(len(hits)):
        recent = (starts > starts[i] - n * HOP_US) & (starts <= starts[i])
        alarm[i] = hits[recent].sum() >= k
    return alarm


def evaluate(scores, threshold, labels, rows, groups, k, n):
    alarm = np.zeros(len(scores), bool)
    episodes, first_alarm, found, drone_streams = 0, [], 0, 0
    for idx in groups.values():
        idx = np.array(idx)
        starts = np.array([int(rows[i]["t_start_us"]) for i in idx])
        a = vote(scores[idx] > threshold, starts, k, n)
        alarm[idx] = a
        if labels[idx].all():
            drone_streams += 1
            if a.any():
                found += 1
                first_alarm.append((starts[a.argmax()] - starts[0]) / 1e6)
        elif not labels[idx].any():
            # a new episode starts at an alarm whose previous window (one hop earlier) did not alarm
            previous = {s: on for s, on in zip(starts, a)}
            episodes += sum(on and not previous.get(s - HOP_US, False) for s, on in zip(starts, a))
    bg_minutes = (~labels).sum() * phase1.HOP_S / 60
    return {"recall": float(alarm[labels].mean()), "fa_per_min": float(alarm[~labels].sum() / bg_minutes),
            "episodes_per_min": float(episodes / bg_minutes), "tracks_found": found / drone_streams,
            "first_alarm_s": float(np.median(first_alarm)) if first_alarm else float("nan"),
            "drone_streams": drone_streams}


@cache
def training_data():
    train_seq, _ = load_splits("phase3")
    train_packets, train_count, train_y, _ = load_packets(train_seq)
    return train_packets, train_count, train_y


def scored_models(arm):
    """Max's 5 Phase 3 models for one arm at rate 0.01, each with its threshold (2% of training
    background windows pass). Returns (feature set, constant gap, count stats, [(model, threshold)])."""
    train_packets, train_count, train_y = training_data()
    feature_set, constant_gap, build = PACKET_ARMS[arm.removesuffix("_aug")]
    stats = (train_count.mean(), train_count.std())
    train_store = tensors(train_packets, train_count, feature_set, constant_gap, stats)
    seeds = []
    for seed in SEEDS:
        model = build(len(FEATURES[feature_set]))
        model.load_state_dict(torch.load(RUNS / arm / f"seed_{seed}.pt", map_location="cpu"))
        seeds.append((model, tune_threshold(predict(model, train_store)[~train_y])))
    return feature_set, constant_gap, stats, seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    # self-check of the rule: hits 1,0,1,1 a hop apart, 2 of the last 2 -> only the last window alarms
    assert vote(np.array([1, 0, 1, 1], bool), np.arange(4) * HOP_US, 2, 2).tolist() == [False, False, False, True]
    torch.set_num_threads(8)
    _, val_seq = load_splits("phase3")
    val_packets, val_count, val_y, val_rows = load_packets(val_seq)
    groups = streams(val_rows)
    mixed = sum(0 < val_y[idx].sum() < len(idx) for idx in groups.values())
    assert mixed == 0, f"{mixed} streams mix drone and background windows"
    report = {"claim": __doc__.split("Claim")[1].split("Reported")[0].strip(), "rules": RULES, "arms": {}}
    for arm in ARMS:
        feature_set, constant_gap, stats, seeds = scored_models(arm)
        val_store = tensors(val_packets, val_count, feature_set, constant_gap, stats)
        saved = json.loads((RUNS / arm / "results.json").read_text())["runs"]
        per_rule = defaultdict(list)
        for seed, (model, threshold) in zip(SEEDS, seeds, strict=True):
            scores = predict(model, val_store)
            # self-check: rescoring Max's weights here must give his validation AUC
            auc = both_aucs(scores, val_y, (np.flatnonzero(val_y), np.flatnonzero(~val_y)))["auc"]
            expected = next(r["auc"] for r in saved if r["seed"] == seed)
            assert abs(auc - expected) < 2e-3, (arm, seed, auc, expected)
            for k, n in RULES:
                per_rule[f"{k} of {n}"].append(evaluate(scores, threshold, val_y, val_rows, groups, k, n))
        report["arms"][arm] = {}
        for rule, results in per_rule.items():
            mean = {key: float(np.mean([r[key] for r in results])) for key in results[0]}
            report["arms"][arm][rule] = {"mean": mean, "seeds": results}
            print(f"{arm:13} {rule:7} recall {mean['recall']:.3f}  false alarms/min {mean['fa_per_min']:6.2f}"
                  f"  episodes/min {mean['episodes_per_min']:5.2f}  tracks found {mean['tracks_found']:.3f}"
                  f"  first alarm {mean['first_alarm_s']:.2f} s", flush=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
