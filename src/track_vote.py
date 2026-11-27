"""Track-level voting on saved validation scores, plus the 5-seed average.

python -m src.track_vote --out FILE

Fixed before any result: the Phase 3 GRU at rate 0.01. A window raises an alarm
when its score passes that seed's threshold (2% of training background windows).
Windows are grouped into streams by recording, FRED track ID and box name, in
time order; a gap of more than one 0.25 s step starts a new stream. A rule
"k of n" confirms at a window when at least k of the last n windows in the
stream alarmed. Windows are 0.5 s long every 0.25 s, so neighbours share half
their events; the "every other window" rules use only windows 0.5 s apart,
which share none.

Reported per rule: drone-window recall, false alarms per background-box minute,
share of drone stream segments confirmed at least once, and median seconds from
a segment's start to its first confirmation.

The seed average uses the mean logit of the 5 seeds, with its threshold set the
same way on training background windows (this needs one pass over training).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src import phase1_fft as phase1
from src.data.cnn_windows import load_splits
from src.data.event_packets import FEATURES, load_packets
from src.models.packet_models import PACKET_ARMS
from src.phase1_multiseq import tune_threshold
from src.train.packets import predict, tensors

RUN = Path("runs/phase3-2026-11-24/lr0.01/time_gru")
STEP_US = 250_000
RULES = {  # name -> (k, n, stride in windows)
    "single window": (1, 1, 1),
    "2 of 2": (2, 2, 1),
    "2 of 3": (2, 3, 1),
    "3 of 4": (3, 4, 1),
    "4 of 4": (4, 4, 1),
    "2 of 2, every other window": (2, 2, 2),
    "2 of 3, every other window": (2, 3, 2),
}


def streams(rows):
    """List of index arrays, one per contiguous stream, in time order."""
    keys = {}
    for i, r in enumerate(rows):
        keys.setdefault((r["sequence"], r["track_id"], r["box_name"]), []).append(i)
    out = []
    for idx in keys.values():
        idx = sorted(idx, key=lambda i: int(rows[i]["t_start_us"]))
        t = np.array([int(rows[i]["t_start_us"]) for i in idx])
        cuts = np.flatnonzero(np.diff(t) != STEP_US) + 1
        out.extend(np.split(np.array(idx), cuts))
    return out


def confirm(alarm, segments, k, n, stride):
    """Boolean per window: confirmed under k of n at the given stride."""
    out = np.zeros_like(alarm)
    for seg in segments:
        a = alarm[seg].astype(int)
        for i in range(len(seg)):
            history = a[i::-stride][:n]
            out[seg[i]] = len(history) == n and history.sum() >= k
    return out


def evaluate(scores, threshold, y, segments, drone_segments, t):
    alarm = scores > threshold
    minutes = (~y).sum() * phase1.HOP_S / 60
    result = {}
    for name, (k, n, stride) in RULES.items():
        c = confirm(alarm, segments, k, n, stride)
        firsts = []
        hit = 0
        for seg in drone_segments:
            on = np.flatnonzero(c[seg])
            if len(on):
                hit += 1
                firsts.append((t[seg[on[0]]] - t[seg[0]]) / 1e6)
        result[name] = {
            "recall": float(c[y].mean()),
            "fa_per_min": float(c[~y].sum() / minutes),
            "segments_confirmed": hit / len(drone_segments),
            "median_s_to_confirm": float(np.median(firsts)) if firsts else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(4)
    train_seq, val_seq = load_splits("phase3")
    val_packets, val_count, y, rows = load_packets(val_seq)
    t = np.array([int(r["t_start_us"]) for r in rows])
    segments = streams(rows)
    drone_segments = [s for s in segments if y[s[0]]]
    results = json.loads((RUN / "results.json").read_text())
    with np.load(RUN / "validation_scores.npz") as data:
        seed_scores = [data[f"seed_{s}"] for s in range(5)]
    report = {"model": str(RUN), "drone_segments": len(drone_segments), "seeds": {}}
    for run, scores in zip(results["runs"], seed_scores):
        report["seeds"][run["seed"]] = evaluate(scores, run["threshold"], y, segments, drone_segments, t)

    # 5-seed average: threshold from the averaged training scores.
    train_packets, train_count, train_y, _ = load_packets(train_seq)
    stats = (train_count.mean(), train_count.std())
    feature_set, constant_gap, build = PACKET_ARMS["time_gru"]
    train_store = tensors(train_packets, train_count, feature_set, constant_gap, stats)
    train_scores = []
    for s in range(5):
        model = build(len(FEATURES[feature_set]))
        model.load_state_dict(torch.load(RUN / f"seed_{s}.pt", map_location="cpu"))
        train_scores.append(predict(model, train_store))
    mean_train = np.mean(train_scores, axis=0)
    mean_val = np.mean(seed_scores, axis=0)
    threshold = tune_threshold(mean_train[~train_y])
    report["average"] = {"auc": float(phase1.auc(mean_val[y], mean_val[~y])),
                         "seed_aucs": [float(phase1.auc(s[y], s[~y])) for s in seed_scores],
                         "rules": evaluate(mean_val, threshold, y, segments, drone_segments, t)}
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"drone stream segments: {len(drone_segments)}")
    for name in RULES:
        seeds = [report["seeds"][s][name] for s in report["seeds"]]
        avg = report["average"]["rules"][name]
        print(f"{name:28} seeds: recall {np.mean([x['recall'] for x in seeds]):.3f}  FA/min {np.mean([x['fa_per_min'] for x in seeds]):6.2f}"
              f"  tracks {np.mean([x['segments_confirmed'] for x in seeds]):.3f}  median s {np.median([x['median_s_to_confirm'] for x in seeds]):.2f}"
              f" | average: recall {avg['recall']:.3f}  FA/min {avg['fa_per_min']:6.2f}  tracks {avg['segments_confirmed']:.3f}")
    print(f"AUC: seeds {np.mean(report['average']['seed_aucs']):.4f}, average of seeds {report['average']['auc']:.4f}")


if __name__ == "__main__":
    main()
