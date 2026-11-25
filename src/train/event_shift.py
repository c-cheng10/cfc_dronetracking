"""Score trained packet arms on validation windows whose EVENTS were thinned or jittered.

python -m src.train.event_shift --runs runs/packets-2026-11-22 --out FILE

Unlike rate_shift.py (which drops whole packets), events are changed first and the
window is then re-packetized, as a farther drone or a noisier sensor would be.
  thin K:   keep each event with probability K (fewer events: range, contrast, smaller sensor)
  jitter S: add Gaussian noise of S ms to every timestamp, then re-sort (sensor latency noise)
Claim fixed before any result was seen (2026-11-24): trained the ordinary way, the CfC and
mixed-memory CfC lose less AUC and matched AUC than the GRU and LSTM under both shifts; and the
drop-trained mixed-memory CfC loses less than the drop-trained GRU, since neither saw these shifts.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from src import phase1_fft as phase1
from src.count_matched import both_aucs, matched_pairs
from src.data.cnn_windows import load_splits
from src.data.event_packets import FEATURES, WINDOW_MS, build_sequence
from src.data.window_cache import load_set
from src.models.packet_models import PACKET_ARMS
from src.train.cnn import SEEDS
from src.train.packets import predict, tensors
from src.train.rate_shift import ARMS

THIN = (0.5, 0.25, 0.1)
JITTER_MS = (0.1, 0.5, 2.0)


def thin(keep, rng):
    def fn(t, p, x, y):
        k = rng.random(len(t)) < keep
        return t[k], p[k], x[k], y[k]
    return fn


def jitter(sigma, rng):
    def fn(t, p, x, y):
        t = np.clip(t + rng.normal(0, sigma, len(t)), 0, WINDOW_MS)
        order = np.argsort(t, kind="stable")
        return t[order], p[order], x[order], y[order]
    return fn


def train_count_stats(sequences):
    """The training log-count mean and std, as packets.py computes them, read from the Phase 1 CSVs
    so the training recordings need not be on disk."""
    counts = []
    for s in sequences:
        with phase1.csv_path(s).open(newline="") as f:
            counts += [float(r["event_count"]) for r in csv.DictReader(f) if r["background_type"] in {"drone", "active"}]
    log_count = np.log1p(np.array(counts, dtype=np.float32))
    return log_count.mean(), log_count.std()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(8)
    train_seq, val_seq = load_splits()
    stats = train_count_stats(train_seq)
    rng = np.random.default_rng(0)          # one stream, so every arm sees the same shifted events
    shifts = {"none": lambda *e: e, **{f"thin {k}": thin(k, rng) for k in THIN},
              **{f"jitter {s} ms": jitter(s, rng) for s in JITTER_MS}}
    n_events = {name: [] for name in shifts}

    def counted(name, fn):                  # record each window's surviving event count
        def wrapped(*e):
            out = fn(*e)
            n_events[name].append(len(out[0]))
            return out
        return wrapped

    # shortcut: shifted packets are rebuilt every run, not cached; cache them if this is rerun often
    shifted, y = {name: [] for name in shifts}, []
    for s in val_seq:
        packets, rows = build_sequence(s, {name: counted(name, fn) for name, fn in shifts.items()})
        for name in shifts:
            shifted[name] += packets[name]
        y += [int(r["label"]) for r in rows]
        print(f"{s}: {len(rows)} windows shifted", flush=True)
    y = np.array(y, dtype=bool)
    counts = {name: np.log1p(np.array(n, dtype=np.float32)) for name, n in n_events.items()}
    pairs = matched_pairs(*load_set(val_seq))
    baseline = json.loads((args.runs / "rate_shift.json").read_text())["arms"]
    report = {"shifts": list(shifts), "matched_pairs": len(pairs[0]), "claim": __doc__.split("Claim")[1].strip(), "arms": {}}
    for arm in ARMS:
        feature_set, constant_gap, build = PACKET_ARMS[arm.removesuffix("_aug")]
        models = []
        for seed in SEEDS:
            model = build(len(FEATURES[feature_set]))
            model.load_state_dict(torch.load(args.runs / arm / f"seed_{seed}.pt", map_location="cpu"))
            models.append(model)
        report["arms"][arm] = {}
        for name in shifts:
            store = tensors(shifted[name], counts[name], feature_set, constant_gap, stats)
            results = [both_aucs(predict(m, store), y, pairs) for m in models]
            report["arms"][arm][name] = {k: [r[k] for r in results] for k in results[0]}
            print(f"{arm:17} {name:14} AUC {np.mean([r['auc'] for r in results]):.3f}"
                  f" matched {np.mean([r['matched_auc'] for r in results]):.3f}", flush=True)
        # self-check: the unshifted rebuild must reproduce Max's 0% numbers exactly
        assert np.allclose(report["arms"][arm]["none"]["auc"], baseline[arm]["0.0"]["auc"], atol=1e-4), arm
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
