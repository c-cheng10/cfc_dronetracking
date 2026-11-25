"""Score models on the naturally sparse part of validation, from scores already saved.

python -m src.sparsity_slice --split phase3 --out FILE

No model is run. Drone windows are sliced by their event count and by box width
(small boxes are distant drones); each slice is compared with every background
window (AUC) and with background windows of nearly the same event count from the
same recordings (matched AUC, as in src/count_matched.py). Slice edges are
quantiles of the drone windows in this validation set.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src import phase1_fft as phase1
from src.count_matched import TOLERANCE
from src.data.cnn_windows import load_splits
from src.data.window_cache import CACHE

MODELS = {   # name -> saved validation scores at the model's best rate (Phase 3)
    "GRU": "runs/phase3-2026-11-24/lr0.01/time_gru/validation_scores.npz",
    "GRU, drop-trained": "runs/phase3-2026-11-24/lr0.01/time_gru_aug/validation_scores.npz",
    "LSTM": "runs/phase3-2026-11-24/lr0.01/time_lstm/validation_scores.npz",
    "CNN": "runs/phase3-2026-11-24/lr0.003/cnn/validation_scores.csv",
}
SLICES = {"sparsest 10%": (0.0, 0.1), "sparsest 25%": (0.0, 0.25), "middle 50%": (0.25, 0.75), "densest 25%": (0.75, 1.0)}


def rows_for(sequences):
    rows = []
    for sequence in sequences:
        with np.load(CACHE / f"{sequence.replace('/', '_')}.npz") as data:
            rows.extend(json.loads(str(data["rows"])))
    return rows


def load_scores(path, rows):
    if path.endswith(".npz"):
        with np.load(path) as data:
            return [data[f"seed_{s}"] for s in range(5)]
    with open(path, newline="") as source:
        table = list(csv.DictReader(source))
    assert [(r["sequence"], r["t_start_us"], r["box_name"]) for r in table] == \
           [(r["sequence"], r["t_start_us"], r["box_name"]) for r in rows]
    return [np.array([float(r[f"logit_seed_{s}"]) for r in table]) for s in range(5)]


def matched(drone, background, log_count, sequence):
    """One-to-one same-recording pairs within TOLERANCE, greedy, seed 0."""
    rng = np.random.default_rng(0)
    pos, neg = [], []
    for name in np.unique(sequence[drone]):
        d = drone[sequence[drone] == name].copy()
        b = background[sequence[background] == name]
        rng.shuffle(d)
        free = np.ones(len(b), bool)
        for i in d:
            gap = np.where(free, np.abs(log_count[b] - log_count[i]), np.inf)
            j = int(gap.argmin()) if len(gap) else -1
            if j >= 0 and gap[j] < TOLERANCE:
                free[j] = False
                pos.append(i)
                neg.append(b[j])
    return np.array(pos, int), np.array(neg, int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="phase3")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = rows_for(load_splits(args.split)[1])
    y = np.array([int(r["label"]) for r in rows], bool)
    count = np.array([float(r["event_count"]) for r in rows])
    width = np.array([float(r["box_width_px"]) for r in rows])
    sequence = np.array([r["sequence"] for r in rows])
    log_count = np.log1p(count)
    drone, background = np.flatnonzero(y), np.flatnonzero(~y)
    scores = {name: load_scores(path, rows) for name, path in MODELS.items()}
    report = {"split": args.split, "drone_windows": len(drone), "background_windows": len(background), "slices": {}}
    for measure, values in (("event count", count), ("box width px", width)):
        for label, (lo, hi) in SLICES.items():
            a, b = np.quantile(values[drone], [lo, hi])
            part = drone[(values[drone] >= a) & (values[drone] <= b)]
            pairs = matched(part, background, log_count, sequence)
            key = f"{measure}, {label}" if measure == "event count" else f"{measure}, " + label.replace("sparsest", "smallest").replace("densest", "largest")
            entry = {"range": [float(a), float(b)], "drone_windows": len(part), "matched_pairs": len(pairs[0]), "models": {}}
            for name, runs in scores.items():
                auc = [phase1.auc(s[part], s[background]) for s in runs]
                m = [phase1.auc(s[pairs[0]], s[pairs[1]]) for s in runs] if len(pairs[0]) >= 50 else []
                entry["models"][name] = {"auc": [float(v) for v in auc], "matched_auc": [float(v) for v in m]}
            entry["count_only_auc"] = float(phase1.auc(count[part], count[background]))
            report["slices"][key] = entry
            line = "  ".join(f"{n} {np.mean(v['auc']):.3f}/{(np.mean(v['matched_auc']) if v['matched_auc'] else float('nan')):.3f}"
                             for n, v in entry["models"].items())
            print(f"{key:28} {a:9.0f}-{b:<9.0f} n={len(part):5d} pairs={len(pairs[0]):4d} count {entry['count_only_auc']:.3f} | {line}", flush=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
