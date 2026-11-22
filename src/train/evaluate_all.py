"""Score every Phase 3 model on one set: overall AUC, count-matched AUC and fixed thresholds.

python -m src.train.evaluate_all --set heldout --recurrent runs/recurrent-2026-11-22 --out FILE
Reference features are fitted or oriented on the training recordings only.
Refuses to overwrite, so the challenging set is scored once per output file.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.count_matched import both_aucs, dispersion, matched_pairs
from src.data.cnn_windows import load_splits
from src.data.event_tokens import tokenise
from src.data.window_cache import load_set
from src.models.cnn import EventCountCNN
from src.models.recurrent import ARMS
from src.phase1_fft import OUT_DIR, load_frozen_threshold
from src.phase1_multiseq import metrics, tune_threshold
from src.train.cnn import SEEDS
from src.train.recurrent import predict


def fit_logistic(X, y, steps=2000, rate=0.5):
    mean, std = X.mean(0), X.std(0)
    Z = np.c_[(X - mean) / std, np.ones(len(X))]
    weights = np.zeros(Z.shape[1])
    sample = np.where(y, 0.5 / y.sum(), 0.5 / (~y).sum())       # balanced classes
    for _ in range(steps):
        p = 1 / (1 + np.exp(-Z @ weights))
        weights -= rate * Z.T @ ((p - y) * sample)
    return lambda F: np.c_[(F - mean) / std, np.ones(len(F))] @ weights


@torch.inference_mode()
def cnn_scores(weights, counts):
    model = EventCountCNN()
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    return torch.cat([model(b) for b in torch.from_numpy(np.log1p(counts)).split(256)]).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=("heldout", "challenging"), required=True)
    parser.add_argument("--cnn", type=Path, default=Path("runs/cnn-fred-2026-11-12"))
    parser.add_argument("--recurrent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    sequences = json.loads((OUT_DIR / "phase1_sequences.json").read_text())[args.set]
    train_counts, train_y, train_rows = load_set(load_splits()[0])
    counts, y, rows = load_set(sequences)
    pairs = matched_pairs(counts, y, rows)
    sequence = np.array([row["sequence"] for row in rows])
    rng = np.random.default_rng(0)
    shuffled = np.take_along_axis(counts, np.argsort(rng.random(counts.shape), axis=1), 1)

    column = lambda source, name: np.array([float(row[name]) for row in source])
    combined = "score_lr_" + load_frozen_threshold()["combined_model"]
    feats = lambda c: np.c_[np.log1p(c.sum(1)), dispersion(c)]
    logistic = fit_logistic(feats(train_counts), train_y)
    disp_train, sign = dispersion(train_counts), 1.0
    if both_aucs(disp_train, train_y, matched_pairs(train_counts, train_y, train_rows))["matched_auc"] < 0.5:
        sign = -1.0
    # name -> list of (training scores for the threshold, scores on this set), one per seed
    models = {
        "count": [(train_counts.sum(1), counts.sum(1))],
        "fft": [(column(train_rows, "score"), column(rows, "score"))],
        "fft_stability": [(column(train_rows, "score_fft_stab"), column(rows, "score_fft_stab"))],
        "fft_count_lr": [(column(train_rows, combined), column(rows, combined))],
        "dispersion": [(sign * disp_train, sign * dispersion(counts))],
        "count_dispersion_lr": [(logistic(feats(train_counts)), logistic(feats(counts)))],
        "cnn": [], "cnn_shuffled": [],
    }
    for seed in SEEDS:
        weights = args.cnn / f"seed_{seed}.pt"
        fit = cnn_scores(weights, train_counts)
        models["cnn"].append((fit, cnn_scores(weights, counts)))
        models["cnn_shuffled"].append((fit, cnn_scores(weights, shuffled)))
    tokens, train_tokens = [tokenise(w) for w in counts], [tokenise(w) for w in train_counts]
    shuffled_tokens = [tokenise(w) for w in shuffled]
    for arm, (build, constant_gap) in ARMS.items():
        models[arm] = []
        for seed in SEEDS:
            model = build()
            model.load_state_dict(torch.load(args.recurrent / arm / f"seed_{seed}.pt"))
            fit = predict(model, train_tokens, constant_gap)
            models[arm].append((fit, predict(model, tokens, constant_gap)))
            models.setdefault(arm + "_shuffled", []).append((fit, predict(model, shuffled_tokens, constant_gap)))

    report = {"set": args.set, "sequences": sequences, "windows": len(y), "drone_windows": int(y.sum()),
              "matched_pairs": len(pairs[0]),
              "matched_per_sequence": {s: int((sequence[pairs[0]] == s).sum()) for s in sequences},
              "dispersion_sign": sign, "models": {}}
    for name, runs in models.items():
        results = []
        for fit, scores in runs:
            result = {**both_aucs(scores, y, pairs),
                      **{k: v for k, v in metrics(scores[y], scores[~y], tune_threshold(fit[~train_y])).items()
                         if k in ("precision", "recall", "fa_per_min")}}
            for s in sequences:
                m = sequence == s
                result[f"auc {s}"] = float(both_aucs(scores[m], y[m], (np.flatnonzero(y[m]), np.flatnonzero(~y[m])))["auc"])
            results.append(result)
        report["models"][name] = {key: {"mean": float(np.mean([r[key] for r in results])),
                                        "std": float(np.std([r[key] for r in results], ddof=1)) if len(results) > 1 else None,
                                        "values": [float(r[key]) for r in results]}
                                  for key in results[0]}
        line = report["models"][name]
        print(f"{name:22}" + "".join(f" {k} {line[k]['mean']:.3f}" + (f"±{line[k]['std']:.3f}" if line[k]["std"] is not None else "      ")
                                     for k in ("auc", "matched_auc", "precision", "recall", "fa_per_min")), flush=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
