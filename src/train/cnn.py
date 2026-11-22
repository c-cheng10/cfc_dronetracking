"""Train and validate the temporal CNN on the Phase 1 FRED crop protocol.

Run from the repo root: python -m src.train.cnn --out runs/cnn-fred
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data.cnn_windows import load_splits, load_windows
from src.models.cnn import EventCountCNN
from src.phase1_fft import BIN_US, WIN, load_frozen_threshold
from src.phase1_multiseq import FA_TARGET, metrics, tune_threshold


SEEDS = (0, 1, 2, 3, 4)
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WINDOW_ID = ("sequence", "track_id", "box_name", "t_start_us", "t_end_us", "label")


def load_split(sequences, device):
    counts_by_sequence = []
    rows = []
    for sequence in sequences:
        counts, sequence_rows = load_windows(sequence)
        counts_by_sequence.append(counts)
        rows.extend(sequence_rows)
        print(f"{sequence}: verified {len(sequence_rows):,} windows", flush=True)
    counts = np.concatenate(counts_by_sequence)
    np.log1p(counts, out=counts)
    labels = np.array([int(row["label"]) for row in rows], dtype=np.float32)
    if len(np.unique(labels)) != 2:
        raise ValueError("Each split needs both drone and background windows")
    return torch.from_numpy(counts).to(device), torch.from_numpy(labels).to(device), rows


@torch.inference_mode()
def predict(model, counts):
    model.eval()
    return torch.cat([model(batch) for batch in counts.split(BATCH_SIZE)]).cpu().numpy()


def evaluate(train_scores, train_positive, validation_scores, validation_positive):
    threshold = tune_threshold(train_scores[~train_positive])
    result = metrics(validation_scores[validation_positive], validation_scores[~validation_positive], threshold)
    # JSON uses null when the model makes no positive predictions and precision is undefined.
    result = {name: value if np.isfinite(value) else None for name, value in result.items()}
    return {"threshold": threshold, **result}


def train_model(counts, labels, seed, epochs):
    torch.manual_seed(seed)
    model = EventCountCNN().to(counts.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    positives = labels.sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=(len(labels) - positives) / positives)
    for epoch in range(1, epochs + 1):
        total_loss = torch.zeros((), device=counts.device)
        order = torch.randperm(len(labels)).to(counts.device)
        for indices in order.split(BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(counts[indices]), labels[indices])
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * len(indices)
        mean_loss = total_loss.item() / len(labels)
        if not np.isfinite(mean_loss):
            raise FloatingPointError(f"Non-finite training loss for seed {seed}, epoch {epoch}")
        print(f"seed {seed} epoch {epoch:02}/{epochs}: loss {mean_loss:.4f}", flush=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new directory for run artifacts")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=False)

    train_sequences, validation_sequences = load_splits()
    print(f"Loading FRED windows for {device} training", flush=True)
    train_counts, train_labels, train_rows = load_split(train_sequences, device)
    val_counts, val_labels, val_rows = load_split(validation_sequences, device)
    train_positive = train_labels.cpu().numpy().astype(bool)
    val_positive = val_labels.cpu().numpy().astype(bool)

    references = {}
    combined = load_frozen_threshold()["combined_model"]
    for name, column in {"count": "event_count", "fft_count": f"score_lr_{combined}"}.items():
        train_scores = np.array([float(row[column]) for row in train_rows])
        val_scores = np.array([float(row[column]) for row in val_rows])
        references[name] = evaluate(train_scores, train_positive, val_scores, val_positive)

    results = []
    validation_scores = {}
    for seed in SEEDS:
        model = train_model(train_counts, train_labels, seed, args.epochs)
        scores = predict(model, val_counts)
        result = evaluate(predict(model, train_counts), train_positive, scores, val_positive)
        results.append({"seed": seed, **result})
        validation_scores[f"logit_seed_{seed}"] = scores
        weights = {name: value.cpu() for name, value in model.state_dict().items()}
        torch.save(weights, args.out / f"seed_{seed}.pt")
        print(f"seed {seed}: validation AUC {result['auc']:.4f}", flush=True)

    summary = {}
    for name in ("auc", "precision", "recall", "fa_rate"):
        values = [result[name] for result in results if result[name] is not None]
        summary[name] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "seeds": len(values),
        }
    report = {
        "device": device,
        "torch_version": torch.__version__,
        "epochs": args.epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "input": {"bin_us": BIN_US, "bins": WIN, "transform": "log1p"},
        "training": {"sequences": train_sequences, "windows": len(train_rows)},
        "validation": {"sequences": validation_sequences, "windows": len(val_rows)},
        "train_background_false_alarm_target": FA_TARGET,
        "references": references,
        "seeds": results,
        "validation_summary": summary,
    }
    with (args.out / "validation_scores.csv").open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow([*WINDOW_ID, *validation_scores])
        for row, scores in zip(val_rows, zip(*validation_scores.values(), strict=True), strict=True):
            writer.writerow([*(row[column] for column in WINDOW_ID), *scores])
    (args.out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Results saved to {args.out / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
