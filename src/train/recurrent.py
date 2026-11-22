"""Train one recurrent arm on the Phase 1 windows, with the CNN's protocol.

Run from the repo root: python -m src.train.recurrent --arm cfc_dt --out runs/recurrent
Fixed before any result was seen: 12 epochs, batch 128, AdamW at 0.001, gradient
norm clipped to 1, last epoch kept, seeds 0-4. No tuning on validation.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.count_matched import both_aucs, matched_pairs
from src.data.cnn_windows import load_splits
from src.data.event_tokens import collate, tokenise
from src.data.window_cache import load_set
from src.models.recurrent import ARMS
from src.phase1_multiseq import metrics, tune_threshold
from src.train.cnn import BATCH_SIZE, LEARNING_RATE, SEEDS

CLIP = 1.0


@torch.inference_mode()
def predict(model, tokens, constant_gap):
    model.eval()
    order = np.argsort([len(values) for values, _ in tokens])      # less padding
    scores = np.empty(len(tokens), dtype=np.float32)
    for start in range(0, len(order), 256):
        index = order[start:start + 256]
        scores[index] = model(*collate([tokens[i] for i in index], constant_gap)).numpy()
    return scores


def train_model(arm, tokens, labels, seed, epochs):
    build, constant_gap = ARMS[arm]
    torch.manual_seed(seed)
    model = build()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    targets = torch.from_numpy(labels.astype(np.float32))
    criterion = nn.BCEWithLogitsLoss(pos_weight=(len(labels) - targets.sum()) / targets.sum())
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        start, total = time.time(), 0.0
        for index in torch.randperm(len(labels)).split(BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            batch = collate([tokens[i] for i in index.tolist()], constant_gap)
            loss = criterion(model(*batch), targets[index])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            total += loss.item() * len(index)
        losses.append(total / len(labels))
        if not np.isfinite(losses[-1]):
            raise FloatingPointError(f"{arm} seed {seed} epoch {epoch}: non-finite loss")
        print(f"{arm} seed {seed} epoch {epoch:02}/{epochs}: loss {losses[-1]:.4f} "
              f"({time.time() - start:.0f} s)", flush=True)
    return model, losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    out = args.out / args.arm
    out.mkdir(parents=True, exist_ok=False)

    train_sequences, val_sequences = load_splits()
    train_counts, train_labels, _ = load_set(train_sequences)
    val_counts, val_labels, val_rows = load_set(val_sequences)
    train_tokens = [tokenise(window) for window in train_counts]
    val_tokens = [tokenise(window) for window in val_counts]
    pairs = matched_pairs(val_counts, val_labels, val_rows)

    runs, all_scores = [], {}
    for seed in SEEDS:
        model, losses = train_model(args.arm, train_tokens, train_labels, seed, args.epochs)
        constant_gap = ARMS[args.arm][1]
        threshold = tune_threshold(predict(model, train_tokens, constant_gap)[~train_labels])
        scores = predict(model, val_tokens, constant_gap)
        result = metrics(scores[val_labels], scores[~val_labels], threshold)
        result.update(both_aucs(scores, val_labels, pairs))
        runs.append({"seed": seed, "threshold": threshold, "losses": losses, **result})
        all_scores[f"seed_{seed}"] = scores
        torch.save(model.state_dict(), out / f"seed_{seed}.pt")
        print(f"{args.arm} seed {seed}: AUC {result['auc']:.4f}, matched {result['matched_auc']:.4f}", flush=True)
    np.savez(out / "validation_scores.npz", **all_scores)
    summary = {name: {"mean": float(np.mean([r[name] for r in runs])),
                      "std": float(np.std([r[name] for r in runs], ddof=1))}
               for name in ("auc", "matched_auc", "precision", "recall", "fa_rate")}
    report = {"arm": args.arm, "epochs": args.epochs, "batch_size": BATCH_SIZE,
              "learning_rate": LEARNING_RATE, "clip": CLIP, "torch_version": torch.__version__,
              "parameters": sum(p.numel() for p in model.parameters()),
              "training": train_sequences, "validation": val_sequences,
              "matched_pairs": len(pairs[0]), "runs": runs, "validation_summary": summary}
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
