"""Train one event-packet arm. Run: python -m src.train.packets --arm time_cfc --out DIR

With --drop-augment R, every training window loses packets at a rate drawn
uniformly from 0 to R, freshly each epoch (src/data/packet_drop.py).

Fixed before any result was seen: 40 epochs (12 was shown to undertrain recurrent
models), batch 128, AdamW at 0.001, gradient norm clipped at 1, last epoch kept,
seeds 0-4, no tuning on validation.
--parallel 5 --threads 1 trains the seeds at once; results match a serial run up to float rounding. Log counts are standardised with training statistics.
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.count_matched import both_aucs, matched_pairs
from src.data.cnn_windows import load_splits
from src.data.event_packets import FEATURES, MAX_PACKETS, load_packets
from src.data.packet_drop import drop
from src.data.window_cache import load_set
from src.models.packet_models import PACKET_ARMS
from src.phase1_multiseq import metrics, tune_threshold
from src.train.cnn import BATCH_SIZE, LEARNING_RATE, SEEDS
from src.train.recurrent import CLIP


def tensors(packets, log_count, feature_set, constant_gap, count_stats):
    """Pad to [N, MAX_PACKETS, features] once; batches slice it to their longest window."""
    columns = FEATURES[feature_set]
    data = np.zeros((len(packets), MAX_PACKETS, len(columns)), dtype=np.float32)
    gap = np.zeros((len(packets), MAX_PACKETS), dtype=np.float32)
    length = np.array([len(p) for p in packets])
    for i, p in enumerate(packets):
        data[i, :len(p)] = p[:, columns]
        gap[i, :len(p)] = p[:, 0]
    if constant_gap:
        gap[:] = 1.0
    gap = np.log1p(gap)
    data[..., 0] = gap
    mean, std = count_stats
    count = torch.from_numpy((log_count - mean) / std)
    return (torch.from_numpy(gap), torch.from_numpy(data), torch.from_numpy(length), count)


def batch(store, index):
    gap, data, length, count = store
    longest = int(length[index].max())
    mask = torch.arange(longest)[None] < length[index][:, None]
    return gap[index, :longest], data[index, :longest], mask, count[index]


@torch.no_grad()  # not inference_mode: TorchScript reuses graphs profiled under it, which then fail in training
def predict(model, store):
    model.eval()
    order = torch.argsort(store[2])
    scores = torch.empty(len(order))
    for index in order.split(256):
        scores[index] = model(*batch(store, index))
    return scores.numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=PACKET_ARMS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--drop-augment", type=float, default=0.0,
                        help="train with each window's packets dropped at a rate drawn from 0 to this value")
    parser.add_argument("--split", choices=("phase1", "phase3"), default="phase1")
    parser.add_argument("--parallel", type=int, default=1, help="seeds trained at once, one process each")
    parser.add_argument("--script", action="store_true",
                        help="TorchScript the model: same maths, about 20%% faster for the CfC arms")
    args = parser.parse_args()
    out = args.out / (args.arm + ("_aug" if args.drop_augment else ""))
    out.mkdir(parents=True, exist_ok=False)
    feature_set, constant_gap, _ = PACKET_ARMS[args.arm]
    train_seq, val_seq = load_splits(args.split)
    jobs = [(vars(args), seed) for seed in SEEDS]
    if args.parallel > 1:
        with ProcessPoolExecutor(args.parallel) as pool:
            done = list(pool.map(train_seed, *zip(*jobs)))
    else:
        done = [train_seed(*job) for job in jobs]
    runs, all_scores = [], {}
    for run, scores, state, parameters, n_pairs in done:
        runs.append(run)
        all_scores[f"seed_{run['seed']}"] = scores
        torch.save(state, out / f"seed_{run['seed']}.pt")
    np.savez(out / "validation_scores.npz", **all_scores)
    summary = {name: {"mean": float(np.mean([r[name] for r in runs])),
                      "std": float(np.std([r[name] for r in runs], ddof=1))}
               for name in ("auc", "matched_auc", "precision", "recall", "fa_per_min", *(f"auc {s}" for s in val_seq))}
    (out / "results.json").write_text(json.dumps({
        "arm": args.arm, "split": args.split, "drop_augment": args.drop_augment, "feature_set": feature_set, "constant_gap": constant_gap,
        "epochs": args.epochs, "parameters": parameters,
        "learning_rate": args.lr, "batch_size": BATCH_SIZE, "clip": CLIP, "threads": args.threads, "script": args.script,
        "training": train_seq, "validation": val_seq, "matched_pairs": n_pairs,
        "runs": runs, "validation_summary": summary}, indent=2) + "\n")


@cache
def setup(arm, split, threads):
    """Everything a seed needs, loaded once per process."""
    torch.set_num_threads(threads)
    feature_set, constant_gap, _ = PACKET_ARMS[arm]
    train_seq, val_seq = load_splits(split)
    train_packets, train_count, train_y, _ = load_packets(train_seq)
    val_packets, val_count, val_y, val_rows = load_packets(val_seq)
    stats = (train_count.mean(), train_count.std())
    train_store = tensors(train_packets, train_count, feature_set, constant_gap, stats)
    val_store = tensors(val_packets, val_count, feature_set, constant_gap, stats)
    # Pairs are made from the same count windows as every earlier result.
    val_counts, val_labels, pair_rows = load_set(val_seq)
    assert [r["t_start_us"] for r in pair_rows] == [r["t_start_us"] for r in val_rows]
    pairs = matched_pairs(val_counts, val_labels, pair_rows)
    return val_seq, train_packets, train_count, train_y, train_store, val_store, val_y, val_rows, stats, pairs


def train_seed(args, seed):
    """Train one seed. Returns (run, validation scores, state dict, parameter count, matched pairs)."""
    args = argparse.Namespace(**args)
    (val_seq, train_packets, train_count, train_y, train_store, val_store,
     val_y, val_rows, stats, pairs) = setup(args.arm, args.split, args.threads)
    feature_set, constant_gap, build = PACKET_ARMS[args.arm]
    targets = torch.from_numpy(train_y.astype(np.float32))
    criterion = nn.BCEWithLogitsLoss(pos_weight=(len(targets) - targets.sum()) / targets.sum())
    sequence = np.array([r["sequence"] for r in val_rows])
    torch.manual_seed(seed)
    augment = np.random.default_rng(1000 + seed)
    model = build(len(FEATURES[feature_set]))
    if args.script:
        model = torch.jit.script(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses, start = [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for index in torch.randperm(len(targets)).split(BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            if args.drop_augment:
                picked = [drop(train_packets[i], np.expm1(train_count[i]).round(),
                               augment.uniform(0, args.drop_augment), augment) for i in index.tolist()]
                store = tensors([p for p, _ in picked], np.log1p([n for _, n in picked]).astype(np.float32),
                                feature_set, constant_gap, stats)
                inputs = batch(store, torch.arange(len(index)))
            else:
                inputs = batch(train_store, index)
            loss = criterion(model(*inputs), targets[index])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            total += loss.item() * len(index)
        losses.append(total / len(targets))
        if not np.isfinite(losses[-1]):
            raise FloatingPointError(f"{args.arm} seed {seed}: non-finite loss")
        if epoch % 5 == 0:
            print(f"{args.arm} seed {seed} epoch {epoch}/{args.epochs}: loss {losses[-1]:.4f} "
                  f"({(time.time() - start) / epoch:.1f} s/epoch)", flush=True)
    threshold = tune_threshold(predict(model, train_store)[~train_y])
    scores = predict(model, val_store)
    result = metrics(scores[val_y], scores[~val_y], threshold)
    result.update(both_aucs(scores, val_y, pairs))
    for s in val_seq:
        m = sequence == s
        result[f"auc {s}"] = float(both_aucs(scores[m], val_y[m], (np.flatnonzero(val_y[m]), np.flatnonzero(~val_y[m])))["auc"])
    print(f"{args.arm} seed {seed}: AUC {result['auc']:.4f}, matched {result['matched_auc']:.4f}", flush=True)
    run = {"seed": seed, "threshold": threshold, "losses": losses, **result}
    return run, scores, model.state_dict(), sum(p.numel() for p in model.parameters()), len(pairs[0])


if __name__ == "__main__":
    main()
