"""Score trained packet arms with validation packets dropped at fixed rates.

python -m src.train.rate_shift --runs runs/packets-2026-11-22 --out FILE
Drop rates, seeds and arms were fixed before any result was seen. Every arm sees
the same dropped packets at a given rate (seed 0). Reports AUC and matched AUC.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.count_matched import both_aucs, matched_pairs
from src.data.cnn_windows import load_splits
from src.data.event_packets import FEATURES, load_packets
from src.data.packet_drop import drop_all
from src.data.window_cache import load_set
from src.models.packet_models import PACKET_ARMS
from src.train.cnn import SEEDS
from src.train.packets import predict, tensors

RATES = (0.0, 0.25, 0.5, 0.75, 0.9)
ARMS = ("time_cfc", "time_cfc_const", "time_mmcfc", "time_mmcfc_const", "time_gru", "time_gru_const",
        "time_lstm", "time_gru_aug", "time_mmcfc_aug", "time_gru_nocount")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("phase1", "phase3"), default="phase1")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(8)
    train_seq, val_seq = load_splits(args.split)
    _, train_count, _, _ = load_packets(train_seq)
    stats = (train_count.mean(), train_count.std())
    packets, log_count, y, rows = load_packets(val_seq)
    pairs = matched_pairs(*load_set(val_seq))
    dropped = {rate: drop_all(packets, log_count, rate, seed=0) for rate in RATES}
    report = {"rates": RATES, "matched_pairs": len(pairs[0]), "arms": {}}
    for arm in ARMS:
        if not (args.runs / arm / "results.json").is_file():
            print(f"{arm}: missing or unfinished, skipped", flush=True)
            continue
        feature_set, constant_gap, build = PACKET_ARMS[arm.removesuffix("_aug")]
        models = []
        for seed in SEEDS:
            model = build(len(FEATURES[feature_set]))
            model.load_state_dict(torch.load(args.runs / arm / f"seed_{seed}.pt", map_location="cpu"))
            models.append(model)
        report["arms"][arm] = {}
        for rate in RATES:
            store = tensors(*dropped[rate], feature_set, constant_gap, stats)
            results = [both_aucs(predict(model, store), y, pairs) for model in models]
            report["arms"][arm][str(rate)] = {k: [r[k] for r in results] for k in results[0]}
            print(f"{arm:17} drop {rate:.2f}: AUC {np.mean([r['auc'] for r in results]):.3f}"
                  f" matched {np.mean([r['matched_auc'] for r in results]):.3f}", flush=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
