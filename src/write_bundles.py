"""Write seed_N.json bundles for packet runs trained before packets.py saved them.

python -m src.write_bundles RUN_DIR [RUN_DIR ...]     (each RUN_DIR holds results.json and seed_N.pt)

Count statistics come from the run's own training recordings, computed as packets.py does;
the threshold is the one results.json already records for each seed.
"""

import json
import sys
from pathlib import Path

import numpy as np

from src.data.event_packets import load_packets
from src.train.event_shift import train_count_stats
from src.train.packets import write_bundle


def main():
    for run_dir in map(Path, sys.argv[1:]):
        results = json.loads((run_dir / "results.json").read_text())
        _, log_count, _, _ = load_packets(results["training"])
        mean, std = float(log_count.mean()), float(log_count.std())
        # self-check: an independent route (the window CSVs) must give the same statistics
        assert np.allclose((mean, std), train_count_stats(results["training"]), atol=1e-5), run_dir
        for run in results["runs"]:
            write_bundle(run_dir, {**run, "count_mean": mean, "count_std": std},
                         results["arm"], results["feature_set"], results["learning_rate"])
        print(f"{run_dir}: {len(results['runs'])} bundles, count mean {mean:.4f} std {std:.4f}", flush=True)


if __name__ == "__main__":
    main()
