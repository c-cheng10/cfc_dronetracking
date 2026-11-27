"""Stage A: grid over finder settings on the Phase 3 training recordings.

python -m src.finder_tune --out FILE

For each setting: share of drone windows with a proposal on the drone (the
recall ceiling for any scorer behind the finder), and proposals per minute that
hit no drone. Validation recordings are not touched.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from src import finder2
from src import phase1_find as finder
from src.data.cnn_windows import load_splits

GRID = {"mean": [8.0, 16.0], "std": [3.0, 5.0, 8.0], "history": [3.0, 5.0, 8.0]}
MAX_CANDS = [1, 3, 5]
BASE = {"cell_px": 48, "min_count": 10, "hot_frac": 0.9}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    settings = [dict(BASE, mode=m, k=k, max_cands=c) for m, ks in GRID.items() for k in ks for c in MAX_CANDS]
    totals = {i: {"found": 0, "drone_windows": 0, "false": 0, "minutes": 0.0} for i in range(len(settings))}
    per_seq = {}
    for name in load_splits("phase3")[0]:
        ev = np.load(finder.seq_dir(name) / "Event/events.npy", mmap_mode="r")
        x, y, t = (np.asarray(ev[c]).astype(np.int64 if c == "t" else np.int32) for c in ("x", "y", "t"))
        tracks = finder.load_tracks(name)
        window_grids = finder2.grids(x, y, t, BASE["cell_px"], BASE["hot_frac"])
        per_seq[name] = []
        for i, prm in enumerate(settings):
            res = finder2.run_sequence(name, prm, (x, y, t), tracks, window_grids)
            found = sum(g[-1] > 0 for g in res["gt"])
            false = int((res["cands"][:, 8] < 0).sum())
            for key, val in (("found", found), ("drone_windows", len(res["gt"])), ("false", false), ("minutes", res["minutes"])):
                totals[i][key] += val
            per_seq[name].append(found / max(len(res["gt"]), 1))
        print(f"{name} done", flush=True)
    rows = []
    for i, prm in enumerate(settings):
        tot = totals[i]
        rows.append({**prm, "recall": tot["found"] / tot["drone_windows"], "false_per_min": tot["false"] / tot["minutes"],
                     "worst_recordings": sorted(((per_seq[n][i], n) for n in per_seq))[:3]})
    args.out.write_text(json.dumps(rows, indent=1) + "\n")
    for r in sorted(rows, key=lambda r: -r["recall"]):
        print(f"{r['mode']:8} k={r['k']:<5} boxes={r['max_cands']}  recall {r['recall']:.3f}  false/min {r['false_per_min']:7.1f}"
              f"  worst {[(n, round(v, 2)) for v, n in r['worst_recordings']]}")


if __name__ == "__main__":
    main()
