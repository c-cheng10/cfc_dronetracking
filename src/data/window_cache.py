"""Cache the verified count windows so several training processes can share them."""

import json

import numpy as np

from src import phase1_fft as phase1
from src.data.cnn_windows import load_windows

CACHE = phase1.REPO / "runs" / "window-cache"


def load_set(sequences):
    """Return counts [N, 1000], bool labels, and source rows for whole recordings."""
    CACHE.mkdir(parents=True, exist_ok=True)
    counts, rows = [], []
    for sequence in sequences:
        path = CACHE / f"{sequence.replace('/', '_')}.npz"
        stat = phase1.csv_path(sequence).stat()
        key = np.array([stat.st_size, stat.st_mtime_ns])
        if path.is_file():
            with np.load(path) as data:
                if np.array_equal(data["key"], key):
                    counts.append(data["counts"])
                    rows.extend(json.loads(str(data["rows"])))
                    continue
        windows, sequence_rows = load_windows(sequence)
        np.savez_compressed(path, key=key, counts=windows, rows=json.dumps(sequence_rows))
        counts.append(windows)
        rows.extend(sequence_rows)
    labels = np.array([int(row["label"]) for row in rows], dtype=bool)
    return np.concatenate(counts), labels, rows
