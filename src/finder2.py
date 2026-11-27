"""Box finder with more proposals and noise-robust thresholds (variants of src/phase1_find.py).

Threshold rules for a 0.5 s window's grid of cell counts:
- "mean":    cell >= max(min_count, k * mean of the window's cells)   (Phase 1's rule, k = 16)
- "std":     cell >= max(min_count, mean + k * std of the window's cells)
- "history": cell >= max(min_count, h + k * sqrt(h + 1)), where h is that cell's median
             count over the previous HISTORY_WINDOWS non-overlapping windows (2 s).
             Static clutter and steady noise raise h, so they stop producing boxes. A drone
             that hovers in one cell for longer than 2 s also raises h; that is a known risk.
Proposals are ranked by count above the threshold, and the top max_cands are kept.
The output matches phase1_find.run_sequence, so src/finder_gru.py can score it.
"""

import time

import numpy as np

from src import phase1_find as finder

HISTORY_WINDOWS = 4          # previous 0.5 s windows, i.e. 2 s


def grids(x, y, t, cell_px, hot_frac):
    hot = finder.hot_pixel_mask(x, y, t, hot_frac)
    rows, cols = -(-finder.HEIGHT // cell_px), -(-finder.WIDTH // cell_px)
    cold = ~hot[y * finder.WIDTH + x]
    slot = t // finder.HOP_US
    n_slots = int(slot.max()) + 1
    cell = (y // cell_px) * cols + (x // cell_px)
    flat = np.bincount(slot[cold] * (rows * cols) + cell[cold], minlength=n_slots * rows * cols)
    slots = flat.reshape(n_slots, rows, cols).astype(float)
    return slots[:-1] + slots[1:]                    # window w covers slots w and w+1


def propose(window_grids, w, prm):
    grid = window_grids[w]
    k, mode = prm["k"], prm["mode"]
    if mode == "mean":
        thr = np.maximum(prm["min_count"], k * grid.mean())
    elif mode == "std":
        thr = np.maximum(prm["min_count"], grid.mean() + k * grid.std())
    else:
        past = [w - 2 * j for j in range(1, HISTORY_WINDOWS + 1) if w - 2 * j >= 0]
        h = np.median(window_grids[past], axis=0) if past else np.zeros_like(grid)
        thr = np.maximum(prm["min_count"], h + k * np.sqrt(h + 1))
    dense = grid >= thr
    if not dense.any():
        return np.zeros((0, 4)), np.zeros(0)
    excess = np.where(dense, grid - thr, 0.0)
    lab = finder.connected_labels(dense)
    rr, cc = np.nonzero(dense)
    ids, inv = np.unique(lab[rr, cc], return_inverse=True)
    counts = np.bincount(inv, weights=grid[rr, cc])
    score = np.bincount(inv, weights=excess[rr, cc])
    r0 = np.full(len(ids), 10**9); r1 = np.full(len(ids), -1)
    c0 = np.full(len(ids), 10**9); c1 = np.full(len(ids), -1)
    np.minimum.at(r0, inv, rr); np.maximum.at(r1, inv, rr)
    np.minimum.at(c0, inv, cc); np.maximum.at(c1, inv, cc)
    # Phase 1 ranked by count; the other rules rank by count above the threshold.
    order = np.argsort(-(counts if mode == "mean" else score))[:prm["max_cands"]]
    s = prm["cell_px"]
    boxes = np.stack([c0[order] * s, r0[order] * s, np.minimum((c1[order] + 1) * s, finder.WIDTH),
                      np.minimum((r1[order] + 1) * s, finder.HEIGHT)], axis=1).astype(float)
    return boxes, counts[order]


def run_sequence(name, prm, ev, tracks, window_grids=None):
    """Same output as phase1_find.run_sequence, without the FFT (prominence columns are 0)."""
    x, y, t = ev
    started = time.perf_counter()
    if window_grids is None:
        window_grids = grids(x, y, t, prm["cell_px"], prm["hot_frac"])
    win_starts = np.arange(0, int(t[-1]) - finder.WIN_US + 1, finder.HOP_US)
    cand_rows, gt_rows = [], []
    for w, w0 in enumerate(win_starts):
        boxes, cnts = propose(window_grids, w, prm)
        gts = finder.gt_in_window(tracks, w0, w0 + finder.WIN_US)
        matched = np.full(len(boxes), -1)
        ov = None
        if gts and len(boxes):
            ov = finder.iou(boxes, np.array([g[2] for g in gts]))
            best, hit = ov.argmax(axis=1), ov.max(axis=1) > finder.IOU_MATCH
            matched[hit] = [gts[j][0] for j in best[hit]]
        for j in range(len(boxes)):
            cand_rows.append((w, *boxes[j], cnts[j], 0.0, 0.0, matched[j]))
        for j, (i, cls, gbox, nf) in enumerate(gts):
            if nf >= finder.MIN_GT_FRAMES:
                found = ov is not None and bool((ov[:, j] > finder.IOU_MATCH).any())
                gt_rows.append((w, i, cls, *gbox, 1.0 if found else -1.0))
    return dict(name=name, cands=np.array(cand_rows, dtype=float).reshape(-1, 9), gt=gt_rows,
                n_windows=len(win_starts), minutes=len(win_starts) * finder.HOP_US / 60e6,
                runtime=time.perf_counter() - started, win_starts=win_starts)
