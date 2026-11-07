"""Phase 1 finder: locate drones WITHOUT FRED's boxes, then FFT-score each guess.

The box-based detector (src/phase1_fft.py) answers "is there a rotor in this
box?" but needs FRED's annotated boxes. A live demo has no boxes, so this script
proposes its own:

  1. Hot pixels. A few sensor pixels fire all the time. We flag any pixel that
     fires in at least HOT_FRAC of the 1-second blocks of the recording and
     ignore it when looking for clusters.
  2. Density grid. For each 0.5 s window (0.25 s step, same windows as the FFT)
     we split the 1280x720 image into square cells (CELL_PX wide) and count the
     events in each cell.
  3. Dense cells. A cell is "dense" if its count is at least K_MEAN times the
     window's average cell count, and at least MIN_COUNT. Using the window's own
     average makes the threshold follow the overall event rate (busy scene vs
     quiet sky).
  4. Clusters. Neighbouring dense cells (8-connected: sides and corners) are
     merged into one cluster by repeated "take the smallest label among my
     neighbours" passes (a flood fill written with NumPy arrays).
  5. Candidates. Each cluster becomes one box (its cells' outer edges). We keep
     the MAX_CANDS clusters with the most events. The box is fixed for the whole
     0.5 s window, so for a moving drone it covers the path swept in that time.
  6. FFT score (locked spec, identical to the box-based detector): count the
     events inside the candidate box in 0.5 ms bins over the window, subtract
     the mean, apply a Hann taper, np.fft.rfft, and take
         prominence = max magnitude / median magnitude, both over 50-1000 Hz.
     The FFT uses ALL events in the box (hot pixels included), exactly as the
     box-based detector does, so the two scores compare.

Evaluation (against coordinates.txt, used ONLY for scoring):
  * A drone is "present" in a window if it has at least MIN_GT_FRAMES of the
    ~15 annotation frames (30 Hz) inside it. Its window box is the union of
    those frames' boxes (the swept box), to match how candidate boxes are made.
  * A candidate MATCHES a drone if IoU(candidate box, swept drone box) > 0.1.
    Any drone with at least one annotation in the window can be matched, so a
    candidate on a briefly annotated drone is not called false.
  * Candidate recall  = share of "present" windows with a matching candidate.
  * Detection recall  = same, counting only candidates with prominence > threshold.
  * False candidates  = candidates matching no drone, per minute of recording,
    before and after the FFT threshold. (One window starts every 0.25 s, so a
    minute has 240 windows.)
  * Precision         = matching / all, among candidates that pass the threshold.

Sets come from experiments/phase1_sequences.json. Parameters are picked on the
"tune" sequences only, frozen in experiments/phase1_find_params.json, then
applied unchanged to "heldout" and "challenging".

Run from the repo root:
    .venv/bin/python -I src/phase1_find.py tune      # grid search, freezes params
    .venv/bin/python -I src/phase1_find.py report    # heldout + challenging with frozen params
"""

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data/FRED"
OUT = REPO / "experiments"
WIDTH, HEIGHT = 1280, 720

# ---- locked FFT spec (same as src/phase1_fft.py) ---------------------------
BIN_US = 500                      # 0.5 ms bins -> 2000 samples per second
FS = 1e6 / BIN_US
WIN_US = 500_000                  # 0.5 s window
HOP_US = 250_000                  # 0.25 s step
F_LO, F_HI = 50.0, 1000.0         # band for peak and median

# ---- finder settings (defaults; the tuned values go to the params json) ----
DEFAULTS = dict(
    cell_px=16,      # side of one grid cell, pixels
    k_mean=4.0,      # dense if count >= k_mean * (window's mean cell count) ...
    min_count=20,    # ... and count >= this many events
    max_cands=5,     # keep at most this many clusters per window (most events first)
    hot_frac=0.9,    # hot pixel: fires in >= this share of 1-second blocks
)
MIN_GT_FRAMES = 12   # drone counts as present if >= 12 of ~15 frames annotated
IOU_MATCH = 0.1
THRESHOLDS = np.round(np.arange(2.0, 15.01, 0.25), 2)   # FFT thresholds swept


# ============================================================================
# 1. Loading
# ============================================================================
def seq_dir(name):
    """'train/91' -> data/FRED/train/91/91"""
    split, num = name.split("/")
    return DATA / split / num / num


def load_events(name):
    """Read x, y, t (t in microseconds) into memory as plain arrays."""
    ev = np.load(seq_dir(name) / "Event/events.npy", mmap_mode="r")
    x = np.asarray(ev["x"]).astype(np.int32)
    y = np.asarray(ev["y"]).astype(np.int32)
    t = np.asarray(ev["t"]).astype(np.int64)
    return x, y, t


def load_tracks(name):
    """coordinates.txt -> {drone_id: {"t": us, "box": (n,4) x1,y1,x2,y2, "cls": name}}.
    Boxes are clipped to the image, as in the box-based detector."""
    rows, cls = [], {}
    for line in open(seq_dir(name) / "coordinates.txt"):
        t_str, rest = line.split(":", 1)
        f = [s.strip() for s in rest.split(",")]
        i = int(float(f[4]))
        cls[i] = f[5]
        rows.append([float(t_str), float(f[0]), float(f[1]), float(f[2]), float(f[3]), i])
    a = np.array(rows)
    tracks = {}
    for i in np.unique(a[:, 5]).astype(int):
        b = a[a[:, 5] == i]
        b = b[np.argsort(b[:, 0])]
        box = np.stack([np.clip(b[:, 1], 0, WIDTH), np.clip(b[:, 2], 0, HEIGHT),
                        np.clip(b[:, 3], 0, WIDTH), np.clip(b[:, 4], 0, HEIGHT)], axis=1)
        tracks[i] = {"t": np.round(b[:, 0] * 1e6).astype(np.int64), "box": box, "cls": cls[i]}
    return tracks


# ============================================================================
# 2. Hot pixels
# ============================================================================
def hot_pixel_mask(x, y, t, hot_frac):
    """Boolean array (HEIGHT*WIDTH,), True for pixels that fire in at least
    hot_frac of all 1-second blocks. A moving drone never stays on one pixel
    that long; a defective pixel does.
    (Offline shortcut: we use the whole recording. Live, you would measure this
    once from a short calibration clip.)"""
    pix = y * WIDTH + x
    sec = t // 1_000_000
    n_sec = int(sec.max()) + 1
    pairs = np.unique(sec * (WIDTH * HEIGHT) + pix)        # each (second, pixel) once
    active_secs = np.bincount(pairs % (WIDTH * HEIGHT), minlength=WIDTH * HEIGHT)
    return active_secs >= hot_frac * n_sec


# ============================================================================
# 3. Clustering one window
# ============================================================================
def connected_labels(mask):
    """Label 8-connected groups of True cells in a 2-D boolean array.

    Start with every True cell holding its own number. Repeatedly replace each
    cell's number by the smallest number among itself and its 8 neighbours.
    When nothing changes, all cells of one group share the same number.
    False cells hold a huge number and never spread."""
    big = mask.size + 1
    lab = np.where(mask, np.arange(mask.size).reshape(mask.shape), big)
    while True:
        p = np.pad(lab, 1, constant_values=big)
        h, w = mask.shape
        neigh = np.min([p[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)], axis=0)
        new = np.where(mask, neigh, big)
        if np.array_equal(new, lab):
            return lab
        lab = new


def find_candidates(grid, prm):
    """grid: (rows, cols) event counts for one window.
    Returns boxes (n,4) in pixels [x1,y1,x2,y2) and their event counts,
    at most prm['max_cands'] of them, largest count first."""
    thr = max(prm["min_count"], prm["k_mean"] * grid.mean())
    dense = grid >= thr
    if not dense.any():
        return np.zeros((0, 4)), np.zeros(0)
    lab = connected_labels(dense)
    rr, cc = np.nonzero(dense)
    ids, inv = np.unique(lab[rr, cc], return_inverse=True)
    counts = np.bincount(inv, weights=grid[rr, cc])
    r0 = np.full(len(ids), 10**9); r1 = np.full(len(ids), -1)
    c0 = np.full(len(ids), 10**9); c1 = np.full(len(ids), -1)
    np.minimum.at(r0, inv, rr); np.maximum.at(r1, inv, rr)
    np.minimum.at(c0, inv, cc); np.maximum.at(c1, inv, cc)
    order = np.argsort(-counts)[:prm["max_cands"]]
    s = prm["cell_px"]
    boxes = np.stack([c0[order] * s, r0[order] * s,
                      np.minimum((c1[order] + 1) * s, WIDTH),
                      np.minimum((r1[order] + 1) * s, HEIGHT)], axis=1).astype(float)
    return boxes, counts[order]


# ============================================================================
# 4. FFT prominence (locked spec)
# ============================================================================
N_WIN_BINS = WIN_US // BIN_US                                  # 1000
FREQS = np.fft.rfftfreq(N_WIN_BINS, d=1 / FS)
BAND = (FREQS >= F_LO) & (FREQS <= F_HI)
HANN = np.hanning(N_WIN_BINS)


def prominence(counts):
    """counts: (n, 1000) events per 0.5 ms bin. Returns prominence and peak
    frequency for each row."""
    seg = counts - counts.mean(axis=1, keepdims=True)
    mag = np.abs(np.fft.rfft(seg * HANN, axis=1))[:, BAND]
    med = np.median(mag, axis=1)
    prom = np.where(med > 0, mag.max(axis=1) / np.where(med > 0, med, 1), 0.0)
    return prom, FREQS[BAND][np.argmax(mag, axis=1)]


# ============================================================================
# 5. Ground truth per window
# ============================================================================
def iou(a, b):
    """IoU between every box in a (n,4) and every box in b (m,4) -> (n,m)."""
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def gt_in_window(tracks, w0, w1):
    """List of (drone_id, class, swept box, n_frames) for drones annotated in [w0, w1)."""
    out = []
    for i, tr in tracks.items():
        m = (tr["t"] >= w0) & (tr["t"] < w1)
        if m.any():
            b = tr["box"][m]
            out.append((i, tr["cls"], np.array([b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()]),
                        int(m.sum())))
    return out


# ============================================================================
# 6. One sequence
# ============================================================================
def run_sequence(name, prm, ev=None, tracks=None):
    """Find candidates and FFT-score them in every window of one sequence.

    Returns a dict with
      cands: one row per candidate, columns: window, x1, y1, x2, y2, count,
             prominence, peak_f, matched drone id (-1 = none)
      gt:    one row per (window, drone) that is "present", with the best
             prominence among candidates matching it (-1 if none match)
      minutes, runtime numbers."""
    if ev is None:
        ev = load_events(name)
    if tracks is None:
        tracks = load_tracks(name)
    x, y, t = ev
    t_start = time.perf_counter()

    hot = hot_pixel_mask(x, y, t, prm["hot_frac"])
    s = prm["cell_px"]
    rows, cols = -(-HEIGHT // s), -(-WIDTH // s)               # ceiling division
    cell = (y // s) * cols + (x // s)                           # grid cell of each event
    cold = ~hot[y * WIDTH + x]                                  # not a hot pixel

    # Cell counts per 0.25 s slot; a 0.5 s window = two consecutive slots.
    slot = t // HOP_US
    n_slots = int(slot.max()) + 1
    flat = np.bincount(slot[cold] * (rows * cols) + cell[cold], minlength=n_slots * rows * cols)
    slot_grid = flat.reshape(n_slots, rows, cols).astype(float)

    win_starts = np.arange(0, int(t[-1]) - WIN_US + 1, HOP_US)
    cand_rows, gt_rows = [], []
    for w, w0 in enumerate(win_starts):
        w1 = w0 + WIN_US
        grid = slot_grid[w] + slot_grid[w + 1]
        boxes, cnts = find_candidates(grid, prm)

        # FFT: events of this window inside each candidate box, 0.5 ms bins.
        if len(boxes):
            i0, i1 = np.searchsorted(t, [w0, w1])
            xs, ys, b = x[i0:i1], y[i0:i1], (t[i0:i1] - w0) // BIN_US
            sig = np.zeros((len(boxes), N_WIN_BINS))
            for j, (bx1, by1, bx2, by2) in enumerate(boxes):
                inside = (xs >= bx1) & (xs < bx2) & (ys >= by1) & (ys < by2)
                sig[j] = np.bincount(b[inside], minlength=N_WIN_BINS)[:N_WIN_BINS]
            prom, pf = prominence(sig)
        else:
            prom, pf = np.zeros(0), np.zeros(0)

        # Match against ground truth (used for scoring only).
        gts = gt_in_window(tracks, w0, w1)
        matched = np.full(len(boxes), -1)
        if gts and len(boxes):
            ov = iou(boxes, np.array([g[2] for g in gts]))
            best = ov.argmax(axis=1)
            hit = ov.max(axis=1) > IOU_MATCH
            matched[hit] = [gts[k][0] for k in best[hit]]
        for j in range(len(boxes)):
            cand_rows.append((w, *boxes[j], cnts[j], prom[j], pf[j], matched[j]))
        for k, (i, cls, gbox, nf) in enumerate(gts):
            if nf >= MIN_GT_FRAMES:
                if len(boxes):
                    m = ov[:, k] > IOU_MATCH
                    best_prom = prom[m].max() if m.any() else -1.0
                else:
                    best_prom = -1.0
                gt_rows.append((w, i, cls, *gbox, best_prom))

    runtime = time.perf_counter() - t_start
    cands = np.array(cand_rows, dtype=float).reshape(-1, 9)
    return dict(name=name, cands=cands, gt=gt_rows, n_windows=len(win_starts),
                minutes=len(win_starts) * HOP_US / 60e6, runtime=runtime,
                data_s=(int(t[-1]) - int(t[0])) / 1e6, n_hot=int(hot.sum()), win_starts=win_starts)


# ============================================================================
# 7. Metrics
# ============================================================================
def metrics(results, thr, cls_filter=None):
    """Pool results (list of run_sequence outputs) and compute the metrics at
    FFT threshold thr. cls_filter: only drones of this class count for recall,
    and only sequences containing that class count for false/min and precision."""
    n_gt = n_cand_hit = n_det_hit = 0
    false_all = false_pass = pass_all = pass_hit = 0
    minutes = 0.0
    for r in results:
        gts = [g for g in r["gt"] if cls_filter is None or g[2] == cls_filter]
        if cls_filter is not None and not gts:
            continue
        best = np.array([g[-1] for g in gts]) if gts else np.zeros(0)
        n_gt += len(best)
        n_cand_hit += int((best >= 0).sum())
        n_det_hit += int((best > thr).sum())
        c = r["cands"]
        prom, matched = c[:, 6], c[:, 8]
        false_all += int((matched < 0).sum())
        false_pass += int(((matched < 0) & (prom > thr)).sum())
        pass_all += int((prom > thr).sum())
        pass_hit += int(((matched >= 0) & (prom > thr)).sum())
        minutes += r["minutes"]
    nan = float("nan")
    return dict(windows=n_gt,
                cand_recall=n_cand_hit / n_gt if n_gt else nan,
                det_recall=n_det_hit / n_gt if n_gt else nan,
                false_pm=false_all / minutes if minutes else nan,
                false_pass_pm=false_pass / minutes if minutes else nan,
                precision=pass_hit / pass_all if pass_all else nan,
                n_pass=pass_all, minutes=minutes)


def f1(m):
    p, r = m["precision"], m["det_recall"]
    if not (p > 0 and r > 0):
        return 0.0
    return 2 * p * r / (p + r)


# ============================================================================
# 8. Tables and plots
# ============================================================================
def table(results, thr, title):
    lines = [f"\n{title}  (FFT threshold {thr:g})",
             f"{'group':38s} {'win':>5s} {'candRec':>8s} {'detRec':>7s} {'false/min':>9s} "
             f"{'falsePass/min':>13s} {'precision':>9s} {'nPass':>6s}"]
    def row(label, m):
        lines.append(f"{label:38s} {m['windows']:5d} {m['cand_recall']:8.3f} {m['det_recall']:7.3f} "
                     f"{m['false_pm']:9.1f} {m['false_pass_pm']:13.2f} {m['precision']:9.3f} {m['n_pass']:6d}")
    for r in results:
        classes = sorted({g[2] for g in r["gt"]})
        row(f"{r['name']} ({', '.join(classes)})", metrics([r], thr))
    lines.append("  -- per drone type (pooled) --")
    for cls in sorted({g[2] for r in results for g in r["gt"]}):
        row(cls, metrics(results, thr, cls))
    row("ALL (pooled)", metrics(results, thr))
    return lines


def plot_frames(r, ev, tracks, prm, thr, path, n=6):
    """Event-density images of a few windows with candidate boxes vs ground truth."""
    x, y, t = ev
    c = r["cands"]
    gt_w = sorted({g[0] for g in r["gt"]})
    picks = [gt_w[int(i)] for i in np.linspace(0, len(gt_w) - 1, n)] if gt_w else list(range(n))
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(6 * ((n + 1) // 2), 7))
    for ax, w in zip(axes.ravel(), picks):
        w0 = r["win_starts"][w]
        i0, i1 = np.searchsorted(t, [w0, w0 + WIN_US])
        img = np.bincount(y[i0:i1] * WIDTH + x[i0:i1], minlength=WIDTH * HEIGHT).reshape(HEIGHT, WIDTH)
        ax.imshow(np.log1p(img), cmap="gray_r", vmax=np.log1p(np.percentile(img[img > 0], 99)) if img.any() else 1)
        for g in gt_in_window(tracks, w0, w0 + WIN_US):
            b = g[2]
            ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="blue", lw=2))
        for row in c[c[:, 0] == w]:
            _, x1, y1, x2, y2, cnt, prom, pf, m = row
            col = "green" if m >= 0 else "red"
            ls = "-" if prom > thr else ":"
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=col, lw=1.5, ls=ls))
            ax.text(x1, y1 - 4, f"{prom:.1f}", color=col, fontsize=7)
        ax.set_title(f"{r['name']}  t={w0 / 1e6:.2f} s", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("blue = ground truth (swept over window); green = candidate matching a drone, red = false; "
                 f"solid = passes FFT (> {thr:g}), dotted = fails; number = prominence", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_sweep(results, path, thr_frozen):
    """Detection recall and false-pass rate vs FFT threshold, per drone type."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    classes = sorted({g[2] for r in results for g in r["gt"]})
    for cls in classes + [None]:
        ms = [metrics(results, th, cls) for th in THRESHOLDS]
        lab = cls or "ALL"
        a1.plot(THRESHOLDS, [m["det_recall"] for m in ms], label=lab, lw=2.5 if cls is None else 1.2)
        a2.plot(THRESHOLDS, [m["false_pass_pm"] for m in ms], label=lab, lw=2.5 if cls is None else 1.2)
    for a in (a1, a2):
        a.axvline(thr_frozen, color="k", ls="--", lw=1)
        a.set_xlabel("FFT prominence threshold")
        a.grid(alpha=0.3)
    a1.set_ylabel("detection recall"); a2.set_ylabel("false FFT-passing candidates per minute")
    a2.set_yscale("symlog", linthresh=1)
    a1.legend(fontsize=8)
    fig.suptitle("Tune sequences: threshold sweep (dashed = frozen threshold)")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def plot_prom_hist(results, path, thr, title):
    """Prominence of candidates matching a drone vs false candidates."""
    c = np.concatenate([r["cands"] for r in results])
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 20, 81)
    ax.hist(np.clip(c[c[:, 8] < 0, 6], 0, 20), bins, alpha=0.6, density=True, label=f"false ({(c[:, 8] < 0).sum()})")
    ax.hist(np.clip(c[c[:, 8] >= 0, 6], 0, 20), bins, alpha=0.6, density=True, label=f"on a drone ({(c[:, 8] >= 0).sum()})")
    ax.axvline(thr, color="k", ls="--")
    ax.set_xlabel("prominence (clipped at 20)"); ax.set_ylabel("density"); ax.legend(); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


# ============================================================================
# 9. Commands
# ============================================================================
def sets():
    return json.load(open(OUT / "phase1_sequences.json"))


# Round 1 searched cell 8/16/32, k 3/5/8, min 10/30, K 3/5/10; the best sat on the edge
# (cell 32, k 8, K 3, min_count made no difference), so round 2 extends past that edge.
# A round-3 probe (cell 48/k 24, 64/16, 64/24, 48/32, all K 1) scored lower than
# round 2's winner (cell 48, k 16, K 1, F1 0.691), so that is a local maximum.
# K = 1 is the floor: with one candidate per window the finder can never
# report two drones at once.
GRID = dict(cell_px=[16, 32, 48], k_mean=[8.0, 12.0, 16.0], min_count=[10], max_cands=[1, 2, 3])


def cmd_tune():
    """Grid search on the tune sequences.
    Rule (stated before looking at heldout): for every cluster setting and
    every FFT threshold, pool the tune sequences and compute F1 of
    (precision of FFT-passing candidates, detection recall). Keep the setting
    and threshold with the highest pooled F1; ties go to the smaller threshold."""
    names = sets()["tune"]
    data = {n: (load_events(n), load_tracks(n)) for n in names}
    best, log = None, []
    import itertools
    for cell, k, mc, K in itertools.product(*GRID.values()):
        prm = dict(DEFAULTS, cell_px=cell, k_mean=k, min_count=mc, max_cands=K)
        res = [run_sequence(n, prm, *data[n]) for n in names]
        scores = [(f1(metrics(res, th)), -th, th) for th in THRESHOLDS]
        sc, _, th = max(scores)
        m = metrics(res, th)
        log.append(f"cell {cell:2d} k {k:4.1f} min {mc:3d} K {K:2d} | best thr {th:5.2f} F1 {sc:.3f} "
                   f"candRec {m['cand_recall']:.3f} detRec {m['det_recall']:.3f} prec {m['precision']:.3f} "
                   f"false/min {m['false_pm']:.0f} falsePass/min {m['false_pass_pm']:.1f}")
        print(log[-1])
        if best is None or sc > best[0]:
            best = (sc, prm, th, res)
    sc, prm, th, res = best
    frozen = dict(prm, fft_threshold=float(th), iou_match=IOU_MATCH, min_gt_frames=MIN_GT_FRAMES,
                  rule="max pooled F1(precision of FFT-passing candidates, detection recall) over tune "
                       "sequences, jointly over the cluster grid and thresholds 2..15 step 0.25",
                  tune_sequences=names, tune_F1=round(sc, 4))
    json.dump(frozen, open(OUT / "phase1_find_params.json", "w"), indent=2)
    lines = ["Phase 1 finder: tuning (tune sequences only)", "Grid search log:"] + log
    lines += [f"\nChosen: {json.dumps(frozen)}"]
    lines += table(res, th, "TUNE (in-sample)")
    (OUT / "phase1_find_tune_log.txt").write_text("\n".join(lines) + "\n")
    plot_sweep(res, OUT / "phase1_find_tune_sweep.png", th)
    print("\n".join(lines[-12:]))


def cmd_report(which=("heldout", "challenging", "development")):
    """Run frozen parameters on the reporting sets; write tables and plots."""
    frozen = json.load(open(OUT / "phase1_find_params.json"))
    prm = {k: frozen[k] for k in DEFAULTS}
    th = frozen["fft_threshold"]
    s = sets()
    out = ["Phase 1 finder results (parameters frozen on tune; see phase1_find_params.json)",
           f"params: {prm}, FFT threshold {th}",
           "Match: IoU(candidate box, drone box swept over the 0.5 s window) > 0.1.",
           f"Present window: drone has >= {MIN_GT_FRAMES} of ~15 annotation frames in it.",
           "false/min: candidates matching no drone, per minute (240 windows/min). "
           "falsePass/min: same, prominence > threshold.",
           "Per-type rows: recall over that type's windows; false/min and precision over the sequences "
           "containing that type.",
           "Runtime: finder + FFT only (file loading excluded), single process."]
    timing = []
    for set_name in ["tune"] + list(which):
        names = [n for n in s[set_name] if (seq_dir(n) / "Event/events.npy").exists()]
        missing = [n for n in s[set_name] if n not in names]
        res = []
        for n in names:
            ev, tr = load_events(n), load_tracks(n)
            r = run_sequence(n, prm, ev, tr)
            res.append(r)
            timing.append((set_name, n, r["data_s"], r["runtime"], r["n_hot"]))
            print(f"{set_name} {n}: {r['runtime']:.1f} s for {r['data_s']:.1f} s of data")
            if set_name != "tune" and n == names[0]:
                plot_frames(r, ev, tr, prm, th, OUT / f"phase1_find_frames_{set_name}_{n.replace('/', '_')}.png")
            del ev
        if not res:
            continue
        label = set_name.upper() + (" (in-sample, tuned here)" if set_name == "tune" else "")
        if missing:
            label += f"  [missing, not run: {', '.join(missing)}]"
        out += table(res, th, label)
        if set_name != "tune":
            plot_prom_hist(res, OUT / f"phase1_find_prominence_{set_name}.png", th,
                           f"{set_name}: candidate prominence")
        # Same set at a few other thresholds, for context (NOT used to choose anything).
        out.append("  threshold context (pooled ALL):")
        for alt in (3, 4, 5, 6, 8):
            m = metrics(res, alt)
            out.append(f"    thr {alt:>2}: detRec {m['det_recall']:.3f} falsePass/min {m['false_pass_pm']:.2f} "
                       f"precision {m['precision']:.3f}")
    out.append("\nRuntime per sequence (s compute / s data):")
    for set_name, n, ds, rt, nh in timing:
        out.append(f"  {set_name:12s} {n:10s} {rt:6.2f} s / {ds:6.1f} s = {rt / ds:.3f}   hot pixels {nh}")
    tot_rt, tot_ds = sum(x[3] for x in timing), sum(x[2] for x in timing)
    out.append(f"  overall {tot_rt / tot_ds:.3f} s of compute per s of data")
    (OUT / "phase1_find_results.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    {"tune": cmd_tune, "report": cmd_report}[cmd]()
