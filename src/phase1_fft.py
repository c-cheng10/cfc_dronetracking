"""Phase 1: rotor-frequency drone detector with no machine learning.

Idea: a spinning propeller blade passes the same pixels many times per second,
so the NUMBER of events inside the drone's box goes up and down at the
blade-pass frequency. If we count events in short time bins and take a Fourier
transform (FFT), a drone should show one tall, narrow peak. A patch of
background (sky, trees) should not.

Pipeline:
  1. Read the raw events (x, y, polarity, t) from events.npy, made once per
     sequence by src/data/ecf_convert.py.
  2. For each annotated drone, follow its bounding box over time and count the
     events inside it in 0.5 ms bins. That gives a signal sampled at 2 kHz.
  3. Do the same for "background" boxes: same size, same motion as the drone
     box, shifted to a place with no drone in it.
  4. Cut each signal into sliding windows, FFT each window, and record
     (a) the peak frequency and (b) a prominence score =
         peak magnitude / median magnitude, both measured in 50-1000 Hz.
  5. Call "drone" when prominence > threshold. Drone windows are positives,
     background windows are negatives -> precision and recall.

Run from the repo root, after converting the sequence:
    python -m src.data.ecf_convert data/FRED/train/36/36/Event/events.hdf5
    python src/phase1_fft.py

Options (defaults reproduce the first run):
    --bin-ms 0.1 --f-hi 5000   finer bins (10 kHz sampling, 5 kHz limit), wider band
    --speed-split              slow-vs-fast-drone analysis (writes *_speedsplit files only)
Any non-default setting adds a suffix such as "_bin0.1ms_50-5000Hz" to the
output file names, so earlier results are never overwritten.

"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
SEQ_DIR = REPO / "data/FRED/train/36/36"
EVENTS = SEQ_DIR / "Event/events.npy"   # written by src/data/ecf_convert.py
OUT_DIR = REPO / "experiments"
WIDTH, HEIGHT = 1280, 720        # sensor size (from the file's 'geometry' attribute)

BIN_US = 500                     # 0.5 ms bins -> 2000 samples per second
FS = 1e6 / BIN_US                # sampling rate of the count signal, Hz (= 2000)
WIN_S = 0.5                      # FFT window length, seconds
HOP_S = 0.25                     # step between windows, seconds (50% overlap)
F_LO, F_HI = 50.0, 1000.0        # band (Hz) for the peak search and median
BOX_TOL_US = 50_000              # a box stays valid this long after its annotation
                                 # (annotations come every 33.3 ms)
CHUNK = 4_000_000                # events read from disk per step (~64 MB)
N_BG = 3                         # background boxes per drone
THRESHOLDS = [2, 3, 4, 5, 6, 8, 10, 15]
SUFFIX = ""                      # added to output file names (set in parse_args)
SPEED_SPLIT = False              # --speed-split: slow-vs-fast analysis instead of the normal outputs
SLOW_BW = 1.0                    # --slow-bw: "slow" = box moves < this many box-widths per second


def parse_args():
    """Read --bin-ms, --f-lo, --f-hi from the command line and overwrite the
    settings above. The FFT can only see frequencies up to half the sampling
    rate (the Nyquist limit), so 0.5 ms bins see up to 1000 Hz and 0.1 ms bins
    up to 5000 Hz. Anything faster folds back ("aliases") to a wrong, lower
    frequency."""
    import argparse
    global BIN_US, FS, F_LO, F_HI, SUFFIX, SPEED_SPLIT, SLOW_BW
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bin-ms", type=float, default=0.5, help="bin width in ms (default 0.5)")
    p.add_argument("--f-lo", type=float, default=50.0, help="band low edge, Hz (default 50)")
    p.add_argument("--f-hi", type=float, default=1000.0, help="band high edge, Hz (default 1000)")
    p.add_argument("--speed-split", action="store_true",
                   help="split windows by drone speed (slow vs fast) and write only those outputs")
    p.add_argument("--slow-bw", type=float, default=1.0,
                   help="absolute 'slow' cut, in box-widths per second (default 1.0)")
    a = p.parse_args()
    SPEED_SPLIT, SLOW_BW = a.speed_split, a.slow_bw
    BIN_US = int(round(a.bin_ms * 1000))
    FS = 1e6 / BIN_US
    F_LO, F_HI = a.f_lo, a.f_hi
    if F_HI > FS / 2:
        sys.exit(f"--f-hi {F_HI} is above the Nyquist limit {FS / 2:.0f} Hz for {a.bin_ms} ms bins")
    if (BIN_US, F_LO, F_HI) != (500, 50.0, 1000.0):
        SUFFIX = f"_bin{a.bin_ms:g}ms_{F_LO:g}-{F_HI:g}Hz"
DRONE_NAMES = {1: "ID 1 DJI Mini 3", 2: "ID 2 DarwinFPV cineape20"}


# ----------------------------------------------------------------------------
# 1. Annotations
# ----------------------------------------------------------------------------
def load_tracks(path):
    """Read coordinates.txt -> {drone_id: dict of arrays}.

    Each line is  'time: x1, y1, x2, y2, id, class'  with time in seconds.
    We convert time to microseconds so it matches the event timestamps.
    Boxes can stick out of the image (the "extended" boxes), so we clip them.
    """
    rows = []
    for line in open(path):
        t_str, rest = line.split(":", 1)
        f = rest.split(",")
        rows.append([float(t_str), float(f[0]), float(f[1]), float(f[2]), float(f[3]), int(f[4])])
    a = np.array(rows)
    tracks = {}
    for i in np.unique(a[:, 5]).astype(int):
        b = a[a[:, 5] == i]
        b = b[np.argsort(b[:, 0])]
        tracks[i] = {
            "t": np.round(b[:, 0] * 1e6).astype(np.int64),   # microseconds
            "x1": np.clip(b[:, 1], 0, WIDTH), "y1": np.clip(b[:, 2], 0, HEIGHT),
            "x2": np.clip(b[:, 3], 0, WIDTH), "y2": np.clip(b[:, 4], 0, HEIGHT),
        }
    return tracks


def frame_index(track_t, t):
    """For each time in t, the index of the latest annotation at or before it,
    and whether that annotation is recent enough (within BOX_TOL_US) to trust."""
    k = np.searchsorted(track_t, t, side="right") - 1
    kc = np.clip(k, 0, len(track_t) - 1)
    ok = (k >= 0) & (t - track_t[kc] <= BOX_TOL_US)
    return kc, ok


def boxes_overlap(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    """True where box A and box B share any area (works on arrays)."""
    return (ax1 < bx2) & (bx1 < ax2) & (ay1 < by2) & (by1 < ay2)


def choose_background_offsets(tracks, drone_id):
    """Pick N_BG shifts (dx, dy) for drone_id's box.

    A background box is the drone's box moved by (dx, dy): same size and the
    same motion, so the only thing that differs is "drone inside or not".
    For each frame it is valid only if it lies fully inside the image and does
    not touch ANY drone's box. We keep the shifts valid in the most frames.
    """
    tr = tracks[drone_id]
    candidates = [(dx, dy) for dx in (-450, -300, -150, 0, 150, 300, 450)
                  for dy in (-250, -150, 0, 150, 250) if (dx, dy) != (0, 0)]
    scored = []
    for dx, dy in candidates:
        valid = per_frame_bg_valid(tracks, drone_id, dx, dy)
        scored.append((valid.mean(), dx, dy))
    scored.sort(reverse=True)
    return [(dx, dy) for _, dx, dy in scored[:N_BG]], scored[:N_BG]


def per_frame_bg_valid(tracks, drone_id, dx, dy):
    tr = tracks[drone_id]
    x1, y1, x2, y2 = tr["x1"] + dx, tr["y1"] + dy, tr["x2"] + dx, tr["y2"] + dy
    valid = (x1 >= 0) & (y1 >= 0) & (x2 <= WIDTH) & (y2 <= HEIGHT)
    for other in tracks.values():                      # includes the drone itself
        k, ok = frame_index(other["t"], tr["t"])
        hit = boxes_overlap(x1, y1, x2, y2, other["x1"][k] - 10, other["y1"][k] - 10,
                            other["x2"][k] + 10, other["y2"][k] + 10)   # 10 px margin
        valid &= ~(hit & ok)
    return valid


# ----------------------------------------------------------------------------
# 2. Event counts per box, in 0.5 ms bins
# ----------------------------------------------------------------------------
def count_events(ev, boxes, n_bins):
    """boxes: list of (track, dx, dy). Returns counts[n_boxes, n_bins].

    We read the file in chunks so we never hold all 53 M events in memory.
    For every event we look up which annotation frame it belongs to, then test
    whether its pixel is inside each (shifted) box at that frame.
    """
    counts = np.zeros((len(boxes), n_bins), dtype=np.int64)
    n = ev.shape[0]
    for start in range(0, n, CHUNK):
        e = ev[start:start + CHUNK]
        x, y = e["x"].astype(np.float32), e["y"].astype(np.float32)
        t = e["t"].astype(np.int64)               # stored as uint32 microseconds
        b = t // BIN_US                           # which time bin
        keep_t = b < n_bins
        for j, (tr, dx, dy) in enumerate(boxes):
            k, ok = frame_index(tr["t"], t)
            inside = (ok & keep_t
                      & (x >= tr["x1"][k] + dx) & (x < tr["x2"][k] + dx)
                      & (y >= tr["y1"][k] + dy) & (y < tr["y2"][k] + dy))
            counts[j] += np.bincount(b[inside], minlength=n_bins)[:n_bins]
        print(f"  read {min(start + CHUNK, n):,} / {n:,} events", end="\r")
    print()
    return counts


# ----------------------------------------------------------------------------
# 3. Sliding-window FFT
# ----------------------------------------------------------------------------
def window_spectra(signal, valid_bins):
    """Cut a count signal into windows; return start times, peak freq,
    prominence, the magnitude spectra, and which windows are usable.

    A window is usable only if its box was valid for EVERY bin in it (otherwise
    the box blinked off and the count drops to 0, which fakes a spectrum).
    """
    win = int(WIN_S * FS)                      # 1000 samples
    hop = int(HOP_S * FS)                      # 500 samples
    freqs = np.fft.rfftfreq(win, d=1 / FS)     # 0, 2, 4, ... 1000 Hz
    band = (freqs >= F_LO) & (freqs <= F_HI)
    hann = np.hanning(win)                     # tapers the window ends so the
                                               # cut edges don't smear the spectrum
    starts = np.arange(0, len(signal) - win + 1, hop)
    peak_f = np.zeros(len(starts))
    prom = np.zeros(len(starts))
    usable = np.zeros(len(starts), bool)
    spectra = np.zeros((len(starts), band.sum()))
    for w, s in enumerate(starts):
        seg = signal[s:s + win].astype(float)
        usable[w] = valid_bins[s:s + win].all()
        seg = seg - seg.mean()                 # remove the DC (average rate) term
        mag = np.abs(np.fft.rfft(seg * hann))  # magnitude at each frequency
        m = mag[band]
        spectra[w] = m
        med = np.median(m)
        peak_f[w] = freqs[band][np.argmax(m)]
        prom[w] = m.max() / med if med > 0 else 0.0
    return starts / FS, peak_f, prom, usable, spectra, freqs[band]


def valid_bin_mask(tr, dx, dy, tracks, drone_id, n_bins, bg):
    """Which 0.5 ms bins have a trustworthy box (and, for background, a box
    that is in-image and away from every drone)."""
    centres = np.arange(n_bins) * BIN_US + BIN_US // 2
    k, ok = frame_index(tr["t"], centres)
    if bg:
        ok &= per_frame_bg_valid(tracks, drone_id, dx, dy)[k]
    return ok


# ----------------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)
    tracks = load_tracks(SEQ_DIR / "coordinates.txt")
    ev = np.load(EVENTS, mmap_mode="r")       # reads from disk on demand, not into RAM
    t_last = int(ev["t"][-1])
    n_bins = t_last // BIN_US + 1
    print(f"recording length {t_last / 1e6:.2f} s -> {n_bins:,} bins of {BIN_US} us, "
          f"sampling {FS:.0f} Hz, band {F_LO:g}-{F_HI:g} Hz")

    # Build the list of boxes: each drone, then its background boxes.
    boxes, labels = [], []                     # labels: (drone_id, is_background, dx, dy)
    for i, tr in tracks.items():
        boxes.append((tr, 0, 0)); labels.append((i, False, 0, 0))
        offsets, scored = choose_background_offsets(tracks, i)
        for (frac, dx, dy) in scored:
            print(f"  drone {i}: background shift ({dx:+d},{dy:+d}) px valid in {frac:.0%} of frames")
        for dx, dy in offsets:
            boxes.append((tr, dx, dy)); labels.append((i, True, dx, dy))

    print(f"counting events in {len(boxes)} boxes ...")
    counts = count_events(ev, boxes, n_bins)

    results = []
    for (i, bg, dx, dy), (tr, _, _), c in zip(labels, boxes, counts):
        vb = valid_bin_mask(tr, dx, dy, tracks, i, n_bins, bg)
        t0, pf, pr, us, sp, fb = window_spectra(c, vb)
        results.append(dict(id=i, bg=bg, dx=dx, dy=dy, counts=c, t0=t0, peak_f=pf,
                            prom=pr, usable=us, spectra=sp, freqs=fb))

    if SPEED_SPLIT:
        speed_split_analysis(tracks, results)
        return

    # ---- numbers ----------------------------------------------------------
    summary_lines = []
    def say(s=""):
        print(s); summary_lines.append(s)

    for i in tracks:
        dr = next(r for r in results if r["id"] == i and not r["bg"])
        bgs = [r for r in results if r["id"] == i and r["bg"]]
        u = dr["usable"]
        say(f"\n=== {DRONE_NAMES.get(i, i)} ===")
        say(f"usable drone windows: {u.sum()} of {len(u)} ({WIN_S}s, hop {HOP_S}s)")
        f = dr["peak_f"][u]
        if len(f):
            med = np.median(f)
            near = np.mean(np.abs(f - med) <= 0.1 * med)
            say(f"drone peak freq: median {med:.0f} Hz, IQR {np.percentile(f,25):.0f}-{np.percentile(f,75):.0f} Hz, "
                f"{near:.0%} of windows within +-10% of median")
            say(f"drone prominence: median {np.median(dr['prom'][u]):.2f}, "
                f"IQR {np.percentile(dr['prom'][u],25):.2f}-{np.percentile(dr['prom'][u],75):.2f}")
        neg = np.concatenate([b["prom"][b["usable"]] for b in bgs])
        negf = np.concatenate([b["peak_f"][b["usable"]] for b in bgs])
        say(f"background windows: {len(neg)}; prominence median {np.median(neg):.2f}, "
            f"IQR {np.percentile(neg,25):.2f}-{np.percentile(neg,75):.2f}; "
            f"peak freq median {np.median(negf):.0f} Hz")
        pos = dr["prom"][u]
        say(f"{'thr':>5} {'TP':>5} {'FP':>5} {'FN':>5} {'precision':>9} {'recall':>7} {'FP rate':>7}")
        for thr in THRESHOLDS:
            tp = int((pos > thr).sum()); fn = len(pos) - tp; fp = int((neg > thr).sum())
            prec = tp / (tp + fp) if tp + fp else float("nan")
            say(f"{thr:>5} {tp:>5} {fp:>5} {fn:>5} {prec:>9.3f} {tp/len(pos):>7.3f} {fp/len(neg):>7.3f}")
        # threshold-free separation: probability a random drone window
        # outscores a random background window (= ROC AUC)
        auc = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
        say(f"AUC (P[drone score > background score]): {auc:.3f}")
    # Mean events per bin inside each box, over the bins where that box is
    # usable. Tells us how much signal each box actually has.
    say("\nmean events per bin (usable bins only):")
    for (i, bg, dx, dy), (tr, _, _), c in zip(labels, boxes, counts):
        vb = valid_bin_mask(tr, dx, dy, tracks, i, n_bins, bg)
        name = f"background ({dx:+d},{dy:+d})" if bg else "drone box"
        say(f"  drone {i} {name:>22}: {c[vb].mean():.3f}")
    (OUT_DIR / f"phase1_fft_seq36_results{SUFFIX}.txt").write_text("\n".join(summary_lines) + "\n")

    make_plots(tracks, results)


# ----------------------------------------------------------------------------
# 4b. Slow vs fast windows (--speed-split)
# ----------------------------------------------------------------------------
# Question: is the FPV's rotor tone hidden by its body motion? When the box
# moves fast, the drone's outline sweeps across many pixels and floods the box
# with edge events that carry no rotor rhythm. If the rotor tone is there but
# drowned, it should reappear in the windows where the drone is nearly still.

def window_motion(tr, t0_s):
    """For each window start (s), the box-centre speed (px/s, path length over
    the window divided by its duration) and the mean box width (px)."""
    cx = (tr["x1"] + tr["x2"]) / 2
    cy = (tr["y1"] + tr["y2"]) / 2
    w = tr["x2"] - tr["x1"]
    speed = np.full(len(t0_s), np.nan)
    width = np.full(len(t0_s), np.nan)
    area = np.full(len(t0_s), np.nan)
    for k, t0 in enumerate(t0_s):
        a = np.searchsorted(tr["t"], t0 * 1e6)
        b = np.searchsorted(tr["t"], (t0 + WIN_S) * 1e6)
        if b - a < 2:
            continue
        path = np.hypot(np.diff(cx[a:b]), np.diff(cy[a:b])).sum()
        speed[k] = path / ((tr["t"][b - 1] - tr["t"][a]) / 1e6)
        width[k] = w[a:b].mean()
        area[k] = (w[a:b] * (tr["y2"][a:b] - tr["y1"][a:b])).mean()
    return speed, width, area


def spearman(a, b):
    """Rank correlation: Pearson correlation of the ranks (no SciPy needed)."""
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def auc(pos, neg):
    return (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()


def speed_split_analysis(tracks, results):
    lines = []
    def say(s=""):
        print(s); lines.append(s)
    say(f"speed split: bins {BIN_US} us, band {F_LO:g}-{F_HI:g} Hz, windows {WIN_S}s hop {HOP_S}s")
    win = int(WIN_S * FS)
    fig, axes = plt.subplots(len(tracks), 2, figsize=(14, 3.6 * len(tracks)))
    axes = np.atleast_2d(axes)

    def norm_median(sp):
        sp = sp / np.median(sp, axis=1, keepdims=True).clip(1e-12)
        return np.median(sp, axis=0)

    for row, i in zip(axes, tracks):
        dr = next(r for r in results if r["id"] == i and not r["bg"])
        bgs = [r for r in results if r["id"] == i and r["bg"]]
        neg = np.concatenate([b["prom"][b["usable"]] for b in bgs])
        bgsp = np.vstack([b["spectra"][b["usable"]] for b in bgs])
        speed, width, area = window_motion(tracks[i], dr["t0"])
        u = dr["usable"] & np.isfinite(speed)
        sp_u, bw_u = speed[u], speed[u] / width[u]       # px/s and box-widths/s
        starts = (dr["t0"][u] * FS).astype(int)
        n_ev = np.array([dr["counts"][s:s + win].sum() for s in starts])
        density = n_ev / area[u] / WIN_S                  # events per px^2 per s

        say(f"\n=== {DRONE_NAMES.get(i, i)} ===  ({u.sum()} usable windows)")
        say("speed px/s percentiles 10/25/50/75/90: " +
            " / ".join(f"{np.percentile(sp_u, q):.0f}" for q in (10, 25, 50, 75, 90)))
        say("speed box-widths/s percentiles 10/25/50/75/90: " +
            " / ".join(f"{np.percentile(bw_u, q):.2f}" for q in (10, 25, 50, 75, 90)))
        say(f"box width px median {np.median(width[u]):.0f}")
        say(f"rank corr(speed px/s, events per window)        = {spearman(sp_u, n_ev):+.2f}")
        say(f"rank corr(speed px/s, events per px^2 per s)    = {spearman(sp_u, density):+.2f}")
        say(f"rank corr(speed px/s, prominence)               = {spearman(sp_u, dr['prom'][u]):+.2f}")
        say(f"background: {len(neg)} windows, prominence median {np.median(neg):.2f}")

        tercile = np.percentile(sp_u, 100 / 3)
        splits = [(f"slow tercile (< {tercile:.0f} px/s)", sp_u < tercile),
                  (f"fast (>= {tercile:.0f} px/s)", sp_u >= tercile),
                  (f"slow absolute (< {SLOW_BW:g} box-widths/s)", bw_u < SLOW_BW),
                  (f"fast absolute (>= {SLOW_BW:g} box-widths/s)", bw_u >= SLOW_BW)]
        say(f"{'group':<38} {'n':>4} {'peak med':>8} {'IQR':>11} {'+-10%':>6} {'prom med':>8} {'AUC':>6} {'ev/win':>8}")
        for k, (name, m) in enumerate(splits):
            n = int(m.sum())
            if n == 0:
                say(f"{name:<38} {0:>4}  (no windows)"); continue
            f = dr["peak_f"][u][m]; pr = dr["prom"][u][m]; med = np.median(f)
            near = np.mean(np.abs(f - med) <= 0.1 * med)
            say(f"{name:<38} {n:>4} {med:>8.0f} {np.percentile(f,25):>5.0f}-{np.percentile(f,75):<5.0f} "
                f"{near:>6.0%} {np.median(pr):>8.2f} {auc(pr, neg):>6.3f} {np.median(n_ev[m]):>8.0f}"
                + ("   <-- fewer than 20 windows: too few to conclude" if n < 20 else ""))
            if k < 2 and n:   # plot the tercile split
                row[0].plot(dr["freqs"], norm_median(dr["spectra"][u][m]), lw=0.8,
                            label=f"drone, {name} (n={n})")
        row[0].plot(dr["freqs"], norm_median(bgsp), lw=0.8, color="0.5", label="background boxes")
        row[0].set_title(f"{DRONE_NAMES[i]}: median normalised spectrum, slow vs fast")
        row[0].set_xlabel("frequency (Hz)"); row[0].set_ylabel("magnitude / band median")
        row[0].legend(fontsize=8)
        row[1].scatter(sp_u, n_ev, s=6, c=dr["prom"][u], cmap="viridis", vmin=2.5, vmax=8)
        row[1].axvline(tercile, color="k", ls="--", lw=0.8, label="speed tercile")
        row[1].set_xscale("log"); row[1].set_yscale("log")
        row[1].set_xlabel("box-centre speed (px/s)"); row[1].set_ylabel("events per 0.5 s window")
        row[1].set_title(f"{DRONE_NAMES[i]}: events vs speed (colour = prominence)")
        row[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"phase1_d_speed_split{SUFFIX}_speedsplit.png", dpi=120); plt.close(fig)
    (OUT_DIR / f"phase1_fft_seq36_results{SUFFIX}_speedsplit.txt").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# 5. Plots
# ----------------------------------------------------------------------------
def make_plots(tracks, results):
    ids = list(tracks)

    # (a) raw event-count signal, 100 ms snippet from the most prominent window
    fig, axes = plt.subplots(len(ids), 1, figsize=(10, 3 * len(ids)))
    for ax, i in zip(np.atleast_1d(axes), ids):
        dr = next(r for r in results if r["id"] == i and not r["bg"])
        bg = next(r for r in results if r["id"] == i and r["bg"])
        w = np.argmax(np.where(dr["usable"], dr["prom"], -1))
        s = int(dr["t0"][w] * FS); n = int(0.1 * FS)
        tt = (np.arange(s, s + n) / FS - dr["t0"][w]) * 1000
        ax.plot(tt, dr["counts"][s:s + n], lw=0.8, label="drone box")
        ax.plot(tt, bg["counts"][s:s + n], lw=0.8, label="background box", alpha=0.8)
        ax.set_title(f"{DRONE_NAMES[i]}: events per 0.5 ms bin, t = {dr['t0'][w]:.2f} s "
                     f"(peak {dr['peak_f'][w]:.0f} Hz)")
        ax.set_xlabel("time in window (ms)"); ax.set_ylabel("events / bin"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT_DIR / f"phase1_a_signal_snippet{SUFFIX}.png", dpi=120); plt.close(fig)

    # (b) spectrum: median over usable windows (normalised per window by its own
    #     band median, so the y axis is "times the noise floor"), drone vs background
    fig, axes = plt.subplots(len(ids), 1, figsize=(10, 3.2 * len(ids)))
    for ax, i in zip(np.atleast_1d(axes), ids):
        dr = next(r for r in results if r["id"] == i and not r["bg"])
        bgs = [r for r in results if r["id"] == i and r["bg"]]
        def norm_median(r):
            sp = r["spectra"][r["usable"]]
            sp = sp / np.median(sp, axis=1, keepdims=True).clip(1e-12)
            return np.median(sp, axis=0)
        ax.plot(dr["freqs"], norm_median(dr), label="drone box")
        bgsp = np.vstack([b["spectra"][b["usable"]] for b in bgs])
        bgsp = bgsp / np.median(bgsp, axis=1, keepdims=True).clip(1e-12)
        ax.plot(dr["freqs"], np.median(bgsp, axis=0), label="background boxes", alpha=0.8)
        ax.set_title(f"{DRONE_NAMES[i]}: median spectrum over usable {WIN_S}s windows")
        ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("magnitude / band median"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT_DIR / f"phase1_b_spectrum_drone_vs_control{SUFFIX}.png", dpi=120); plt.close(fig)

    # (c) prominence over time, and (d) peak frequency over time
    fig, axes = plt.subplots(len(ids), 2, figsize=(14, 3.2 * len(ids)))
    axes = np.atleast_2d(axes)
    for row, i in zip(axes, ids):
        dr = next(r for r in results if r["id"] == i and not r["bg"])
        bgs = [r for r in results if r["id"] == i and r["bg"]]
        u = dr["usable"]
        row[0].plot(dr["t0"][u], dr["prom"][u], ".", ms=3, label="drone box")
        row[1].plot(dr["t0"][u], dr["peak_f"][u], ".", ms=3, label="drone box")
        for k, b in enumerate(bgs):
            ub = b["usable"]
            row[0].plot(b["t0"][ub], b["prom"][ub], ".", ms=2, alpha=0.5, color="C1",
                        label="background boxes" if k == 0 else None)
            row[1].plot(b["t0"][ub], b["peak_f"][ub], ".", ms=2, alpha=0.3, color="C1",
                        label="background boxes" if k == 0 else None)
        row[0].set_yscale("log"); row[0].set_ylabel("prominence (peak / median)")
        row[0].set_title(f"{DRONE_NAMES[i]}: prominence per window")
        row[1].set_ylabel("peak frequency (Hz)"); row[1].set_title(f"{DRONE_NAMES[i]}: peak frequency per window")
        for ax in row:
            ax.set_xlabel("time (s)"); ax.legend(markerscale=3)
    fig.tight_layout(); fig.savefig(OUT_DIR / f"phase1_c_prominence_over_time{SUFFIX}.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    if not EVENTS.exists():
        sys.exit(f"missing {EVENTS}\nconvert first: python -m src.data.ecf_convert "
                 f"{SEQ_DIR / 'Event/events.hdf5'}")
    parse_args()
    main()
