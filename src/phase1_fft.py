"""Phase 1: rotor-frequency drone detector with no machine learning.

Idea: a spinning propeller blade passes the same pixels many times per second,
so the NUMBER of events inside the drone's box goes up and down at the
blade-pass frequency. If we count events in short time bins and take a Fourier
transform (FFT), a drone should show one tall, narrow peak. A patch of
background (sky, trees) should not.

Pipeline for one FRED sequence:
  1. Read the raw events (x, y, polarity, t) from events.npy, made once per
     sequence by src/data/ecf_convert.py, and the boxes from coordinates.txt.
  2. Build a list of "box streams": each drone's own box, plus background boxes
     of the same size (two kinds, see BACKGROUND BOXES below).
  3. Count the events inside every box in 0.5 ms bins (a signal sampled at
     2 kHz).
  4. Cut each signal into 0.5 s windows (0.25 s step), Hann taper, FFT, and
     score each window by prominence = tallest peak / median magnitude, both
     inside 50-1000 Hz.
  5. Write one CSV row per usable window per box. Drone windows are positives,
     background windows are negatives. A threshold on the score gives
     precision and recall (done across sequences by src/phase1_multiseq.py).

LOCKED SETTINGS (every reported number uses these; there are no options to
change them): 0.5 ms bins, 50-1000 Hz band, 0.5 s windows, 0.25 s step.
A wider band would raise the false-alarm rate at the same threshold, because
the tallest of more noise bins is taller. Earlier exploratory runs at 0.1 ms /
50-5000 Hz live in git history and in experiments/*_bin0.1ms_* files.

BACKGROUND BOXES (negatives)
  "shifted": the drone's box moved by a fixed (dx, dy), so it has the same
      size and the same motion as the drone box. Candidate shifts are
      +-150/300/450 px in x and +-150/250 px in y; we keep the N_SHIFTED shifts
      that are valid in the most annotation frames. A frame is valid if the
      shifted box is fully inside the image and at least 10 px from EVERY
      drone box (all tracks). These land mostly on empty sky: weak negatives.
  "active": static boxes on busy parts of the image (trees, buildings,
      lights, noisy pixels, anything that makes events). No RGB frames are
      available, so we pick them from the events themselves:
        a. Split the recording into 5 s chunks. For each chunk, count all
           events per pixel (an event-density map).
        b. For each drone annotated in that chunk, take its median box size
           in the chunk. Slide a box of that size over the image on an 8 px
           grid. A position is allowed only if it never comes within 10 px of
           ANY drone box annotated in the chunk (all tracks).
        c. Keep only "active" positions: event rate at or above the 75th
           percentile of the allowed positions in that chunk.
        d. Among those, pick N_ACTIVE positions, not overlapping each other,
           whose event rate is closest (smallest |log ratio|) to the drone
           box's own event rate in that chunk. So they are busy AND
           count-matched to the drone where the scene allows it.
        e. Ignore times when any drone is inside its annotated life span
           but has no box (an annotation gap): an un-annotated drone could
           sit inside an "active" box. Those events are left out of the
           density map and active-box windows there are not used.
      An active box stays still for its 5 s chunk, so its windows never cross
      a chunk boundary. How well the counts actually match is reported by
      src/phase1_multiseq.py; when the scene has nothing as busy as the drone,
      the busiest allowed positions are used and the match is imperfect.

CSV OUTPUT: experiments/csv/<split>_<seq>.csv (with _ds2 / _ds4 for the
simulated smaller sensors). One row per usable window per box. A window is
usable only if its box was valid for the whole window. Columns, in order:
  sequence        FRED sequence, e.g. train/36
  track_id        drone ID from coordinates.txt (for a background box: the
                  drone whose box size it copies)
  is_background   1 for a background box, 0 for the drone's own box
  dx, dy          shift in px of a "shifted" box (0, 0 for the drone box).
                  For an "active" box: offset of its centre from the drone
                  box centre at mid-window, informational only (the active
                  box does NOT follow the drone; use box_x1..box_y2)
  t_start_us      window start, microseconds (same clock as events.npy)
  t_end_us        window end, microseconds (start + 500000)
  score           FFT prominence, peak / median magnitude in 50-1000 Hz
  label           1 = drone inside the box, 0 = no drone
  event_count     events inside the box during the window (the count-only
                  baseline score)
  -- extra columns --
  drone_class     class name from coordinates.txt for track_id
  background_type "drone", "shifted" or "active"
  box_name        e.g. "shifted(-450,+0)" or "active1"; one name = one stream
  box_width_px    box width at mid-window, in this sensor's pixels
  box_height_px   box height at mid-window
  box_x1..box_y2  box corners at mid-window (exact for active boxes, which
                  are static; drone and shifted boxes move during the window)
  peak_hz         frequency of the tallest peak in 50-1000 Hz
  downsample      1, 2 or 4 (sensor pixel size relative to 1280x720)

SIMULATED SMALLER SENSOR (--downsample 2 or 4): event x, y and box
coordinates are integer-divided by 2 or 4 (a 640 or 320 px wide sensor).
Events that land on the same coarse pixel stay separate events. Caveats:
coarse pixels merge rotors with the drone body, and a real small sensor
differs in noise, contrast and refractory behaviour, so this is only a
geometric approximation. Because the detector sums ALL events in the box,
this approximation mostly changes box-edge rounding, not the counts.

Run from the repo root, after converting the sequence:
    python -m src.data.ecf_convert data/FRED/train/36/36/Event/events.hdf5
    python src/phase1_fft.py --seq train/36
    python src/phase1_fft.py --seq train/91 --downsample 2 --no-plots
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "experiments"
CSV_DIR = OUT_DIR / "csv"
THRESHOLD_FILE = OUT_DIR / "phase1_threshold.json"   # written by phase1_multiseq.py

# ----------------------------------------------------------------------------
# Locked settings
# ----------------------------------------------------------------------------
SENSOR_W, SENSOR_H = 1280, 720   # full sensor size (from the file's 'geometry' attribute)
BIN_US = 500                     # 0.5 ms bins -> 2000 samples per second
FS = 1e6 / BIN_US                # sampling rate of the count signal, Hz (= 2000)
WIN_S, HOP_S = 0.5, 0.25         # FFT window length and step, seconds
WIN = int(WIN_S * FS)            # 1000 bins per window
HOP = int(HOP_S * FS)            # 500 bins per step
F_LO, F_HI = 50.0, 1000.0        # band (Hz) for the peak search and median

# ----------------------------------------------------------------------------
# Box and background settings
# ----------------------------------------------------------------------------
BOX_TOL_US = 50_000              # a box stays valid this long after its annotation
                                 # (annotations come every 33.3 ms)
MARGIN_PX = 10                   # background boxes stay this far from every drone box
N_SHIFTED = 3                    # shifted background boxes per drone
N_ACTIVE = 3                     # active background boxes per drone per chunk
ACTIVE_CHUNK_S = 5.0             # active boxes are re-chosen every 5 s
ACTIVE_STRIDE_PX = 8             # grid step when sliding the candidate box
ACTIVE_PCTL = 75                 # "active" = event rate >= this percentile of allowed spots
CHUNK = 4_000_000                # events read from disk per step (~36 MB)


# ----------------------------------------------------------------------------
# 0. Paths
# ----------------------------------------------------------------------------
def seq_dir(seq):
    """'train/91' -> data/FRED/train/91/91 (where coordinates.txt lives)."""
    split, num = seq.split("/")
    return REPO / "data" / "FRED" / split / num / num


def events_path(seq):
    return seq_dir(seq) / "Event" / "events.npy"


def csv_path(seq, downsample=1):
    split, num = seq.split("/")
    tag = "" if downsample == 1 else f"_ds{downsample}"
    return CSV_DIR / f"{split}_{num}{tag}.csv"


# ----------------------------------------------------------------------------
# 1. Annotations
# ----------------------------------------------------------------------------
def load_tracks(path, downsample=1):
    """Read coordinates.txt -> {drone_id: dict of arrays}.

    Each line is  'time: x1, y1, x2, y2, id, class'  with time in seconds.
    We convert time to microseconds so it matches the event timestamps.
    Boxes can stick out of the image (the "extended" boxes), so we clip them.
    The class name is read from the file, so any drone type works.
    With downsample > 1 the box corners are integer-divided like the events.
    """
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        t_str, rest = line.split(":", 1)
        f = [s.strip() for s in rest.split(",")]
        cls = ",".join(f[5:]).strip() or "unknown"
        rows.append((float(t_str), float(f[0]), float(f[1]), float(f[2]), float(f[3]),
                     int(float(f[4])), cls))
    tracks = {}
    for i in sorted({r[5] for r in rows}):
        b = sorted((r for r in rows if r[5] == i), key=lambda r: r[0])
        a = np.array([r[:5] for r in b])
        names = [r[6] for r in b]
        tr = {
            "t": np.round(a[:, 0] * 1e6).astype(np.int64),   # microseconds
            "x1": np.clip(a[:, 1], 0, SENSOR_W), "y1": np.clip(a[:, 2], 0, SENSOR_H),
            "x2": np.clip(a[:, 3], 0, SENSOR_W), "y2": np.clip(a[:, 4], 0, SENSOR_H),
            "cls": max(set(names), key=names.count),           # most common name for this ID
        }
        if downsample > 1:
            for k in ("x1", "y1", "x2", "y2"):
                tr[k] = np.floor(tr[k] / downsample)
        tracks[i] = tr
    return tracks


def frame_index(key_t, t, tol=BOX_TOL_US):
    """For each time in t, the index of the latest key time at or before it,
    and whether that key is recent enough (within tol) to trust."""
    k = np.searchsorted(key_t, t, side="right") - 1
    kc = np.clip(k, 0, len(key_t) - 1)
    ok = (k >= 0) & (t - key_t[kc] <= tol)
    return kc, ok


def boxes_overlap(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    """True where box A and box B share any area (works on arrays)."""
    return (ax1 < bx2) & (bx1 < ax2) & (ay1 < by2) & (by1 < ay2)


# ----------------------------------------------------------------------------
# 2a. Shifted background boxes
# ----------------------------------------------------------------------------
def per_frame_bg_valid(tracks, drone_id, dx, dy, downsample=1):
    """For each annotation frame of drone_id: is the box shifted by (dx, dy)
    fully inside the image and >= MARGIN_PX from every drone box?"""
    W, H = SENSOR_W // downsample, SENSOR_H // downsample
    m = MARGIN_PX / downsample
    tr = tracks[drone_id]
    x1, y1, x2, y2 = tr["x1"] + dx, tr["y1"] + dy, tr["x2"] + dx, tr["y2"] + dy
    valid = (x1 >= 0) & (y1 >= 0) & (x2 <= W) & (y2 <= H)
    for other in tracks.values():                      # includes the drone itself
        k, ok = frame_index(other["t"], tr["t"])
        hit = boxes_overlap(x1, y1, x2, y2, other["x1"][k] - m, other["y1"][k] - m,
                            other["x2"][k] + m, other["y2"][k] + m)
        valid &= ~(hit & ok)
    return valid


def choose_background_offsets(tracks, drone_id, downsample=1):
    """Pick N_SHIFTED shifts (dx, dy) for drone_id's box, the ones valid in
    the most annotation frames. Returns [(dx, dy)], [(fraction, dx, dy)]."""
    candidates = [(dx, dy) for dx in (-450, -300, -150, 0, 150, 300, 450)
                  for dy in (-250, -150, 0, 150, 250) if (dx, dy) != (0, 0)]
    scored = []
    for dx, dy in candidates:
        dxs, dys = dx // downsample, dy // downsample      # shift in this sensor's pixels
        valid = per_frame_bg_valid(tracks, drone_id, dxs, dys, downsample)
        scored.append((valid.mean(), dxs, dys))
    scored.sort(reverse=True)
    scored = [s for s in scored[:N_SHIFTED] if s[0] > 0]
    return [(dx, dy) for _, dx, dy in scored], scored


# ----------------------------------------------------------------------------
# 2b. Active background boxes (busy regions, chosen from the events)
# ----------------------------------------------------------------------------
def integral_image(a):
    """S[y, x] = sum of a[:y, :x]. Any box sum is then 4 lookups."""
    S = np.zeros((a.shape[0] + 1, a.shape[1] + 1), np.int64)
    S[1:, 1:] = a.cumsum(0).cumsum(1)
    return S


def box_sums(S, xs, ys, bw, bh):
    """Sum inside boxes with top-left (xs, ys) and size (bw, bh), on grid xs x ys."""
    X, Y = np.meshgrid(xs, ys)
    return S[Y + bh, X + bw] - S[Y, X + bw] - S[Y + bh, X] + S[Y, X], X, Y


def safe_times(tracks, t):
    """True at times when no drone can be in the image without a box: every
    track is either outside its annotated life span [first, last box] or has
    a box less than BOX_TOL_US old. During an annotation gap an un-annotated
    drone could sit inside an 'active' box, so those times are not used."""
    ok = np.ones(len(t), bool)
    for tr in tracks.values():
        inside_life = (t >= tr["t"][0]) & (t <= tr["t"][-1])
        ok &= ~inside_life | frame_index(tr["t"], t)[1]
    return ok


def choose_active_boxes(ev, tracks, t_end_us, downsample=1):
    """Pick the 'active' background boxes (rule in the module docstring).

    Returns
      chunk_t : start time (us) of every 5 s chunk
      boxes   : {drone_id: array [N_ACTIVE, n_chunks, 4]} of x1, y1, x2, y2,
                NaN where no box was chosen
      log     : list of dicts, one per (drone, chunk), for reporting
    """
    W, H = SENSOR_W // downsample, SENSOR_H // downsample
    m = MARGIN_PX / downsample
    stride = max(1, ACTIVE_STRIDE_PX // downsample)
    ch_us = int(ACTIVE_CHUNK_S * 1e6)
    n_chunks = t_end_us // ch_us + 1
    chunk_t = np.arange(n_chunks, dtype=np.int64) * ch_us
    edges = np.searchsorted(ev["t"], np.append(chunk_t, chunk_t[-1] + ch_us))
    boxes = {i: np.full((N_ACTIVE, n_chunks, 4), np.nan) for i in tracks}
    log = []

    for c in range(n_chunks):
        c0, c1 = int(chunk_t[c]), int(chunk_t[c] + ch_us)
        dur_s = (min(c1, t_end_us) - c0) / 1e6
        if edges[c + 1] <= edges[c] or dur_s <= 0:
            continue

        excl = np.zeros((H, W), bool)          # pixels within MARGIN of any drone box
        for tr in tracks.values():
            sel = np.flatnonzero((tr["t"] >= c0 - BOX_TOL_US) & (tr["t"] < c1))
            for k in sel:                      # paint every annotated box, grown by the margin
                xa, xb = int(max(0, np.floor(tr["x1"][k] - m))), int(min(W, np.ceil(tr["x2"][k] + m)))
                ya, yb = int(max(0, np.floor(tr["y1"][k] - m))), int(min(H, np.ceil(tr["y2"][k] + m)))
                excl[ya:yb, xa:xb] = True

        # (a) event-density map for this chunk
        e = ev[edges[c]:edges[c + 1]]
        x = e["x"].astype(np.int64) // downsample
        y = e["y"].astype(np.int64) // downsample
        t = e["t"].astype(np.int64)
        good = (x < W) & (y < H) & safe_times(tracks, t)   # (e) only events from safe times
        if not good.any():
            continue
        dens = np.bincount(y[good] * W + x[good], minlength=W * H).reshape(H, W)
        S, E = integral_image(dens), integral_image(excl.astype(np.int64))

        for i, tr in tracks.items():
            in_chunk = (tr["t"] >= c0) & (tr["t"] < c1)
            if in_chunk.sum() < 15:             # drone needs >= 0.5 s of boxes in the chunk
                continue
            bw = max(1, int(round(np.median(tr["x2"][in_chunk] - tr["x1"][in_chunk]))))
            bh = max(1, int(round(np.median(tr["y2"][in_chunk] - tr["y1"][in_chunk]))))
            if bw >= W or bh >= H:
                continue
            # the drone's own event rate in this chunk (events per second)
            k, ok = frame_index(tr["t"], t)
            inside = ok & (x >= tr["x1"][k]) & (x < tr["x2"][k]) & (y >= tr["y1"][k]) & (y < tr["y2"][k])
            centres = np.arange(c0, min(c1, t_end_us), BIN_US) + BIN_US // 2
            covered_s = frame_index(tr["t"], centres)[1].sum() * BIN_US / 1e6
            if covered_s <= 0:
                continue
            drone_rate = inside.sum() / covered_s

            # (b) candidate positions on the grid, allowed if they touch no drone box
            xs = np.arange(0, W - bw + 1, stride)
            ys = np.arange(0, H - bh + 1, stride)
            cnt, X, Y = box_sums(S, xs, ys, bw, bh)
            hit, _, _ = box_sums(E, xs, ys, bw, bh)
            allowed = (hit == 0).ravel()
            if not allowed.any():
                continue
            rate = cnt.ravel() / dur_s
            X, Y = X.ravel(), Y.ravel()
            # (c) active = at or above the 75th percentile of allowed positions
            floor = np.percentile(rate[allowed], ACTIVE_PCTL)
            cand = np.flatnonzero(allowed & (rate >= floor))
            # (d) closest event rate to the drone's, greedy, no overlap between picks
            mismatch = np.abs(np.log((rate[cand] + 1) / (drone_rate + 1)))
            chosen, chosen_rates = [], []
            for j in cand[np.argsort(mismatch, kind="stable")]:
                bx, by = X[j], Y[j]
                if all(not boxes_overlap(bx, by, bx + bw, by + bh, *b) for b in chosen):
                    chosen.append((bx, by, bx + bw, by + bh))
                    chosen_rates.append(rate[j])
                if len(chosen) == N_ACTIVE:
                    break
            for s, b in enumerate(chosen):
                boxes[i][s, c] = b
            log.append(dict(track_id=i, chunk=c, drone_rate=drone_rate, active_rates=chosen_rates,
                            max_allowed_rate=rate[allowed].max()))
    return chunk_t, boxes, log


# ----------------------------------------------------------------------------
# 2c. Box streams: drone box + background boxes, in one common format
# ----------------------------------------------------------------------------
def build_streams(tracks, chunk_t, active, downsample=1):
    """One 'stream' per box that we count events in.

    Every stream is a list of key times (key_t) with a box per key. At time t
    the box is the one of the latest key at or before t, if that key is less
    than 'tol' old and ok_key is True. 'seg_key' marks which keys belong to
    the same static placement; a window may not span two segments.
    """
    streams = []
    for i, tr in tracks.items():
        n = len(tr["t"])
        base = dict(track_id=i, cls=tr["cls"], key_t=tr["t"], tol=BOX_TOL_US,
                    seg_key=np.zeros(n, np.int64))
        streams.append(dict(base, kind="drone", name="drone", dx=0, dy=0,
                            x1=tr["x1"], y1=tr["y1"], x2=tr["x2"], y2=tr["y2"],
                            ok_key=np.ones(n, bool)))
        offsets, _ = choose_background_offsets(tracks, i, downsample)
        for dx, dy in offsets:
            streams.append(dict(base, kind="shifted", name=f"shifted({dx:+d},{dy:+d})", dx=dx, dy=dy,
                                x1=tr["x1"] + dx, y1=tr["y1"] + dy, x2=tr["x2"] + dx, y2=tr["y2"] + dy,
                                ok_key=per_frame_bg_valid(tracks, i, dx, dy, downsample)))
        for s in range(N_ACTIVE):
            b = active[i][s]                                    # [n_chunks, 4]
            streams.append(dict(track_id=i, cls=tr["cls"], kind="active", name=f"active{s + 1}",
                                dx=0, dy=0, key_t=chunk_t, tol=int(ACTIVE_CHUNK_S * 1e6) - 1,
                                seg_key=np.arange(len(chunk_t)), ok_key=~np.isnan(b[:, 0]),
                                x1=np.nan_to_num(b[:, 0]), y1=np.nan_to_num(b[:, 1]),
                                x2=np.nan_to_num(b[:, 2]), y2=np.nan_to_num(b[:, 3])))
    return streams


# ----------------------------------------------------------------------------
# 3. Event counts per box, in 0.5 ms bins
# ----------------------------------------------------------------------------
def count_events(ev, streams, n_bins, downsample=1):
    """Returns counts[n_streams, n_bins].

    We read the file in chunks so we never hold all events in memory. For
    every event we look up which key (annotation frame or 5 s chunk) it
    belongs to, then test whether its pixel is inside each box at that key.
    Streams that share key times share the lookup (cache).
    """
    counts = np.zeros((len(streams), n_bins), dtype=np.int64)
    n = ev.shape[0]
    for start in range(0, n, CHUNK):
        e = ev[start:start + CHUNK]
        x = e["x"].astype(np.int32) // downsample
        y = e["y"].astype(np.int32) // downsample
        t = e["t"].astype(np.int64)               # stored as uint32 microseconds
        b = t // BIN_US                           # which time bin
        keep_t = b < n_bins
        cache = {}
        for j, s in enumerate(streams):
            key = (id(s["key_t"]), s["tol"])
            if key not in cache:
                cache[key] = frame_index(s["key_t"], t, s["tol"])
            k, ok = cache[key]
            inside = (ok & keep_t & s["ok_key"][k]
                      & (x >= s["x1"][k]) & (x < s["x2"][k])
                      & (y >= s["y1"][k]) & (y < s["y2"][k]))
            counts[j] += np.bincount(b[inside], minlength=n_bins)[:n_bins]
        print(f"  read {min(start + CHUNK, n):,} / {n:,} events", end="\r", file=sys.stderr)
    print(file=sys.stderr)
    return counts


def valid_bins(s, n_bins):
    """Which 0.5 ms bins have a trustworthy box, and the segment of each bin."""
    centres = np.arange(n_bins, dtype=np.int64) * BIN_US + BIN_US // 2
    k, ok = frame_index(s["key_t"], centres, s["tol"])
    return ok & s["ok_key"][k], s["seg_key"][k]


# ----------------------------------------------------------------------------
# 4. Sliding-window FFT
# ----------------------------------------------------------------------------
def window_spectra(signal, valid, seg):
    """Cut a count signal into windows and score each one.

    A window is usable only if its box was valid for EVERY bin in it (otherwise
    the box blinked off and the count drops to 0, which fakes a spectrum) and,
    for active boxes, it stays inside one 5 s placement.
    Returns a dict of arrays, one entry per usable window.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    freqs = np.fft.rfftfreq(WIN, d=1 / FS)     # 0, 2, 4, ... 1000 Hz
    band = (freqs >= F_LO) & (freqs <= F_HI)
    starts = np.arange(0, len(signal) - WIN + 1, HOP)
    usable = (sliding_window_view(valid, WIN)[::HOP].all(axis=1)
              & (seg[starts] == seg[starts + WIN - 1]))
    segs = sliding_window_view(signal, WIN)[::HOP][usable].astype(float)
    n_ev = segs.sum(axis=1)                    # events in the window (count baseline)
    segs = segs - segs.mean(axis=1, keepdims=True)   # remove the DC (average rate) term
    hann = np.hanning(WIN)                     # tapers the window ends so the cut
                                               # edges don't smear the spectrum
    mag = np.abs(np.fft.rfft(segs * hann, axis=1))[:, band]
    if len(mag):
        med = np.median(mag, axis=1)
        prom = np.where(med > 0, mag.max(axis=1) / np.where(med > 0, med, 1), 0.0)
        peak_f = freqs[band][np.argmax(mag, axis=1)]
    else:
        prom = peak_f = np.zeros(0)
    return dict(start_bin=starts[usable], prom=prom, peak_f=peak_f, n_ev=n_ev,
                spectra=mag, freqs=freqs[band])


# ----------------------------------------------------------------------------
# 5. One sequence -> CSV
# ----------------------------------------------------------------------------
CSV_COLUMNS = ["sequence", "track_id", "is_background", "dx", "dy", "t_start_us", "t_end_us",
               "score", "label", "event_count",
               "drone_class", "background_type", "box_name", "box_width_px", "box_height_px",
               "box_x1", "box_y1", "box_x2", "box_y2", "peak_hz", "downsample"]


def box_at(s, t_us):
    """Box corners of stream s at times t_us (array)."""
    k, _ = frame_index(s["key_t"], t_us, s["tol"])
    return s["x1"][k], s["y1"][k], s["x2"][k], s["y2"][k]


def run_sequence(seq, downsample=1, write_csv=True):
    """Run the detector on one sequence. Returns (tracks, streams, results,
    rows, active_log); results[j] holds the window scores of streams[j]."""
    ev = np.load(events_path(seq), mmap_mode="r")   # reads from disk on demand, not into RAM
    tracks = load_tracks(seq_dir(seq) / "coordinates.txt", downsample)
    t_end = int(ev["t"][-1])
    n_bins = t_end // BIN_US + 1
    print(f"{seq} ds{downsample}: {t_end / 1e6:.1f} s, {ev.shape[0]:,} events, "
          f"{len(tracks)} drone(s): " + ", ".join(f"{i} {tr['cls']}" for i, tr in tracks.items()),
          file=sys.stderr)

    chunk_t, active, active_log = choose_active_boxes(ev, tracks, t_end, downsample)
    streams = build_streams(tracks, chunk_t, active, downsample)
    counts = count_events(ev, streams, n_bins, downsample)

    centres = np.arange(n_bins, dtype=np.int64) * BIN_US + BIN_US // 2
    safe_bins = safe_times(tracks, centres)
    results, rows = [], []
    drone_stream = {s["track_id"]: s for s in streams if s["kind"] == "drone"}
    for s, c in zip(streams, counts):
        vb, seg = valid_bins(s, n_bins)
        if s["kind"] == "active":              # step (e): no un-annotated drone around
            vb &= safe_bins
        r = window_spectra(c, vb, seg)
        results.append(r)
        t0 = r["start_bin"].astype(np.int64) * BIN_US
        mid = t0 + int(WIN_S * 1e6) // 2
        x1, y1, x2, y2 = box_at(s, mid)
        if s["kind"] == "active":              # informational offset from the drone box
            dx1, dy1, dx2, dy2 = box_at(drone_stream[s["track_id"]], mid)
            dxs = np.round((x1 + x2 - dx1 - dx2) / 2).astype(int)
            dys = np.round((y1 + y2 - dy1 - dy2) / 2).astype(int)
        else:
            dxs = np.full(len(t0), s["dx"]); dys = np.full(len(t0), s["dy"])
        is_bg = int(s["kind"] != "drone")
        for w in range(len(t0)):
            rows.append([seq, s["track_id"], is_bg, int(dxs[w]), int(dys[w]), int(t0[w]),
                         int(t0[w] + WIN_S * 1e6), round(float(r["prom"][w]), 4), 1 - is_bg,
                         int(r["n_ev"][w]), s["cls"], s["kind"], s["name"],
                         round(float(x2[w] - x1[w]), 2), round(float(y2[w] - y1[w]), 2),
                         round(float(x1[w]), 2), round(float(y1[w]), 2),
                         round(float(x2[w]), 2), round(float(y2[w]), 2),
                         float(r["peak_f"][w]), downsample])
    if write_csv:
        CSV_DIR.mkdir(parents=True, exist_ok=True)
        with open(csv_path(seq, downsample), "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(CSV_COLUMNS)
            wr.writerows(rows)
        print(f"wrote {csv_path(seq, downsample)} ({len(rows)} rows)", file=sys.stderr)
    return tracks, streams, results, rows, active_log


# ----------------------------------------------------------------------------
# 6. Single-sequence summary and plots
# ----------------------------------------------------------------------------
def auc(pos, neg):
    """P[random positive scores higher than random negative], ties count half
    (= ROC AUC, computed from ranks so it is fast for big arrays)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    _, first, cnt = np.unique(allv[order], return_index=True, return_counts=True)
    ranks = np.empty(len(allv))
    ranks[order] = np.repeat(first + (cnt - 1) / 2.0 + 1, cnt)   # average rank for ties
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load_frozen_threshold():
    """The frozen thresholds from phase1_multiseq.py, or None before tuning."""
    import json
    if THRESHOLD_FILE.exists():
        return json.loads(THRESHOLD_FILE.read_text())
    return None


def summarise(seq, tracks, streams, results, active_log):
    lines = []
    def say(s=""):
        print(s); lines.append(s)
    thr = load_frozen_threshold()
    say(f"Phase 1 FFT detector, {seq}. Locked: {BIN_US / 1000:g} ms bins, {F_LO:g}-{F_HI:g} Hz, "
        f"{WIN_S} s windows, {HOP_S} s step.")
    if thr:
        say(f"Frozen thresholds from {THRESHOLD_FILE.name}: FFT score > {thr['fft_threshold']:.3f}, "
            f"count > {thr['count_threshold']:.0f} ({thr['rule']})")
    for i, tr in tracks.items():
        idx = [j for j, s in enumerate(streams) if s["track_id"] == i]
        dr = next(results[j] for j in idx if streams[j]["kind"] == "drone")
        say(f"\n=== ID {i} {tr['cls']} ===")
        say(f"usable drone windows: {len(dr['prom'])}")
        if len(dr["prom"]):
            f = dr["peak_f"]; med = np.median(f)
            say(f"drone peak freq: median {med:.0f} Hz, IQR {np.percentile(f, 25):.0f}-{np.percentile(f, 75):.0f} Hz, "
                f"{np.mean(np.abs(f - med) <= 0.1 * med):.0%} within +-10% of median")
            say(f"drone prominence: median {np.median(dr['prom']):.2f}; events/window median {np.median(dr['n_ev']):.0f}")
        for kind in ("shifted", "active"):
            bg = [results[j] for j in idx if streams[j]["kind"] == kind]
            neg = np.concatenate([b["prom"] for b in bg]) if bg else np.zeros(0)
            negc = np.concatenate([b["n_ev"] for b in bg]) if bg else np.zeros(0)
            if len(neg) == 0:
                say(f"{kind:>8}: no usable windows"); continue
            line = (f"{kind:>8}: {len(neg)} windows, events/window median {np.median(negc):.0f}; "
                    f"AUC FFT {auc(dr['prom'], neg):.3f}, AUC count {auc(dr['n_ev'], negc):.3f}")
            if thr:
                ft, ct = thr["fft_threshold"], thr["count_threshold"]
                line += (f"; at frozen thr: FFT recall {np.mean(dr['prom'] > ft):.3f} FA {np.mean(neg > ft):.3f}, "
                         f"count recall {np.mean(dr['n_ev'] > ct):.3f} FA {np.mean(negc > ct):.3f}")
            say(line)
        lg = [a for a in active_log if a["track_id"] == i and a["active_rates"]]
        if lg:
            ratio = np.concatenate([np.array(a["active_rates"]) / max(a["drone_rate"], 1e-9) for a in lg])
            say(f"active box event rate / drone event rate (per chunk): median {np.median(ratio):.2f}, "
                f"IQR {np.percentile(ratio, 25):.2f}-{np.percentile(ratio, 75):.2f} over {len(lg)} chunks; "
                f"busiest allowed spot / drone: median "
                f"{np.median([a['max_allowed_rate'] / max(a['drone_rate'], 1e-9) for a in lg]):.2f}")
    split, num = seq.split("/")
    (OUT_DIR / f"phase1_fft_{split}_{num}_results.txt").write_text("\n".join(lines) + "\n")


def make_plots(seq, tracks, streams, results):
    """Median spectrum (drone vs shifted vs active) and prominence over time."""
    split, num = seq.split("/")
    ids = list(tracks)
    fig, axes = plt.subplots(len(ids), 2, figsize=(14, 3.4 * len(ids)), squeeze=False)
    for row, i in zip(axes, ids):
        name = f"ID {i} {tracks[i]['cls']}"
        for kind, col in (("drone", "C0"), ("shifted", "0.5"), ("active", "C3")):
            rs = [r for s, r in zip(streams, results)
                  if s["track_id"] == i and s["kind"] == kind and len(r["prom"])]
            if rs:
                sp = np.vstack([r["spectra"] for r in rs])
                sp = sp / np.median(sp, axis=1, keepdims=True).clip(1e-12)
                row[0].plot(rs[0]["freqs"], np.median(sp, axis=0), color=col, lw=0.9, label=f"{kind} boxes")
            for k, r in enumerate(rs):
                row[1].plot(r["start_bin"] / FS, r["prom"], ".", ms=3 if kind == "drone" else 2,
                            color=col, alpha=0.9 if kind == "drone" else 0.5,
                            label=f"{kind} boxes" if k == 0 else None)
        row[0].set_title(f"{seq} {name}: median spectrum (each window / its band median)", fontsize=9)
        row[0].set_xlabel("frequency (Hz)"); row[0].set_ylabel("magnitude / band median")
        row[1].set_yscale("log"); row[1].set_title(f"{seq} {name}: prominence per window", fontsize=9)
        row[1].set_xlabel("window start (s)"); row[1].set_ylabel("prominence")
        for ax in row:
            ax.legend(fontsize=8, markerscale=3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"phase1_spectrum_prominence_{split}_{num}.png", dpi=110)
    plt.close(fig)


def parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seq", default="train/36", help="FRED sequence as split/number (default train/36)")
    p.add_argument("--downsample", type=int, default=1, choices=(1, 2, 4),
                   help="simulate a smaller sensor: divide pixel coordinates by 2 or 4 (default 1)")
    p.add_argument("--no-plots", action="store_true", help="skip the per-sequence plot")
    return p.parse_args(argv)


def main():
    a = parse_args()
    if not events_path(a.seq).exists():
        sys.exit(f"missing {events_path(a.seq)}\nconvert first: python -m src.data.ecf_convert "
                 f"{seq_dir(a.seq) / 'Event/events.hdf5'}")
    OUT_DIR.mkdir(exist_ok=True)
    tracks, streams, results, rows, active_log = run_sequence(a.seq, a.downsample)
    if a.downsample == 1:
        summarise(a.seq, tracks, streams, results, active_log)
        if not a.no_plots:
            make_plots(a.seq, tracks, streams, results)


if __name__ == "__main__":
    main()
