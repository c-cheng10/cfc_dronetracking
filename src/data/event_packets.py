"""Group each window's events into packets of N consecutive events.

N is chosen per window so a window has at most MAX_PACKETS packets. A packet
keeps its real collection time, so the gaps are irregular and come from the
event stream itself. Columns of every packet:
  0 gap in ms      time from the previous packet's last event to this packet's last event
                   (first packet: from the window start)
  1 ON fraction    share of events with polarity 1
  2, 3 mean x, y   position inside the box, 0 to 1
  4, 5 std x, y    spread inside the box, in box widths/heights
Columns 2 to 5 may reveal how the boxes were placed, so "time" uses only 0 and 1.
"""

import json
import os

import numpy as np

from src import phase1_fft as phase1
from src.data.cnn_windows import window_streams
from src.data.window_cache import CACHE

MAX_PACKETS = 500
WINDOW_MS = phase1.WIN * phase1.BIN_US / 1000
FEATURES = {"time": [0, 1], "space": [0, 1, 2, 3, 4, 5]}


def packets_from_events(t_ms, polarity, x, y):
    """Columns described above, from one window's events. t_ms is relative to the window start."""
    n = len(t_ms)
    if n == 0:
        return np.array([[WINDOW_MS, 0.5, 0.5, 0.5, 0.0, 0.0]], dtype=np.float32)
    size = int(np.ceil(n / MAX_PACKETS))
    edges = np.append(np.arange(0, n, size), n)
    ends = edges[1:]
    width = np.diff(edges)
    total = lambda v: np.add.reduceat(v, edges[:-1])
    mean = lambda v: total(v) / width
    mx, my = mean(x), mean(y)
    sx = np.sqrt(np.maximum(mean(x * x) - mx * mx, 0))
    sy = np.sqrt(np.maximum(mean(y * y) - my * my, 0))
    gap = np.diff(t_ms[ends - 1], prepend=0.0)
    return np.stack((gap, mean(polarity), mx, my, sx, sy), axis=1).astype(np.float32)


def build_sequence(sequence, shifts=None):
    """Packets and rows for one recording. With shifts {name: fn(t_ms, p, x, y) -> same},
    each window's events go through every fn first and {name: packets} is returned instead."""
    rows, events, streams, starts, stream_index, n_bins = window_streams(sequence)
    collected = [[] for _ in streams]
    for begin in range(0, events.shape[0], phase1.CHUNK):
        e = events[begin:begin + phase1.CHUNK]
        x, y = e["x"].astype(np.int32), e["y"].astype(np.int32)
        t = e["t"].astype(np.int64)
        p = e["p"].astype(np.float32)
        cache = {}
        for j, s in enumerate(streams):
            key = (id(s["key_t"]), s["tol"])
            if key not in cache:
                cache[key] = phase1.frame_index(s["key_t"], t, s["tol"])
            k, ok = cache[key]
            i = np.flatnonzero(ok & (t // phase1.BIN_US < n_bins) & s["ok_key"][k]
                               & (x >= s["x1"][k]) & (x < s["x2"][k])
                               & (y >= s["y1"][k]) & (y < s["y2"][k]))
            if not len(i):
                continue
            k = k[i]
            w = np.maximum(s["x2"][k] - s["x1"][k], 1)
            h = np.maximum(s["y2"][k] - s["y1"][k], 1)
            collected[j].append((t[i], p[i], ((x[i] - s["x1"][k]) / w).astype(np.float32),
                                 ((y[i] - s["y1"][k]) / h).astype(np.float32)))
    merged = [tuple(np.concatenate(col) for col in zip(*parts, strict=True)) if parts else None
              for parts in collected]
    out = {name: [] for name in (shifts or {None: None})}
    for row, start, j in zip(rows, starts, stream_index, strict=True):
        begin = int(start) * phase1.BIN_US
        if merged[j] is None:
            window = [np.empty(0)] * 4
        else:
            t = merged[j][0]
            lo, hi = np.searchsorted(t, [begin, begin + phase1.WIN * phase1.BIN_US])
            window = [col[lo:hi] for col in merged[j]]
        if len(window[0]) != int(row["event_count"]):
            raise ValueError(f"{sequence} at {row['t_start_us']} µs: {len(window[0])} events, expected {row['event_count']}")
        events_ms = ((window[0] - begin) / 1000.0, *window[1:])
        for name, fn in (shifts or {None: None}).items():
            out[name].append(packets_from_events(*(fn(*events_ms) if fn else events_ms)))
    return (out if shifts else out[None]), rows


def load_packets(sequences):
    """Return (packet arrays per window, log1p event counts, bool labels, rows)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    packets, rows = [], []
    for sequence in sequences:
        path = CACHE / f"packets_{sequence.replace('/', '_')}.npz"
        if path.is_file():
            with np.load(path) as data:
                offsets = data["offsets"]
                flat = data["packets"]
                packets.extend(np.split(flat, offsets[1:-1]))
                rows.extend(json.loads(str(data["rows"])))
            continue
        sequence_packets, sequence_rows = build_sequence(sequence)
        offsets = np.concatenate(([0], np.cumsum([len(p) for p in sequence_packets])))
        partial = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")    # rename, so parallel builders never read half a file
        np.savez(partial, packets=np.concatenate(sequence_packets), offsets=offsets, rows=json.dumps(sequence_rows))
        partial.replace(path)
        packets.extend(sequence_packets)
        rows.extend(sequence_rows)
    log_count = np.log1p(np.array([float(row["event_count"]) for row in rows], dtype=np.float32))
    labels = np.array([int(row["label"]) for row in rows], dtype=bool)
    return packets, log_count, labels, rows
