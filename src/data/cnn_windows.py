"""Rebuild the Phase 1 event-count windows from annotated FRED recordings."""

import csv
import json

import numpy as np

from src import phase1_fft as phase1


def load_splits() -> tuple[list[str], list[str]]:
    """Return whole recordings for CNN training and validation."""
    with (phase1.OUT_DIR / "phase1_sequences.json").open() as source:
        splits = json.load(source)
    train, validation = splits["tune"], splits["heldout"]
    reserved = set(splits["challenging"]) | set(splits["development"])
    if set(train) & set(validation) or reserved & (set(train) | set(validation)):
        raise ValueError("Training, validation, and reserved recordings must be disjoint")
    return train, validation


def window_streams(sequence: str):
    """Return everything needed to find the events of each drone and active window.

    Returns (rows, events, streams, start bin per row, stream index per row, n_bins).
    """
    with phase1.csv_path(sequence).open(newline="") as source:
        rows = [
            row for row in csv.DictReader(source)
            if row["background_type"] in {"drone", "active"}
        ]
    if not rows:
        raise ValueError(f"{sequence} has no drone or active background windows")

    events = np.load(phase1.events_path(sequence), mmap_mode="r")
    tracks = phase1.load_tracks(phase1.seq_dir(sequence) / "coordinates.txt")
    end_us = int(events["t"][-1])
    n_bins = end_us // phase1.BIN_US + 1
    chunk_us = int(phase1.ACTIVE_CHUNK_S * 1e6)
    chunk_starts = np.arange(end_us // chunk_us + 1, dtype=np.int64) * chunk_us

    # Phase 1 keeps each active box fixed for its five-second placement.
    # Its integer pixel coordinates can be recovered exactly from the CSV.
    active_boxes = {
        track_id: np.full((phase1.N_ACTIVE, len(chunk_starts), 4), np.nan)
        for track_id in tracks
    }
    starts = []
    stream_keys = []
    for row in rows:
        start_us, stop_us = int(row["t_start_us"]), int(row["t_end_us"])
        start, remainder = divmod(start_us, phase1.BIN_US)
        if (remainder or stop_us - start_us != phase1.WIN * phase1.BIN_US
                or start < 0 or start + phase1.WIN > n_bins):
            raise ValueError(f"{sequence} has an invalid window at {start_us} µs")
        starts.append(start)
        track_id, name = int(row["track_id"]), row["box_name"]
        stream_keys.append((track_id, name))
        if row["background_type"] == "active":
            chunk = start_us // chunk_us
            if (stop_us - 1) // chunk_us != chunk:
                raise ValueError(f"{sequence} active box spans placements at {start_us} µs")
            slot = int(name.removeprefix("active")) - 1
            corners = np.array([float(row[f"box_{coord}"]) for coord in ("x1", "y1", "x2", "y2")])
            previous = active_boxes[track_id][slot, chunk]
            if not np.isnan(previous[0]) and not np.array_equal(previous, corners):
                raise ValueError(f"{sequence} active box changes within placement at {start_us} µs")
            active_boxes[track_id][slot, chunk] = corners

    streams = [
        stream for stream in phase1.build_streams(tracks, chunk_starts, active_boxes)
        if stream["kind"] in {"drone", "active"}
    ]
    indices = {(stream["track_id"], stream["name"]): index for index, stream in enumerate(streams)}
    stream_index = [indices[key] for key in stream_keys]
    return rows, events, streams, starts, stream_index, n_bins


def load_windows(sequence: str) -> tuple[np.ndarray, list[dict[str, str]]]:
    """Return the counts and source rows for drone and busy-background windows."""
    rows, events, streams, starts, stream_index, n_bins = window_streams(sequence)
    counts = phase1.count_events(events, streams, n_bins)
    windows = np.empty((len(rows), phase1.WIN), dtype=np.float32)
    for index, (stream, start, row) in enumerate(zip(stream_index, starts, rows, strict=True)):
        window = counts[stream, start:start + phase1.WIN]
        expected = int(row["event_count"])
        actual = int(window.sum())
        if actual != expected:
            raise ValueError(
                f"{sequence} stream {stream} at {row['t_start_us']} µs: counted {actual}, expected {expected}"
            )
        windows[index] = window
    return windows, rows
