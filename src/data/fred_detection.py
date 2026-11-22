"""Prepare full-view event frames and boxes from stored FRED recordings."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from src import phase1_fft as phase1


SCALE = 2
HEIGHT = phase1.SENSOR_H // SCALE
WIDTH = phase1.SENSOR_W // SCALE
FRAME_US = 33_300
HOP_US = 200_000
CACHE_VERSION = 1


def _sample_annotations(tracks, first_us, last_us):
    # Use exact annotation times so all boxes come from the same source frame.
    # A missing row is ambiguous because coordinates.txt has no empty-frame flag.
    if not tracks:
        return [], []
    annotation_times = np.unique(np.concatenate([track["t"] for track in tracks.values()]))
    complete = (annotation_times >= first_us + FRAME_US) & (annotation_times <= last_us)
    for track in tracks.values():
        track_times = track["t"]
        in_lifetime = (annotation_times >= track_times[0]) & (annotation_times <= track_times[-1])
        complete &= ~in_lifetime | np.isin(annotation_times, track_times)

    # Pick the nearest complete annotation to each fixed 5 Hz sampling time.
    # The half-frame tolerance prevents a sample in a gap using a distant box.
    eligible = annotation_times[complete]
    if not len(eligible):
        return [], []
    grid = np.arange(0, last_us + 1, HOP_US, dtype=np.int64)
    right = np.searchsorted(eligible, grid)
    left = eligible[np.maximum(right - 1, 0)]
    right = eligible[np.minimum(right, len(eligible) - 1)]
    nearest = np.where(grid - left <= right - grid, left, right)
    nearest[grid < eligible[0]] = eligible[0]
    nearest[grid > eligible[-1]] = eligible[-1]
    times = nearest[np.abs(nearest - grid) <= FRAME_US // 2]

    boxes_at_time = {time: [] for time in times}
    for track in tracks.values():
        indices = np.searchsorted(track["t"], times)
        indices = np.minimum(indices, len(track["t"]) - 1)
        present = track["t"][indices] == times
        coords = np.column_stack([track[key][indices[present]] for key in ("x1", "y1", "x2", "y2")])
        for time, box in zip(times[present], coords, strict=True):
            boxes_at_time[time].append(box)

    valid_times = []
    boxes = []
    for time, frame_boxes in boxes_at_time.items():
        corners = np.asarray(frame_boxes, dtype=np.float32) / SCALE
        # Dropping the frame avoids silently losing a labelled instance.
        if np.any(corners[:, 2:] <= corners[:, :2]):
            continue
        valid_times.append(time)
        boxes.append(corners)
    return np.asarray(valid_times, dtype=np.int64), boxes


def _build_sequence(sequence, event_path, annotation_path):
    events = np.load(event_path, mmap_mode="r", allow_pickle=False)
    if not len(events):
        raise ValueError(f"{sequence} has no recorded events")
    tracks = phase1.load_tracks(annotation_path)
    timestamps = events["t"]
    times, boxes = _sample_annotations(tracks, int(timestamps[0]), int(timestamps[-1]))
    if not len(times):
        raise ValueError(f"{sequence} has no fully annotated sample times")

    # FRED stores time as uint32. Match that dtype so searchsorted can use the mmap
    # view directly, without converting the recording's entire timestamp column.
    starts = np.searchsorted(timestamps, (times - FRAME_US).astype(timestamps.dtype))
    ends = np.searchsorted(timestamps, times.astype(timestamps.dtype))
    images = np.empty((len(times), 2, HEIGHT, WIDTH), dtype=np.uint8)
    for image, start, end in zip(images, starts, ends, strict=True):
        frame = events[start:end]
        x = frame["x"].astype(np.int32) // SCALE
        y = frame["y"].astype(np.int32) // SCALE
        polarity = frame["p"].astype(np.int32)
        valid = (x < WIDTH) & (y < HEIGHT) & (polarity < 2)
        indices = (polarity[valid] * HEIGHT + y[valid]) * WIDTH + x[valid]
        counts = np.bincount(indices, minlength=image.size).reshape(image.shape)
        image[:] = counts.clip(max=255)
    return images, boxes


def _load_sequence(sequence, cache_dir):
    event_path = phase1.events_path(sequence)
    annotation_path = phase1.seq_dir(sequence) / "coordinates.txt"
    source_stats = [path.stat() for path in (event_path, annotation_path)]
    cache_key = np.array([CACHE_VERSION, *(item for stat in source_stats for item in (stat.st_size, stat.st_mtime_ns))])
    cache_path = cache_dir / f"{sequence.replace('/', '_')}.npz"
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as data:
            if np.array_equal(data["cache_key"], cache_key):
                offsets = data["offsets"]
                boxes = data["boxes"]
                return data["images"], list(np.split(boxes, offsets[1:-1]))

    images, boxes = _build_sequence(sequence, event_path, annotation_path)
    offsets = np.concatenate(([0], np.cumsum([len(frame_boxes) for frame_boxes in boxes])))
    with NamedTemporaryFile(dir=cache_dir, suffix=".npz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            np.savez(temporary, cache_key=cache_key, images=images, boxes=np.concatenate(boxes), offsets=offsets)
            temporary.close()
            temporary_path.replace(cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return images, boxes


def load_detection_split(sequences, cache_dir) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return uint8 [N, 2, 360, 640] counts and per-frame float32 xyxy boxes."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_parts = []
    boxes = []
    for sequence in sequences:
        images, frame_boxes = _load_sequence(sequence, cache_dir)
        image_parts.append(images)
        boxes.extend(frame_boxes)
    if not image_parts:
        raise ValueError("At least one FRED recording is required")
    return np.concatenate(image_parts) if len(image_parts) > 1 else image_parts[0], boxes
