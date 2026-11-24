"""Remove packets at random to imitate a lower or uneven event rate.

A surviving packet's gap becomes the time since the previous surviving packet's
last event, so the timestamps stay real. At least one packet always survives.
The event count is recomputed from the surviving packets.
"""

import numpy as np

from src.data.event_packets import MAX_PACKETS


def packet_sizes(n_events, n_packets):
    """Events per packet, as packets_from_events made them."""
    if n_events == 0:
        return np.zeros(1)
    size = int(np.ceil(n_events / MAX_PACKETS))
    sizes = np.full(n_packets, size, dtype=np.float64)
    sizes[-1] = n_events - size * (n_packets - 1)
    return sizes


def drop(packets, n_events, rate, rng):
    """Return (kept packets, surviving event count) for one window."""
    if rate == 0:
        return packets, float(n_events)
    keep = rng.random(len(packets)) >= rate
    if not keep.any():
        keep[rng.integers(len(packets))] = True
    ends = np.cumsum(packets[:, 0].astype(np.float64))
    kept = packets[keep].copy()
    kept[:, 0] = np.diff(ends[keep], prepend=0.0)
    return kept, float(packet_sizes(n_events, len(packets))[keep].sum())


def drop_all(packets, log_count, rate, seed):
    """Drop packets from every window with a fixed seed. Returns packets and log1p counts."""
    rng = np.random.default_rng(seed)
    pairs = [drop(p, np.expm1(c).round(), rate, rng) for p, c in zip(packets, log_count, strict=True)]
    return [p for p, _ in pairs], np.log1p(np.array([n for _, n in pairs], dtype=np.float32))
