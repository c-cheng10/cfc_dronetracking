"""Pair each drone window with a background window holding the same event count."""

import numpy as np

from src import phase1_fft as phase1

TOLERANCE = 0.15      # largest allowed gap in log1p(event count)


def matched_pairs(counts, labels, rows, seed=0):
    """One-to-one pairs within a recording. Returns drone and background indices."""
    log_count = np.log1p(counts.sum(1))
    sequence = np.array([row["sequence"] for row in rows])
    rng = np.random.default_rng(seed)
    positives, negatives = [], []
    for name in dict.fromkeys(sequence):
        drone = np.flatnonzero(labels & (sequence == name))
        background = np.flatnonzero(~labels & (sequence == name))
        rng.shuffle(drone)
        free = np.ones(len(background), dtype=bool)
        for i in drone:
            distance = np.where(free, np.abs(log_count[background] - log_count[i]), np.inf)
            best = int(distance.argmin()) if len(distance) else -1
            if best >= 0 and distance[best] < TOLERANCE:
                free[best] = False
                positives.append(i)
                negatives.append(background[best])
    return np.array(positives), np.array(negatives)


def dispersion(counts):
    """Variance over mean of the bin counts, in logs. Ignores the order of the bins."""
    return np.log((counts.var(1) + 1e-6) / (counts.mean(1) + 1e-6))


def both_aucs(scores, labels, pairs):
    return {"auc": float(phase1.auc(scores[labels], scores[~labels])),
            "matched_auc": float(phase1.auc(scores[pairs[0]], scores[pairs[1]]))}
