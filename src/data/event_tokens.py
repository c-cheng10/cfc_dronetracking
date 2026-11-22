"""Turn 0.5 ms count windows into irregular sequences of non-empty bins.

Each token holds log1p(count) and the gap, in bins, since the previous token.
A closing zero token is added when the last bin is empty, so the tokens carry
exactly the information in the 1,000-bin window.
"""

import numpy as np
import torch


def tokenise(window):
    index = np.flatnonzero(window)
    if not len(index) or index[-1] != len(window) - 1:
        index = np.append(index, len(window) - 1)
    values = np.log1p(window[index])
    gaps = np.diff(index, prepend=-1)          # first gap counts from the window start
    return values.astype(np.float32), gaps.astype(np.float32)


def reconstruct(values, gaps, length):
    """Invert tokenise; used only to verify that no information is lost."""
    window = np.zeros(length, dtype=np.float32)
    window[np.cumsum(gaps).astype(int) - 1] = np.expm1(values)
    return window


def collate(tokens, constant_gap=False):
    """Pad to [B, T]. Returns values, gaps and a mask of real tokens."""
    longest = max(len(values) for values, _ in tokens)
    values = torch.zeros(len(tokens), longest)
    gaps = torch.ones(len(tokens), longest)
    mask = torch.zeros(len(tokens), longest, dtype=torch.bool)
    for row, (token_values, token_gaps) in enumerate(tokens):
        n = len(token_values)
        values[row, :n] = torch.from_numpy(token_values)
        if not constant_gap:
            gaps[row, :n] = torch.from_numpy(token_gaps)
        mask[row, :n] = True
    return values, gaps, mask
