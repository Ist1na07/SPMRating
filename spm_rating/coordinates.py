"""Smooth near-synchronous boundary coordinates."""
import numpy as np

def coordinate(s, bd, pos, tau):
    if not 0 < bd < 7:
        return np.full(len(pos), float(min(max(bd, 0), 6)))
    q = s.tg[pos]
    weights = []
    for col in [bd - 1, bd]:
        times = s.tg[s.col_chain[col]['pos']]
        if len(times):
            hi = np.minimum(np.searchsorted(times, q), len(times) - 1)
            lo = np.maximum(hi - 1, 0)
            d = np.minimum(abs(q - times[hi]), abs(q - times[lo]))
            u = np.clip(d / tau, 0.0, 1.0)
            weights.append(np.cos(np.pi * u / 2.0) ** 2)
        else:
            weights.append(np.zeros(len(q)))
    left, right = weights
    return ((bd - 1) * left + bd * right) / np.maximum(left + right, 1e-12)
