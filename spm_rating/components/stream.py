"""
SPM Rating — Stream / Pressing difficulty (Pbar).

Ported from SunnyRework algorithm.py.
Measures overall note density with LN body weighting and stream booster.
"""

import numpy as np
from ..utils import smooth_on_corners, LN_sum


def compute_Pbar(K, x, note_seq, LN_rep, anchor, base_corners,
                 stream_booster_scale=1.7e-7):
    """
    Compute Pbar — stream/pressing difficulty.

    For each pair of consecutive notes, computes difficulty based on
    interval timing, LN body presence, and anchor modulation.

    Args:
        K: number of columns
        x: hit leniency
        note_seq: sorted list of (col, head, tail) tuples
        LN_rep: sparse LN body representation
        anchor: anchor values on base_corners
        base_corners: time grid
        stream_booster_scale: booster intensity (now tunable)

    Returns:
        Pbar: array on base_corners
    """
    def stream_booster(delta):
        if 160 < (7.5 / delta) < 360:
            return 1 + stream_booster_scale * ((7.5 / delta) - 160) * ((7.5 / delta) - 360) ** 2
        return 1

    P_step = np.zeros(len(base_corners))
    for i in range(len(note_seq) - 1):
        h_l = note_seq[i][1]
        h_r = note_seq[i + 1][1]
        delta_time = h_r - h_l

        if delta_time < 1e-9:
            # Dirac delta: simultaneous notes
            spike = 1000 * (0.02 * (4 / x - 24)) ** (1 / 4)
            left_idx = np.searchsorted(base_corners, h_l, side='left')
            left_idx = min(left_idx, len(base_corners) - 1)
            right_idx = np.searchsorted(base_corners, h_l, side='right')
            right_idx = min(right_idx, len(base_corners))
            idx = np.arange(left_idx, right_idx)
            if len(idx) > 0:
                P_step[idx] += spike
            continue

        left_idx = np.searchsorted(base_corners, h_l, side='left')
        right_idx = np.searchsorted(base_corners, h_r, side='left')
        idx = np.arange(left_idx, right_idx)
        if len(idx) == 0:
            continue

        delta = 0.001 * delta_time
        v = 1 + 6 * 0.001 * LN_sum(h_l, h_r, LN_rep)
        b_val = stream_booster(delta)

        if delta < 2 * x / 3:
            inc = delta ** (-1) * (
                0.08 * x ** (-1) * (1 - 24 * x ** (-1) * (delta - x / 2) ** 2)
            ) ** (1 / 4) * max(b_val, v)
        else:
            inc = delta ** (-1) * (
                0.08 * x ** (-1) * (1 - 24 * x ** (-1) * (x / 6) ** 2)
            ) ** (1 / 4) * max(b_val, v)

        P_step[idx] += np.minimum(
            inc * anchor[idx],
            np.maximum(inc, inc * 2 - 10)
        )

    Pbar = smooth_on_corners(base_corners, P_step, window=500, scale=0.001, mode='sum')
    return Pbar
