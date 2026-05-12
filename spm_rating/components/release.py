"""
SPM Rating — Release difficulty (Rbar).

Ported from SunnyRework algorithm.py.
V1: clones original behavior (LN tail intervals, release index I).
"""

import math
import numpy as np
from ..utils import smooth_on_corners


def compute_Rbar(K, x, note_seq_by_column, tail_seq, base_corners):
    """
    Compute Rbar — LN release difficulty.

    Measures difficulty of LN tail timing: release-to-next-note intervals
    and release-to-next-release intervals.

    Args:
        K: number of columns
        x: hit leniency
        note_seq_by_column: notes grouped by column
        tail_seq: LN notes sorted by tail time
        base_corners: time grid

    Returns:
        Rbar: array on base_corners
        I_list: list of release index values per tail
    """
    I_arr = np.zeros(len(base_corners))
    R_step = np.zeros(len(base_corners))

    times_by_column = {
        i: [note[1] for note in column]
        for i, column in enumerate(note_seq_by_column)
    }

    # Release Index: measures the timing quality of release-to-next-note
    I_list = []
    for i in range(len(tail_seq)):
        k, h_i, t_i = tail_seq[i]
        # Find next note in same column
        idx = 0
        times = times_by_column[k]
        for j, t in enumerate(times):
            if t >= h_i:
                idx = j
                break
        next_note = (0, 10**9, 10**9)
        if idx + 1 < len(note_seq_by_column[k]):
            next_note = note_seq_by_column[k][idx + 1]

        _, h_j, _ = next_note
        I_h = 0.001 * abs(t_i - h_i - 80) / x
        I_t = 0.001 * abs(h_j - t_i - 80) / x
        I_val = 2 / (2 + math.exp(-5 * (I_h - 0.75)) + math.exp(-5 * (I_t - 0.75)))
        I_list.append(I_val)

    # For each interval between successive tail times
    for i in range(len(tail_seq) - 1):
        t_start = tail_seq[i][2]
        t_end = tail_seq[i + 1][2]
        left_idx = np.searchsorted(base_corners, t_start, side='left')
        right_idx = np.searchsorted(base_corners, t_end, side='left')
        idx = np.arange(left_idx, right_idx)
        if len(idx) == 0:
            continue
        I_arr[idx] = 1 + I_list[i]
        delta_r = 0.001 * (tail_seq[i + 1][2] - tail_seq[i][2])
        R_step[idx] = 0.08 * (delta_r) ** (-0.5) * x ** (-1) * (
            1 + 0.8 * (I_list[i] + I_list[i + 1])
        )

    Rbar = smooth_on_corners(base_corners, R_step, window=500, scale=0.001, mode='sum')
    return Rbar, I_list
