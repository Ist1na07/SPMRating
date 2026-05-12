"""
SPM Rating — Cross difficulty (Xbar).

Ported from SunnyRework algorithm.py.
Measures cross-column coordination difficulty.
V1: clones original behavior (no column distance weighting).
"""

import numpy as np
import heapq
from ..utils import smooth_on_corners


# Cross matrix weights — original SunnyRework values
CROSS_MATRIX = [
    [-1],
    [0.075, 0.075],
    [0.125, 0.05, 0.125],
    [0.125, 0.125, 0.125, 0.125],
    [0.175, 0.25, 0.05, 0.25, 0.175],
    [0.175, 0.25, 0.175, 0.175, 0.25, 0.175],
    [0.225, 0.35, 0.25, 0.05, 0.25, 0.35, 0.225],
    [0.225, 0.35, 0.25, 0.225, 0.225, 0.25, 0.35, 0.225],
    [0.275, 0.45, 0.35, 0.25, 0.05, 0.25, 0.35, 0.45, 0.275],
    [0.275, 0.45, 0.35, 0.25, 0.275, 0.275, 0.25, 0.35, 0.45, 0.275],
    [0.325, 0.55, 0.45, 0.35, 0.25, 0.05, 0.25, 0.35, 0.45, 0.55, 0.325],
]


def compute_Xbar(K, x, note_seq_by_column, active_columns, base_corners,
                 use_column_distance=False):
    """
    Compute Xbar — cross-column coordination difficulty.

    Args:
        K: number of columns
        x: hit leniency
        note_seq_by_column: notes grouped by column
        active_columns: which columns active at each time point
        base_corners: time grid
        use_column_distance: if True, apply column distance weighting (phase 3)

    Returns:
        Xbar: array on base_corners
    """
    X_ks = {k: np.zeros(len(base_corners)) for k in range(K + 1)}
    fast_cross = {k: np.zeros(len(base_corners)) for k in range(K + 1)}
    cross_coeff = CROSS_MATRIX[K]

    for k in range(K + 1):
        # Build note pairs:
        # k=0: pair within column 0
        # k=K: pair within last column
        # otherwise: merge of columns k-1 and k
        if k == 0:
            notes_in_pair = note_seq_by_column[0]
        elif k == K:
            notes_in_pair = note_seq_by_column[K - 1]
        else:
            notes_in_pair = list(heapq.merge(
                note_seq_by_column[k - 1],
                note_seq_by_column[k],
                key=lambda tup: tup[1]
            ))

        for i in range(1, len(notes_in_pair)):
            start = notes_in_pair[i - 1][1]
            end = notes_in_pair[i][1]
            idx_start = np.searchsorted(base_corners, start, side='left')
            idx_end = np.searchsorted(base_corners, end, side='left')
            idx = np.arange(idx_start, idx_end)
            if len(idx) == 0:
                continue
            delta = 0.001 * (notes_in_pair[i][1] - notes_in_pair[i - 1][1])
            val = 0.16 * max(x, delta) ** (-2)

            # If one side of the pair is inactive, reduce the contribution
            col_a = k - 1
            col_b = k
            if ((col_a not in active_columns[idx_start] and
                 col_a not in active_columns[idx_end]) or
                (col_b not in active_columns[idx_start] and
                 col_b not in active_columns[idx_end])):
                val *= (1 - cross_coeff[k])

            X_ks[k][idx] = val
            fast_cross[k][idx] = max(0, 0.4 * max(delta, 0.06, 0.75 * x) ** (-2) - 80)

    # Combine
    X_base = np.zeros(len(base_corners))
    for i in range(len(base_corners)):
        X_base[i] = (
            sum(X_ks[k][i] * cross_coeff[k] for k in range(K + 1))
            + sum(
                np.sqrt(fast_cross[k][i] * cross_coeff[k] *
                        fast_cross[k + 1][i] * cross_coeff[k + 1])
                for k in range(0, K)
            )
        )

    Xbar = smooth_on_corners(base_corners, X_base, window=500, scale=0.001, mode='sum')
    return Xbar
