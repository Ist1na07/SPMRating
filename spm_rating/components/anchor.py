"""
SPM Rating — Anchor / Unevenness (Abar).

Ported from SunnyRework algorithm.py.
Computes column usage imbalance — penalizes extreme favoritism of certain columns.
"""

import numpy as np
from ..utils import smooth_on_corners, interp_values


def compute_anchor(K, key_usage_400, base_corners):
    """
    Compute the anchor metric — measures how "anchored" the play is
    (concentration on a subset of columns).

    Args:
        K: number of columns
        key_usage_400: dict of {k: array} weighted key usage per column
        base_corners: time grid

    Returns:
        anchor: array of anchor values on base_corners
    """
    anchor = np.zeros(len(base_corners))
    for idx in range(len(base_corners)):
        counts = np.array([key_usage_400[k][idx] for k in range(K)])
        counts[::-1].sort()  # descending
        nonzero_counts = counts[counts != 0]
        if nonzero_counts.size > 1:
            ratios = nonzero_counts[1:] / nonzero_counts[:-1]
            walk = np.sum(nonzero_counts[:-1] * (1 - 4 * (0.5 - ratios) ** 2))
            max_walk = np.sum(nonzero_counts[:-1])
            anchor[idx] = walk / max_walk if max_walk > 0 else 0
        else:
            anchor[idx] = 0
    anchor = 1 + np.minimum(anchor - 0.18, 5 * (anchor - 0.22) ** 3)
    return anchor


def compute_Abar(K, delta_ks, active_columns, A_corners, base_corners):
    """
    Compute Abar — anchor unevenness difficulty.

    At each time point, looks at active columns and their delta differences
    to penalize patterns where column difficulty is highly uneven.

    Args:
        K: number of columns
        delta_ks: dict {k: array} per-column delta values (same column interval, scaled)
        active_columns: list of lists — which columns are active at each base_corner
        A_corners: time grid for A
        base_corners: reference time grid

    Returns:
        Abar: array on A_corners
        dks: dict {k: array} difference of adjacent delta_ks
    """
    # Compute dks — difference between adjacent active columns
    dks = {k: np.zeros(len(base_corners)) for k in range(K - 1)}
    for i in range(len(base_corners)):
        cols = active_columns[i]
        for j in range(len(cols) - 1):
            k0 = cols[j]
            k1 = cols[j + 1]
            dks[k0][i] = abs(delta_ks[k0][i] - delta_ks[k1][i]) + 0.4 * max(
                0, max(delta_ks[k0][i], delta_ks[k1][i]) - 0.11
            )

    A_step = np.ones(len(A_corners))

    for i, s in enumerate(A_corners):
        idx = np.searchsorted(base_corners, s)
        if idx >= len(base_corners):
            idx = len(base_corners) - 1
        cols = active_columns[idx]
        for j in range(len(cols) - 1):
            k0 = cols[j]
            k1 = cols[j + 1]
            d_val = dks[k0][idx]
            if d_val < 0.02:
                # Low difference: pattern is uniform — reduce difficulty
                A_step[i] *= min(
                    0.75 + 0.5 * max(delta_ks[k0][idx], delta_ks[k1][idx]), 1
                )
            elif d_val < 0.07:
                # Medium difference: moderate penalty
                A_step[i] *= min(
                    0.65 + 5 * d_val + 0.5 * max(delta_ks[k0][idx], delta_ks[k1][idx]), 1
                )

    Abar = smooth_on_corners(A_corners, A_step, window=250, mode='avg')
    return Abar, dks
