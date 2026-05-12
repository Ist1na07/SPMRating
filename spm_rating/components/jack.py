"""
SPM Rating — Jack difficulty (Jbar).

Ported from SunnyRework algorithm.py.
Measures same-column rapid note intervals with nonlinear speed scaling.
"""

import numpy as np
from ..utils import smooth_on_corners


def compute_Jbar(K, x, note_seq_by_column, base_corners, aggregation_power=5, multi_jack_boost=0.0):
    """
    Compute Jbar — same-column jack difficulty.

    For each column, computes per-interval jack difficulty based on
    the time gap between consecutive notes in the same column.
    Smooths over 500ms window and aggregates across columns.

    Args:
        K: number of columns
        x: hit leniency
        note_seq_by_column: list of lists — notes grouped by column
        base_corners: time grid
        aggregation_power: power for cross-column aggregation (default 5)
        multi_jack_boost: boost factor when multiple columns have jacks simultaneously (default 0)

    Returns:
        delta_ks: dict {k: array} — per-column delta values (normalized interval)
        Jbar: array on base_corners
    """
    J_ks = {k: np.zeros(len(base_corners)) for k in range(K)}
    delta_ks = {k: np.full(len(base_corners), 1e9) for k in range(K)}

    def jack_nerfer(delta):
        return 1 - 7e-5 * (0.15 + abs(delta - 0.08)) ** (-4)

    for k in range(K):
        notes = note_seq_by_column[k]
        for i in range(len(notes) - 1):
            start = notes[i][1]
            end = notes[i + 1][1]
            left_idx = np.searchsorted(base_corners, start, side='left')
            right_idx = np.searchsorted(base_corners, end, side='left')
            idx = np.arange(left_idx, right_idx)
            if len(idx) == 0:
                continue
            delta = 0.001 * (end - start)
            val = (delta ** (-1)) * (delta + 0.11 * x ** (1 / 4)) ** (-1)
            J_val = val * jack_nerfer(delta)
            J_ks[k][idx] = J_val
            delta_ks[k][idx] = delta

    # Smooth each column
    Jbar_ks = {}
    for k in range(K):
        Jbar_ks[k] = smooth_on_corners(base_corners, J_ks[k], window=500, scale=0.001, mode='sum')

    # Aggregate: weighted average across columns
    Jbar = np.empty(len(base_corners))
    for i, s in enumerate(base_corners):
        vals = [Jbar_ks[k][i] for k in range(K)]
        weights = [1 / delta_ks[k][i] for k in range(K)]
        num = sum((max(v, 0) ** aggregation_power) * w for v, w in zip(vals, weights))
        den = sum(weights)
        Jbar[i] = num / max(1e-9, den)
        Jbar[i] = Jbar[i] ** (1 / aggregation_power)
        # Multi-column jack boost: chordjack patterns reward simultaneous jacks across columns
        if multi_jack_boost > 1e-12:
            active_count = sum(1 for v in vals if v > 1e-9)
            if active_count >= 2:
                Jbar[i] *= (1 + multi_jack_boost * (active_count - 1))

    return delta_ks, Jbar, Jbar_ks


def aggregate_Jbar(K, Jbar_ks, delta_ks, base_corners, aggregation_power=5, multi_jack_boost=0.0):
    """
    Re-aggregate Jbar from precomputed per-column smoothed values.

    Vectorized NumPy implementation: ~100x faster than per-corner Python loop.

    Args:
        K: number of columns
        Jbar_ks: dict {k: array} — smoothed per-column jack values
        delta_ks: dict {k: array} — per-column delta values
        base_corners: time grid
        aggregation_power: power for cross-column aggregation
        multi_jack_boost: boost for multi-column simultaneous jacks

    Returns:
        Jbar: array on base_corners
    """
    n = len(base_corners)
    vals = np.array([Jbar_ks[k] for k in range(K)])      # (K, n)
    deltas = np.array([delta_ks[k] for k in range(K)])   # (K, n)
    weights = 1.0 / np.maximum(deltas, 1e-12)             # (K, n)
    vals_pos = np.maximum(vals, 0)                        # (K, n)

    num = np.sum((vals_pos ** aggregation_power) * weights, axis=0)  # (n,)
    den = np.sum(weights, axis=0)                                    # (n,)
    Jbar = (num / np.maximum(den, 1e-9)) ** (1.0 / aggregation_power)

    if multi_jack_boost > 1e-12:
        active_count = np.sum(vals > 1e-9, axis=0)  # (n,)
        mask = active_count >= 2
        Jbar[mask] *= (1.0 + multi_jack_boost * (active_count[mask] - 1))

    return Jbar
