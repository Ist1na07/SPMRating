"""
SPM Rating — Aggregation and final SR computation.

Ported from SunnyRework algorithm.py.
"""

import numpy as np
import pandas as pd


def compute_SR(all_corners, C_arr, D_all, total_notes, LN_seq,
               rescale=True,
               w_93=0.25, w_83=0.20, w_mean=0.55,
               coeff_93=0.88, coeff_83=0.94,
               mean_power=5,
               note_norm_N0=60,
               rescale_threshold=9, rescale_divisor=1.2,
               global_scale=0.975):
    """
    Compute final Star Rating from per-point difficulty.

    Uses percentile-weighted averaging: 93rd and 83rd percentiles
    weighted by local note count, plus power-weighted mean.

    Args:
        all_corners: time grid
        C_arr: note count on all_corners
        D_all: difficulty on all_corners
        total_notes: note count (with LN length bonus)
        LN_seq: LN notes for total_notes computation
        rescale: if True, apply final rescale
        w_93, w_83, w_mean: aggregation weights
        coeff_93, coeff_83: percentile scaling coefficients
        mean_power: power for weighted mean
        note_norm_N0: note count normalization offset
        rescale_threshold: threshold for high-SR rescale
        rescale_divisor: divisor for high-SR rescale
        global_scale: global output scale factor

    Returns:
        SR: final star rating
        details: dict with intermediate values
    """
    # Compute gaps for weighting
    gaps = np.empty_like(all_corners, dtype=float)
    gaps[0] = (all_corners[1] - all_corners[0]) / 2.0
    gaps[-1] = (all_corners[-1] - all_corners[-2]) / 2.0
    gaps[1:-1] = (all_corners[2:] - all_corners[:-2]) / 2.0

    effective_weights = C_arr * gaps

    # Sort by difficulty
    sort_idx = np.argsort(D_all)
    D_sorted = D_all[sort_idx]
    w_sorted = effective_weights[sort_idx]

    # Cumulative normalized weights
    cum_weights = np.cumsum(w_sorted)
    total_weight = cum_weights[-1]
    norm_cum_weights = cum_weights / total_weight

    target_percentiles = np.array([
        0.945, 0.935, 0.925, 0.915,
        0.845, 0.835, 0.825, 0.815,
    ])

    indices = np.searchsorted(norm_cum_weights, target_percentiles, side='left')
    indices = np.clip(indices, 0, len(D_sorted) - 1)

    percentile_93 = np.mean(D_sorted[indices[:4]])
    percentile_83 = np.mean(D_sorted[indices[4:8]])

    weighted_mean = (np.sum(D_sorted ** mean_power * w_sorted) / np.sum(w_sorted)) ** (1 / mean_power)

    # Final SR with tunable weights
    SR = (coeff_93 * percentile_93) * w_93 + (coeff_83 * percentile_83) * w_83 + weighted_mean * w_mean
    SR = SR ** 1.0 / (8 ** 1.0) * 8

    # Note count normalization
    n_effective = total_notes
    SR *= n_effective / (n_effective + note_norm_N0)

    if rescale:
        SR = _rescale_high(SR, threshold=rescale_threshold, divisor=rescale_divisor)
        SR *= global_scale

    details = {
        "percentile_93": percentile_93,
        "percentile_83": percentile_83,
        "weighted_mean": weighted_mean,
        "n_effective": n_effective,
    }

    return SR, details


def compute_total_notes(note_seq, LN_seq):
    """Compute effective note count with LN length bonus."""
    total = len(note_seq) + 0.5 * sum(
        min(t - h, 1000) / 200 for (_, h, t) in LN_seq
    )
    return total


def _rescale_high(sr, threshold=9, divisor=1.2):
    """Rescale SR above threshold."""
    if sr <= threshold:
        return sr
    return threshold + (sr - threshold) * (1.0 / divisor)
