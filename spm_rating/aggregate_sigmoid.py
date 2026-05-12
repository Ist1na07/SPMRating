"""
SPM Rating — Sigmoid-based player accuracy aggregation.

Replaces the percentile + power-mean aggregation with a physically
motivated player accuracy model:

  A(d) = A_min + (A_max - A_min) / (C + e^(k(d - D)))

A player who achieves reference accuracy at difficulty D will have
accuracy A(d) on segments of difficulty d. The overall difficulty is
found by solving:

  Σ w_i / (C + e^(k(D_i - D))) = total_weight / (C + 1)

using bisection, after compressing the D(t) array into ~N segments
by difficulty to reduce computation.

Reference: Ist1na_7KRating (original sigmoid concept).
"""

import numpy as np


def segment_by_difficulty(D_all, weights, n_segments=30):
    """
    Compress D_all array into n_segments of similar difficulty.

    Sorts by D, divides into equal-cumulative-weight segments.
    Each segment is represented by its weighted-mean D and total weight.

    Args:
        D_all: 1D array of instantaneous difficulty values
        weights: 1D array of per-point weights (C * gap)
        n_segments: number of output segments (10-100)

    Returns:
        D_seg: array of shape (n_segments,)  — representative D per segment
        w_seg: array of shape (n_segments,)  — total weight per segment
    """
    n = len(D_all)
    if n == 0:
        return np.array([]), np.array([])

    # Sort by difficulty
    sort_idx = np.argsort(D_all)
    D_sorted = D_all[sort_idx]
    w_sorted = weights[sort_idx].astype(float)

    # Cumulative weight
    cum_w = np.cumsum(w_sorted)
    total_w = cum_w[-1]

    if total_w <= 0:
        return np.array([float(np.mean(D_all))]), np.array([1.0])

    # Divide into n_segments equal-weight buckets
    n_seg = min(n_segments, n)
    # Target cumulative weights at segment boundaries
    boundaries = np.linspace(0, total_w, n_seg + 1)

    D_seg = np.empty(n_seg)
    w_seg = np.empty(n_seg)

    for i in range(n_seg):
        lo = boundaries[i]
        hi = boundaries[i + 1]
        # Find indices in this bucket
        start = int(np.searchsorted(cum_w, lo, side='right'))
        end = int(np.searchsorted(cum_w, hi, side='right'))
        if end <= start:
            end = start + 1
        bucket_w = w_sorted[start:end]
        bucket_D = D_sorted[start:end]
        w_seg[i] = float(np.sum(bucket_w))
        if w_seg[i] > 0:
            D_seg[i] = float(np.sum(bucket_D * bucket_w) / w_seg[i])
        else:
            D_seg[i] = float(np.mean(bucket_D)) if len(bucket_D) > 0 else 0.0

    return D_seg, w_seg


def _sigmoid_sum(D_seg, w_seg, D_target, k, C):
    """Compute f(D_target) = Σ w_i/(C + e^(k(D_i - D_target)))."""
    # Clip exponent for numerical stability
    arg = np.clip(k * (D_seg - D_target), -50, 50)
    contributions = w_seg / (C + np.exp(arg))
    return float(np.sum(contributions))


def solve_D_bisection(D_seg, w_seg, k=0.5, C=4.0,
                      gamma=0.2, high_weight_power=0.0,
                      delta=5.0, tol=0.0001, max_iter=100):
    """
    Solve for D using bisection.

    Full model: A(d) = A_min + (A_max - A_min) / (C + e^(k(d - D)))

    Solving: Σ w_i/(C + e^(k(D_i - D))) = total_weight * gamma

    where gamma = (A_ref - A_min) / (A_max - A_min) is the accuracy
    fraction the reference player achieves on a perfectly matched segment.

    If high_weight_power > 0, weights are modified: w'_i = w_i * D_i^power,
    giving more influence to high-difficulty segments.

    f(D) is monotonically increasing in D (as D increases, D_i - D
    decreases, denominator decreases, contribution increases).

    Args:
        D_seg: segment difficulty values
        w_seg: segment weights
        k: sigmoid steepness (x-scaling)
        C: curve shape / denominator offset
        gamma: reference accuracy fraction (y-scaling control)
        high_weight_power: power for high-D weight emphasis (0 = equal)
        delta: search margin beyond [min(D), max(D)]
        tol: convergence tolerance
        max_iter: max bisection iterations

    Returns:
        D_solution: solved overall difficulty
        n_iter: iterations used
    """
    # Apply high-D weighting: w'_i = w_i * D_i^power
    if high_weight_power > 1e-9:
        w_seg = w_seg * (np.maximum(D_seg, 0.01) ** high_weight_power)

    total_weight = float(np.sum(w_seg))
    if total_weight <= 0:
        return float(np.mean(D_seg)) if len(D_seg) > 0 else 0.0, 0

    target = total_weight * gamma

    lo = float(np.min(D_seg)) - delta
    hi = float(np.max(D_seg)) + delta

    # Check bounds
    f_lo = _sigmoid_sum(D_seg, w_seg, lo, k, C)
    f_hi = _sigmoid_sum(D_seg, w_seg, hi, k, C)

    if f_lo >= target:
        return lo, 0
    if f_hi <= target:
        return hi, 0

    n_iter = 0
    while hi - lo > tol and n_iter < max_iter:
        mid = (lo + hi) / 2.0
        f_mid = _sigmoid_sum(D_seg, w_seg, mid, k, C)
        if f_mid < target:
            lo = mid
        else:
            hi = mid
        n_iter += 1

    return (lo + hi) / 2.0, n_iter


def _compute_effective_weights(all_corners, C_arr):
    """Compute per-point time-weighted note density weights."""
    gaps = np.empty_like(all_corners, dtype=float)
    gaps[0] = (all_corners[1] - all_corners[0]) / 2.0
    gaps[-1] = (all_corners[-1] - all_corners[-2]) / 2.0
    gaps[1:-1] = (all_corners[2:] - all_corners[:-2]) / 2.0
    return C_arr * gaps


def _compute_total_notes(note_seq, LN_seq):
    """Compute effective note count with LN length bonus."""
    total = len(note_seq) + 0.5 * sum(
        min(t - h, 1000) / 200 for (_, h, t) in LN_seq
    )
    return total


def _rescale_high(sr, threshold=9, divisor=1.2):
    """Compress SR above threshold."""
    if sr <= threshold:
        return sr
    return threshold + (sr - threshold) * (1.0 / divisor)


def compute_SR_sigmoid(all_corners, C_arr, D_all, total_notes, LN_seq,
                       n_segments=30,
                       sigmoid_k=0.5, sigmoid_C=4.0,
                       sigmoid_gamma=0.2,
                       sigmoid_high_power=0.0,
                       bisect_tol=0.0001, bisect_delta=5.0,
                       note_norm_N0=60,
                       rescale_threshold=9, rescale_divisor=1.2,
                       global_scale=0.975):
    """
    Compute Star Rating using sigmoid player-accuracy aggregation.

    Full model: A(d) = A_min + (A_max - A_min) / (C + e^(k(d - D)))
    gamma = (A_ref - A_min) / (A_max - A_min)

    If sigmoid_high_power > 0, segment weights are scaled by D_i^power
    to emphasize high-difficulty segments.

    Args:
        all_corners: time grid
        C_arr: note count per time point (on all_corners)
        D_all: instantaneous difficulty per time point
        total_notes: effective note count (with LN bonus)
        LN_seq: LN notes (for total_notes if needed)
        n_segments: number of difficulty segments for compression
        sigmoid_k: sigmoid steepness (x-scaling)
        sigmoid_C: curve shape / denominator offset
        sigmoid_gamma: reference accuracy fraction (y-scaling)
        sigmoid_high_power: power for high-D weight emphasis
        bisect_tol: bisection convergence tolerance
        bisect_delta: search margin beyond D range
        note_norm_N0: note count normalization offset
        rescale_threshold, rescale_divisor: high-SR compression
        global_scale: final output scale

    Returns:
        SR: final star rating
        details: dict with diagnostic values
    """
    # 1. Effective weights
    eff_w = _compute_effective_weights(all_corners, C_arr)

    # 2. Segment
    D_seg, w_seg = segment_by_difficulty(D_all, eff_w, n_segments)

    if len(D_seg) == 0:
        return 0.0, {"error": "empty segmentation"}

    # 3. Solve for D by bisection
    D_solved, n_iter = solve_D_bisection(
        D_seg, w_seg,
        k=sigmoid_k, C=sigmoid_C, gamma=sigmoid_gamma,
        high_weight_power=sigmoid_high_power,
        delta=bisect_delta, tol=bisect_tol,
    )

    # 4. Raw SR = solved D (which is in the same units as D_all)
    SR = float(D_solved)

    # 5. Note count normalization (same as original)
    n_eff = total_notes
    SR *= n_eff / (n_eff + note_norm_N0)

    # 6. Rescale + global scale (same as original)
    SR = _rescale_high(SR, threshold=rescale_threshold, divisor=rescale_divisor)
    SR *= global_scale

    details = {
        "D_solved": D_solved,
        "n_segments": len(D_seg),
        "n_bisect_iters": n_iter,
        "D_min": float(np.min(D_seg)) if len(D_seg) > 0 else 0.0,
        "D_max": float(np.max(D_seg)) if len(D_seg) > 0 else 0.0,
        "D_weighted_mean": float(np.sum(D_seg * w_seg) / np.sum(w_seg)) if np.sum(w_seg) > 0 else 0.0,
        "total_weight": float(np.sum(w_seg)),
        "n_effective": n_eff,
        "aggregation": "sigmoid",
        "sigmoid_k": sigmoid_k, "sigmoid_C": sigmoid_C, "sigmoid_gamma": sigmoid_gamma,
    }

    return SR, details
