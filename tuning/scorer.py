"""
SPM Rating — Scoring function.

Primary loss: piecewise-linear with dead zone (v2):
  - delta/eps <= 0.5  →  loss = 0           (dead zone: within half-error = perfect)
  - 0.5 < delta/eps <= 1.0  →  loss = (ratio-0.5)*2    (linear 0→1)
  - delta/eps > 1.0   →  loss = 1.0 + (ratio-1.0)*6    (3x slope beyond error)

Legacy loss (ratio^2 / ratio^4) kept as score_single_legacy().
"""

import numpy as np


def score_single(sr_pred, sr_ref, error_bound):
    """
    Score a single prediction against reference (v2 piecewise-linear).

    Dead-zone piecewise-linear loss:
    - Within half error bound: loss = 0 (prediction is "perfect")
    - From half to full error bound: loss ramps linearly from 0 to 1
    - Beyond full error bound: loss grows at 3x slope (steeper penalty)

    Args:
        sr_pred: predicted star rating
        sr_ref: reference star rating
        error_bound: error tolerance (in SR units)

    Returns:
        score: lower is better, >= 0
    """
    delta = abs(sr_pred - sr_ref)
    eps = max(error_bound, 0.01)
    ratio = delta / eps

    if ratio <= 0.5:
        return 0.0
    elif ratio <= 1.0:
        return (ratio - 0.5) * 2.0
    else:
        return 1.0 + (ratio - 1.0) * 6.0


def score_single_legacy(sr_pred, sr_ref, error_bound):
    """
    Legacy scoring function (ratio^2 / ratio^4).

    - Inside error bounds: ratio^2 (gentle quadratic)
    - Outside error bounds: ratio^4 (steep quartic penalty)
    """
    delta = abs(sr_pred - sr_ref)
    eps = max(error_bound, 0.01)
    ratio = delta / eps

    if ratio <= 1.0:
        return ratio ** 2
    else:
        return ratio ** 4


def score_batch(predictions, references, error_bounds):
    """
    Score a batch of predictions.

    Args:
        predictions: list/array of predicted SR values
        references: list/array of reference SR values
        error_bounds: list/array of error tolerance per entry

    Returns:
        total_loss: mean score across all entries
        per_entry_scores: list of individual scores
        details: dict with extra statistics
    """
    scores = []
    inside_count = 0
    outside_count = 0

    for pred, ref, err in zip(predictions, references, error_bounds):
        s = score_single(pred, ref, err)
        scores.append(s)
        delta = abs(pred - ref)
        if delta <= err:
            inside_count += 1
        else:
            outside_count += 1

    total_loss = np.mean(scores)

    details = {
        "mean_loss": total_loss,
        "median_loss": np.median(scores),
        "max_loss": np.max(scores),
        "inside_error": inside_count,
        "outside_error": outside_count,
        "inside_ratio": inside_count / len(scores) if scores else 0,
    }

    return total_loss, scores, details


def score_weighted(predictions, references, error_bounds, weights=None):
    """
    Score with per-entry weights (e.g., to balance difficulty ranges).

    Args:
        predictions, references, error_bounds: as above
        weights: optional list of per-entry weights (uniform if None)

    Returns:
        weighted_loss: weighted mean score
    """
    if weights is None:
        weights = np.ones(len(predictions))

    weights = np.array(weights)
    weights = weights / weights.sum()

    total = 0.0
    for pred, ref, err, w in zip(predictions, references, error_bounds, weights):
        total += w * score_single(pred, ref, err)

    return total
