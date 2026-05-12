"""
SPM Rating — Math utility functions.

Ported from SunnyRework algorithm.py with slight refactoring.
"""

import numpy as np
import math


def cumulative_sum(x, f):
    """
    Given sorted positions x (length N) and function values f defined
    piecewise constant on [x[i], x[i+1]), return an array F of cumulative
    integrals such that F[0]=0 and for i>=1:
        F[i] = sum_{j=0}^{i-1} f[j]*(x[j+1]-x[j])

    Vectorized for performance.
    """
    F = np.zeros(len(x))
    F[1:] = np.cumsum(f[:-1] * np.diff(x))
    return F


def query_cumsum(q, x, F, f):
    """
    Given cumulative data (x, F, f), return cumulative sum at point q.
    Assumes f is constant on each interval.
    """
    if q <= x[0]:
        return 0.0
    if q >= x[-1]:
        return F[-1]
    i = np.searchsorted(x, q) - 1
    return F[i] + f[i]*(q - x[i])


def _query_cumsum_vec(q_array, x, F, f):
    """Vectorized cumsum query for multiple points at once."""
    i = np.searchsorted(x, q_array) - 1
    below = q_array <= x[0]
    above = q_array >= x[-1]
    i = np.clip(i, 0, len(x) - 1)
    result = F[i] + f[i] * (q_array - x[i])
    result[below] = 0.0
    result[above] = F[-1]
    return result


def smooth_on_corners(x, f, window, scale=1.0, mode='sum'):
    """
    Apply a symmetric sliding window to piecewise-constant function f
    defined on positions x.

    Parameters:
        x: sorted 1D array of positions
        f: function values (piecewise constant on intervals defined by x)
        window: half-width of sliding window
        scale: multiplier applied to result
        mode: 'sum' -> g(s) = scale * ∫[s-window, s+window] f(t) dt
              'avg' -> g(s) = ∫f / (window length)

    Returns:
        g: array of same length as x, smoothed values

    Fully vectorized for performance.
    """
    F = cumulative_sum(x, f)

    left_bounds = np.maximum(x - window, x[0])
    right_bounds = np.minimum(x + window, x[-1])

    left_vals = _query_cumsum_vec(left_bounds, x, F, f)
    right_vals = _query_cumsum_vec(right_bounds, x, F, f)

    g = right_vals - left_vals
    if mode == 'avg':
        width = np.maximum(right_bounds - left_bounds, 1e-12)
        g = g / width
    else:
        g = scale * g
    return g


def interp_values(new_x, old_x, old_vals):
    """Linear interpolation from (old_x, old_vals) to new_x."""
    return np.interp(new_x, old_x, old_vals)


def step_interp(new_x, old_x, old_vals):
    """
    Zero-order hold: for each position in new_x, return value from old_vals
    corresponding to the greatest old_x <= new_x.
    """
    indices = np.searchsorted(old_x, new_x, side='right') - 1
    indices = np.clip(indices, 0, len(old_vals)-1)
    return old_vals[indices]


def rescale_high(sr, threshold=9, divisor=1.2):
    """Rescale SR above threshold to compress high end."""
    if sr <= threshold:
        return sr
    return threshold + (sr - threshold) * (1.0 / divisor)


def find_next_note_in_column(note, times, note_seq_by_column):
    """Find the next note in the same column after a given note."""
    k, h, t = note
    import bisect
    idx = bisect.bisect_left(times, h)
    if idx + 1 < len(note_seq_by_column[k]):
        return note_seq_by_column[k][idx + 1]
    return (0, 10**9, 10**9)


def LN_sum(a, b, LN_rep):
    """
    Compute cumulative LN body value between a and b using sparse representation.
    """
    import bisect
    points, cumsum, values = LN_rep
    i = bisect.bisect_right(points, a) - 1
    j = bisect.bisect_right(points, b) - 1

    total = 0.0
    if i == j:
        total = (b - a) * values[i]
    else:
        total += (points[i+1] - a) * values[i]
        total += cumsum[j] - cumsum[i+1]
        total += (b - points[j]) * values[j]
    return total


def logistic(x, midpoint=0, steepness=1):
    """Standard logistic function: 1 / (1 + exp(-steepness * (x - midpoint)))"""
    return 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))
