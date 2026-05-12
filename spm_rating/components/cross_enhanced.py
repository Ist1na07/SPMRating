"""
SPM Rating — Cross difficulty (Xbar) — Enhanced with column distance.

Extends the original SunnyRework cross difficulty with column distance
weighting. Now distinguishes same-hand vs cross-hand coordination.

Phase 4: Support RC/LN differentiated parameters via ln_ratio blending.
Precompute data → fast recompute at combine-time (like Shield/Inverse).

Optimization: base values precomputed, pairs grouped by k,
vectorized combine stage for fast runtime recomputation.
"""

import numpy as np
import heapq
from ..utils import smooth_on_corners


# Original cross matrix weights
_CROSS_MATRIX = [
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

# 7K hand mapping
_HAND_MAP = {0: "L", 1: "L", 2: "L", 3: "T", 4: "R", 5: "R", 6: "R"}


def _get_dist_weight(k1, k2, K, dist_exponent, same_hand_penalty, thumb_bridge):
    """Compute column distance weight between two columns."""
    if k1 < 0 or k2 < 0 or k1 >= K or k2 >= K:
        return 1.0
    raw_dist = abs(k1 - k2)
    if raw_dist == 0:
        return 1.0  # Same column: no cross-column modifier
    h1, h2 = _HAND_MAP.get(k1, ""), _HAND_MAP.get(k2, "")
    if h1 == h2 and h1 != "T":
        # Same hand: harder — closer columns = more interfering
        return 1.0 + same_hand_penalty * (1.0 / (raw_dist ** dist_exponent))
    elif h1 == "T" or h2 == "T":
        # Thumb bridge: thumb-to-hand transitions
        return 1.0 - thumb_bridge * (1.0 / max(raw_dist, 1))
    else:
        # Opposite hands: easier
        return 1.0 - same_hand_penalty * min(raw_dist / K, 1.0)


def _build_dist_w_cache(K, dist_exponent, same_hand_penalty, thumb_bridge):
    """Precompute dist_w for all possible column pairs."""
    cache = {}
    for k1 in range(K):
        for k2 in range(K):
            cache[(k1, k2)] = _get_dist_weight(
                k1, k2, K, dist_exponent, same_hand_penalty, thumb_bridge)
    return cache


# Map column pair to precomputed hand relationship type
# 0=same_col, 1=same_hand, 2=thumb_bridge, 3=opposite_hand
def _classify_col_pair(k1, k2, K):
    raw_dist = abs(k1 - k2)
    if raw_dist == 0:
        return 0  # same column — dist_w = 1.0 always
    h1, h2 = _HAND_MAP.get(k1, ""), _HAND_MAP.get(k2, "")
    if h1 == h2 and h1 != "T":
        return 1  # same hand
    elif h1 == "T" or h2 == "T":
        return 2  # thumb bridge
    else:
        return 3  # opposite hand


def precompute_cross_enhanced_data(K, x, note_seq_by_column, active_columns, base_corners):
    """
    Precompute structured data for fast runtime Xbar recomputation.

    Returns:
        {
            'K': K,
            'cross_coeff': list of K+1 floats,
            'pairs_by_k': [list of (idx_start, idx_end, col_type, raw_dist, val_base, fc_val), ...]
                          col_type: 0=same_col, 1=same_hand, 2=thumb, 3=opposite
            'base_corners_len': int,
        }
    """
    cross_coeff = _CROSS_MATRIX[K]
    n = len(base_corners)
    pairs_by_k = [[] for _ in range(K + 1)]

    for k in range(K + 1):
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
            idx_start = int(np.searchsorted(base_corners, start, side='left'))
            idx_end = int(np.searchsorted(base_corners, end, side='left'))
            if idx_end <= idx_start:
                continue

            n1 = notes_in_pair[i - 1]
            n2 = notes_in_pair[i]
            k1, k2 = int(n1[0]), int(n2[0])
            delta = 0.001 * (n2[1] - n1[1])

            # Classify column pair for fast runtime lookup
            col_type = _classify_col_pair(k1, k2, K)
            raw_dist = abs(k1 - k2)

            # Check active columns at boundaries
            col_a, col_b = k - 1, k
            not_active = (
                (col_a not in active_columns[idx_start] and
                 col_a not in active_columns[idx_end]) or
                (col_b not in active_columns[idx_start] and
                 col_b not in active_columns[idx_end])
            )

            # Precompute base values (param-independent parts)
            val_base = 0.16 * max(x, delta) ** (-2)
            if not_active:
                val_base *= (1 - cross_coeff[k])
            fc_val = max(0, 0.4 * max(delta, 0.06, 0.75 * x) ** (-2) - 80)

            pairs_by_k[k].append((idx_start, idx_end, col_type, raw_dist, val_base, fc_val))

    # Convert pairs_by_k to tuple arrays for faster iteration
    pairs_by_k_tup = tuple(pairs_by_k)

    return {
        'K': K,
        'cross_coeff': cross_coeff,
        'pairs_by_k': pairs_by_k_tup,
        'base_corners_len': n,
    }


def compute_Xbar_enhanced_fast(cross_data, base_corners, x,
                                dist_exponent=1.0, same_hand_penalty=0.3,
                                thumb_bridge=0.5):
    """
    Fast recompute of Xbar from precomputed cross_data.

    Uses precomputed base values + column pair classification
    for minimal runtime computation. Vectorized combine stage.

    Args:
        cross_data: from precompute_cross_enhanced_data()
        base_corners: base corner timestamps
        x: speed multiplier (from cache)
        dist_exponent, same_hand_penalty, thumb_bridge: current params

    Returns:
        Xbar: array on base_corners
    """
    K = cross_data['K']
    n = cross_data['base_corners_len']
    cross_coeff = cross_data['cross_coeff']
    pairs_by_k = cross_data['pairs_by_k']

    X_ks = {k: np.zeros(n) for k in range(K + 1)}
    fast_cross = {k: np.zeros(n) for k in range(K + 1)}

    for k in range(K + 1):
        pk = pairs_by_k[k]
        if not pk:
            continue
        Xk = X_ks[k]
        fck = fast_cross[k]
        for idx_s, idx_e, col_type, raw_dist, val_base, fc_val in pk:
            if col_type == 0:
                dist_w = 1.0  # same column
            elif col_type == 1:
                dist_w = 1.0 + same_hand_penalty * (1.0 / (raw_dist ** dist_exponent))
            elif col_type == 2:
                dist_w = 1.0 - thumb_bridge * (1.0 / max(raw_dist, 1))
            else:
                dist_w = 1.0 - same_hand_penalty * min(raw_dist / K, 1.0)
            Xk[idx_s:idx_e] = val_base * dist_w
            fck[idx_s:idx_e] = fc_val

    # Vectorized combine (was O(n*K) Python loop, now O(K) numpy ops)
    X_base = np.zeros(n)
    for k in range(K + 1):
        X_base += X_ks[k] * cross_coeff[k]
    for k in range(K):
        X_base += np.sqrt(np.maximum(
            fast_cross[k] * cross_coeff[k] *
            fast_cross[k + 1] * cross_coeff[k + 1], 0.0))

    Xbar = smooth_on_corners(base_corners, X_base, window=500, scale=0.001, mode='sum')
    return Xbar


def compute_Xbar_enhanced(K, x, note_seq_by_column, active_columns, base_corners,
                          dist_exponent=1.0, same_hand_penalty=0.3,
                          thumb_bridge=0.5):
    """
    Compute enhanced Xbar with column distance weighting. (Original direct method)

    Column layout (7K): L(0-2) Thumb(3) R(4-6)

    Args:
        dist_exponent: exponent for distance weighting
        same_hand_penalty: bonus difficulty for same-hand coordination
        thumb_bridge: how much the thumb bridges the hand gap (0=full, 1=none)

    Returns:
        Xbar: array on base_corners
    """
    X_ks = {k: np.zeros(len(base_corners)) for k in range(K + 1)}
    fast_cross = {k: np.zeros(len(base_corners)) for k in range(K + 1)}
    cross_coeff = _CROSS_MATRIX[K]

    for k in range(K + 1):
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

            n1 = notes_in_pair[i - 1]
            n2 = notes_in_pair[i]
            delta = 0.001 * (n2[1] - n1[1])

            dist_w = _get_dist_weight(int(n1[0]), int(n2[0]), K,
                                       dist_exponent, same_hand_penalty, thumb_bridge)
            val = 0.16 * dist_w * max(x, delta) ** (-2)

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
