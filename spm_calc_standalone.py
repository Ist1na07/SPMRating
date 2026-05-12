#!/usr/bin/env python
"""
SPM Rating — 独立 SR 计算器 (Sigmoid 玩家准度聚合)

单文件, 零外部依赖 (除 numpy)。放入任意目录即可使用。

用法:
  python spm_calc_standalone.py                     # 扫描当前目录的 .osu
  python spm_calc_standalone.py "D:/maps/"          # 扫描指定目录
  python spm_calc_standalone.py chart.osu           # 计算单张谱面

模型:
  - Enhanced 模式 (Cross/Release/Shield/Inverse 全量分量)
  - Sigmoid 准确度聚合 (k=1.56, C=4.0, γ=0.20)
  - 205 张 playtest 谱面调优, MAE=0.2253

构建时间: """ + __import__("datetime").datetime.now().strftime("%Y-%m-%d") + """
"""


import sys, os, json, time, glob, re, pickle, warnings
import numpy as np
import math
import bisect
import heapq
from collections import defaultdict
from dataclasses import dataclass, field
import pandas as pd

# scipy only needed for Nelder-Mead (optional)
try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None

# pandas only needed for playtest loading (optional)
try:
    import pandas as pd
except ImportError:
    pd = None


# ============================================================================
# Tuned Parameters (from tuned_params_sigmoid.json, MAE=0.2253)
# ============================================================================

TUNED_PARAMS = {
        "use_enhanced": True,
        "use_enhanced_release": 1,
        "use_column_distance": 1,
        "use_shield": 1,
        "use_inverse": 1,
        "use_stamina": 0,
        "use_comprehensiveness": 0,
        "D_gamma_e": 0.0,
        "w_mean": 0.572,
        "rescale_threshold": 9.54,
        "rescale_divisor": 2.0,
        "jack_aggregation_power": 3.98,
        "multi_jack_boost": 0.003,
        "Abar_scale": 1.0159974528704387,
        "inverse_peak_width": 2.0,
        "shield_anchor_mod": 0.8062210781592527,
        "shield_coord_factor": 1.0025438218559561,
        "stream_booster_scale": 1.75e-07,
        "cross_dist_exponent": 1.0,
        "cross_same_hand_penalty": 0.3,
        "cross_thumb_bridge_factor": 0.5768675268883898,
        "release_tail_coeff": 0.1301386626710443,
        "release_tail_to_tap": 2.8102788921489603,
        "release_same_col_bonus": 0.2510782776793202,
        "release_coord_exponent": 0.6855610516647912,
        "S_w1": 0.5127205484014595,
        "S_p": 1.1350727423836298,
        "alpha_P": 0.7243971981192697,
        "alpha_R": 28.68398732935888,
        "alpha_C": 9.597140096247783,
        "alpha_S": 0.5017118508840599,
        "alpha_V": 0.4348924525605874,
        "D_beta1": 1.1645638403719478,
        "D_beta2": 0.3889790083690824,
        "w_93": 0.1820607086443739,
        "w_83": 0.2338096935124762,
        "coeff_93": 0.9642799087643721,
        "coeff_83": 0.6025543835022324,
        "mean_power": 2.137380905141331,
        "note_norm_N0": 10.0,
        "global_scale": 1.0548667350346,
        "inv_amplitude": 3.448859525864742,
        "inv_tau": 32.502710313970894,
        "inv_power": 0.8537687907693771,
        "guide_depth": 0.7971215272422785,
        "guide_center": 78.53359863949761,
        "guide_width": 41.581395044572375,
        "cross_guide_scale": 0.5894104667719972,
        "inverse_same_col_bonus": 2.6084806897704493,
        "shield_tau_ms": 56.24289518646123,
        "release_seq_coeff": 0.07876946297853933,
        "lock_interaction_coeff": 0.13816209949139724,
        "cross_dist_exponent_rc": 0.9732421726222308,
        "cross_dist_exponent_ln": 0.974803204070658,
        "cross_same_hand_penalty_rc": 0.3613101110984365,
        "cross_same_hand_penalty_ln": 0.30422710249481105,
        "use_sigmoid_aggregation": 1,
        "agg_sigmoid_k": 1.5636,
        "agg_sigmoid_C": 3.9862,
        "agg_sigmoid_ref_gamma": 0.2011,
        "calib_a": 0.8899,
        "calib_b": 0.0436
    }
TUNED_PARAMS["use_sigmoid_aggregation"] = 1




# === anchor.py ===
"""
SPM Rating — Anchor / Unevenness (Abar).

Ported from SunnyRework algorithm.py.
Computes column usage imbalance — penalizes extreme favoritism of certain columns.
"""



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


# === jack.py ===
"""
SPM Rating — Jack difficulty (Jbar).

Ported from SunnyRework algorithm.py.
Measures same-column rapid note intervals with nonlinear speed scaling.
"""



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


# === cross.py ===
"""
SPM Rating — Cross difficulty (Xbar).

Ported from SunnyRework algorithm.py.
Measures cross-column coordination difficulty.
V1: clones original behavior (no column distance weighting).
"""



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


# === cross_enhanced.py ===
"""
SPM Rating — Cross difficulty (Xbar) — Enhanced with column distance.

Extends the original SunnyRework cross difficulty with column distance
weighting. Now distinguishes same-hand vs cross-hand coordination.

Phase 4: Support RC/LN differentiated parameters via ln_ratio blending.
Precompute data → fast recompute at combine-time (like Shield/Inverse).

Optimization: base values precomputed, pairs grouped by k,
vectorized combine stage for fast runtime recomputation.
"""



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


# === stream.py ===
"""
SPM Rating — Stream / Pressing difficulty (Pbar).

Ported from SunnyRework algorithm.py.
Measures overall note density with LN body weighting and stream booster.
"""



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


# === release.py ===
"""
SPM Rating — Release difficulty (Rbar).

Ported from SunnyRework algorithm.py.
V1: clones original behavior (LN tail intervals, release index I).
"""



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


# === release_enhanced.py ===
"""
SPM Rating — Release difficulty (Rbar) — Enhanced.

Major rework: treats LN tails as independent objects.
All releases contribute, not just overlapping ones.

Per experience.md §4(4):
  (a) Sequential releases harder than simultaneous — tail-to-tail intervals
  (b) U-shaped inverse curve — delegated to Vbar component
  (c) Cross-column stream guide — delegated to Vbar component
  (d) Shield-release interaction — other locked columns increase release difficulty
  (e) Short LN → reduced release difficulty (player can tap-release quickly)

Key improvements over original:
  1. Treat each LN tail as an independent time point
  2. Measure release-to-next-release intervals
  3. Measure release-to-next-tap intervals
  4. Apply column distance (coordination) weighting
  5. Short-LN release difficulty reduction
  6. Lock-hand interaction: release harder when other columns are locked
"""


# Hand layout for coordination weighting
_HAND_MAP = {0: "L", 1: "L", 2: "L", 3: "T", 4: "R", 5: "R", 6: "R"}

# Sentinel: no valid next event found
_NO_NEXT_EVENT = 10**9


def _coord_weight(k1, k2):
    """Coordination weight for release-to-next-event.
    Same-hand coordination is harder than opposite-hand.
    Per experience.md §4(4a):
      "同速度的213轨道依次放手比147轨道依次放手更难"
    """
    if k1 == k2:
        return 1.0
    h1, h2 = _HAND_MAP.get(k1, ""), _HAND_MAP.get(k2, "")
    if h1 == h2 and h1 != "T":
        return 0.8   # Same hand (both L or both R)
    elif h1 == "T" or h2 == "T":
        return 0.4   # Thumb bridge
    else:
        return 0.2   # Opposite hands


def _find_next_event(t_i, note_seq, tail_seq):
    """Find the closest event (head or tail) after time t_i.

    Returns:
        (next_time, next_col, next_is_tail) or (NO_NEXT_EVENT, -1, False)
    """
    next_time = _NO_NEXT_EVENT
    next_col = -1
    next_is_tail = False

    # Find closest note head after t_i
    for n in note_seq:
        if n[1] > t_i:
            next_time = n[1]
            next_col = n[0]
            next_is_tail = False
            break  # Notes sorted by head time → first match is closest

    # Find closest tail after t_i (may be closer than the note head)
    for n in tail_seq:
        if n[2] > t_i and n[2] < next_time:
            next_time = n[2]
            next_col = n[0]
            next_is_tail = True

    return next_time, next_col, next_is_tail


def precompute_release_data(K, x, note_seq_by_column, tail_seq, note_seq):
    """Precompute structured release data for fast runtime Rbar computation.

    This extracts all note-dependent information (release index I, next-event
    mapping, lock states) into a compact dict.  The runtime function
    compute_Rbar_enhanced_fast() then applies tunable parameters on top.

    Returns:
        dict with keys: tails, I_list, lock_data, K, x
    """
    n_tails = len(tail_seq)
    if n_tails == 0:
        return {"tails": [], "I_list": [], "lock_data": [], "K": K, "x": x}

    # Build per-column timing for I computation
    times_by_column = {
        i: np.array([note[1] for note in column], dtype=np.float64)
        for i, column in enumerate(note_seq_by_column)
    }

    # Precompute release index I for each tail (same as original)
    I_list = []
    for i in range(n_tails):
        k, h_i, t_i = tail_seq[i]
        times = times_by_column[k]
        # Binary search for note index at or after head
        idx = int(np.searchsorted(times, h_i, side='left'))
        if idx < len(note_seq_by_column[k]):
            next_note = note_seq_by_column[k][idx + 1] if idx + 1 < len(note_seq_by_column[k]) else (0, 1e9, 1e9)
        else:
            next_note = (0, 1e9, 1e9)
        _, h_j, _ = next_note
        I_h = 0.001 * abs(t_i - h_i - 80) / x
        I_t = 0.001 * abs(h_j - t_i - 80) / x
        I_val = 2.0 / (2.0 + math.exp(-5.0 * (I_h - 0.75)) + math.exp(-5.0 * (I_t - 0.75)))
        I_list.append(float(I_val))

    # Build sorted arrays for vectorized next-event search
    note_times = np.array([n[1] for n in note_seq], dtype=np.float64)
    note_cols = np.array([n[0] for n in note_seq], dtype=np.int32)
    if len(tail_seq) > 0:
        tail_times_arr = np.array([t[2] for t in tail_seq], dtype=np.float64)
        tail_cols_arr = np.array([t[0] for t in tail_seq], dtype=np.int32)
    else:
        tail_times_arr = np.array([], dtype=np.float64)
        tail_cols_arr = np.array([], dtype=np.int32)

    # Precompute per-tail data
    tails = []
    for i in range(n_tails):
        k_i, h_i, t_i = tail_seq[i]

        # Next note head after t_i
        next_head_idx = int(np.searchsorted(note_times, t_i, side='right'))
        next_note_time = float(note_times[next_head_idx]) if next_head_idx < len(note_times) else _NO_NEXT_EVENT
        next_note_col = int(note_cols[next_head_idx]) if next_head_idx < len(note_times) else -1

        # Next tail after t_i
        next_tail_idx = int(np.searchsorted(tail_times_arr, t_i, side='right'))
        next_tail_time = float(tail_times_arr[next_tail_idx]) if next_tail_idx < len(tail_times_arr) else _NO_NEXT_EVENT
        next_tail_col = int(tail_cols_arr[next_tail_idx]) if next_tail_idx < len(tail_times_arr) else -1

        # Pick the closer event
        if next_tail_time < next_note_time:
            next_time = next_tail_time
            next_col = next_tail_col
            next_is_tail = True
        else:
            next_time = next_note_time
            next_col = next_note_col
            next_is_tail = False

        tails.append({
            "col": int(k_i),
            "tail_time": float(t_i),
            "ln_duration": float(t_i - h_i),
            "I": float(I_list[i]),
            "next_time": float(next_time),
            "next_col": int(next_col),
            "next_is_tail": bool(next_is_tail),
        })

    # Precompute lock states: per tail, which other columns are locked
    lock_data = []
    for i in range(n_tails):
        k_i, _, t_i = tail_seq[i]
        locks = []
        for j in range(K):
            if j == k_i:
                continue
            for kj, hj, tj in tail_seq:
                if kj != j:
                    continue
                if hj <= t_i <= tj:
                    locks.append((int(j), float(_coord_weight(k_i, j))))
                    break
        lock_data.append(locks)

    return {
        "tails": tails,
        "I_list": [float(v) for v in I_list],
        "lock_data": lock_data,
        "tail_seq_cols": [(int(t[0]), int(t[0])) for t in tail_seq],  # (k_i, k_j) for sequential pairs
        "K": int(K),
        "x": float(x),
    }


def compute_Rbar_enhanced_fast(release_data, base_corners,
                                release_tail_coeff=0.08,
                                release_tail_to_tap_factor=1.0,
                                release_same_col_bonus=1.5,
                                release_coord_exponent=1.0,
                                release_seq_coeff=0.03,
                                short_ln_threshold=200,
                                short_ln_reduction=0.5,
                                lock_interaction_coeff=0.3,
                                smooth_window=500,
                                smooth_scale=0.001):
    """Compute Enhanced Rbar at runtime from precomputed release_data.

    Uses precomputed per-tail info + tunable parameters.
    Mirrors the original compute_Rbar_enhanced() logic but reads from cached data.

    Args:
        release_data: dict from precompute_release_data()
        base_corners: time grid
        release_tail_coeff: base per-tail coefficient
        release_tail_to_tap_factor: multiplier for release-to-tap vs release-to-tail
        release_same_col_bonus: same-column release difficulty multiplier
        release_coord_exponent: exponent for column distance weighting
        short_ln_threshold: LN duration below which release is reduced (ms)
        short_ln_reduction: factor to reduce short-LN release (0=full reduce, 1=no reduce)
        lock_interaction_coeff: how much other-column locks increase release difficulty
        smooth_window: smoothing window (ms)
        smooth_scale: scale multiplier in smoothing

    Returns:
        Rbar: array on base_corners
    """
    tails = release_data.get("tails", [])
    I_list = release_data.get("I_list", [])
    lock_data = release_data.get("lock_data", [])
    tail_seq_cols = release_data.get("tail_seq_cols", [])
    K = release_data["K"]
    x = release_data["x"]
    n_tails = len(tails)

    R_step = np.zeros(len(base_corners), dtype=np.float64)

    # === Per-tail release difficulty ===
    for i in range(n_tails):
        td = tails[i]
        next_time = td["next_time"]
        if next_time >= _NO_NEXT_EVENT:
            continue

        dt = next_time - td["tail_time"]
        if dt <= 0 or dt > 5000:
            continue

        delta = 0.001 * dt

        # Base release difficulty
        release_val = release_tail_coeff * (delta) ** (-0.5) * x ** (-1) * (1.0 + td["I"])

        # Same-column inverse bonus
        if td["col"] == td["next_col"] and not td["next_is_tail"]:
            release_val *= release_same_col_bonus

        # Column distance (coordination) weighting
        if td["col"] != td["next_col"]:
            cw = _coord_weight(td["col"], td["next_col"])
            if td["next_is_tail"]:
                release_val *= 1.0 + (cw - 1.0) * release_coord_exponent * 0.5
            else:
                release_val *= 1.0 + (cw - 1.0) * release_coord_exponent * release_tail_to_tap_factor

        # Short-LN reduction
        ln_dur = td["ln_duration"]
        if ln_dur < short_ln_threshold:
            reduction = short_ln_reduction + (1.0 - short_ln_reduction) * (ln_dur / short_ln_threshold)
            release_val *= reduction

        # Lock-hand interaction
        if lock_interaction_coeff > 1e-9 and i < len(lock_data):
            lock_count = sum(cw for _, cw in lock_data[i])
            release_val *= (1.0 + lock_interaction_coeff * lock_count)

        # Safety clamp: prevent overflow in downstream power computation
        release_val = float(np.clip(release_val, 0.0, 1e6))

        # Map to base_corners: distribute over [tail_time, next_time]
        left_idx = int(np.searchsorted(base_corners, td["tail_time"], side='left'))
        right_idx = int(np.searchsorted(base_corners, min(next_time, base_corners[-1]), side='left'))
        if right_idx > left_idx:
            R_step[left_idx:right_idx] += release_val

    # === Tail-to-tail sequential release difficulty ===
    for i in range(n_tails - 1):
        td_i = tails[i]
        td_j = tails[i + 1]
        t_start = td_i["tail_time"]
        t_end = td_j["tail_time"]

        left_idx = int(np.searchsorted(base_corners, t_start, side='left'))
        right_idx = int(np.searchsorted(base_corners, t_end, side='left'))
        if right_idx <= left_idx:
            continue

        delta_r = 0.001 * (t_end - t_start)

        cw = _coord_weight(td_i["col"], td_j["col"])
        coord_factor = 1.0 + (cw - 1.0) * release_coord_exponent

        seq_val = (
            release_seq_coeff * (delta_r) ** (-0.5) * x ** (-1)
            * (1.0 + 0.8 * (td_i["I"] + td_j["I"]))
            * coord_factor
        )
        seq_val = float(np.clip(seq_val, 0.0, 1e6))
        R_step[left_idx:right_idx] += seq_val

    # Final safety clamp before smoothing
    R_step = np.clip(R_step, 0.0, 1e8)
    Rbar = smooth_on_corners(base_corners, R_step, window=smooth_window,
                             scale=smooth_scale, mode='sum')
    return Rbar


def compute_Rbar_enhanced(K, x, note_seq_by_column, tail_seq, base_corners,
                          note_seq,
                          release_tail_coeff=0.08,
                          release_tail_to_tap_factor=1.0,
                          release_same_col_bonus=1.5,
                          release_coord_exponent=1.0,
                          release_seq_coeff=0.03,
                          short_ln_threshold=200,
                          short_ln_reduction=0.5,
                          lock_interaction_coeff=0.3,
                          smooth_window=500,
                          smooth_scale=0.001):
    """Compute enhanced Rbar — LN tail based release difficulty.

    Args:
        K: number of columns
        x: hit leniency
        note_seq_by_column: notes grouped by column
        tail_seq: LN notes sorted by tail time [(col, head, tail), ...]
        base_corners: time grid
        note_seq: all notes sorted by head time [(col, head, tail), ...]
        release_tail_coeff: base per-tail coefficient
        release_tail_to_tap_factor: multiplier for release-to-tap vs release-to-tail
        release_same_col_bonus: same-column release difficulty multiplier
        release_coord_exponent: exponent for column distance weighting
        short_ln_threshold: LN duration below which release is reduced (ms)
        short_ln_reduction: factor to reduce short-LN release (0=full reduce, 1=no reduce)
        lock_interaction_coeff: how much other-column locks increase release difficulty
        smooth_window: smoothing window (ms)
        smooth_scale: scale multiplier in smoothing

    Returns:
        Rbar: array on base_corners
    """
    R_step = np.zeros(len(base_corners))

    times_by_column = {
        i: [note[1] for note in column]
        for i, column in enumerate(note_seq_by_column)
    }

    # === Release index (I) computation (same as original) ===
    I_list = []
    for i in range(len(tail_seq)):
        k, h_i, t_i = tail_seq[i]
        times = times_by_column[k]
        idx = 0
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

    # === Enhanced: per-tail release difficulty ===
    # Each LN tail is treated as an independent release event.
    # Release difficulty is based on the interval to the next event.
    for i in range(len(tail_seq)):
        k_i, h_i, t_i = tail_seq[i]
        I_i = I_list[i]

        # Find next event after this tail
        next_time, next_col, next_is_tail = _find_next_event(t_i, note_seq, tail_seq)

        if next_time >= _NO_NEXT_EVENT:
            continue  # No valid next event (end of map)

        dt = next_time - t_i
        if dt <= 0 or dt > 5000:
            continue

        delta = 0.001 * dt

        # === Base release difficulty ===
        # Same structure as original: δt^(-0.5) * x^(-1) * (1 + I)
        release_val = release_tail_coeff * (delta) ** (-0.5) * x ** (-1) * (1 + I_i)

        # === Same-column inverse bonus ===
        # Per experience.md §4(4b): same-column note after release creates
        # extra difficulty (simple multiplier; U-curve handled by Vbar)
        if k_i == next_col and not next_is_tail:
            release_val *= release_same_col_bonus

        # === Column distance (coordination) weighting ===
        # Per experience.md §4(4a): sequential releases across columns
        # have coordination-dependant difficulty
        if k_i != next_col:
            cw = _coord_weight(k_i, next_col)
            # Release-to-tap gets different coordination weight than release-to-tail
            if next_is_tail:
                release_val *= 1.0 + (cw - 1.0) * release_coord_exponent * 0.5
            else:
                release_val *= 1.0 + (cw - 1.0) * release_coord_exponent * release_tail_to_tap_factor

        # === Short-LN reduction ===
        # Per experience.md §4(4e): very short LNs have reduced release difficulty
        # (player can tap-release quickly for decent tail judgment)
        ln_duration = t_i - h_i
        if ln_duration < short_ln_threshold:
            # Smooth reduction: from short_ln_reduction at 0 to 1.0 at threshold
            reduction = short_ln_reduction + (1.0 - short_ln_reduction) * (ln_duration / short_ln_threshold)
            release_val *= reduction

        # === Lock-hand interaction ===
        # Per experience.md §4(4d): release harder when other columns are locked
        if lock_interaction_coeff > 1e-9:
            lock_count = 0.0
            for j in range(K):
                if j == k_i:
                    continue
                for (kj, hj, tj) in tail_seq:
                    if kj != j:
                        continue
                    if hj <= t_i <= tj:
                        lock_count += _coord_weight(k_i, j)
                        break
            release_val *= (1.0 + lock_interaction_coeff * lock_count)

        # Map to base_corners: distribute over the interval [t_i, next_time]
        left_idx = np.searchsorted(base_corners, t_i, side='left')
        right_idx = np.searchsorted(base_corners, min(next_time, base_corners[-1]), side='left')
        idx = np.arange(left_idx, right_idx)
        if len(idx) > 0:
            R_step[idx] += release_val

    # === Tail-to-tail sequential release difficulty ===
    # Per experience.md §4(4a): sequential releases are harder than isolated ones.
    # This captures the coordination cost of consecutive releases.
    for i in range(len(tail_seq) - 1):
        k_i, h_i, t_start = tail_seq[i]
        k_j, h_j, t_end = tail_seq[i + 1]
        left_idx = np.searchsorted(base_corners, t_start, side='left')
        right_idx = np.searchsorted(base_corners, t_end, side='left')
        idx = np.arange(left_idx, right_idx)
        if len(idx) == 0:
            continue

        delta_r = 0.001 * (t_end - t_start)

        # Coordination weighting for sequential releases
        cw = _coord_weight(k_i, k_j)
        coord_factor = 1.0 + (cw - 1.0) * release_coord_exponent

        # Sequential release coefficient (tunable) to avoid double-counting with per-tail section
        R_step[idx] += release_seq_coeff * (delta_r) ** (-0.5) * x ** (-1) * (
            1 + 0.8 * (I_list[i] + I_list[i + 1])
        ) * coord_factor

    Rbar = smooth_on_corners(base_corners, R_step, window=smooth_window,
                             scale=smooth_scale, mode='sum')
    return Rbar


# === shield.py ===
"""
SPM Rating — Shield difficulty (Sbar).

Models difficulty when same-column notes appear shortly before LN heads.
The preceding note primes a jack-like arm motion that conflicts with the
sustained press required for the LN, making accidental LN breaks likely.

Per experience.md §4(5):
  - Closer preceding notes → higher shield difficulty
  - Lock-hand interaction: other columns holding LNs increase shield difficulty
  - Coordination-weighted: same-hand locks are harder than opposite-hand locks

Formula: shield_sum = Σ exp(-δt / tau_ms)
  - tau_ms controls decay: smaller tau → shorter-range shield effect
  - Shield is applied around the LN head (h-100 to h+100)

Supports two modes:
  - compute_Sbar(): full computation from raw data (slow)
  - precompute_shield_data() + compute_Sbar_fast(): precompute structure, fast runtime
"""


_HAND_MAP = {0: "L", 1: "L", 2: "L", 3: "T", 4: "R", 5: "R", 6: "R"}


def _coord_weight(k1, k2):
    if k1 == k2:
        return 1.0
    h1, h2 = _HAND_MAP.get(k1, ""), _HAND_MAP.get(k2, "")
    if h1 == h2 and h1 != "T":
        return 0.8
    elif h1 == "T" or h2 == "T":
        return 0.4
    else:
        return 0.2


def precompute_shield_data(K, note_seq_by_column, LN_seq, shield_window_ms=500):
    """Precompute per-LN structural data for fast Sbar computation.

    Stores δt values and lock columns so runtime doesn't scan all notes.
    Returns list of dicts, one per LN.
    """
    data = []
    col_head_times = {k: [n[1] for n in note_seq_by_column[k]] for k in range(K)}

    for (k, h, t) in LN_seq:
        # Same-column preceding notes within window
        prev_dts = []
        for note_h in col_head_times[k]:
            dt = h - note_h
            if 0 < dt <= shield_window_ms:
                prev_dts.append(dt)

        # Lock-hand columns (active LNs at time h, excluding k)
        lock_cols = []
        for j in range(K):
            if j == k:
                continue
            for (kj, hj, tj) in LN_seq:
                if kj == j and hj <= h <= tj:
                    lock_cols.append(j)
                    break

        if prev_dts:  # Only store if there's shield content
            data.append({
                "col": k, "head_time": h, "tail_time": t,
                "prev_dts": np.array(prev_dts, dtype=np.float64),
                "lock_cols": lock_cols,
            })

    return data


def compute_Sbar_fast(shield_data, base_corners,
                      shield_tau_ms=100, shield_anchor_mod=1.0,
                      shield_coord_factor=1.0,
                      smooth_window=500, smooth_scale=0.001):
    """Fast Sbar from precomputed structural data.

    Uses pre-stored δt lists + lock column indices.
    Formula: shield_sum = Σ exp(-δt / tau_ms)
    """
    S_step = np.zeros(len(base_corners))

    for ln_data in shield_data:
        k = ln_data["col"]
        h = ln_data["head_time"]
        t = ln_data["tail_time"]
        dts = ln_data["prev_dts"]

        if len(dts) == 0:
            continue

        # Exponential decay: closer notes → higher shield
        shield_sum = float(np.sum(np.exp(-dts / shield_tau_ms)))

        if shield_sum < 1e-12:
            continue

        # Lock-hand interaction
        lock_bonus = 0.0
        for j in ln_data["lock_cols"]:
            lock_bonus += _coord_weight(k, j)

        shield_val = shield_sum * (1.0 + shield_anchor_mod * shield_coord_factor * lock_bonus)

        # Time region around LN head
        earliest_prev = h - np.max(dts)
        start_time = max(h - 100, earliest_prev)
        end_time = min(h + 100, t)

        left_idx = np.searchsorted(base_corners, start_time, side='left')
        right_idx = np.searchsorted(base_corners, end_time, side='left')
        if right_idx > left_idx:
            S_step[left_idx:right_idx] += shield_val

    Sbar = smooth_on_corners(base_corners, S_step, window=smooth_window,
                             scale=smooth_scale, mode='sum')
    return Sbar


# Fallback: full computation (used when precomputed data unavailable)
def compute_Sbar(K, note_seq_by_column, LN_seq, base_corners,
                 shield_tau_ms=100, shield_anchor_mod=1.0,
                 shield_coord_factor=1.0, shield_window_ms=500,
                 smooth_window=500, smooth_scale=0.001):
    """Full Sbar computation from raw data (slow fallback)."""
    S_step = np.zeros(len(base_corners))
    col_notes = {k: note_seq_by_column[k] for k in range(K)}

    for (k, h, t) in LN_seq:
        shield_sum = 0.0
        earliest_prev = h
        for note in col_notes[k]:
            note_h = note[1]
            dt = h - note_h
            if dt <= 0 or dt > shield_window_ms:
                continue
            shield_sum += np.exp(-dt / shield_tau_ms)
            if note_h < earliest_prev:
                earliest_prev = note_h

        if shield_sum < 1e-12:
            continue

        lock_bonus = 0.0
        for j in range(K):
            if j == k:
                continue
            for (kj, hj, tj) in LN_seq:
                if kj != j:
                    continue
                if hj <= h <= tj:
                    lock_bonus += _coord_weight(k, j)
                    break

        shield_val = shield_sum * (1.0 + shield_anchor_mod * shield_coord_factor * lock_bonus)
        start_time = max(h - 100, earliest_prev)
        end_time = min(h + 100, t)
        left_idx = np.searchsorted(base_corners, start_time, side='left')
        right_idx = np.searchsorted(base_corners, end_time, side='left')
        if right_idx > left_idx:
            S_step[left_idx:right_idx] += shield_val

    Sbar = smooth_on_corners(base_corners, S_step, window=smooth_window,
                             scale=smooth_scale, mode='sum')
    return Sbar


# === inverse.py ===
"""
SPM Rating — Inverse / Guide effect (Vbar).

Models two SEPARATE effects when notes appear shortly after LN releases:

1. Inverse spike (same-column only): very close note after LN release
   → rapid release→repress in same column = harder
   Formula: inv_amplitude * exp(-(dt / inv_tau) ^ inv_power)

2. Guide dip (same-column + cross-column): medium-distance note after LN release
   → upcoming note provides timing reference → easier
   Formula: -guide_depth * exp(-((dt - guide_center) / guide_width) ^ 2)
   Cross-column: same formula × cross_guide_scale × coordination_weight

These were previously combined into a single U-shaped curve. Now they are
independent mechanisms with separate time scales and amplitudes.

Supports two modes:
  - compute_Vbar(): full computation from raw data (slow)
  - precompute_inverse_data() + compute_Vbar_fast(): precompute structure, fast runtime
"""


_HAND_MAP = {0: "L", 1: "L", 2: "L", 3: "T", 4: "R", 5: "R", 6: "R"}


def _coord_weight(k1, k2):
    if k1 == k2:
        return 1.0
    h1, h2 = _HAND_MAP.get(k1, ""), _HAND_MAP.get(k2, "")
    if h1 == h2 and h1 != "T":
        return 0.8
    elif h1 == "T" or h2 == "T":
        return 0.4
    else:
        return 0.2


def precompute_inverse_data(K, note_seq_by_column, LN_seq, window_ms=200):
    """Precompute per-LN structural data for fast Vbar computation.

    Returns list of dicts, one per LN release.
    """
    data = []
    col_head_times = {k: np.array([n[1] for n in note_seq_by_column[k]], dtype=float)
                      for k in range(K)}

    for (k, h, t) in LN_seq:
        if t < 0:
            continue

        # Same-column notes after release
        same_dts = col_head_times[k] - t
        same_mask = (same_dts > 0) & (same_dts <= window_ms)
        same_col_dts = same_dts[same_mask].tolist()

        # Cross-column notes after release
        cross_dts = []
        cross_k1 = []
        cross_k2 = []
        for other_k in range(K):
            if other_k == k:
                continue
            cross_all = col_head_times[other_k] - t
            cross_mask = (cross_all > 0) & (cross_all <= window_ms)
            valid_dts = cross_all[cross_mask]
            for dt in valid_dts:
                cross_dts.append(float(dt))
                cross_k1.append(k)
                cross_k2.append(other_k)

        if same_col_dts or cross_dts:
            data.append({
                "col": k, "head_time": h, "tail_time": t,
                "same_col_dts": np.array(same_col_dts, dtype=np.float64),
                "cross_col_dts": np.array(cross_dts, dtype=np.float64) if cross_dts else np.array([], dtype=np.float64),
                "cross_col_k1": np.array(cross_k1, dtype=np.int32) if cross_k1 else np.array([], dtype=np.int32),
                "cross_col_k2": np.array(cross_k2, dtype=np.int32) if cross_k2 else np.array([], dtype=np.int32),
            })

    return data


def compute_Vbar_fast(inverse_data, base_corners,
                      inv_amplitude=3.0, inv_tau=31, inv_power=1.0,
                      guide_depth=0.9, guide_center=78, guide_width=31,
                      cross_guide_scale=0.67,
                      same_col_bonus=3.6,
                      window_ms=200):
    """Fast Vbar from precomputed structural data.

    Two independent mechanisms:

    1. Inverse spike (same-column only):
       inv_amplitude * exp(-(dt / inv_tau) ^ inv_power)
       inv_power=1 → exponential, inv_power=2 → Gaussian

    2. Guide dip (same + cross column):
       -guide_depth * exp(-((dt - guide_center) / guide_width) ^ 2)
       Cross-column: same dip × cross_guide_scale × coordination_weight

    Vbar = spike + same_col_guide + cross_col_guide
    Vbar > 0 → release harder,  Vbar < 0 → release easier
    """

    V_step = np.zeros(len(base_corners))

    for ln_data in inverse_data:
        k = ln_data["col"]
        t = ln_data["tail_time"]

        # Common left bound for all notes of this LN tail
        left_idx = np.searchsorted(base_corners, t, side='left')

        # === 1. Same-column: inverse spike + guide dip ===
        same_dts = ln_data["same_col_dts"]
        if len(same_dts) > 0:
            # Inverse spike: very close → harder
            spike_vals = inv_amplitude * np.exp(-(same_dts / inv_tau) ** inv_power)
            # Guide dip: medium distance → easier
            dip_vals = guide_depth * np.exp(-((same_dts - guide_center) / guide_width) ** 2)
            # Combined same-col: (spike - dip) * same_col_bonus
            v_vals = (spike_vals - dip_vals) * same_col_bonus

            note_times = t + same_dts
            right_indices = np.searchsorted(base_corners, note_times, side='left')
            for i in range(len(same_dts)):
                right_idx = right_indices[i]
                if right_idx > left_idx:
                    V_step[left_idx:right_idx] += v_vals[i]

        # === 2. Cross-column: guide dip only (no inverse spike) ===
        cross_dts = ln_data["cross_col_dts"]
        if len(cross_dts) > 0:
            cw_arr = np.array([_coord_weight(ln_data["cross_col_k1"][i],
                                             ln_data["cross_col_k2"][i])
                               for i in range(len(cross_dts))])
            # Cross guide: same dip shape × cross_guide_scale × coordination weight
            cross_vals = -guide_depth * cross_guide_scale * cw_arr * np.exp(
                -((cross_dts - guide_center) / guide_width) ** 2
            )

            note_times = t + cross_dts
            right_indices = np.searchsorted(base_corners, note_times, side='left')
            for i in range(len(cross_dts)):
                right_idx = right_indices[i]
                if right_idx > left_idx:
                    V_step[left_idx:right_idx] += cross_vals[i]

    Vbar = smooth_on_corners(base_corners, V_step, window=500, scale=0.001, mode='sum')
    return Vbar


# Fallback: full computation (used when precomputed data unavailable)
def compute_Vbar(K, note_seq_by_column, LN_seq, base_corners,
                 inv_amplitude=3.0, inv_tau=31, inv_power=1.0,
                 guide_depth=0.9, guide_center=78, guide_width=31,
                 cross_guide_scale=0.67,
                 same_col_bonus=3.6,
                 window_ms=200):
    """Full Vbar computation from raw data (slow fallback)."""
    V_step = np.zeros(len(base_corners))
    col_notes = {k: note_seq_by_column[k] for k in range(K)}

    for (k, h, t) in LN_seq:
        if t < 0:
            continue

        # Same-column: spike + dip
        for note in col_notes[k]:
            note_h = note[1]
            dt = note_h - t
            if dt <= 0 or dt > window_ms:
                continue
            spike_val = inv_amplitude * np.exp(-(dt / inv_tau) ** inv_power)
            dip_val = guide_depth * np.exp(-((dt - guide_center) / guide_width) ** 2)
            v_val = (spike_val - dip_val) * same_col_bonus
            left_idx = np.searchsorted(base_corners, t, side='left')
            right_idx = np.searchsorted(base_corners, note_h, side='left')
            if right_idx > left_idx:
                V_step[left_idx:right_idx] += v_val

        # Cross-column: guide dip only
        for other_k in range(K):
            if other_k == k:
                continue
            cw = _coord_weight(k, other_k)
            for note in col_notes[other_k]:
                note_h = note[1]
                dt = note_h - t
                if dt <= 0 or dt > window_ms:
                    continue
                guide_val = -guide_depth * cross_guide_scale * cw * np.exp(
                    -((dt - guide_center) / guide_width) ** 2
                )
                left_idx = np.searchsorted(base_corners, t, side='left')
                right_idx = np.searchsorted(base_corners, note_h, side='left')
                if right_idx > left_idx:
                    V_step[left_idx:right_idx] += guide_val

    Vbar = smooth_on_corners(base_corners, V_step, window=500, scale=0.001, mode='sum')
    return Vbar


# === stamina.py ===
"""
SPM Rating — Stamina / Endurance difficulty (Ebar).

NEW component: Models accumulated fatigue with recovery periods.

Leaky-integrator fatigue model:
- Fatigue accumulates during dense sections
- Decays during rest periods
- Recovery is proportional to rest duration

Also provides rhythmic complexity bonus for irregular patterns.
"""



def compute_Ebar(K, note_seq, base_corners, anchor,
                 fatigue_tau_ms=8000, fatigue_increment=1.0,
                 recovery_threshold_ms=2000, recovery_tau_ms=3000):
    """
    Compute Ebar — stamina/endurance difficulty.

    Uses a leaky-integrator model:
        fatigue(t + δt) = fatigue(t) * exp(-δt / τ_f) + inc * (1 - exp(-δt / τ_f))

    Args:
        K: number of columns
        note_seq: sorted list of (col, head, tail) tuples
        base_corners: time grid
        anchor: anchor values on base_corners
        fatigue_tau_ms: fatigue decay time constant (ms)
        fatigue_increment: base increment per active note
        recovery_threshold_ms: gap size that counts as "rest" (ms)
        recovery_tau_ms: recovery time constant (ms)

    Returns:
        Ebar: array on base_corners (multiplier for D)
    """
    fatigue = np.zeros(len(base_corners))
    recovery = np.ones(len(base_corners))

    # Build note density and intervals
    note_times = sorted(n[1] for n in note_seq)
    prev_time = 0

    for s_idx, s in enumerate(base_corners):
        if s <= 0:
            continue

        # Count notes in a window around s (二分搜索替代线性扫描)
        window_left = max(0, s - 2000)
        window_right = min(s, base_corners[-1])
        left_idx = bisect.bisect_left(note_times, window_left)
        right_idx = bisect.bisect_right(note_times, window_right)
        note_count = right_idx - left_idx

        # Density: notes per second in recent window
        density = note_count / max((window_right - window_left) / 1000.0, 0.001)

        # Anchor factor: more anchored = more fatiguing
        anchor_factor = max(anchor[s_idx] - 0.5, 0)

        # Fatigue increment
        inc = fatigue_increment * density * anchor_factor

        # Time since last corner
        dt = (base_corners[s_idx] - base_corners[s_idx - 1]) if s_idx > 0 else 0

        # Leaky integrator update
        decay_factor = np.exp(-dt / fatigue_tau_ms) if fatigue_tau_ms > 0 else 0
        fatigue[s_idx] = fatigue[s_idx - 1] * decay_factor + inc * (1 - decay_factor)

        # Recovery: detect rest periods (二分搜索替代线性扫描)
        idx = bisect.bisect_right(note_times, s) - 1
        last_note_before = note_times[idx] if idx >= 0 else 0
        gap_since_last = s - last_note_before

        if gap_since_last > recovery_threshold_ms:
            # Active recovery
            recovery_progress = (gap_since_last - recovery_threshold_ms) / recovery_tau_ms
            recovery[s_idx] = max(0, min(1, 1 - np.exp(-recovery_progress)))
        else:
            recovery[s_idx] = 0

    # Combined: fatigue moderated by recovery
    Ebar_raw = fatigue * (1 - recovery)

    # Smooth over longer window
    Ebar = smooth_on_corners(base_corners, Ebar_raw, window=2000, mode='avg')

    # Normalize: keep in reasonable range as multiplier
    # Ebar is used as multiplier on D: D_final = D * (1 + gamma * Ebar)
    e_max = np.max(Ebar)
    if e_max > 1e-6:
        Ebar = Ebar / e_max  # normalize to [0, 1]

    return Ebar


def compute_rhythm_complexity(note_seq, base_corners):
    """
    Compute rhythmic complexity from inter-onset interval (IOI) variety.

    Higher variance in IOIs means more complex rhythms.
    Smooth over a larger window.
    """
    note_times = sorted(n[1] for n in note_seq)
    if len(note_times) < 2:
        return np.zeros(len(base_corners))

    # Compute IOIs
    iois = np.diff(note_times)
    ioi_times = [(note_times[i] + note_times[i+1]) / 2 for i in range(len(iois))]

    # Pre-sort IOI times for binary search
    ioi_times_arr = np.array([(note_times[i] + note_times[i+1]) / 2 for i in range(len(iois))])

    for idx, s in enumerate(base_corners):
        # Find IOIs within window using binary search
        lo = np.searchsorted(ioi_times_arr, s - 1000, side='left')
        hi = np.searchsorted(ioi_times_arr, s + 1000, side='right')
        window_iois = iois[lo:hi]

        if len(window_iois) >= 3:
            # Coefficient of variation as rhythm complexity
            mean_ioi = np.mean(window_iois)
            std_ioi = np.std(window_iois)
            if mean_ioi > 0:
                cv = std_ioi / mean_ioi
                # More variance = more rhythm complexity
                rhythm_raw[idx] = min(cv, 3.0) / 3.0  # normalize

    rhythm = smooth_on_corners(base_corners, rhythm_raw, window=2000, mode='avg')
    return rhythm


# === utils.py ===
"""
SPM Rating — Math utility functions.

Ported from SunnyRework algorithm.py with slight refactoring.
"""



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
    idx = bisect.bisect_left(times, h)
    if idx + 1 < len(note_seq_by_column[k]):
        return note_seq_by_column[k][idx + 1]
    return (0, 10**9, 10**9)


def LN_sum(a, b, LN_rep):
    """
    Compute cumulative LN body value between a and b using sparse representation.
    """
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


# === config.py ===
"""
SPM Rating — All tunable parameters.

Each parameter is defined as a dict entry:
    key: (default_value, lower_bound, upper_bound, description)

Values are internally normalized to [0,1] for optimization.
The optimizer reads/writes this file as the single parameter source.
"""

# ============================================================
# Column / Hand layout
# ============================================================
COLUMN_LAYOUT = {
    "num_columns": 7,
    # Hand assignment: 'L'=left, 'T'=thumb, 'R'=right
    "hand_map": {0: "L", 1: "L", 2: "L", 3: "T", 4: "R", 5: "R", 6: "R"},
}

# ============================================================
# Hit leniency (x)
# ============================================================
HIT_LENIENCY = {
    "x_od_coeff_A":   (0.3,  0.1,  1.0,  "x = A * sqrt((B - ceil(OD*3)) / 500)"),
    "x_od_coeff_B":   (64.5, 60.0, 70.0, "OD reference point"),
    "x_clip_scale":   (0.6,  0.3,  1.0,  "clamp scale: min(x, scale*(x-0.09)+0.09)"),
    "x_clip_offset":  (0.09, 0.05, 0.15, "clamp offset"),
}

# ============================================================
# Key usage windows
# ============================================================
KEY_USAGE = {
    "ku_active_before":   (150, 50,  300, "ms before note head: column active"),
    "ku_active_after":    (150, 50,  300, "ms after note head / LN tail: column active"),
}

# ============================================================
# Anchor (key_usage_400)
# ============================================================
ANCHOR = {
    "anchor_center_scale":  (3.75, 1.0, 10.0, "base weight for active key period"),
    "anchor_ln_bonus":      (0.025, 0.0, 0.1, "LN duration bonus per ms (/150 scaled, max 1500)"),
    "anchor_ramp_radius":   (400,  200, 600,  "ms: quadratic ramp window each side"),
    "anchor_sort_cap_clip": (0.18, 0.0, 0.5,  "anchor lower clamp threshold"),
    "anchor_cubic_scale":   (5.0,  1.0, 20.0, "cubic transform scale"),
    "anchor_cubic_offset":  (0.22, 0.0, 0.5,  "cubic transform offset"),
}

# ============================================================
# Corners / time grid
# ============================================================
CORNERS = {
    "corner_radius_A":  (1000, 500,  2000, "A-corners offset radius"),
    "corner_radius_base": (500, 250,  1000, "base-corners offset radius"),
    "corner_delta_spike": (1,    0.5,  5,    "resolve Dirac-Delta for simultaneous notes"),
}

# ============================================================
# Jack difficulty (Jbar)
# ============================================================
JACK = {
    "jack_nerfer_A":       (7e-5, 1e-6, 1e-3, "nerf coefficient"),
    "jack_nerfer_B":       (4,    1,    8,    "nerf exponent"),
    "jack_nerfer_offset":  (0.08, 0.01, 0.2,  "nerf delta offset"),
    "jack_nerfer_floor":   (0.15, 0.05, 0.5,  "nerf baseline addend"),
    "jack_x_power":        (0.25, 0.1,  0.5,  "x exponent in jack formula"),
    "jack_delta_coeff":    (0.11, 0.05, 0.3,  "delta coefficient in jack"),
    "jack_smooth_window":  (500,  250,  1000, "smoothing window (ms)"),
    "jack_scale":          (0.001, 0.0001, 0.01, "scale multiplier in smoothing"),
    "jack_aggregation_power": (5, 2, 10, "power for cross-column aggregation"),
}

# ============================================================
# Cross difficulty (Xbar)
# ============================================================
CROSS = {
    "cross_coeff_scale":   (0.16, 0.05, 0.5,  "base coefficient in X"),
    "cross_fast_scale":    (0.4,  0.1,  1.0,  "fast cross base scale"),
    "cross_fast_floor":    (0.06, 0.02, 0.15, "delta floor for fast cross"),
    "cross_fast_subtract": (80,   20,   200,  "subtract term for fast cross"),
    "cross_smooth_window": (500,  250,  1000, "smoothing window (ms)"),
    "cross_scale":         (0.001, 0.0001, 0.01, "scale multiplier in smoothing"),
    # Column distance integration (unified params, used by old cache)
    "cross_dist_exponent":       (1.0,  0.0, 3.0, "distance exponent (unified/old cache)"),
    "cross_same_hand_penalty":   (0.3,  0.0, 1.0, "same-hand penalty (unified/old cache)"),
    # Phase 4: RC/LN differentiated params (used by new cache with cross_data)
    "cross_dist_exponent_rc":    (1.0,  0.0, 3.0, "distance exponent for RC/speed maps"),
    "cross_dist_exponent_ln":    (1.0,  0.0, 3.0, "distance exponent for LN maps"),
    "cross_same_hand_penalty_rc":(0.3,  0.0, 1.0, "same-hand penalty for RC/speed maps"),
    "cross_same_hand_penalty_ln":(0.3,  0.0, 1.0, "same-hand penalty for LN maps"),
    "cross_thumb_bridge_factor": (0.5,  0.0, 1.0, "thumb bridges hand gap: 0=full bridge, 1=no bridge"),
    "cross_inactive_nerf":       (1.0,  0.0, 2.0, "multiplier when one side inactive"),
}

# ============================================================
# Stream / Pressing difficulty (Pbar)
# ============================================================
STREAM = {
    "stream_booster_tau":    (7.5, 3.0, 15.0, "stream booster tau (delta normalizer)"),
    "stream_booster_lo":     (160, 100, 250,  "stream booster low threshold"),
    "stream_booster_hi":     (360, 250, 500,  "stream booster high threshold"),
    "stream_booster_scale":  (1.7e-7, 1e-8, 1e-5, "stream booster scale"),
    "stream_ln_body_factor": (6,    1,   20,   "LN body multiplier factor"),
    "stream_ln_body_scale":  (0.001, 0.0001, 0.01, "LN body to Pbar scaling"),
    "stream_dirac_spike":    (0.02, 0.005, 0.1, "simultaneous note spike base"),
    "stream_dirac_x_exp":    (4,    2,   6,    "x exponent in spike formula"),
    "stream_x_threshold_ratio": (2/3, 0.3, 1.0, "delta < ratio*x case threshold"),
    "stream_burst_coeff":    (0.08, 0.02, 0.5, "P step coefficient"),
    "stream_burst_x_exp":    (1,    0.5,  2,   "x exponent in burst formula"),
    "stream_saturate_width": (24,   10,   50,   "saturation quadratic width"),
    "stream_saturate_offset":(0.5,  0.1,  1.0,  "saturation center offset"),
    "stream_anchor_min":     (1,    0.5,  3,   "anchor modulation min"),
    "stream_anchor_max":     (2,    1,    5,   "anchor modulation max"),
    "stream_smooth_window":  (500,  250,  1000,"smoothing window (ms)"),
    "stream_scale":          (0.001, 0.0001, 0.01, "scale multiplier in smoothing"),
}

# ============================================================
# Anchor unevenness (Abar)
# ============================================================
ABAR = {
    "abar_diff_threshold_1": (0.02, 0.005, 0.08, "d_val threshold: low regime"),
    "abar_diff_threshold_2": (0.07, 0.02,  0.2,  "d_val threshold: mid regime"),
    "abar_coeff_low_A":      (0.75, 0.3, 1.5, "A_step low-regime base"),
    "abar_coeff_low_B":      (0.5,  0.1, 1.0, "A_step low-regime delta multiplier"),
    "abar_coeff_mid_A":      (0.65, 0.3, 1.5, "A_step mid-regime base"),
    "abar_coeff_mid_B":      (5.0,  1.0, 15.0, "A_step mid-regime d_val multiplier"),
    "abar_coeff_mid_C":      (0.5,  0.1, 1.0, "A_step mid-regime delta multiplier"),
    "abar_smooth_window":    (250,  100, 500,  "smoothing window (ms)"),
}

# ============================================================
# Release difficulty (Rbar)
# ============================================================
RELEASE = {
    "release_interval_power":  (-0.5, -2.0, -0.1, "delta release exponent"),
    "release_x_power":         (-1,   -3,   -0.2, "x exponent in release formula"),
    "release_base_scale":      (0.08, 0.01, 0.5, "base release scale"),
    "release_ln_timing_coeff": (0.001, 0.0001, 0.01, "LN head timing to release coeff"),
    "release_smooth_window":   (500,  250,  1000, "smoothing window (ms)"),
    "release_scale":           (0.001, 0.0001, 0.01, "scale multiplier in smoothing"),
    # Enhanced release: LN tail as independent object (now runtime-tunable)
    "release_tail_coeff":      (0.08, 0.01, 2.0,  "base per-tail release coeff"),
    "release_tail_to_tap":     (1.0,  0.1,  5.0,  "release-to-tap difficulty multiplier"),
    "release_same_col_bonus":  (1.5,  1.0,  5.0,  "same-column release bonus"),
    "release_coord_exponent":  (1.0,  0.1,  3.0,  "column distance exponent for release"),
    "short_ln_threshold":      (200,  50,   500,  "LN duration below which release reduced (ms)"),
    "short_ln_reduction":      (0.5,  0.0,  1.0,  "short-LN release reduction factor"),
    "lock_interaction_coeff":  (0.3,  0.0,  2.0,  "lock-hand interaction multiplier"),
    "release_seq_coeff":       (0.03, 0.005, 0.2, "sequential release difficulty coefficient"),
}

# ============================================================
# Shield difficulty (Sbar) — NEW
# ============================================================
SHIELD = {
    "shield_enabled":         (True, 0, 1,    "enable shield component"),
    "shield_window_ms":       (500,  100, 1000, "max lookback for shield detection (ms)"),
    "shield_tau_ms":          (100,  20,  500,  "exponential decay time constant (ms)"),
    "shield_anchor_mod":      (1.0,  0.1,  5.0, "anchor interaction multiplier"),
    "shield_coord_factor":    (1.0,  0.1,  3.0, "coordination difficulty multiplier"),
    "shield_smooth_window":   (500,  250,  1000, "smoothing window (ms)"),
    "shield_scale":           (0.001, 0.0001, 0.01, "scale multiplier"),
}

# ============================================================
# Inverse difficulty (Vbar) — separated spike + guide dip
# ============================================================
INVERSE = {
    "inverse_enabled":        (True,  0,  1,   "enable inverse component"),
    # Inverse spike (same-column only): very close → harder
    "inv_amplitude":          (3.0,   0.5, 15.0, "spike amplitude at dt=0"),
    "inv_tau":                (31,    5,   80,   "spike decay time constant (ms)"),
    "inv_power":              (1.0,   0.5, 3.0,  "spike shape: 1=exponential, 2=Gaussian"),
    # Guide dip (same + cross column): medium distance → easier
    "guide_depth":            (0.9,   0.1, 5.0,  "guide dip depth"),
    "guide_center":           (78,    30,  200,  "guide dip center (ms)"),
    "guide_width":            (31,    10,  100,  "guide dip width (ms)"),
    # Cross-column guide scale (relative to guide_depth)
    "cross_guide_scale":      (0.67,  0.1, 2.0,  "cross-col guide = guide_depth * cross_guide_scale"),
    # Same-column bonus (applied to spike+dip combined)
    "inverse_same_col_bonus": (3.6,   1.0, 8.0,  "same-column multiplier for inverse"),
    # Window
    "inverse_window_ms":      (200,   50,  500,  "max detection window (ms)"),
    "inverse_smooth_window":  (500,   250, 1000, "smoothing window (ms)"),
    "inverse_scale":          (0.001, 0.0001, 0.01, "scale multiplier"),
}

# ============================================================
# Stamina difficulty (Ebar) — NEW
# ============================================================
STAMINA = {
    "stamina_enabled":         (True, 0, 1,     "enable stamina component"),
    "fatigue_tau_ms":          (8000, 2000, 30000, "fatigue decay time constant (ms)"),
    "fatigue_increment_scale": (1.0,  0.1,  5.0,  "fatigue increment per event"),
    "recovery_threshold_ms":   (2000, 500,  10000, "min gap for recovery (ms)"),
    "recovery_tau_ms":         (3000, 1000, 15000, "recovery time constant (ms)"),
    "stamina_note_weight":     (1.0,  0.1,  5.0,  "note count weight for fatigue"),
    "stamina_anchor_weight":   (1.0,  0.1,  5.0,  "anchor weight for fatigue"),
    "stamina_smooth_window":   (2000, 500,  10000, "fatigue smoothing window (ms)"),
}

# ============================================================
# Combination (S, T, D)
# ============================================================
COMBINE = {
    "S_w1":        (0.4,  0.1,  0.9,  "weight for jack branch in S"),
    "S_p":         (1.5,  0.5,  4.0,  "p-norm exponent for S"),
    "alpha_P":     (0.8,  0.1,  3.0,  "Pbar weight in stream branch"),
    "alpha_R":     (35.0, 5.0,  100., "Rbar weight numerator in stream branch"),
    "alpha_C":     (8.0,  2.0,  30.0, "C offset in Rbar denominator"),
    "alpha_S":     (1.0,  0.0,  5.0,  "Sbar weight in stream branch"),
    "alpha_V":     (1.0,  0.0,  5.0,  "Vbar weight in stream branch"),
    "D_beta1":     (2.7,  0.5,  10.0, "coefficient for S^0.5 * T^1.5"),
    "D_beta2":     (0.27, 0.05, 1.0,  "coefficient for linear S term"),
    "D_gamma_e":   (0.0,  0.0,  2.0,  "stamina multiplier (0=disabled by default)"),
}

# ============================================================
# Aggregation
# ============================================================
AGGREGATE = {
    "percentile_targets": ([0.945, 0.935, 0.925, 0.915, 0.845, 0.835, 0.825, 0.815],
                           None, None, "target percentiles for 93rd and 83rd"),
    "w_93":          (0.25, 0.05, 0.5, "93rd percentile weight"),
    "w_83":          (0.20, 0.05, 0.5, "83rd percentile weight"),
    "w_mean":        (0.55, 0.2,  0.8, "weighted mean weight"),
    "coeff_93":      (0.88, 0.5,  1.5, "93rd SR multiplier"),
    "coeff_83":      (0.94, 0.5,  1.5, "83rd SR multiplier"),
    "mean_power":    (5,    1,    10,   "power for weighted mean"),
    "note_norm_N0":  (60,   10,   200,  "note count normalization offset"),
    "rescale_threshold": (9,  7,   12,   "threshold for high-SR rescale"),
    "rescale_divisor":   (1.2, 1.05, 2.0, "rescale divisor"),
    "global_scale":      (0.975, 0.9, 1.1, "global output scale"),
    "comprehensiveness_bonus": (0.0, 0.0, 0.05, "RC+LN combined bonus (0=disabled by default)"),
    # Sigmoid-based player accuracy aggregation (alternative to percentile)
    "agg_sigmoid_k":      (0.5,   0.1, 5.0,   "sigmoid steepness / x-scaling"),
    "agg_sigmoid_C":      (4.0,   1.0, 20.0,  "sigmoid curve shape / denominator offset"),
    "agg_sigmoid_ref_gamma": (0.2, 0.01, 0.99, "ref accuracy fraction gamma=(A_ref-A_min)/(A_max-A_min)"),
    "agg_sigmoid_high_power": (0.0, 0.0, 3.0, "high-D weight emphasis: w_i'=w_i*D_i^power"),
    "calib_a":             (1.0,  0.6,  1.5,   "D pre-calibration scale for sigmoid: D' = a*D + b"),
    "calib_b":             (0.0, -1.0,  1.0,   "D pre-calibration offset for sigmoid"),
    "agg_n_segments":     (30,    10,  100,   "number of difficulty segments"),
    "agg_bisect_tol":     (0.0001, 1e-6, 0.01, "bisection convergence tolerance"),
    "agg_bisect_delta":   (5.0,   1.0, 20.0,  "bisection search margin beyond D range"),
}

# ============================================================
# LN representation
# ============================================================
LN_REP = {
    "ln_body_head_delay_1": (60,  20,  200,  "first LN body ramp point (ms)"),
    "ln_body_head_delay_2": (120, 40,  300,  "second LN body ramp point (ms)"),
    "ln_body_ramp_val_1":   (1.3, 0.5, 3.0,  "LN body value at point 1"),
    "ln_body_ramp_val_2":   (0.3, 0.0, 2.0,  "LN body remaining from point 1 (net = -val1 + val2)"),
    "ln_body_cap_A":        (2.5, 1.0, 5.0,  "LN body cap base"),
    "ln_body_cap_B":        (0.5, 0.1, 2.0,  "LN body cap linear coeff"),
}

# ============================================================
# Master switch: use_enhanced_features (0=SunnyRework clone, 1=all enhancements)
# ============================================================
MASTER = {
    "use_enhanced_release":      (0, 0, 1, "0=clone Rbar, 1=enhanced Rbar"),
    "use_column_distance":       (0, 0, 1, "0=uniform columns, 1=distance weights"),
    "use_shield":                (0, 0, 1, "enable Sbar"),
    "use_inverse":               (0, 0, 1, "enable Vbar"),
    "use_stamina":               (0, 0, 1, "enable Ebar"),
    "use_comprehensiveness":     (0, 0, 1, "enable RC+LN bonus"),
    "use_sigmoid_aggregation":   (0, 0, 1, "1=sigmoid acc model aggregation, 0=percentile-based"),
}


def get_default_params():
    """Return flat dict of (default_value, lower, upper, description)."""
    all_params = {}
    for module in [HIT_LENIENCY, KEY_USAGE, ANCHOR, CORNERS,
                   JACK, CROSS, STREAM, ABAR, RELEASE,
                   SHIELD, INVERSE, STAMINA, COMBINE, AGGREGATE,
                   LN_REP, MASTER]:
        all_params.update(module)
    return all_params


def get_bounds(params=None):
    """Return (lower_bounds, upper_bounds) as lists for optimizer."""
    if params is None:
        params = get_default_params()
    lowers, uppers = [], []
    for key, val in params.items():
        if isinstance(val, tuple) and len(val) >= 3:
            if val[1] is not None and val[2] is not None:
                lowers.append(val[1])
                uppers.append(val[2])
            else:
                # Non-tunable (e.g. percentile_targets)
                pass
    return lowers, uppers


def params_to_dict(params, template=None):
    """Convert list of parameter values back to dict using template keys."""
    if template is None:
        template = get_default_params()
    result = {}
    idx = 0
    for key, val in template.items():
        if isinstance(val, tuple) and len(val) >= 3:
            if val[1] is not None and val[2] is not None:
                result[key] = params[idx]
                idx += 1
            else:
                result[key] = val[0]
        else:
            result[key] = val
    return result


def dict_to_params(param_dict, template=None):
    """Convert parameter dict to list of values for optimizer."""
    if template is None:
        template = get_default_params()
    result = []
    for key, val in template.items():
        if isinstance(val, tuple) and len(val) >= 3:
            if val[1] is not None and val[2] is not None:
                result.append(param_dict.get(key, val[0]))
    return result


# === py ===
"""
SPM Rating — .osu file 

Refactored from sunnyrework/osu_file_parser.py.
Parses osu!mania beatmap files and returns structured note data.
"""



@dataclass
class NoteData:
    """Structured parsed beatmap data."""
    column_count: int = 0
    columns: list = field(default_factory=list)
    note_starts: list = field(default_factory=list)
    note_ends: list = field(default_factory=list)
    note_types: list = field(default_factory=list)
    od: float = -1.0
    file_path: str = ""
    metadata: dict = field(default_factory=dict)


def _str_to_int(s):
    """Convert string to int (handle float strings like '6.0')."""
    return int(float(s))


# Column remapping tables: non-7K modes → 7K physical column indices (0-indexed)
_COLUMN_REMAP = {
    4: {0: 1, 1: 2, 2: 4, 3: 5},       # 4K → 7K columns 2356
    5: {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},  # 5K → 7K columns 23456
    6: {0: 0, 1: 1, 2: 2, 3: 4, 4: 5, 5: 6},  # 6K → 7K columns 123567
}


def _build_remap_table(original_k):
    """Build a column remap array for the given key count.

    Returns an array where arr[original_col] = 7k_col, or None if no remap needed.
    """
    return _COLUMN_REMAP.get(original_k, None)


class Parser:
    """Parser for osu! .osu beatmap files.

    Non-7K maps (4K/5K/6K) have their columns remapped to 7K physical
    column indices, so the rating pipeline always operates in 7K space:
      4K → 7K columns 2,3,5,6 (indices 1,2,4,5)
      5K → 7K columns 2,3,4,5,6 (indices 1,2,3,4,5)
      6K → 7K columns 1,2,3,5,6,7 (indices 0,1,2,4,5,6)
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = NoteData()
        self.data.file_path = file_path
        self._original_column_count = 7
        self._remap = None

    def process(self):
        """Parse the .osu file and populate self.data."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            try:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue

                    # Read Metadata section
                    if stripped == "[Metadata]":
                        self._read_metadata(f)

                    # Read OverallDifficulty
                    if stripped.startswith("OverallDifficulty"):
                        self.data.od = self._read_od(stripped)

                    # Read CircleSize (key count for mania)
                    if stripped.startswith("CircleSize"):
                        self.data.column_count = self._read_circle_size(stripped)

                    # Read HitObjects
                    if stripped == "[HitObjects]":
                        self._read_notes(f)

            except StopIteration:
                pass

    def _read_metadata(self, f):
        """Read metadata lines until we find the next section."""
        for line in f:
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                return
            if ":" in stripped:
                key, _, val = stripped.partition(":")
                self.data.metadata[key.strip()] = val.strip()

    def _read_od(self, line):
        """Parse OverallDifficulty value."""
        try:
            pos = line.index(':')
            return float(line[pos+1:].strip())
        except (ValueError, IndexError):
            return -1.0

    def _read_circle_size(self, line):
        """Parse CircleSize (column count in mania mode).

        Stores the original key count and builds a column remap table
        for non-7K modes. After parsing notes, column_count is set to 7.
        """
        try:
            pos = line.index(':')
            val = line[pos+1:].strip()
            if val == '0':
                self._original_column_count = 10
                self.data.column_count = 10
                return 10
            k = _str_to_int(val)
            self._original_column_count = k
            self._remap = _build_remap_table(k)
            # Set column_count to 7 for 4K/5K/6K (operate in 7K space)
            self.data.column_count = 7 if k in (4, 5, 6) else k
            return self.data.column_count
        except (ValueError, IndexError):
            return -1

    def _read_notes(self, f):
        """Parse all hit objects."""
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("["):
                return
            self._parse_hit_object(stripped)

    def _parse_hit_object(self, obj_line):
        """Parse a single hit object line.

        Format: x,y,time,type,hitSound,objectParams,hitSample
        For mania: x determines column, hold notes have endTime in objectParams.

        Non-7K maps (4K/5K/6K): X is first mapped to an original column
        (0..K-1) using the original key-count's column width, then remapped
        to the corresponding 7K physical column index.
        """
        params = obj_line.split(",")
        if len(params) < 4:
            return

        # Column from x-coordinate using ORIGINAL key count
        x = int(params[0])
        original_k = self._original_column_count
        column_width = 512.0 / original_k
        column = min(int(x / column_width), original_k - 1)

        # Remap to 7K column space for 4K/5K/6K
        if self._remap is not None:
            column = self._remap.get(column, column)

        # Note start time
        note_start = int(params[2])

        # Note type (bit 7 = hold/LN)
        note_type = int(params[3])

        # Note end time (hold notes only, from objectParams)
        # format of params[5]: endTime:...
        note_end = 0
        if len(params) >= 6:
            last_chunk = params[5].split(":")
            try:
                note_end = int(last_chunk[0])
            except (ValueError, IndexError):
                note_end = 0

        self.data.columns.append(column)
        self.data.note_starts.append(note_start)
        self.data.note_ends.append(note_end)
        self.data.note_types.append(note_type)

    def get_parsed_data(self):
        """Return parsed data as a list matching the SunnyRework format.

        Returns:
            [column_count, columns, note_starts, note_ends, note_types, od]
        """
        return [
            self.data.column_count,
            self.data.columns,
            self.data.note_starts,
            self.data.note_ends,
            self.data.note_types,
            self.data.od,
        ]


def parse_file(file_path):
    """Convenience function to parse a .osu file and return raw data."""
    p = Parser(file_path)
    p.process()
    return p.get_parsed_data()


# === py ===
"""
SPM Rating — Preprocessing pipeline.

Converts parsed .osu data into the canonical format used by all
difficulty components: time grids, key usage, LN representation, etc.
"""



def preprocess(parsed_data, mod="", params=None):
    """
    Preprocess parsed beatmap data into the canonical format.

    Args:
        parsed_data: output of parse_file() — [K, cols, starts, ends, types, od]
        mod: speed modifier — "" (none), "DT", "HT"
        params: optional parameter dict from config (uses defaults if None)

    Returns:
        dict with all preprocessed data
    """
    K = parsed_data[0]
    cols = parsed_data[1]
    starts = parsed_data[2]
    ends = parsed_data[3]
    types = parsed_data[4]
    od = parsed_data[5]

    # ================================================================
    # Hit leniency x
    # ================================================================
    x = 0.3 * ((64.5 - math.ceil(od * 3)) / 500) ** 0.5
    x = min(x, 0.6 * (x - 0.09) + 0.09)

    # ================================================================
    # Build note_seq: (column, head_time, tail_time)
    # ================================================================
    note_seq = []
    for i in range(len(cols)):
        k = cols[i]
        h = starts[i]
        # LN if type has bit 7 set (128)
        t = ends[i] if (types[i] & 128) else -1
        if mod == "DT":
            h = int(math.floor(h * 2 / 3))
            t = int(math.floor(t * 2 / 3)) if t >= 0 else t
        elif mod == "HT":
            h = int(math.floor(h * 4 / 3))
            t = int(math.floor(t * 4 / 3)) if t >= 0 else t
        note_seq.append((k, h, t))

    # Sort by head time, then column
    note_seq.sort(key=lambda tup: (tup[1], tup[0]))

    # ================================================================
    # Group notes by column — sorted lists (K entries, empty for unused columns)
    # ================================================================
    note_seq_by_column = [[] for _ in range(K)]
    for tup in note_seq:
        note_seq_by_column[tup[0]].append(tup)

    # ================================================================
    # LN sequences
    # ================================================================
    LN_seq = [n for n in note_seq if n[2] >= 0]
    tail_seq = sorted(LN_seq, key=lambda tup: tup[2])

    LN_seq_by_column = [[] for _ in range(K)]
    for tup in LN_seq:
        LN_seq_by_column[tup[0]].append(tup)

    # ================================================================
    # Time range T
    # ================================================================
    max_head = max(n[1] for n in note_seq) if note_seq else 0
    max_tail = max(n[2] for n in note_seq) if note_seq else 0
    T = max(max_head, max_tail) + 1

    # ================================================================
    # Compute corners (time grid breakpoints)
    # ================================================================
    all_corners, base_corners, A_corners = _compute_corners(T, note_seq)

    # ================================================================
    # Key usage
    # ================================================================
    key_usage = _compute_key_usage(K, T, note_seq, base_corners)
    active_columns = [
        [k for k in range(K) if key_usage[k][i]]
        for i in range(len(base_corners))
    ]

    # ================================================================
    # Key usage 400 (for anchor)
    # ================================================================
    key_usage_400 = _compute_key_usage_400(K, T, note_seq, base_corners)

    # ================================================================
    # LN bodies sparse representation
    # ================================================================
    LN_rep = _compute_LN_rep(LN_seq, T)

    # ================================================================
    # Per-column note times (for release lookup)
    # ================================================================
    times_by_column = {
        i: [note[1] for note in column]
        for i, column in enumerate(note_seq_by_column)
    }

    # ================================================================
    # Note hit times sorted (for C/note count window)
    # ================================================================
    note_hit_times = sorted(n[1] for n in note_seq)

    # ================================================================
    # Flatten LN_seq_by_column for cross-column LN processing
    # ================================================================
    # Already have LN_seq and tail_seq

    return {
        "x": x,
        "K": K,
        "T": T,
        "od": od,
        "note_seq": note_seq,
        "note_seq_by_column": note_seq_by_column,
        "LN_seq": LN_seq,
        "tail_seq": tail_seq,
        "LN_seq_by_column": LN_seq_by_column,
        "all_corners": all_corners,
        "base_corners": base_corners,
        "A_corners": A_corners,
        "key_usage": key_usage,
        "active_columns": active_columns,
        "key_usage_400": key_usage_400,
        "LN_rep": LN_rep,
        "times_by_column": times_by_column,
        "note_hit_times": note_hit_times,
    }


def _compute_corners(T, note_seq):
    """Compute time grid breakpoints."""
    # base_corners: ±500 around note bounds
    corners_base = set()
    for (_, h, t) in note_seq:
        corners_base.add(h)
        if t >= 0:
            corners_base.add(t)
    for s in list(corners_base):
        corners_base.add(s + 501)
        corners_base.add(s - 499)
        corners_base.add(s + 1)  # resolve Dirac-Delta
    corners_base.add(0)
    corners_base.add(T)
    corners_base = sorted(s for s in corners_base if 0 <= s <= T)

    # A_corners: ±1000 around note bounds
    corners_A = set()
    for (_, h, t) in note_seq:
        corners_A.add(h)
        if t >= 0:
            corners_A.add(t)
    for s in list(corners_A):
        corners_A.add(s + 1000)
        corners_A.add(s - 1000)
    corners_A.add(0)
    corners_A.add(T)
    corners_A = sorted(s for s in corners_A if 0 <= s <= T)

    # Union
    all_corners = sorted(set(corners_base) | set(corners_A))
    all_corners = np.array(all_corners, dtype=float)
    base_corners = np.array(corners_base, dtype=float)
    A_corners = np.array(corners_A, dtype=float)

    return all_corners, base_corners, A_corners


def _compute_key_usage(K, T, note_seq, base_corners):
    """Boolean key usage: column k active at base_corners[i]?"""
    key_usage = {k: np.zeros(len(base_corners), dtype=bool) for k in range(K)}
    for (k, h, t) in note_seq:
        startTime = max(h - 150, 0)
        endTime = (h + 150) if t < 0 else min(t + 150, T - 1)
        left_idx = np.searchsorted(base_corners, startTime, side='left')
        right_idx = np.searchsorted(base_corners, endTime, side='left')
        idx = np.arange(left_idx, right_idx)
        if len(idx) > 0:
            key_usage[k][idx] = True
    return key_usage


def _compute_key_usage_400(K, T, note_seq, base_corners):
    """Weighted key usage for anchor computation."""
    key_usage_400 = {k: np.zeros(len(base_corners), dtype=float) for k in range(K)}
    for (k, h, t) in note_seq:
        startTime = max(h, 0)
        endTime = h if t < 0 else min(t, T - 1)
        left400_idx = np.searchsorted(base_corners, startTime - 400, side='left')
        left_idx = np.searchsorted(base_corners, startTime, side='left')
        right_idx = np.searchsorted(base_corners, endTime, side='left')
        right400_idx = np.searchsorted(base_corners, endTime + 400, side='left')

        # Active region
        idx = np.arange(left_idx, right_idx)
        if len(idx) > 0:
            duration = min(endTime - startTime, 1500)
            key_usage_400[k][idx] += 3.75 + duration / 150.0

        # Left ramp
        idx = np.arange(left400_idx, left_idx)
        if len(idx) > 0:
            key_usage_400[k][idx] += 3.75 - 3.75 / 400**2 * (base_corners[idx] - startTime) ** 2

        # Right ramp
        idx = np.arange(right_idx, right400_idx)
        if len(idx) > 0:
            key_usage_400[k][idx] += 3.75 - 3.75 / 400**2 * np.abs(base_corners[idx] - endTime) ** 2

    return key_usage_400


def _compute_LN_rep(LN_seq, T):
    """Sparse representation of LN bodies."""
    diff = {}
    for (k, h, t) in LN_seq:
        t0 = min(h + 60, t)
        t1 = min(h + 120, t)
        diff[t0] = diff.get(t0, 0) + 1.3
        diff[t1] = diff.get(t1, 0) + (-1.3 + 1)  # net: -0.3
        diff[t] = diff.get(t, 0) - 1

    points = sorted(set([0, T] + list(diff.keys())))
    values = []
    cumsum = [0]
    curr = 0.0

    for i in range(len(points) - 1):
        t = points[i]
        if t in diff:
            curr += diff[t]
        v = min(curr, 2.5 + 0.5 * curr)
        values.append(v)
        seg_length = points[i+1] - points[i]
        cumsum.append(cumsum[-1] + seg_length * v)

    return (points, cumsum, values)


def find_next_note_in_column(note, times_by_column, note_seq_by_column):
    """Find the next note in the same column after a given note."""
    k, h, t = note
    idx = bisect.bisect_left(times_by_column[k], h)
    if idx + 1 < len(note_seq_by_column[k]):
        return note_seq_by_column[k][idx + 1]
    return (0, 10**9, 10**9)


# === combine.py ===
"""
SPM Rating — S/T/D combination formulas and C/Ks computation.

Ported from SunnyRework algorithm.py.
"""



def compute_C_and_Ks(K, note_seq, key_usage, base_corners):
    """
    Compute C (note count in 500ms window) and Ks (local key usage count).

    Args:
        K: number of columns
        note_seq: sorted list of (col, head, tail) tuples
        key_usage: dict {k: bool_array} per-column activity
        base_corners: time grid

    Returns:
        C_step: note count per base_corner
        Ks_step: active key count per base_corner
    """
    note_hit_times = sorted(n[1] for n in note_seq)
    C_step = np.zeros(len(base_corners))
    for i, s in enumerate(base_corners):
        low = s - 500
        high = s + 500
        cnt = bisect.bisect_left(note_hit_times, high) - bisect.bisect_left(note_hit_times, low)
        C_step[i] = cnt

    Ks_step = np.array([
        max(sum(1 for k in range(K) if key_usage[k][i]), 1)
        for i in range(len(base_corners))
    ])

    return C_step, Ks_step


def compute_D(all_corners, base_corners, Abar, Jbar, Xbar, Pbar, Rbar,
              C_step, Ks_step, alpha_S=0, Vbar=None, Sbar_input=None,
              stamina_factor=None,
              S_w1=0.4, S_p=1.5,
              alpha_P=0.8, alpha_R=35.0, alpha_C=8.0,
              alpha_S_val=1.0, alpha_V=1.0,
              D_beta1=2.7, D_beta2=0.27,
              D_gamma_e=0.0, Abar_scale=1.0):
    """
    Compute per-point difficulty D on all_corners.

    S(s) = [w1 * (A^(3/Ks) * min(J, 8+0.85J))^p
          + (1-w1) * (A^(2/3) * (alpha_P*P + alpha_R*R/(C+alpha_C) + alpha_S*Sbar + alpha_V*Vbar))^p]^(1/p)

    T(s) = A^(3/Ks) * X / (X + S + 1)

    D(s) = beta1 * S^0.5 * T^1.5 + beta2 * S

    Args:
        all_corners: full time grid
        base_corners: base time grid
        <component arrays>
        S_w1: weight for jack branch in S
        S_p: p-norm exponent for S
        alpha_P: Pbar weight in stream branch
        alpha_R: Rbar weight numerator in stream branch
        alpha_C: C offset in Rbar denominator
        alpha_S_val: Sbar weight
        alpha_V: Vbar weight
        D_beta1: coefficient for S^0.5 * T^1.5
        D_beta2: coefficient for linear S term
        D_gamma_e: stamina multiplier
        Abar_scale: multiplier for Abar (tunable anchor/unevenness sensitivity)

    Returns:
        D_all, S_all, T_all, C_arr, Ks_arr
    """
    # Apply tunable anchor scale
    Abar = Abar * Abar_scale

    # Step-interpolate C, Ks to all_corners
    C_arr = step_interp(all_corners, base_corners, C_step)
    Ks_arr = step_interp(all_corners, base_corners, Ks_step)

    # Apply Vbar as multiplicative modifier on Rbar (inverse/guide effect)
    # Vbar > 0 → release harder (very close inverse), Vbar < 0 → release easier (guide)
    # Clamp to [0.15, 3.0] to prevent negative/exploding Rbar while preserving intent
    if Vbar is not None and alpha_V > 1e-9:
        multiplier = np.clip(1.0 + alpha_V * Vbar, 0.15, 3.0)
        Rbar = Rbar * multiplier

    # Build stream branch with tunable coefficients
    stream_branch = alpha_P * Pbar + alpha_R * Rbar / (C_arr + alpha_C)
    if Sbar_input is not None and alpha_S > 0:
        stream_branch += alpha_S_val * Sbar_input

    # S: sustained difficulty
    w2 = 1.0 - S_w1
    jack_branch = Abar ** (3 / Ks_arr) * np.minimum(Jbar, 8 + 0.85 * Jbar)
    stream_branch_full = Abar ** (2 / 3) * stream_branch

    S_all = ((S_w1 * jack_branch ** S_p)
             + (w2 * stream_branch_full ** S_p)) ** (1.0 / S_p)

    # T: technicality
    T_all = (Abar ** (3 / Ks_arr) * Xbar) / (Xbar + S_all + 1)

    # D: instantaneous difficulty
    D_all = D_beta1 * (S_all ** 0.5) * (T_all ** 1.5) + D_beta2 * S_all

    # Apply stamina if enabled
    if stamina_factor is not None and D_gamma_e > 1e-9:
        D_all = D_all * (1 + D_gamma_e * stamina_factor)

    return D_all, S_all, T_all, C_arr, Ks_arr


# === aggregate.py ===
"""
SPM Rating — Aggregation and final SR computation.

Ported from SunnyRework algorithm.py.
"""



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


# === aggregate_sigmoid.py ===
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


# === rating.py ===
"""
SPM Rating — Top-level SR calculation pipeline.

# ===== 生命周期管理 =====
# 完整计算 (解析+分量+组合):
#   SR, details = calculate(file_path, ...)
#
# 优化后: 预计算+组合分离 (用于调参加速):
#   cache = precompute(file_path, mod, use_enhanced)   # 只跑一次
#   SR, details = combine(cache, params)                # 参数调优时只跑这个
#   SR, details = combine(cache, different_params)      # 再跑一遍, 秒级
"""



def _p(params, key, default):
    """Get parameter value from params dict, with fallback."""
    if params and key in params:
        return params[key]
    return default


# ====================================================================
# 预计算：解析 + 分量（仅依赖谱面本身，不依赖调参参数）
# ====================================================================
def precompute(file_path, mod="", use_enhanced=False, params=None):
    """
    预计算谱面的所有分量（不依赖调参参数的部分）。

    Args:
        file_path: .osu 文件路径
        mod: 模组标识
        use_enhanced: 是否计算 enhanced cross/release
        params: 可选的参数字典（用于依赖参数的分量如 stream_booster）

    返回:
        cache dict，包含解析+预处理+所有分量数据。
    """
    # 解析
    parsed = parse_file(file_path)
    data = preprocess(parsed, mod=mod)

    x = data["x"]
    K = data["K"]
    note_seq = data["note_seq"]
    note_seq_by_column = data["note_seq_by_column"]
    LN_seq = data["LN_seq"]
    tail_seq = data["tail_seq"]
    all_corners = data["all_corners"]
    base_corners = data["base_corners"]
    A_corners = data["A_corners"]
    key_usage = data["key_usage"]
    active_columns = data["active_columns"]
    key_usage_400 = data["key_usage_400"]
    LN_rep = data["LN_rep"]

    # 提取预计算时使用的分量参数，存入缓存供 combine 按需重算
    _comp_params = {}
    if params:
        for k in ("stream_booster_scale", "jack_aggregation_power", "cross_fast_scale", "multi_jack_boost"):
            if k in params:
                _comp_params[k] = params[k]

    # --- 不依赖参数的分量 ---
    anchor_arr = compute_anchor(K, key_usage_400, base_corners)
    delta_ks, Jbar_base, Jbar_ks = compute_Jbar(K, x, note_seq_by_column, base_corners,
                                                        aggregation_power=_comp_params.get("jack_aggregation_power", 5),
                                                        multi_jack_boost=_comp_params.get("multi_jack_boost", 0.0))
    Pbar_base = compute_Pbar(K, x, note_seq, LN_rep, anchor_arr, base_corners,
                                     stream_booster_scale=_comp_params.get("stream_booster_scale", 1.7e-7))
    Abar_A, dks = compute_Abar(K, delta_ks, active_columns, A_corners, base_corners)
    C_step, Ks_step = compute_C_and_Ks(K, note_seq, key_usage, base_corners)

    cache = {
        "_version": 8,
        "_comp_params": _comp_params,
        "x": x,
        "K": K,
        "T": data["T"],
        "od": data["od"],
        "note_seq": note_seq,
        "LN_seq": LN_seq,
        "tail_seq": tail_seq,
        "note_seq_by_column": note_seq_by_column,
        "all_corners": all_corners,
        "base_corners": base_corners,
        "A_corners": A_corners,
        "key_usage": key_usage,
        "active_columns": active_columns,
        "LN_rep": LN_rep,
        # 分量（不依赖参数）
        "anchor_arr": anchor_arr,
        "delta_ks": delta_ks,
        "Jbar_base": Jbar_base,
        "Jbar_ks": Jbar_ks,       # per-column smoothed jack values for runtime re-aggregation
        "Pbar_base": Pbar_base,
        "Abar_A": Abar_A,
        "C_step": C_step,
        "Ks_step": Ks_step,
        # 这些分量在预计算时按指定模式计算
        "use_enhanced": use_enhanced,
    }

    # Cross 和 Release 在两种模式下都预计算（clone 版本的为默认）
    cache["Xbar_base_clone"] = compute_Xbar(K, x, note_seq_by_column, active_columns, base_corners)
    cache["Rbar_base_clone"], _ = compute_Rbar(K, x, note_seq_by_column, tail_seq, base_corners)

    if use_enhanced:
        # Cross: precompute structured data for runtime RC/LN parameter blending (Phase 4)
        cache["cross_data"] = precompute_cross_enhanced_data(
            K, x, note_seq_by_column, active_columns, base_corners)
        # LN ratio for RC/LN parameter interpolation (0 = pure RC, higher = more LN)
        n_taps = len(note_seq) - len(LN_seq)
        n_total_objects = len(note_seq)
        cache["ln_ratio"] = len(LN_seq) / max(n_total_objects, 1)

        # Release: precompute structured data (like Shield/Inverse)
        # so parameters can be tuned at runtime without recaching
        cache["release_data"] = precompute_release_data(
            K, x, note_seq_by_column, tail_seq, note_seq
        )
        # Also bake a default Rbar for backward compat (fast from structured data)
        cache["Rbar_base_enhanced"] = compute_Rbar_enhanced_fast(
            cache["release_data"], base_corners,
            release_tail_coeff=RELEASE["release_tail_coeff"][0],
            release_tail_to_tap_factor=RELEASE["release_tail_to_tap"][0],
            release_same_col_bonus=RELEASE["release_same_col_bonus"][0],
            release_coord_exponent=RELEASE["release_coord_exponent"][0],
            short_ln_threshold=RELEASE.get("short_ln_threshold", (200,))[0],
            short_ln_reduction=RELEASE.get("short_ln_reduction", (0.5,))[0],
            lock_interaction_coeff=RELEASE.get("lock_interaction_coeff", (0.3,))[0],
            release_seq_coeff=RELEASE.get("release_seq_coeff", (0.03,))[0],
            smooth_window=RELEASE["release_smooth_window"][0],
            smooth_scale=RELEASE["release_scale"][0],
        )
        # Precompute structured data for fast Shield and Inverse at combine-time
        cache["shield_data"] = precompute_shield_data(
            K, note_seq_by_column, LN_seq)
        cache["inverse_data"] = precompute_inverse_data(
            K, note_seq_by_column, LN_seq)

    return cache


# ====================================================================
# 组合：从预计算缓存 + 参数 → 最终SR（秒级）
# ====================================================================
def combine(cache, params=None):
    """
    从预计算缓存 + 调参参数快速计算SR。

    Args:
        cache: precompute() 返回的dict
        params: 调参参数字典（可选，默认用config默认值）

    Returns:
        SR: 最终星位
        details: 中间数据
    """
    use_enhanced = cache.get("use_enhanced", False)
    all_corners = cache["all_corners"]
    base_corners = cache["base_corners"]
    A_corners = cache["A_corners"]

    # 分量从缓存读取
    Jbar_base = cache["Jbar_base"]
    Pbar_base = cache["Pbar_base"]
    C_step = cache["C_step"]
    Ks_step = cache["Ks_step"]
    anchor_arr = cache["anchor_arr"]

    # 按需重算 Jbar（当 jack 聚合参数与预计算时不同时）— cheap: just re-aggregation
    comp_params_cache = cache.get("_comp_params", {})
    cur_jack_agg = _p(params, "jack_aggregation_power", 5)
    cur_multi_jack = _p(params, "multi_jack_boost", 0.0)
    if "Jbar_ks" in cache and (
        abs(cur_jack_agg - comp_params_cache.get("jack_aggregation_power", 5)) > 1e-12 or
        abs(cur_multi_jack - comp_params_cache.get("multi_jack_boost", 0.0)) > 1e-12):
        Jbar_base = aggregate_Jbar(
            cache["K"], cache["Jbar_ks"], cache["delta_ks"], cache["base_corners"],
            aggregation_power=cur_jack_agg,
            multi_jack_boost=cur_multi_jack,
        )

    # 按需重算 Pbar（当 stream_booster_scale 与预计算时不同时）
    cur_boost = _p(params, "stream_booster_scale", 1.7e-7)
    if abs(cur_boost - comp_params_cache.get("stream_booster_scale", 1.7e-7)) > 1e-12 and False:  # DISABLED for speed; use recache to change booster
        Pbar_base = compute_Pbar(
            cache["K"], cache["x"], cache["note_seq"],
            cache["LN_rep"], cache["anchor_arr"], cache["base_corners"],
            stream_booster_scale=cur_boost,
        )

    # 根据模式选择合适的 Xbar/Rbar（enhanced 模式也可降级用 clone 的）
    if use_enhanced:
        # Phase 4: Cross runtime computation with RC/LN parameter blending
        cross_data = cache.get("cross_data")
        if cross_data is not None:
            ln_ratio = cache.get("ln_ratio", 0.0)
            # Interpolate cross params: ln_ratio=0 → RC params, ln_ratio=1 → LN params
            t = ln_ratio
            dist_exp_rc = _p(params, "cross_dist_exponent_rc",
                             _p(params, "cross_dist_exponent", 1.0))
            dist_exp_ln = _p(params, "cross_dist_exponent_ln",
                             _p(params, "cross_dist_exponent", 1.0))
            penalty_rc = _p(params, "cross_same_hand_penalty_rc",
                            _p(params, "cross_same_hand_penalty", 0.3))
            penalty_ln = _p(params, "cross_same_hand_penalty_ln",
                            _p(params, "cross_same_hand_penalty", 0.3))
            dist_exponent = dist_exp_rc + (dist_exp_ln - dist_exp_rc) * t
            same_hand_penalty = penalty_rc + (penalty_ln - penalty_rc) * t
            Xbar_base = compute_Xbar_enhanced_fast(
                cross_data, base_corners, cache["x"],
                dist_exponent=dist_exponent,
                same_hand_penalty=same_hand_penalty,
                thumb_bridge=_p(params, "cross_thumb_bridge_factor", 0.5),
            )
        else:
            Xbar_base = cache.get("Xbar_base_enhanced", cache["Xbar_base_clone"])
        # Release: runtime computation from structured data (like Shield/Inverse)
        release_data = cache.get("release_data")
        if release_data is not None:
            Rbar_base = compute_Rbar_enhanced_fast(
                release_data, base_corners,
                release_tail_coeff=_p(params, "release_tail_coeff", 0.08),
                release_tail_to_tap_factor=_p(params, "release_tail_to_tap", 1.0),
                release_same_col_bonus=_p(params, "release_same_col_bonus", 1.5),
                release_coord_exponent=_p(params, "release_coord_exponent", 1.0),
                short_ln_threshold=_p(params, "short_ln_threshold", 200),
                short_ln_reduction=_p(params, "short_ln_reduction", 0.5),
                lock_interaction_coeff=_p(params, "lock_interaction_coeff", 0.3),
                release_seq_coeff=_p(params, "release_seq_coeff", 0.03),
                smooth_window=_p(params, "release_smooth_window", 500),
                smooth_scale=_p(params, "release_scale", 0.001),
            )
        else:
            Rbar_base = cache.get("Rbar_base_enhanced", cache["Rbar_base_clone"])
    else:
        Xbar_base = cache["Xbar_base_clone"]
        Rbar_base = cache["Rbar_base_clone"]

    # Abar
    Abar_A = cache["Abar_A"]

    # --- Enhanced 新分量（按需计算，因为它们依赖参数） ---
    Sbar_base = None
    Vbar_base = None
    Ebar_base = None

    if use_enhanced:
        if _p(params, "use_shield", 1) > 0.5:
            shield_data = cache.get("shield_data")
            if shield_data is not None:
                Sbar_base = compute_Sbar_fast(
                    shield_data, base_corners,
                    shield_tau_ms=_p(params, "shield_tau_ms", 100),
                    shield_anchor_mod=_p(params, "shield_anchor_mod", 1.0),
                    shield_coord_factor=_p(params, "shield_coord_factor", 1.0),
                    smooth_window=_p(params, "shield_smooth_window", 500),
                    smooth_scale=_p(params, "shield_scale", 0.001),
                )
            else:
                Sbar_base = compute_Sbar(
                    cache["K"], cache["note_seq_by_column"], cache["LN_seq"],
                    base_corners,
                    shield_tau_ms=_p(params, "shield_tau_ms", 100),
                    shield_anchor_mod=_p(params, "shield_anchor_mod", 1.0),
                    shield_coord_factor=_p(params, "shield_coord_factor", 1.0),
                    smooth_window=_p(params, "shield_smooth_window", 500),
                    smooth_scale=_p(params, "shield_scale", 0.001),
                )
        if _p(params, "use_inverse", 1) > 0.5:
            inverse_data = cache.get("inverse_data")
            if inverse_data is not None:
                Vbar_base = compute_Vbar_fast(
                    inverse_data, base_corners,
                    inv_amplitude=_p(params, "inv_amplitude", 3.0),
                    inv_tau=_p(params, "inv_tau", 31),
                    inv_power=_p(params, "inv_power", 1.0),
                    guide_depth=_p(params, "guide_depth", 0.9),
                    guide_center=_p(params, "guide_center", 78),
                    guide_width=_p(params, "guide_width", 31),
                    cross_guide_scale=_p(params, "cross_guide_scale", 0.67),
                    same_col_bonus=_p(params, "inverse_same_col_bonus", 3.6),
                    window_ms=_p(params, "inverse_window_ms", 200),
                )
            else:
                Vbar_base = compute_Vbar(
                    cache["K"], cache["note_seq_by_column"], cache["LN_seq"],
                    base_corners,
                    inv_amplitude=_p(params, "inv_amplitude", 3.0),
                    inv_tau=_p(params, "inv_tau", 31),
                    inv_power=_p(params, "inv_power", 1.0),
                    guide_depth=_p(params, "guide_depth", 0.9),
                    guide_center=_p(params, "guide_center", 78),
                    guide_width=_p(params, "guide_width", 31),
                    cross_guide_scale=_p(params, "cross_guide_scale", 0.67),
                    same_col_bonus=_p(params, "inverse_same_col_bonus", 3.6),
                    window_ms=_p(params, "inverse_window_ms", 200),
                )
        if _p(params, "use_stamina", 1) > 0.5:
            Ebar_base = compute_Ebar(
                cache["K"], cache["note_seq"], base_corners, anchor_arr,
                fatigue_tau_ms=_p(params, "fatigue_tau_ms", 8000),
                fatigue_increment=_p(params, "fatigue_increment_scale", 1.0),
                recovery_threshold_ms=_p(params, "recovery_threshold_ms", 2000),
                recovery_tau_ms=_p(params, "recovery_tau_ms", 3000),
            )

    # --- 插值到 all_corners ---
    Jbar = interp_values(all_corners, base_corners, Jbar_base)
    Xbar = interp_values(all_corners, base_corners, Xbar_base)
    Pbar = interp_values(all_corners, base_corners, Pbar_base)
    Rbar = interp_values(all_corners, base_corners, Rbar_base)
    Abar = interp_values(all_corners, A_corners, Abar_A)

    Sbar_all = interp_values(all_corners, base_corners, Sbar_base) if Sbar_base is not None else None
    Vbar_all = interp_values(all_corners, base_corners, Vbar_base) if Vbar_base is not None else None
    Ebar_all = interp_values(all_corners, base_corners, Ebar_base) if Ebar_base is not None else None

    # --- 组合 S/T/D ---
    D_all, S_all, T_all, C_arr, Ks_arr = compute_D(
        all_corners, base_corners,
        Abar, Jbar, Xbar, Pbar, Rbar,
        C_step, Ks_step,
        alpha_S=(1.0 if use_enhanced else 0),
        Vbar=Vbar_all,
        Sbar_input=Sbar_all,
        stamina_factor=Ebar_all,
        S_w1=_p(params, "S_w1", 0.4),
        S_p=_p(params, "S_p", 1.5),
        alpha_P=_p(params, "alpha_P", 0.8),
        alpha_R=_p(params, "alpha_R", 35.0),
        alpha_C=_p(params, "alpha_C", 8.0),
        alpha_S_val=_p(params, "alpha_S", 1.0),
        alpha_V=_p(params, "alpha_V", 1.0),
        D_beta1=_p(params, "D_beta1", 2.7),
        D_beta2=_p(params, "D_beta2", 0.27),
        D_gamma_e=_p(params, "D_gamma_e", 0.0),
        Abar_scale=_p(params, "Abar_scale", 1.0),
    )

    # --- 聚合 SR ---
    total_notes = compute_total_notes(cache["note_seq"], cache["LN_seq"])

    use_sigmoid_agg = _p(params, "use_sigmoid_aggregation", 0)
    if use_sigmoid_agg:
        # D pre-calibration for sigmoid: D' = a*D + b
        calib_a = _p(params, "calib_a", 1.0)
        calib_b = _p(params, "calib_b", 0.0)
        D_all_calib = calib_a * D_all + calib_b if abs(calib_a - 1.0) > 1e-12 or abs(calib_b) > 1e-12 else D_all
        SR, agg_details = compute_SR_sigmoid(
            all_corners, C_arr, D_all_calib, total_notes, cache["LN_seq"],
            n_segments=int(_p(params, "agg_n_segments", 30)),
            sigmoid_k=_p(params, "agg_sigmoid_k", 0.5),
            sigmoid_C=_p(params, "agg_sigmoid_C", 4.0),
            sigmoid_gamma=_p(params, "agg_sigmoid_ref_gamma", 0.2),
            sigmoid_high_power=_p(params, "agg_sigmoid_high_power", 0.0),
            bisect_tol=_p(params, "agg_bisect_tol", 0.0001),
            bisect_delta=_p(params, "agg_bisect_delta", 5.0),
            note_norm_N0=_p(params, "note_norm_N0", 60),
            rescale_threshold=_p(params, "rescale_threshold", 9),
            rescale_divisor=_p(params, "rescale_divisor", 1.2),
            global_scale=_p(params, "global_scale", 0.975),
        )
    else:
        SR, agg_details = compute_SR(
            all_corners, C_arr, D_all, total_notes, cache["LN_seq"],
            w_93=_p(params, "w_93", 0.25),
            w_83=_p(params, "w_83", 0.20),
            w_mean=_p(params, "w_mean", 0.55),
            coeff_93=_p(params, "coeff_93", 0.88),
            coeff_83=_p(params, "coeff_83", 0.94),
            mean_power=_p(params, "mean_power", 5),
            note_norm_N0=_p(params, "note_norm_N0", 60),
            rescale_threshold=_p(params, "rescale_threshold", 9),
            rescale_divisor=_p(params, "rescale_divisor", 1.2),
            global_scale=_p(params, "global_scale", 0.975),
        )

    details = {
        "total_notes": total_notes,
        "n_raw": len(cache["note_seq"]),
        "n_LN": len(cache["LN_seq"]),
        "D_all": D_all, "S_all": S_all, "T_all": T_all,
        "Jbar": Jbar, "Xbar": Xbar, "Pbar": Pbar,
        "Abar": Abar, "Rbar": Rbar,
        "Sbar": Sbar_all, "Vbar": Vbar_all, "Ebar": Ebar_all,
        "C_arr": C_arr, "Ks_arr": Ks_arr,
        **agg_details,
    }

    return SR, details


# ====================================================================
# 原始完整接口（向后兼容）
# ====================================================================
def calculate(file_path, mod="", use_enhanced=False, params=None):
    """
    完整计算：解析 + 预计算 + 组合。

    等同于:
        c = precompute(file_path, mod, use_enhanced)
        return combine(c, params)
    """
    c = precompute(file_path, mod, use_enhanced)
    return combine(c, params)

# ============================================================================
# CLI Entry Point
# ============================================================================

def _p(params, key, default):
    """Get param from dict with default."""
    if params is None:
        return default
    return params.get(key, default)


def _load_params():
    """Return tuned params as the base parameter dict."""
    return dict(TUNED_PARAMS)


def compute_sr_map(osu_path, params=None):
    """Compute Star Rating for a single .osu chart.

    Args:
        osu_path: path to .osu file
        params: optional param overrides (dict), uses TUNED_PARAMS if None

    Returns:
        sr: float Star Rating value
        details: dict with diagnostic info (D_all, D_solved, component values, etc.)
    """
    if params is None:
        params = dict(TUNED_PARAMS)
    else:
        # Merge with tuned params
        p = dict(TUNED_PARAMS)
        p.update(params)
        params = p
    params["use_sigmoid_aggregation"] = 1
    cache = precompute(osu_path, use_enhanced=True, params=params)
    sr, details = combine(cache, params=params)
    return sr, details


def main():
    print("=" * 55)
    print("  SPM Rating — Sigmoid 聚合 SR 计算器")
    print("=" * 55)
    print(f"  k={TUNED_PARAMS['agg_sigmoid_k']:.2f}, C={TUNED_PARAMS['agg_sigmoid_C']:.2f}, "
          f"gamma={TUNED_PARAMS['agg_sigmoid_ref_gamma']:.3f}")
    print(f"  训练 MAE={0.2253}, Pass@0.5={92.2}%")
    print()

    args = sys.argv[1:]
    target = args[0] if args else os.path.dirname(os.path.abspath(__file__))

    # Collect .osu files
    if os.path.isfile(target) and target.endswith(".osu"):
        osu_files = [target]
    elif os.path.isdir(target):
        osu_files = []
        for root, dirs, files in os.walk(target):
            for f in files:
                if f.endswith('.osu'):
                    osu_files.append(os.path.join(root, f))
        osu_files.sort()
    else:
        print(f"  无效目标: {target}")
        sys.exit(1)

    if not osu_files:
        print(f"  未找到 .osu 文件")
        sys.exit(1)

    if len(osu_files) == 1:
        fpath = osu_files[0]
        print(f"  计算: {os.path.basename(fpath)}")
        sr, d = compute_sr_map(fpath)
        print(f"  SR = {sr:.4f}")
        print(f"  D_all 范围: [{d.get('D_min',0):.2f}, {d.get('D_max',0):.2f}]")
        print(f"  D_solved: {d.get('D_solved',0):.2f}")
        if 'n_raw' in d:
            print(f"  notes: {d.get('n_raw',0)}, LN: {d.get('n_LN',0)}")
        return

    print(f"  计算 {len(osu_files)} 张谱面...")
    print()
    results = []
    errors = 0
    t0 = time.time()
    for i, fpath in enumerate(osu_files):
        fname = os.path.basename(fpath)
        try:
            sr, _ = compute_sr_map(fpath)
            results.append((fname, sr))
            print(f"  [{i+1}/{len(osu_files)}] {fname}  SR={sr:.4f}")
        except Exception as e:
            errors += 1
            print(f"  [{i+1}/{len(osu_files)}] {fname}  [失败: {e}]")

    elapsed = time.time() - t0
    print()
    print("-" * 55)
    print(f"  完成 {len(results)} OK, {errors} 失败 ({elapsed:.1f}s)")
    if results:
        srs = [r[1] for r in results]
        print(f"  SR 范围: {min(srs):.4f} ~ {max(srs):.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
