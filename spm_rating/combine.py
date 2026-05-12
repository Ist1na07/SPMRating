"""
SPM Rating — S/T/D combination formulas and C/Ks computation.

Ported from SunnyRework algorithm.py.
"""

import bisect
import numpy as np
from .utils import step_interp


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
