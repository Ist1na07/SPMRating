"""
SPM Rating — RC/LN sub-model evaluation functions.

Provides compute_rc_sr() and compute_ln_sr() that estimate
RC-only and LN-only difficulty from the same precomputed cache.

RC model: same D formula structure as total, but Rbar=Sbar=Vbar=0.
LN model: additive D formula using only Rbar, Sbar, Vbar.
Both route through sigmoid aggregation with their own parameters.
"""

import numpy as np
from .utils import interp_values, step_interp
from .aggregate_sigmoid import compute_SR_sigmoid
from .aggregate import compute_total_notes


def _p(params, key, default):
    if params and key in params:
        return params[key]
    return default


# ============================================================
# RC 模型
# ============================================================

def compute_D_rc(all_corners, base_corners, Abar, Jbar, Xbar, Pbar,
                 C_step, Ks_step,
                 S_w1=0.514, S_p=1.117,
                 alpha_P=0.724,
                 D_beta1=1.170, D_beta2=0.389,
                 Abar_scale=1.016):
    """
    RC-only D formula. Same structure as compute_D() but:
    - No Rbar, Sbar, Vbar (LN-only components)
    - Uses Pbar as-is (LN_sum contribution is small and gets absorbed by RC tuning)
    """
    Abar = Abar * Abar_scale
    C_arr = step_interp(all_corners, base_corners, C_step)
    Ks_arr = step_interp(all_corners, base_corners, Ks_step)

    # Stream branch: ONLY Pbar (no Rbar/Sbar/Vbar)
    stream_branch = alpha_P * Pbar

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

    return D_all, S_all, T_all, C_arr, Ks_arr


def compute_rc_sr(cache, params):
    """
    Compute RC-only Star Rating from cache + RC params.

    Uses precomputed Jbar/Pbar and runtime Xbar with RC cross params.
    Sets Rbar=Sbar=Vbar=0.
    Routes through sigmoid aggregation with RC-specific params.
    """
    all_corners = cache["all_corners"]
    base_corners = cache["base_corners"]
    A_corners = cache["A_corners"]

    # Components from cache (same as combine())
    Jbar_base = cache["Jbar_base"]
    Pbar_base = cache["Pbar_base"]
    C_step = cache["C_step"]
    Ks_step = cache["Ks_step"]
    Abar_A = cache["Abar_A"]

    # Xbar via Phase 4 RC/LN blending — lock to RC params (ln_ratio=0)
    cross_data = cache.get("cross_data")
    if cross_data is not None:
        ln_ratio = cache.get("ln_ratio", 0.0)
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
        from .components import cross_enhanced as _cross_enh
        Xbar_base = _cross_enh.compute_Xbar_enhanced_fast(
            cross_data, base_corners, cache["x"],
            dist_exponent=dist_exponent,
            same_hand_penalty=same_hand_penalty,
            thumb_bridge=_p(params, "cross_thumb_bridge_factor", 0.5),
        )
    else:
        Xbar_base = cache.get("Xbar_base_clone")

    # Interpolate to all_corners
    Jbar = interp_values(all_corners, base_corners, Jbar_base)
    Xbar = interp_values(all_corners, base_corners, Xbar_base)
    Pbar = interp_values(all_corners, base_corners, Pbar_base)
    Abar = interp_values(all_corners, A_corners, Abar_A)

    # Compute RC D
    D_rc, S_rc, T_rc, C_arr, Ks_arr = compute_D_rc(
        all_corners, base_corners, Abar, Jbar, Xbar, Pbar,
        C_step, Ks_step,
        S_w1=_p(params, "S_w1_rc", 0.514),
        S_p=_p(params, "S_p_rc", 1.117),
        alpha_P=_p(params, "alpha_P_rc", 0.724),
        D_beta1=_p(params, "D_beta1_rc", 1.170),
        D_beta2=_p(params, "D_beta2_rc", 0.389),
        Abar_scale=_p(params, "Abar_scale_rc", 1.016),
    )

    # D calibration
    calib_a = _p(params, "calib_a_rc", 0.89)
    calib_b = _p(params, "calib_b_rc", 0.04)
    D_calib = calib_a * D_rc + calib_b if abs(calib_a - 1.0) > 1e-12 or abs(calib_b) > 1e-12 else D_rc

    # Sigmoid aggregation
    total_notes = compute_total_notes(cache["note_seq"], cache["LN_seq"])
    SR, agg_details = compute_SR_sigmoid(
        all_corners, C_arr, D_calib, total_notes, cache["LN_seq"],
        n_segments=int(_p(params, "agg_n_segments", 30)),
        sigmoid_k=_p(params, "agg_sigmoid_k_rc", 1.5),
        sigmoid_C=_p(params, "agg_sigmoid_C_rc", 4.0),
        sigmoid_gamma=_p(params, "agg_sigmoid_gamma_rc", 0.20),
        sigmoid_high_power=_p(params, "agg_sigmoid_high_power", 0.0),
        bisect_tol=_p(params, "agg_bisect_tol", 0.0001),
        bisect_delta=_p(params, "agg_bisect_delta", 5.0),
        note_norm_N0=_p(params, "note_norm_N0_rc", 10.0),
        rescale_threshold=_p(params, "rescale_threshold_rc", 9.54),
        rescale_divisor=_p(params, "rescale_divisor_rc", 2.00),
        global_scale=_p(params, "global_scale_rc", 1.055),
    )

    details = {
        "D_all": D_rc, "S_all": S_rc, "T_all": T_rc,
        "Jbar": Jbar, "Xbar": Xbar, "Pbar": Pbar,
        "Abar": Abar, "Rbar": np.zeros_like(Pbar),
        "Sbar": None, "Vbar": None,
        "C_arr": C_arr, "Ks_arr": Ks_arr,
        "total_notes": total_notes,
        **agg_details,
    }
    return SR, details


# ============================================================
# LN 模型
# ============================================================

def compute_D_ln(all_corners, base_corners, Rbar, Sbar, Vbar,
                 alpha_R=0.5, alpha_S=0.1, alpha_V=0.5):
    """
    LN-only D formula: additive combination of LN-specific components.

    D_ln = alpha_R * Rbar + alpha_S * Sbar + alpha_V * Vbar

    Rbar: release coordination difficulty
    Sbar: shield/blocking difficulty
    Vbar: inverse U-curve (can be negative — guide dip)
    """
    n = len(all_corners)
    D = np.zeros(n)

    if Rbar is not None and alpha_R > 0:
        D += alpha_R * Rbar
    if Sbar is not None and alpha_S > 0:
        D += alpha_S * Sbar
    if Vbar is not None and alpha_V > 0:
        D += alpha_V * Vbar

    return D


def compute_ln_sr(cache, params):
    """
    Compute LN-only Star Rating from cache + LN params.

    Uses Rbar/Sbar/Vbar computed at runtime from precomputed structured data.
    Routes through sigmoid aggregation with LN-specific params.
    """
    all_corners = cache["all_corners"]
    base_corners = cache["base_corners"]

    from .components import release_enhanced as _release_enh
    from .components import shield as _shield
    from .components import inverse as _inverse

    # Rbar from release data
    release_data = cache.get("release_data")
    if release_data is not None:
        Rbar_base = _release_enh.compute_Rbar_enhanced_fast(
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
        Rbar_base = cache.get("Rbar_base_clone")

    # Sbar from shield data
    shield_data = cache.get("shield_data")
    if shield_data is not None:
        Sbar_base = _shield.compute_Sbar_fast(
            shield_data, base_corners,
            shield_tau_ms=_p(params, "shield_tau_ms", 100),
            shield_anchor_mod=_p(params, "shield_anchor_mod", 1.0),
            shield_coord_factor=_p(params, "shield_coord_factor", 1.0),
            smooth_window=_p(params, "shield_smooth_window", 500),
            smooth_scale=_p(params, "shield_scale", 0.001),
        )
    else:
        Sbar_base = None

    # Vbar from inverse data
    inverse_data = cache.get("inverse_data")
    if inverse_data is not None:
        Vbar_base = _inverse.compute_Vbar_fast(
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
        Vbar_base = None

    # Interpolate to all_corners
    Rbar = interp_values(all_corners, base_corners, Rbar_base)
    Sbar = interp_values(all_corners, base_corners, Sbar_base) if Sbar_base is not None else None
    Vbar = interp_values(all_corners, base_corners, Vbar_base) if Vbar_base is not None else None

    # C array (needed for sigmoid aggregation weights) — compute from note_seq
    from .components import cross, jack
    from .combine import compute_C_and_Ks
    from .components.stream import compute_Pbar

    # Reuse C_step from cache
    C_step = cache["C_step"]
    C_arr = step_interp(all_corners, base_corners, C_step)

    # D_ln
    D_ln_raw = compute_D_ln(
        all_corners, base_corners, Rbar, Sbar, Vbar,
        alpha_R=_p(params, "alpha_R_ln", 0.5),
        alpha_S=_p(params, "alpha_S_ln", 0.1),
        alpha_V=_p(params, "alpha_V_ln", 0.5),
    )

    # D calibration
    calib_a = _p(params, "calib_a_ln", 3.0)
    calib_b = _p(params, "calib_b_ln", 0.0)
    D_calib = calib_a * D_ln_raw + calib_b if abs(calib_a - 1.0) > 1e-12 or abs(calib_b) > 1e-12 else D_ln_raw

    # LN total notes: use LN count for normalization
    total_notes_ln = len(cache["LN_seq"])
    if total_notes_ln < 1:
        return 0.0, {"D_all": D_ln_raw, "total_notes": 0, "error": "no LNs"}

    SR, agg_details = compute_SR_sigmoid(
        all_corners, C_arr, D_calib, total_notes_ln, cache["LN_seq"],
        n_segments=int(_p(params, "agg_n_segments", 30)),
        sigmoid_k=_p(params, "agg_sigmoid_k_ln", 1.5),
        sigmoid_C=_p(params, "agg_sigmoid_C_ln", 4.0),
        sigmoid_gamma=_p(params, "agg_sigmoid_gamma_ln", 0.20),
        sigmoid_high_power=_p(params, "agg_sigmoid_high_power", 0.0),
        bisect_tol=_p(params, "agg_bisect_tol", 0.0001),
        bisect_delta=_p(params, "agg_bisect_delta", 5.0),
        note_norm_N0=_p(params, "note_norm_N0_ln", 10.0),
        rescale_threshold=_p(params, "rescale_threshold_ln", 9.54),
        rescale_divisor=_p(params, "rescale_divisor_ln", 2.00),
        global_scale=_p(params, "global_scale_ln", 1.00),
    )

    details = {
        "D_all": D_ln_raw, "Rbar": Rbar, "Sbar": Sbar, "Vbar": Vbar,
        "C_arr": C_arr,
        "total_notes": total_notes_ln,
        **agg_details,
    }
    return SR, details
