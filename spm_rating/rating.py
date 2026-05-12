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

import pickle
import numpy as np
from . import parser
from . import preprocessor
from .components import anchor as _anchor
from .components import jack as _jack
from .components import cross as _cross
from .components import cross_enhanced as _cross_enh
from .components import stream as _stream
from .components import release as _release
from .components import release_enhanced as _release_enh
from .components import shield as _shield
from .components import inverse as _inverse
from .components import stamina as _stamina
from .combine import compute_C_and_Ks, compute_D
from .aggregate import compute_SR, compute_total_notes
from .utils import interp_values
from .config import get_default_params, CROSS as _CROSS_CFG, RELEASE as _RELEASE_CFG, SHIELD as _SHIELD_CFG


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
    parsed = parser.parse_file(file_path)
    data = preprocessor.preprocess(parsed, mod=mod)

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
    anchor_arr = _anchor.compute_anchor(K, key_usage_400, base_corners)
    delta_ks, Jbar_base, Jbar_ks = _jack.compute_Jbar(K, x, note_seq_by_column, base_corners,
                                                        aggregation_power=_comp_params.get("jack_aggregation_power", 5),
                                                        multi_jack_boost=_comp_params.get("multi_jack_boost", 0.0))
    Pbar_base = _stream.compute_Pbar(K, x, note_seq, LN_rep, anchor_arr, base_corners,
                                     stream_booster_scale=_comp_params.get("stream_booster_scale", 1.7e-7))
    Abar_A, dks = _anchor.compute_Abar(K, delta_ks, active_columns, A_corners, base_corners)
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
    cache["Xbar_base_clone"] = _cross.compute_Xbar(K, x, note_seq_by_column, active_columns, base_corners)
    cache["Rbar_base_clone"], _ = _release.compute_Rbar(K, x, note_seq_by_column, tail_seq, base_corners)

    if use_enhanced:
        # Cross: precompute structured data for runtime RC/LN parameter blending (Phase 4)
        cache["cross_data"] = _cross_enh.precompute_cross_enhanced_data(
            K, x, note_seq_by_column, active_columns, base_corners)
        # LN ratio for RC/LN parameter interpolation (0 = pure RC, higher = more LN)
        n_taps = len(note_seq) - len(LN_seq)
        n_total_objects = len(note_seq)
        cache["ln_ratio"] = len(LN_seq) / max(n_total_objects, 1)

        # Release: precompute structured data (like Shield/Inverse)
        # so parameters can be tuned at runtime without recaching
        cache["release_data"] = _release_enh.precompute_release_data(
            K, x, note_seq_by_column, tail_seq, note_seq
        )
        # Also bake a default Rbar for backward compat (fast from structured data)
        cache["Rbar_base_enhanced"] = _release_enh.compute_Rbar_enhanced_fast(
            cache["release_data"], base_corners,
            release_tail_coeff=_RELEASE_CFG["release_tail_coeff"][0],
            release_tail_to_tap_factor=_RELEASE_CFG["release_tail_to_tap"][0],
            release_same_col_bonus=_RELEASE_CFG["release_same_col_bonus"][0],
            release_coord_exponent=_RELEASE_CFG["release_coord_exponent"][0],
            short_ln_threshold=_RELEASE_CFG.get("short_ln_threshold", (200,))[0],
            short_ln_reduction=_RELEASE_CFG.get("short_ln_reduction", (0.5,))[0],
            lock_interaction_coeff=_RELEASE_CFG.get("lock_interaction_coeff", (0.3,))[0],
            release_seq_coeff=_RELEASE_CFG.get("release_seq_coeff", (0.03,))[0],
            smooth_window=_RELEASE_CFG["release_smooth_window"][0],
            smooth_scale=_RELEASE_CFG["release_scale"][0],
        )
        # Precompute structured data for fast Shield and Inverse at combine-time
        cache["shield_data"] = _shield.precompute_shield_data(
            K, note_seq_by_column, LN_seq)
        cache["inverse_data"] = _inverse.precompute_inverse_data(
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
        Jbar_base = _jack.aggregate_Jbar(
            cache["K"], cache["Jbar_ks"], cache["delta_ks"], cache["base_corners"],
            aggregation_power=cur_jack_agg,
            multi_jack_boost=cur_multi_jack,
        )

    # 按需重算 Pbar（当 stream_booster_scale 与预计算时不同时）
    cur_boost = _p(params, "stream_booster_scale", 1.7e-7)
    if abs(cur_boost - comp_params_cache.get("stream_booster_scale", 1.7e-7)) > 1e-12 and False:  # DISABLED for speed; use recache to change booster
        Pbar_base = _stream.compute_Pbar(
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
            Xbar_base = _cross_enh.compute_Xbar_enhanced_fast(
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
                Sbar_base = _shield.compute_Sbar_fast(
                    shield_data, base_corners,
                    shield_tau_ms=_p(params, "shield_tau_ms", 100),
                    shield_anchor_mod=_p(params, "shield_anchor_mod", 1.0),
                    shield_coord_factor=_p(params, "shield_coord_factor", 1.0),
                    smooth_window=_p(params, "shield_smooth_window", 500),
                    smooth_scale=_p(params, "shield_scale", 0.001),
                )
            else:
                Sbar_base = _shield.compute_Sbar(
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
                Vbar_base = _inverse.compute_Vbar(
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
            Ebar_base = _stamina.compute_Ebar(
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
        from .aggregate_sigmoid import compute_SR_sigmoid
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
