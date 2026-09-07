"""SPM Rating 参数注册表。

每个可调参数都收在一个扁平字典里，键按通道分组组织。系数允许为负，
通道系数为 0 即表示该通道不存在。

命名：<通道>_<角色>
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List

HERE = os.path.dirname(os.path.abspath(__file__))
PARAM_DIR = os.path.join(os.path.dirname(HERE), "params")

DEFAULTS: Dict[str, float] = {
    # ---------------------------------------------------------------- OD
    # x_f = x ** od_mult_<f>  (x = hit leniency).  1.0 表示沿用基准行为。
    # od_mult_jc/od_mult_ja 为 -1 表示沿用 od_mult_jm 的值。
    "od_mult_jm": 0.85,
    "od_mult_jc": -1.0,
    "od_mult_ja": -1.0,
    "od_mult_p": 1.15,
    "od_mult_x": 1.15,
    "od_mult_r": 1.0,

    # ------------------------------------------------------- smoothing (ms)
    # w_jc/w_ja 为 0 表示沿用 w_jm 的值（与上面 OD 系数的继承规则一致）。
    "w_jm": 1000.0,
    "w_jc": 0.0,
    "w_ja": 0.0,
    "w_p": 1000.0,
    "w_x": 1000.0,
    "w_r": 1000.0,
    "w_cb": 1000.0,
    "w_a": 150.0,

    # ------------------------------------------------------------ jack core
    # kernel:  d^-1 * (d + c1*x^0.25)^-1 * nerfer
    "j_c1": 0.11,
    "j_nerf_a": 7e-5,          # 0 disables the nerfer
    "j_nerf_off": 0.15,
    "j_nerf_c": 0.08,
    "j_agg_pow": 4.27,          # cross-column power mean for every jack channel
    "j_triple_tau": 120.0,     # ms; same-column triple bonus window
    "j_triple_pow": 1.0,
    "j_triple_gain": 0.0,      # 0 = off  (h_jack3 mechanism, time-series form)
    # per-frame variant: 4K wants the triple gain much lower than 7K
    # (研究结论)。小于 0 表示继承 j_triple_gain。
    "j_triple_gain_4k": -1.0,
    # fast same-column (fj) channel added into Jm.  Per-frame coefficients
    # (0 = off for that frame; c_fj_4k does NOT inherit).
    "c_fj": 0.0,
    "c_fj_4k": 0.0,
    # 同手和弦密度折扣曲线，进入 stream 分支。
    "d_sh": 0.0,
    "shd_w": 1000.0,           # ms smoothing window

    # minijack: dilution by what else is going on around the jack
    "mj_dilute": 0.0,          # 0 = plain Sunny Jbar
    "mj_dilute_exp": 1.0,

    # chordjack: chord load around the jack
    "cj_size_exp": 1.0,        # g(load) = (load/cj_norm)**size_exp
    "cj_norm": 2.0,            # keeps the chordjack channel on Jbar's scale

    # anchor: run length x coordination
    "ja_gap_thr": 250.0,       # ms; gaps above this break a run
    "ja_len_exp": 1.0,
    "ja_coord_exp": 0.0,       # weight of other-column activity
    "ja_len_floor": 4.0,       # runs shorter than this contribute ~0
    "ja_norm": 6.0,            # half-saturation run length for the anchor term

    # --------------------------------------------------------------- Pbar
    "p_lam3": 24.0,            # synchronous-note penalty
    "p_scale": 0.08,
    "p_lam2": 0.006,           # LN body weight (per ms of body)
    "p_v_norm": 0.0,           # 0 = delta form (x ms), 1 = normalised by gap
    "p_boost": 1.75e-7,
    "p_boost_lo": 160.0,
    "p_boost_hi": 360.0,
    "p_bv_max": 1.0,           # 1 = max(b, v) like delta; 0 = b*v
    "p_sat_a": 10.0,           # soft clamp: min(inc*anchor, max(inc, inc*p_sat_b - p_sat_a))
    "p_sat_b": 2.0,
    # per chord size multiplier (index 0 == size 1, fixed at 1.0)
    "p_chw2": 1.0, "p_chw3": 1.0, "p_chw4": 1.0,
    "p_chw5": 1.0, "p_chw6": 1.0, "p_chw7": 1.0,
    # 和弦柔性耦合，毫秒：三角窗正负 chord_tau_c
    # window turns chord size into a continuous effective size (chw table,
    # SHd, chord2), a row dt ms after another is discounted to dt/tau_c of
    # a new attack in Pbar, and jack nrows are counted in attack groups --
    # a 1 ms chord split stops being a full re-pricing.
    # 0 = off (legacy exact-timestamp chords).
    "chord_tau_c": 0.0,
    # short-window burst channel (3/4-note bursts, the `burst` / `lb` feature)
    "p_burst3": 0.0,
    "p_burst4": 0.0,

    # ------------------------------------------------------- anchor / Abar
    "an_a0": 0.18,
    "an_a1": 0.22,
    "an_cubic": 5.0,
    "an_on_p": 1.0,            # strength of the column-imbalance multiplier
    "abar_scale": 1.0163,
    # 0 = pair frame-index-adjacent columns (v0.5.0delta shipped behaviour);
    # 1 = pair adjacent ACTIVE columns (for 4K embeds this also spans the
    #     empty thumb column 3).  Kept as an ablation switch, not assumed.
    "abar_active": 0.0,
    "abar_thr_lo": 0.02,
    "abar_thr_hi": 0.07,
    "abar_c0": 0.75,
    "abar_c1": 0.65,
    "abar_k": 5.0,
    "abar_mx": 0.5,
    "abar_mx_thr": 0.11,
    "abar_mx_w": 0.6,
    # 1 = weld the mid branch to c0 + k*(dd - thr_lo) + ... so the column-
    # 列对乘子在 abar_thr_lo 处连续。
    # 0 = legacy (c1 + k*dd; jumps by c1 + k*thr_lo - c0 at the seam).
    "abar_weld": 0.0,

    # --------------------------------------------------------------- Xbar
    "x_amp": 0.16,
    "x_cw0": 0.225,            # boundary groups {0,7} {1,6} {2,5} {3,4}
    "x_cw1": 0.35,
    "x_cw2": 0.25,
    "x_cw3": 0.225,
    # inward/outward (内外切) split of the pairwise term: pair multiplier
    # 1 + x_din*d+ + x_dout*d-, d = |e_from-3| - |e_to-3| (+1 full inward,
    # -1 full outward, chord-involved pairs half).  0/0 = symmetric legacy.
    "x_din": 0.0,
    "x_dout": 0.0,
    "x_fc_a": 0.4,
    "x_fc_off": 80.0,
    "x_fc_floor": 0.06,
    "x_fc_w": 1.0,
    # geometric cross matrix (jump component)
    "x_jump_w": 0.0,           # 0 = off
    "x_jh_p": 0.45,           # same-hand penalty
    "x_jh_e": 1.5,           # same-hand distance exponent
    "x_jt": 0.5536,            # thumb bridge
    "x_jd_p": 0.0,             # generic distance penalty
    "x_jd_e": 1.0,
    "x_dir_in": 0.0,           # inward-flow multiplier (toward col 3)
    "x_dir_out": 0.0,

    # --------------------------------------------------------------- Rbar
    "r_tail": 0.14488,
    "r_seq": 0.108498,
    "r_I_steep": 5.0,
    "r_I_off": 0.75,
    "r_I_w": 0.8,
    "r_dt_max": 5000.0,
    # physical floor (seconds) for the dtr^-0.5 release-gap pricing
    # 物理下限：以感知尺度下限替代原先纯数值的钳位。
    # perception-scale minimum.  0.0001 = legacy clamp.
    "r_dtr_min": 0.0001,
    # ms; blend zone for near-simultaneous next events (tail vs head):
    # the coord exponent interpolates linearly instead of the 1e-9 hard
    # 判定混合带。0 表示关闭。
    "r_sim_tau": 0.0,
    # 1 = raised-cosine ramps on the r_dt_max / r_order_tau cutoffs and a
    # 零宽释放区间给予单行份额。
    # B2/B3/B5).  0 = legacy.
    "r_soft_edges": 0.0,
    "r_same_col": 0.26775,
    "r_coord_e": 0.71931,
    "r_tt": 4.2242,
    "r_short_thr": 331.62,
    "r_short_red": 0.10009,
    "r_lock": 0.11649,
    "r_simul": 0.05,
    "r_cw_same": 1.0, "r_cw_hand": 0.8, "r_cw_thumb": 0.4, "r_cw_cross": 0.2,
    # new: release-order triple + lock-straddle matrix
    "r_order_pen": 0.0,        # 102/120 harder than 012/210
    "r_order_tau": 400.0,      # ms; max span of the triple
    "r_straddle": 0.0,         # lock-straddle weight
    "r_ain0": 1.0, "r_ain1": 1.0, "r_ain2": 1.0, "r_ain3": 1.0,

    # --------------------------------------------------------------- Cbar
    "cb_shield": 0.0,
    "cb_shield_tau": 56.24,
    "cb_shield_lock": 0.8062,
    "cb_lockjack": 0.0,
    "cb_lock_pow": 1.0,
    "cb_lockanchor": 0.0,
    "cb_par": 0.0,             # true LN overlap (non-nested) density
    "cb_holdtap": 0.0,
    "cb_holdtap_pow": 1.0,

    # ------------------------------------------------- Cbar v2 (lncoord_v2)
    # Row-grid LN coordination.  `use_cbar_v2` is the master gate: while it
    # is 0 the channel returns exact zeros and Cbar == Cbar v1 bit-for-bit.
    # Each sub-term additionally has its own `use_cbv_<term>` switch and its
    # own `cbv_<term>` weight, so terms ablate independently.
    "use_cbar_v2": 0.0,
    # 1 = count churn release events on tail times (grid-free rate;
    # 残留开关。会改变 churn 尺度，
    # cbv_churn needs re-fitting when enabled.  0 = legacy.
    "churn_event_rate": 0.0,
    "cbv_w": 1000.0,           # smoothing window (ms) shared by all v2 terms
    # --- term gates + weights
    "use_cbv_shield": 0.0,     "cbv_shield": 0.0,
    "use_cbv_straddle": 0.0,   "cbv_straddle": 0.0,
    "use_cbv_lockstack": 0.0,  "cbv_lockstack": 0.0,
    "use_cbv_lockanchor": 0.0, "cbv_lockanchor": 0.0,
    "use_cbv_overlap": 0.0,    "cbv_overlap": 0.0,
    "use_cbv_holdtap": 0.0,    "cbv_holdtap": 0.0,
    # --- shield shape: tap/short-LN -> LN on the same column
    "cbv_sh_tau": 60.0,        # ms; decay of the predecessor gap
    "cbv_sh_dt_max": 250.0,    # ms; max gap that still counts as a shield
    "cbv_sh_ln_w": 1.0,        # weight of a zero-length LN predecessor vs tap
    "cbv_sh_ln_tau": 120.0,    # ms; long LN predecessors stop being a shield
    "cbv_sh_lock": 1.0,        # amplification by held other columns
    # --- straddle (衩叠) pair matrix + pinned-finger weights
    "cbv_str_sh_adj": 0.0,     # unused at d==2; kept for matrix completeness
    "cbv_str_sh_split": 1.0,   # (0,2) (4,6) -- the true straddle
    "cbv_str_1h_split": 0.45,  # (1,3) (3,5) -- thumb-bridged
    "cbv_str_2h_split": 0.20,  # (2,4)
    "cbv_str_cross": 0.10,     # d >= 3
    "cbv_str_h0": 0.70,        # pinned-column weight indexed by AIN_GROUP
    "cbv_str_h1": 1.00,        #   h1 = middle finger -- the worst pin
    "cbv_str_h2": 0.70,
    "cbv_str_h3": 0.40,        #   h3 = thumb
    "cbv_str_bal": 1.0,        # exponent on the alternation-balance factor
    "cbv_str_maxd": 2.0,       # max pair distance considered (2 = 衩叠)
    # --- lock matrix: acting on column a while column j is held
    "cbv_lock_pow": 1.0,     # lock-mass exponent for lockstack/lockanchor
    "cbv_ls_sh_adj": 0.60,
    "cbv_ls_sh_split": 1.00,
    "cbv_ls_1h_split": 0.80,
    "cbv_ls_2h_split": 0.50,
    "cbv_ls_cross": 0.30,
    # --- overlap (partial, non-nested LN pairs) pair matrix
    "cbv_ov_sh_adj": 1.00,
    "cbv_ov_sh_split": 1.00,
    "cbv_ov_1h_split": 0.70,
    "cbv_ov_2h_split": 0.50,
    "cbv_ov_cross": 0.30,
    "cbv_ov_dt_max": 1500.0,   # ms; max head-to-head distance for a pair
    "cbv_ov_cap": 2000.0,      # ms; clamp on the priced overlap length

    # ------------------------------------------------- 混合与长键
    # Four curve-level mechanisms targeting the 7K hb/ln gaps vs delta.
    # use_hb = master gate; each term has its own weight (0 = off).
    # Per-frame overrides (0 = inherit the 7K value), same pattern as
    # a_p_4k / a_r_4k / a_cb_4k: LN physics differs across frames.
    "use_hb": 0.0,
    # release churn (concave), added into the Cbar curve
    "cbv_churn": 0.0,
    "cbv_churn_4k": 0.0,
    "churn_sat": 15.0,         # releases/s soft saturation
    "churn_w": 1000.0,         # ms window
    # LN hold-age (occupation duration), added into the Cbar curve
    "cbv_holdage": 0.0,
    "cbv_holdage_4k": 0.0,
    "holdage_th": 0.25,        # seconds before a hold starts costing
    "holdage_cap": 1.0,        # seconds cap per hold
    # chord2 density rate, additive into the stream branch
    "c_chord2": 0.0,
    "chord2_w": 1000.0,        # ms window
    # recovery-aware burst discount on the stream branch
    "d_rec": 0.0,
    "rec_half": 0.5,           # half-saturation of the burst ratio
    "recov_w": 1000.0,         # short window (ms)
    "recov_wl": 20000.0,       # baseline window (ms)

    # ------------------------------------------------------------- combine
    "aj": 3.0,                 # Abar exponent /Ks on the jack branches
    "ap": 0.6667,              # Abar exponent on the stream branch
    "at": 3.0,                 # Abar exponent /Ks on T
    "cap_a": 8.0,
    "cap_b": 0.85,
    "c_jm": 0.484,             # jack-branch weights inside S
    "c_jc": 0.0,
    "c_ja": 0.0,
    "c_s": 0.516,
    "s_p": 1.0,                # power-mean exponent of S branches
    "a_p": 0.72454,            # Pbar weight inside the stream branch
    "a_r": 32.2385,            # Rbar weight
    "a_c": 11.0205,            # C offset for Rbar
    "a_cb": 0.0,               # Cbar weight
    "a_cbc": 11.0205,
    # Per-frame overrides for 4K-embedded charts (gate = s.is_4k).
    # 7K and 4K diverge on stream/LN pricing: 4K high-end LN and speed are
    # under-priced while 7K LN over-prices at the same coefficient.  Each
    # *_4k override: 0 = inherit the shared value (legacy behaviour).
    "a_p_4k": 0.0,
    "a_r_4k": 0.0,
    "a_cb_4k": 0.0,
    "t_jack_mix": 0.0,         # jack contribution to the twist numerator
    "t_s_off": 1.0,
    "d_b1": 1.1879,
    "d_ds": 0.5,
    "d_dt": 1.5,
    "d_b2": 0.3845,

    # ----------------------------------------------------------- aggregate
    "agg_mode": 3.0,           # 0 percentile | 1 soft-max | 2 sigmoid | 3 power mean
    "agg_k": 4.0,              # mode1: lambda | mode2: sigmoid k | mode3: power
    "agg_C": 3.96885,
    "agg_gamma": 0.19562,
    "agg_nseg": 30.0,
    "agg_wc": 1.0,             # weight exponent on the local note count C
    "agg_gap_w": 1.0,          # weight exponent on the row gap
    # 聚合前保留峰值的包络：
    # D <- (1-env_w)*D + env_w * rolling_max(D, env_ms).  0 = off.
    "env_w": 0.0,
    "env_ms": 800.0,
    # percentile mode (agg_mode < 0.5)
    "p93_c": 0.88, "w93": 0.25,
    "p83_c": 0.94, "w83": 0.20,
    "wmean": 0.55, "mean_pow": 5.0,

    # ---------------------------------------------------------- postprocess
    "pp_n0": 1.0286,
    "pp_thr": 9.1064,
    "pp_div": 1.9709,
    "pp_scale": 1.0944,
    "calib_a": 1.0,
    "calib_b": 0.0,
    # 按住感知的聚合权重
    "hw_a": 0.0,
    "hw_half": 30.0,
    # ln_eff_tail_ms: LN effective tail reduction (ms).  STRUCT-LEVEL: it
    # changes s.held / s.heldmask / s.ln_body_ms, so it lives in STRUCT_KEYS
    # and is tuned by grid + inner NM, never inside a plain NM block.
    "ln_eff_tail_ms": 0.0,
    # ln_note_level: 1 = build held/heldmask from per-note tail arrays
    # 残留注记：行级聚合会把与长键同行的点键误标，
    # sharing a row with an LN head as LN heads with the row's MAX tail).
    # STRUCT-LEVEL like ln_eff_tail_ms.  0 = legacy row-level.
    "ln_note_level": 0.0,
    # read (眼难度) component -- legacy reg port (kept for ablation)
    "use_read": 0.0,         # 0 = off, 1 = on
    "d_read": 0.0,           # read coefficient (7K)
    "d_read_4k": 0.0,        # read coefficient (4K, per-frame)
    "read_tol": 0.18,        # regularity tolerance
    "read_n_win": 16.0,      # local median window (events)
    "read_clip_lo": 0.5,     # clip range for read modifier
    "read_clip_hi": 1.5,
    # eye (眼难度) -- E4 beat alignment, density-gated stream interaction
    "use_eye": 0.0,          # 0 = off, 1 = on
    "d_eye": 0.0,            # 7K coefficient (sign free)
    "d_eye_4k": 0.0,         # 4K override; 0 = same as d_eye
    "eye_w_ref": 4000.0,     # local rate reference window (ms)
    "eye_w_s": 500.0,        # smoothing window (ms)
    "eye_tau": 0.2,          # alignment decay scale
    # output-domain affine (per dataset; only 4K needs a non-trivial one)
    "out_a": 1.0,
    "out_b": 0.0,
}

# 参数分组。各通道的键按前缀组织，供评分代码与验证脚本按组取用。
BLOCKS: Dict[str, List[str]] = {
    # od_mult_ja、w_ja、c_ja 不参与调整：长连通道被证明冗余，
    # 它的形状参数保留默认值（继承语义），以保持旧参数文件的兼容。
    "od": [k for k in DEFAULTS if k.startswith("od_mult_")
           and k != "od_mult_ja"],
    "smooth": [k for k in DEFAULTS if k.startswith("w_") and k != "w_ja"],
    "jack": [k for k in DEFAULTS if k.startswith(("j_", "mj_", "cj_", "ja_"))],
    "jackshape": ["cj_size_exp", "cj_norm", "ja_len_exp", "ja_len_floor",
                  "ja_norm", "ja_coord_exp", "mj_dilute_exp"],
    "jackgain": ["mj_dilute", "j_triple_gain", "j_triple_gain_4k", "j_triple_tau",
                 "c_fj", "c_fj_4k", "shd_w"],
    "pbar": [k for k in DEFAULTS if k.startswith("p_")],
    "abar": [k for k in DEFAULTS if k.startswith(("an_", "abar_"))
             and k != "abar_weld"],
    "xbar": [k for k in DEFAULTS if k.startswith("x_")],
    "rbar": [k for k in DEFAULTS if k.startswith("r_")
             and k != "r_soft_edges"],
    "cbar": [k for k in DEFAULTS if k.startswith("cb_")],
    # 长键协调通道 v2。`sat_cbar_v2` 是权重与形状参数组；`use_*` 开关
    # 是 0/1 手工设置，用于对照实验，不参与数值优化。
    "sat_cbar_v2": [k for k in DEFAULTS if k.startswith("cbv_")],
    "gates": ["use_cbar_v2", "r_soft_edges", "abar_weld", "ln_note_level",
              "churn_event_rate"] +
             [k for k in DEFAULTS if k.startswith("use_cbv_")],
    # 和弦耦合参数（密度通道的和弦尺寸、同手折扣项、混合补充项的双押
    # 密度率共用），单独成组，不放在上述通道组里。
    "chordsoft": ["chord_tau_c"],
    "eye": ["d_eye", "d_eye_4k"],
    # cbv_churn / cbv_holdage carry the cbv_ prefix, so they live in the
    # sat_cbar_v2 block automatically; the hb block holds the rest.
    "hb": ["c_chord2", "d_rec",
           "churn_sat", "churn_w", "holdage_th", "holdage_cap",
           "chord2_w", "rec_half", "recov_w", "recov_wl"],
    "combine": ["aj", "ap", "at", "cap_a", "cap_b", "c_jm", "c_jc",
                "c_s", "s_p", "a_p", "a_r", "a_c", "a_cb", "a_cbc",
                "a_p_4k", "a_r_4k", "a_cb_4k", "d_sh",
                "t_jack_mix", "t_s_off", "d_b1", "d_ds", "d_dt", "d_b2"],
    "agg": ["agg_k", "agg_C", "agg_gamma", "agg_wc", "agg_gap_w",
            "p93_c", "w93", "p83_c", "w83", "wmean", "mean_pow",
            "hw_a", "hw_half", "env_w", "env_ms"],
    "post": ["pp_n0", "pp_thr", "pp_div", "pp_scale", "calib_a", "calib_b",
             "out_a", "out_b"],
    "affine": ["out_a", "out_b", "calib_a", "calib_b"],
}

# Keys whose change forces a structural (per-chart) recompute.  Everything
# else can reuse the cached structural arrays.  (Kept explicit and audited:
# a struct key missing from this list silently turns every NM step into a
# 全量重算。)
STRUCT_KEYS: List[str] = [
    "ja_gap_thr", "j_triple_tau", "r_dt_max", "r_short_thr",
    "cb_shield_tau", "r_order_tau", "agg_nseg",
    "cbv_sh_dt_max", "cbv_ov_dt_max", "cbv_str_maxd", "cbv_w",
    "ln_eff_tail_ms", "ln_note_level",
]


def get(params: Dict[str, float], key: str) -> float:
    v = params.get(key)
    return DEFAULTS[key] if v is None else v


def make(overrides: Dict[str, float] | None = None,
         strict: bool = True) -> Dict[str, float]:
    p = dict(DEFAULTS)
    if overrides:
        unknown = set(overrides) - set(DEFAULTS)
        if unknown:
            if strict:
                raise KeyError(f"unknown parameter(s): {sorted(unknown)}")
            overrides = {k: v for k, v in overrides.items() if k in DEFAULTS}
        p.update(overrides)
    return p


def load(path: str, strict: bool = False) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, dict) and "params" in raw:
        raw = raw["params"]
    return make(raw, strict=strict)


def save(params: Dict[str, float], path: str, note: str = "") -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {"note": note,
               "params": {k: float(v) for k, v in params.items()}}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)


def block_keys(names: Iterable[str]) -> List[str]:
    out: List[str] = []
    for n in names:
        out.extend(BLOCKS.get(n, [n]))
    return sorted(set(out))


def nm_block_keys(names: Iterable[str]) -> List[str]:
    """分组键减去结构级键。

    结构级键改变行结构的输出；在分片计算里，这类键的每一步都会触发
    全进程的结构重建。结构级键用网格加内部搜索单独处理，不放在普通
    参数组里。
    """
    sk = set(STRUCT_KEYS)
    return [k for k in block_keys(names) if k not in sk]


def struct_fingerprint(params: Dict[str, float]) -> tuple:
    return tuple(round(float(params.get(k, DEFAULTS[k])), 9) for k in STRUCT_KEYS)


# 不改变难度曲线输出的参数：合并层系数、聚合、标定，以及合并层的
# 视觉/读谱系数。其他参数（连打/密度/不均匀度/跨列/释放/协调/击打
# 宽容度/平滑窗 + 结构级键）都会影响曲线。curve_fingerprint 让调用方在
# 只有非曲线参数变化时复用已缓存的曲线，这是最大的加速来源。
NON_CURVE_KEYS: set = set(
    BLOCKS["combine"] + BLOCKS["agg"] + BLOCKS["post"] + BLOCKS["eye"]
) | {"use_read", "d_read", "d_read_4k", "read_clip_lo",
     "read_clip_hi", "read_tol", "read_n_win",
     # c_chord2 / d_rec are applied in combine() (they scale/add to the
     # stream branch), not in compute_curves -- they must NOT trigger a
     # curve recompute.  (use_hb and the *_w / *_sat / *_th / *_cap shape
     # params DO change the curves and stay curve-affecting.)
     "c_chord2", "d_rec"}
# NB: use_eye is deliberately NOT here -- it gates EyeR computation inside
# compute_curves, so it is curve-affecting.


def curve_fingerprint(params: Dict[str, float]) -> tuple:
    return tuple(round(float(params.get(k, DEFAULTS[k])), 9)
                 for k in sorted(DEFAULTS) if k not in NON_CURVE_KEYS)


if __name__ == "__main__":
    print(f"{len(DEFAULTS)} parameters")
    seen = set()
    for b, keys in BLOCKS.items():
        dup = seen & set(keys)
        if dup:
            print(f"!! block {b} overlaps: {sorted(dup)}")
        seen |= set(keys)
    missing = set(DEFAULTS) - seen
    print("unblocked:", sorted(missing))
    for k in STRUCT_KEYS:
        if k not in DEFAULTS:
            print("!! struct key not in DEFAULTS:", k)
