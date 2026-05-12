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
