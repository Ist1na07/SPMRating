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

import numpy as np
from ..utils import smooth_on_corners

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
