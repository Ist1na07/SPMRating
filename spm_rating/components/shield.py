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
