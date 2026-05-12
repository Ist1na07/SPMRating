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

import math
import numpy as np
from ..utils import smooth_on_corners

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
