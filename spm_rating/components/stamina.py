"""
SPM Rating — Stamina / Endurance difficulty (Ebar).

NEW component: Models accumulated fatigue with recovery periods.

Leaky-integrator fatigue model:
- Fatigue accumulates during dense sections
- Decays during rest periods
- Recovery is proportional to rest duration

Also provides rhythmic complexity bonus for irregular patterns.
"""

import numpy as np
import bisect
from ..utils import smooth_on_corners


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
