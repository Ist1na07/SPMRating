"""
SPM Rating — Preprocessing pipeline.

Converts parsed .osu data into the canonical format used by all
difficulty components: time grids, key usage, LN representation, etc.
"""

import bisect
import math
from collections import defaultdict
import numpy as np
import heapq


def preprocess(parsed_data, mod="", params=None):
    """
    Preprocess parsed beatmap data into the canonical format.

    Args:
        parsed_data: output of parser.parse_file() — [K, cols, starts, ends, types, od]
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
