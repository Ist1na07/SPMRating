"""Vectorised time-grid primitives.

The whole model is evaluated on the **row grid**: the sorted sequence of
distinct note timestamps.  Row r carries
    t[r]     timestamp (ms, strictly increasing)
    mask[r]  7-bit column mask of the chord at that row
    size[r]  popcount(mask[r])
A chord is therefore a single event; chord pricing is explicit instead of
being smuggled in through a Dirac-delta spike.

All per-row quantities are *step functions*: value v[r] holds on the
half-open interval [t[r], t[r+1]).
"""
from __future__ import annotations

import numpy as np

K = 7                      # fixed 7-column frame
N_BOUND = K + 1            # 8 column boundaries for Xbar

POPCOUNT = np.array([bin(i).count("1") for i in range(128)], dtype=np.int64)


class Grid:
    """Row grid derived from a parsed chart."""

    __slots__ = ("t", "mask", "size", "n", "dt", "col_rows", "col_dt",
                 "bound_rows", "bound_dt", "note_times", "is_ln", "ln_end",
                 "key_count", "od", "x", "n_notes", "duration", "rate",
                 "col_notes", "row_of_note")

    def __init__(self, notes, od_leniency: float):
        time = notes.time
        # ---- rows: distinct timestamps
        first = np.empty(len(time), dtype=bool)
        first[0] = True
        np.not_equal(time[1:], time[:-1], out=first[1:])
        row_of_note = np.cumsum(first) - 1
        self.row_of_note = row_of_note
        R = int(row_of_note[-1]) + 1
        t = np.zeros(R, dtype=np.float64)
        t[row_of_note] = time
        mask = np.zeros(R, dtype=np.int32)
        np.bitwise_or.at(mask, row_of_note, (1 << notes.col).astype(np.int32))
        # LN: row is "ln" if any note there is an LN
        is_ln_row = np.zeros(R, dtype=bool)
        np.logical_or.at(is_ln_row, row_of_note, notes.is_ln)
        ln_end = np.full(R, -1.0, dtype=np.float64)
        np.maximum.at(ln_end, row_of_note,
                      np.where(notes.is_ln, notes.ln_end, -1.0))

        self.t = t
        self.mask = mask
        self.size = POPCOUNT[mask]
        self.n = R
        self.is_ln = is_ln_row
        self.ln_end = ln_end
        self.note_times = time
        self.col = notes.col
        self.n_notes = notes.n_notes
        self.key_count = notes.key_count
        self.od = notes.od
        self.x = float(od_leniency)
        self.duration = float(t[-1] - t[0]) if R > 1 else 0.0
        self.rate = notes.rate

        # ---- row gaps (step function on [t[r], t[r+1]) )
        dt = np.diff(t) / 1000.0                      # seconds
        self.dt = dt                                  # (R-1,)

        # ---- per-column note rows + gaps
        self.col_rows = []
        self.col_dt = []
        self.col_notes = []
        for k in range(K):
            idx = np.flatnonzero((mask >> k) & 1)
            self.col_rows.append(idx)
            self.col_notes.append(idx)
            if idx.size >= 2:
                self.col_dt.append(np.diff(t[idx]) / 1000.0)
            else:
                self.col_dt.append(np.zeros(0, dtype=np.float64))

        # ---- per-boundary (two-column union) rows + gaps
        self.bound_rows = []
        self.bound_dt = []
        for b in range(N_BOUND):
            cols = []
            if b - 1 >= 0:
                cols.append(b - 1)
            if b < K:
                cols.append(b)
            if not cols:
                self.bound_rows.append(np.zeros(0, dtype=np.int64))
                self.bound_dt.append(np.zeros(0, dtype=np.float64))
                continue
            merged = np.unique(np.concatenate([self.col_rows[c] for c in cols]))
            self.bound_rows.append(merged)
            self.bound_dt.append(np.diff(t[merged]) / 1000.0
                                 if merged.size >= 2 else
                                 np.zeros(0, dtype=np.float64))

    # ------------------------------------------------------------------
    @property
    def n_ln(self) -> int:
        return int(self.is_ln.sum())

    def col_step(self, k: int, values: np.ndarray) -> np.ndarray:
        """Scatter per-column-gap `values` (len = len(col_rows[k])-1) onto
        the row grid: value[j] holds on rows [idx[j], idx[j+1])."""
        out = np.zeros(self.n, dtype=np.float64)
        idx = self.col_rows[k]
        if idx.size < 2:
            return out
        lengths = np.diff(idx)
        # a chord containing column k at row idx[j] -> interval length >= 0
        seg = np.repeat(values[:len(lengths)], np.maximum(lengths, 0))
        start = idx[0]
        end = start + len(seg)
        out[start:min(end, self.n)] = seg[:min(end, self.n) - start]
        return out

    def bound_step(self, b: int, values: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float64)
        idx = self.bound_rows[b]
        if idx.size < 2:
            return out
        lengths = np.diff(idx)
        seg = np.repeat(values[:len(lengths)], np.maximum(lengths, 0))
        start = idx[0]
        end = start + len(seg)
        out[start:min(end, self.n)] = seg[:min(end, self.n) - start]
        return out


# ----------------------------------------------------------------------
def step_to_row(t: np.ndarray, src_idx: np.ndarray, values: np.ndarray,
                n: int) -> np.ndarray:
    """Scatter values defined on consecutive src_idx intervals onto rows."""
    out = np.zeros(n, dtype=np.float64)
    if src_idx.size < 2:
        return out
    lengths = np.diff(src_idx)
    seg = np.repeat(values[:len(lengths)], np.maximum(lengths, 0))
    start = int(src_idx[0])
    end = min(start + len(seg), n)
    if end > start:
        out[start:end] = seg[:end - start]
    return out


def window_mean(t: np.ndarray, val: np.ndarray, W: float) -> np.ndarray:
    """Centred moving average of the step function `val` with window W (ms).

    val[r] is constant on [t[r], t[r+1]); t must be strictly increasing.
    Returns (1/W) * int_{s-W/2}^{s+W/2} val, sampled at s = t[r].
    """
    n = t.size
    if n < 2:
        return val.copy()
    dt = np.diff(t)
    inc = val[:-1] * dt
    I = np.concatenate(([0.0], np.cumsum(inc)))        # I[j] = int up to t[j]

    def integ(z: np.ndarray) -> np.ndarray:
        zc = np.clip(z, t[0], t[-1])
        j = np.clip(np.searchsorted(t, zc, side="right") - 1, 0, n - 2)
        return I[j] + val[j] * (zc - t[j])

    lo = t - 0.5 * W
    hi = t + 0.5 * W
    out = (integ(hi) - integ(lo)) / W
    return np.maximum(out, 0.0)


def window_sum(t: np.ndarray, val: np.ndarray, W: float) -> np.ndarray:
    """Centred moving *sum* (not average) over window W, in ms*value."""
    return window_mean(t, val, W) * W


def local_count(t: np.ndarray, weights: np.ndarray, W: float) -> np.ndarray:
    """Number of `weights` mass inside [s-W/2, s+W/2) at s = t[r].

    Used for C(s) (local note count) and for density measures.
    """
    n = t.size
    csum = np.concatenate(([0.0], np.cumsum(weights)))

    def cnt(z: np.ndarray) -> np.ndarray:
        zc = np.clip(z, t[0], t[-1] + 1.0)
        return csum[np.searchsorted(t, zc, side="left")]

    return cnt(t + 0.5 * W) - cnt(t - 0.5 * W)


def held_count(notes_ln_head: np.ndarray, notes_ln_tail: np.ndarray,
               t: np.ndarray) -> np.ndarray:
    """Number of LNs whose body strictly covers time t (row grid).

    An LN with head h and tail e is *held* on (h, e).
    """
    n = t.size
    if notes_ln_head.size == 0:
        return np.zeros(n, dtype=np.float64)
    order = np.argsort(notes_ln_head, kind="stable")
    h = notes_ln_head[order]
    e = notes_ln_tail[order]
    # events: +1 at h (exclusive) / -1 at e (inclusive of t==e? use strict)
    starts = np.searchsorted(t, h, side="right")
    ends = np.searchsorted(t, e, side="left")
    d = np.zeros(n + 2, dtype=np.float64)
    np.add.at(d, starts, 1.0)
    np.add.at(d, ends, -1.0)
    return np.maximum(np.cumsum(d)[:n], 0.0)


def run_lengths(keep: np.ndarray) -> np.ndarray:
    """Length of the run of consecutive True values each element belongs to.

    `keep` is boolean over a sequence; runs are maximal True streaks.
    Elements with keep == False get 0.
    """
    n = keep.size
    out = np.zeros(n, dtype=np.int64)
    if n == 0:
        return out
    brk = np.concatenate(([True], ~keep[:-1])) & keep
    ids = np.cumsum(brk) - 1
    if not np.any(brk):
        return out
    lens = np.bincount(ids[keep])
    out[keep] = lens[ids[keep]]
    return out


def active_col_count(grid: Grid, W: float = 300.0) -> np.ndarray:
    """Number of distinct columns with a note in [s-W/2, s+W/2)."""
    n = grid.n
    cnt = np.zeros((K, n), dtype=np.float64)
    for k in range(K):
        idx = grid.col_rows[k]
        if idx.size == 0:
            continue
        v = np.zeros(n, dtype=np.float64)
        np.add.at(v, idx, 1.0)
        cnt[k] = local_count(grid.t, v, W)
    return (cnt > 0).sum(axis=0)
