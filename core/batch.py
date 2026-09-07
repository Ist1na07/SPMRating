"""Flat-batched chart representation.

Instead of looping over charts (thousands of small numpy calls), every chart
is concatenated into one flat array.  Timestamps get a per-chart offset so
that the whole batch is a single increasing time axis -- that makes a single
global `searchsorted` behave like a per-chart one.

Segmented reductions (cumsum / sort / percentile) are implemented with
segment-aware tricks so nothing ever loops over charts in Python.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .parser import Notes, cached_parse, parse_file
from .grid import K, N_BOUND, POPCOUNT

CHART_STRIDE = 1 << 30          # ms offset between charts (>> any window)


# --------------------------------------------------------------- primitives
def seg_cumsum(x: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Cumulative sum restarted at every segment boundary."""
    c = np.cumsum(x)
    n_seg = starts.size - 1
    base = np.zeros(n_seg, dtype=c.dtype)
    if n_seg > 1:
        prev_end = starts[1:-1] - 1
        base[1:] = c[prev_end]
    return c - np.repeat(base, np.diff(starts))


def seg_cumsum_from(c: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Same as seg_cumsum but takes an already-computed global cumsum."""
    n_seg = starts.size - 1
    base = np.zeros(n_seg, dtype=c.dtype)
    if n_seg > 1:
        base[1:] = c[starts[1:-1] - 1]
    return c - np.repeat(base, np.diff(starts))


@dataclass
class Batch:
    """Flat arrays covering every chart in the evaluation set."""
    # --- identity
    ids: List[str] = field(default_factory=list)
    starts: np.ndarray = field(default_factory=lambda: np.zeros(1, np.int64))
    # --- row grid (flat)
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))       # local ms
    tg: np.ndarray = field(default_factory=lambda: np.zeros(0))      # global ms
    chart: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    mask: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    size: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    is_ln: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    ln_end: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dt: np.ndarray = field(default_factory=lambda: np.zeros(0))      # sec, len n-1
    n: int = 0
    # --- per chart scalars
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))       # hit leniency
    od: np.ndarray = field(default_factory=lambda: np.zeros(0))
    key_count: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    n_notes: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    duration: np.ndarray = field(default_factory=lambda: np.zeros(0))
    is_4k: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    # --- per-column index chains (flat, per column)
    col_pos: List[np.ndarray] = field(default_factory=list)   # flat row positions
    col_seg: List[np.ndarray] = field(default_factory=list)   # starts per chart
    col_dt: List[np.ndarray] = field(default_factory=list)    # gaps (sec)
    # --- per-boundary index chains
    bnd_pos: List[np.ndarray] = field(default_factory=list)
    bnd_seg: List[np.ndarray] = field(default_factory=list)
    bnd_dt: List[np.ndarray] = field(default_factory=list)
    # --- LN tails (flat, per chart sorted by tail time)
    tail_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tail_h: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tail_col: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    tail_chart: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    tail_seg: np.ndarray = field(default_factory=lambda: np.zeros(1, np.int64))
    tail_next_head: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tail_next_head_col: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    # --- note-level (not row-level) flat arrays, for C(s) and chord stats
    note_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    note_chart: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    note_seg: np.ndarray = field(default_factory=lambda: np.zeros(1, np.int64))
    note_row: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    note_col: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))

    @property
    def n_charts(self) -> int:
        return len(self.ids)

    def row_of_chart(self, c: int) -> slice:
        return slice(int(self.starts[c]), int(self.starts[c + 1]))


# ------------------------------------------------------------------ building
def _od_leniency(od: float) -> float:
    q = (64.5 - float(np.ceil(3.0 * od))) / 500.0
    x = 0.3 * np.sqrt(max(q, 1e-6))
    return float(min(x, 0.6 * (x - 0.09) + 0.09))


def build_batch(charts: List, rate: float = 1.0,
                parse_cache: bool = True) -> Batch:
    """Parse a list of dataset Chart objects into a flat Batch."""
    b = Batch()
    ids, t_list, mask_list, isln_list, lne_list = [], [], [], [], []
    note_t_list, note_chart_list, note_row_list = [], [], []
    x_list, od_list, kc_list, nnotes_list, dur_list, rows_list = [], [], [], [], [], []
    tail_t, tail_h, tail_col, tail_chart, tail_rows = [], [], [], [], []

    parsed: List[Optional[Notes]] = []
    for ci, ch in enumerate(charts):
        nt = cached_parse(ch.path, rate) if parse_cache else parse_file(ch.path, rate)
        parsed.append(nt)
        if nt is None or nt.n_notes < 2:
            continue
        ids.append(ch.id)
        # rows
        first = np.empty(nt.n_notes, dtype=bool)
        first[0] = True
        np.not_equal(nt.time[1:], nt.time[:-1], out=first[1:])
        row_of_note = np.cumsum(first) - 1
        R = int(row_of_note[-1]) + 1
        t = np.zeros(R, dtype=np.float64)
        t[row_of_note] = nt.time
        mask = np.zeros(R, dtype=np.int32)
        np.bitwise_or.at(mask, row_of_note, (1 << nt.col).astype(np.int32))
        isln = np.zeros(R, dtype=bool)
        np.logical_or.at(isln, row_of_note, nt.is_ln)
        lne = np.full(R, -1.0)
        np.maximum.at(lne, row_of_note, np.where(nt.is_ln, nt.ln_end, -1.0))

        t_list.append(t)
        mask_list.append(mask)
        isln_list.append(isln)
        lne_list.append(lne)
        rows_list.append(R)
        x_list.append(_od_leniency(nt.od))
        od_list.append(nt.od)
        kc_list.append(nt.key_count)
        nnotes_list.append(nt.n_notes)
        dur_list.append(float(t[-1] - t[0]))
        note_t_list.append(nt.time)
        note_chart_list.append(np.full(nt.n_notes, len(ids) - 1, np.int32))
        note_row_list.append(row_of_note)

        # LN tails
        li = np.flatnonzero(nt.is_ln)
        if li.size:
            order = np.argsort(nt.ln_end[li], kind="stable")
            li = li[order]
            tail_t.append(nt.ln_end[li])
            tail_h.append(nt.time[li])
            tail_col.append(nt.col[li].astype(np.int32))
            tail_chart.append(np.full(li.size, len(ids) - 1, np.int32))
            tail_rows.append(row_of_note[li])
        else:
            tail_t.append(np.zeros(0))
            tail_h.append(np.zeros(0))
            tail_col.append(np.zeros(0, np.int32))
            tail_chart.append(np.zeros(0, np.int32))
            tail_rows.append(np.zeros(0, np.int64))

    # ---- next head in column after each tail (vectorised per column)
    next_head = np.full(sum(len(a) for a in tail_t), np.inf)
    next_head_col = np.full(sum(len(a) for a in tail_t), -1, np.int32)
    if tail_t:
        tt = np.concatenate(tail_t)
        tc = np.concatenate(tail_chart)
        tl = np.concatenate(tail_col)
        for k in range(K):
            m = tl == k
            if not m.any():
                continue
            # for each chart, heads in column k
            for ci in np.unique(tc[m]):
                cm = m & (tc == ci)
                heads = None
                # gather heads of column k in chart ci
                nt = parsed[ci]
                if nt is None:
                    continue
                hk = nt.time[(nt.col == k)]
                j = np.searchsorted(hk, tt[cm], side="right")
                ok = j < hk.size
                idx = np.flatnonzero(cm)
                next_head[idx[ok]] = hk[j[ok]]
                next_head_col[idx[ok]] = k
        # fall back: next head in ANY column
        miss = ~np.isfinite(next_head)
        if miss.any():
            for ci in np.unique(tc[miss]):
                nt = parsed[ci]
                if nt is None:
                    continue
                mm = miss & (tc == ci)
                j = np.searchsorted(nt.time, tt[mm], side="right")
                ok = j < nt.n_notes
                idx = np.flatnonzero(mm)
                next_head[idx[ok]] = nt.time[j[ok]]
                next_head_col[idx[ok]] = nt.col[j[ok]]

    # ---- assemble flat arrays
    b.ids = ids
    n_per = np.array(rows_list, dtype=np.int64)
    b.starts = np.concatenate(([0], np.cumsum(n_per))).astype(np.int64)
    b.n = int(n_per.sum())
    b.t = np.concatenate(t_list) if t_list else np.zeros(0)
    b.chart = np.repeat(np.arange(len(ids), dtype=np.int32), n_per)
    off = (b.chart.astype(np.int64) * CHART_STRIDE)
    b.tg = b.t + off
    b.mask = np.concatenate(mask_list).astype(np.int32) if mask_list else np.zeros(0, np.int32)
    b.size = POPCOUNT[b.mask].astype(np.int32)
    b.is_ln = np.concatenate(isln_list) if isln_list else np.zeros(0, bool)
    b.ln_end = np.concatenate(lne_list) if lne_list else np.zeros(0)
    b.x = np.array(x_list)
    b.od = np.array(od_list)
    b.key_count = np.array(kc_list, np.int32)
    b.n_notes = np.array(nnotes_list, np.int64)
    b.duration = np.array(dur_list)
    b.is_4k = (b.key_count == 4)

    # dt (per row, step function length n-1 within chart)
    dt = np.diff(b.tg) / 1000.0
    dt = np.where(np.diff(b.chart) == 0, dt, 1.0)
    b.dt = dt

    # note-level flat
    b.note_t = np.concatenate(note_t_list) if note_t_list else np.zeros(0)
    b.note_chart = np.concatenate(note_chart_list) if note_chart_list else np.zeros(0, np.int32)
    b.note_row = np.concatenate(note_row_list) if note_row_list else np.zeros(0, np.int64)
    b.note_col = np.concatenate([nt.col for nt in parsed if nt is not None and nt.n_notes >= 2]).astype(np.int32) if note_t_list else np.zeros(0, np.int32)
    b.note_seg = np.concatenate(([0], np.cumsum(
        np.array([a.size for a in note_t_list], np.int64)))).astype(np.int64)

    # tails
    if tail_t:
        b.tail_t = np.concatenate(tail_t)
        b.tail_h = np.concatenate(tail_h)
        b.tail_col = np.concatenate(tail_col).astype(np.int32)
        b.tail_chart = np.concatenate(tail_chart).astype(np.int32)
        b.tail_seg = np.concatenate(([0], np.cumsum(
            np.array([a.size for a in tail_t], np.int64)))).astype(np.int64)
        b.tail_next_head = next_head
        b.tail_next_head_col = next_head_col
    else:
        b.tail_seg = np.zeros(1, np.int64)

    # ---- per-column chains
    kc = b.key_count
    for k in range(K):
        rows_k = np.flatnonzero((b.mask >> k) & 1).astype(np.int64)
        keep = np.ones(rows_k.size, dtype=bool)
        # drop duplicates within a chord (can't happen: one note per col per row)
        pos = rows_k[keep]
        b.col_pos.append(pos)
        cids = b.chart[pos]
        seg = np.searchsorted(cids, np.arange(len(ids) + 1), side="left").astype(np.int64)
        b.col_seg.append(seg)
        g = np.diff(b.tg[pos]) / 1000.0
        g = np.where(np.diff(cids) == 0, g, np.inf)
        b.col_dt.append(g)
    # ---- per-boundary chains
    for bd in range(N_BOUND):
        cols = [c for c in (bd - 1, bd) if 0 <= c < K]
        if not cols:
            b.bnd_pos.append(np.zeros(0, np.int64))
            b.bnd_seg.append(np.zeros(len(ids) + 1, np.int64))
            b.bnd_dt.append(np.zeros(0))
            continue
        flat = np.unique(np.concatenate([b.col_pos[c] for c in cols]))
        # b.col_pos are already globally sorted per column; merge sorted
        parts = [b.col_pos[c] for c in cols]
        if len(parts) == 1:
            merged = parts[0]
        else:
            merged = np.union1d(parts[0], parts[1])
        # remove rows where the boundary has <2 distinct columns used
        #  (single-column boundaries are legitimate: they measure that column)
        b.bnd_pos.append(merged.astype(np.int64))
        cids = b.chart[merged]
        seg = np.searchsorted(cids, np.arange(len(ids) + 1), side="left").astype(np.int64)
        b.bnd_seg.append(seg)
        g = np.diff(b.tg[merged]) / 1000.0
        g = np.where(np.diff(cids) == 0, g, np.inf)
        b.bnd_dt.append(g)
    return b


# --------------------------------------------------------------- disk cache
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "cache")


def batch_cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"batch_{name}.npz")


def save_batch(b: Batch, name: str) -> None:
    data = {
        "starts": b.starts, "t": b.t, "mask": b.mask, "is_ln": b.is_ln,
        "ln_end": b.ln_end, "x": b.x, "od": b.od, "key_count": b.key_count,
        "n_notes": b.n_notes, "duration": b.duration,
        "note_t": b.note_t, "note_chart": b.note_chart, "note_row": b.note_row,
        "note_col": b.note_col,
        "note_seg": b.note_seg,
        "tail_t": b.tail_t, "tail_h": b.tail_h, "tail_col": b.tail_col,
        "tail_chart": b.tail_chart, "tail_seg": b.tail_seg,
        "tail_next_head": b.tail_next_head,
        "tail_next_head_col": b.tail_next_head_col,
        "ids": np.array(b.ids),
    }
    for k in range(K):
        data[f"cp{k}"] = b.col_pos[k]
        data[f"cs{k}"] = b.col_seg[k]
        data[f"cd{k}"] = b.col_dt[k]
    for bd in range(N_BOUND):
        data[f"bp{bd}"] = b.bnd_pos[bd]
        data[f"bs{bd}"] = b.bnd_seg[bd]
        data[f"bd{bd}"] = b.bnd_dt[bd]
    np.savez_compressed(batch_cache_path(name), **data)


def load_batch(name: str) -> Optional[Batch]:
    p = batch_cache_path(name)
    if not os.path.exists(p):
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            b = Batch()
            b.ids = [str(s) for s in z["ids"]]
            b.starts = z["starts"]
            b.t = z["t"]
            b.mask = z["mask"]
            b.is_ln = z["is_ln"]
            b.ln_end = z["ln_end"]
            b.x = z["x"]
            b.od = z["od"]
            b.key_count = z["key_count"]
            b.n_notes = z["n_notes"]
            b.duration = z["duration"]
            b.note_t = z["note_t"]
            b.note_chart = z["note_chart"]
            b.note_row = z["note_row"]
            b.note_col = z["note_col"]
            b.note_seg = z["note_seg"]
            b.tail_t = z["tail_t"]
            b.tail_h = z["tail_h"]
            b.tail_col = z["tail_col"]
            b.tail_chart = z["tail_chart"]
            b.tail_seg = z["tail_seg"]
            b.tail_next_head = z["tail_next_head"]
            b.tail_next_head_col = z["tail_next_head_col"]
            b.n = int(b.t.size)
            b.chart = np.repeat(np.arange(len(b.ids), dtype=np.int32),
                                np.diff(b.starts))
            b.tg = b.t + b.chart.astype(np.int64) * CHART_STRIDE
            b.size = POPCOUNT[b.mask].astype(np.int32)
            b.is_4k = (b.key_count == 4)
            dt = np.diff(b.tg) / 1000.0
            b.dt = np.where(np.diff(b.chart) == 0, dt, 1.0)
            b.col_pos = [z[f"cp{k}"] for k in range(K)]
            b.col_seg = [z[f"cs{k}"] for k in range(K)]
            b.col_dt = [z[f"cd{k}"] for k in range(K)]
            b.bnd_pos = [z[f"bp{bd}"] for bd in range(N_BOUND)]
            b.bnd_seg = [z[f"bs{bd}"] for bd in range(N_BOUND)]
            b.bnd_dt = [z[f"bd{bd}"] for bd in range(N_BOUND)]
            return b
    except Exception:                                        # noqa: BLE001
        return None
