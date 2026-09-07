"""SPM Rating 难度模型。

Everything runs on the flat `Batch` representation: one numpy op covers all
charts at once; per-chart isolation comes from the chart-id time offset plus
segmented reductions.

Pipeline
--------
    build_struct(b)       parameter-free per-row arrays (cached)
    compute_curves(...)   Pbar / Jbar / Jm / Jc / Ja / Xbar / Abar / Rbar / Cbar
    combine(...)          S(t), T(t), D(t)
    aggregate(...)        per-chart SR
"""
from __future__ import annotations

import numpy as np

from .batch import seg_cumsum

K = 7
N_BOUND = 8
EPS = 1e-9

# popcount lookup for 7-bit row masks
_POPCOUNT7 = np.array([bin(i).count("1") for i in range(128)], dtype=np.int32)

# 7K: 0,1,2 left hand | 3 thumb | 4,5,6 right hand
HAND = np.array([0, 0, 0, 1, 2, 2, 2], dtype=np.int32)
_IDX7 = np.arange(K, dtype=np.int64)
AIN_GROUP = np.array([0, 1, 2, 3, 2, 1, 0], dtype=np.int32)   # ring/mid/idx/thumb
BOUND_GROUP = np.array([0, 1, 2, 3, 3, 2, 1, 0], dtype=np.int32)
SHIELD_LOOKBACK = 16


# ------------------------------------------------------------------ helpers
def _step_scatter(idx0: np.ndarray, idx1: np.ndarray, val: np.ndarray,
                  n: int) -> np.ndarray:
    """Step function equal to val[j] on rows [idx0[j], idx1[j]).

    Ends at or beyond the last row are absorbed into a sentinel slot so the
    prefix sum always returns to zero (never leaks to the end of the array).
    """
    m = n + 1
    a = np.bincount(np.minimum(idx0, m), weights=val, minlength=m + 1)
    b = np.bincount(np.minimum(idx1, m), weights=val, minlength=m + 1)
    return np.cumsum(a[:m] - b[:m])[:n]


def _chain_scatter(pos: np.ndarray, val: np.ndarray, n: int,
                   same: np.ndarray) -> np.ndarray:
    v = np.where(same, val, 0.0)
    return _step_scatter(pos[:-1], pos[1:], v, n)


def _run_lengths(keep: np.ndarray) -> np.ndarray:
    out = np.zeros(keep.size, dtype=np.float64)
    if keep.size == 0 or not keep.any():
        return out
    brk = np.concatenate(([True], ~keep[:-1])) & keep
    ids = np.cumsum(brk) - 1
    lens = np.bincount(ids[keep])
    out[keep] = lens[ids[keep]]
    return out


class Struct:
    """Parameter-free per-row arrays, built once per batch."""
    __slots__ = ("n", "n_charts", "starts", "counts", "chart", "t", "tg",
                 "tlo", "thi", "dt", "dt_ms", "mask", "size", "x", "is_ln",
                 "ln_end", "dcol", "act", "usage", "anchor_raw", "held",
                 "heldmask", "ln_body_ms", "C", "Ks", "cumsize", "note_row",
                 "note_impulse", "col_chain", "bnd_chain", "tail", "is_4k",
                 "_tail_lo", "_tail_hi", "jlo", "jhi", "_wmean_ops",
                 "_chord_ops",
                 "n_notes", "duration", "dt_diff", "chart_same", "batch")


# _wmean interpolation operators are cached ON the struct object itself.
# (Keying a module-level dict by id(s) is unsound: a rebuilt struct can
# reuse a freed id and silently inherit the previous dataset's index arrays.)


def _wmean_index(s: Struct, W: float):
    """Cached (j_hi, z_hi, j_lo, z_lo) interpolation operator for window W."""
    ops = getattr(s, "_wmean_ops", None)
    if ops is None:
        ops = {}
        s._wmean_ops = ops
    op = ops.get(W)
    if op is not None:
        return op
    tg = s.tg
    zhi = np.minimum(tg + 0.5 * W, s.thi)
    zlo = np.maximum(tg - 0.5 * W, s.tlo)
    jhi = np.clip(np.searchsorted(tg, zhi, side="right") - 1, s.jlo, s.jhi)
    jlo = np.clip(np.searchsorted(tg, zlo, side="right") - 1, s.jlo, s.jhi)
    op = (jhi, zhi - tg[jhi], jlo, zlo - tg[jlo])
    if len(ops) > 64:
        ops.clear()
    ops[W] = op
    return op


def _wmean(s: Struct, val: np.ndarray, W: float) -> np.ndarray:
    """Centred moving average of the step function `val` over a W-ms window.

    The interpolation index is clipped to the *owning chart's* row range --
    clipping to [0, n-2] would silently reach across the chart-id time gap.
    """
    inc = np.zeros(s.n)
    inc[:-1] = val[:-1] * s.dt_diff
    # I[i] = integral from the chart's first row up to t[i]  (exclusive of i)
    I = seg_cumsum(inc, s.starts) - inc
    jhi, dhi, jlo, dlo = _wmean_index(s, W)
    out = ((I[jhi] + val[jhi] * dhi) - (I[jlo] + val[jlo] * dlo)) / W
    return np.maximum(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


def _wcount(s: Struct, w: np.ndarray, W: float) -> np.ndarray:
    tg = s.tg
    c = np.concatenate(([0.0], np.cumsum(w)))

    def cnt(z):
        zc = np.clip(z, s.tlo, s.thi + 1.0)
        return c[np.searchsorted(tg, zc, side="left")]

    return cnt(tg + 0.5 * W) - cnt(tg - 0.5 * W)


# ------------------------------------------------------ soft chord coupling
# 注记：精确时间戳的和弦在 1 毫秒扰动下会分裂。
# chord_tau_c > 0 gives notes in adjacent rows a graded chord-coupling
# w(gap) = clip(1 - gap/tau_c, 0, 1) so effective chord size varies
# continuously.  All of this lives in compute_curves (param-dependent);
# only the cached (wf, wb) operator hangs off the struct.
def _chord_coupling(s: Struct, tau_c: float):
    """(wf, wb): forward/backward adjacent-row coupling weights.

    wf[r] = w(t[r+1] - t[r]) for same-chart successor, else 0;
    wb[r] = wf[r-1].  Cached on the struct per tau_c (same pattern as
    _wmean_ops).
    """
    ops = getattr(s, "_chord_ops", None)
    if ops is None:
        ops = {}
        s._chord_ops = ops
    got = ops.get(tau_c)
    if got is None:
        w = np.clip(1.0 - s.dt_ms / tau_c, 0.0, 1.0)
        wf = np.zeros(s.n)
        wf[:-1] = np.where(s.chart_same, w[:-1], 0.0)
        wb = np.zeros(s.n)
        wb[1:] = np.where(s.chart_same, w[:-1], 0.0)
        got = (wf, wb)
        if len(ops) > 16:
            ops.clear()
        ops[tau_c] = got
    return got


def _couple(x: np.ndarray, wf: np.ndarray, wb: np.ndarray) -> np.ndarray:
    """x plus adjacent-row neighbours discounted by the coupling weights."""
    fwd = np.zeros_like(x)
    fwd[:-1] = x[1:]
    bwd = np.zeros_like(x)
    bwd[1:] = x[:-1]
    return x + wf * fwd + wb * bwd


def _tri_window_count(s: Struct, val: np.ndarray, tau_c: float) -> np.ndarray:
    """Triangular-window count: sum_r' val[r'] * max(0, 1 - |t[r']-t[r]|/tau_c).

    Chart-safe O(n) via cumsums + searchsorted (window clipped to the
    owning chart).  With val = row size this is the graded effective chord
    size: it reaches the full group size for a chord split into rows a few
    ms apart, and equals size exactly for isolated rows.
    """
    tg = s.tg
    t = s.t.astype(np.float64)
    n = s.n
    cv = np.concatenate(([0.0], np.cumsum(val)))
    cvt = np.concatenate(([0.0], np.cumsum(val * t)))
    lo = np.clip(tg - tau_c, s.tlo, s.thi + 1.0)
    hi = np.clip(tg + tau_c, s.tlo, s.thi + 1.0)
    jl = np.searchsorted(tg, lo, side="left")
    jh = np.searchsorted(tg, hi, side="left")
    r = np.arange(n)
    # left part: rows [jl, r) with |dt| = t[r] - t[r']
    sl = cv[r] - cv[jl]
    slt = cvt[r] - cvt[jl]
    left = sl - (t * sl - slt) / tau_c
    # right part: rows [r, jh) with |dt| = t[r'] - t[r]
    sr = cv[jh] - cv[r]
    srt = cvt[jh] - cvt[r]
    right = sr - (srt - t * sr) / tau_c
    return np.maximum(
        np.nan_to_num(left + right, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


def _edge_ramp(x, cutoff) -> np.ndarray:
    """Raised-cosine fade: 1 up to 80% of `cutoff`, 0 at/above `cutoff`.

    Used by r_soft_edges to replace hard time cutoffs (r_dt_max,
    r_order_tau) with a continuous outer-20% fade.
    """
    x0 = 0.8 * cutoff
    u = np.clip((x - x0) / max(cutoff - x0, 1e-9), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * u))


def _abar_val(dd, mx, p) -> np.ndarray:
    """Abar column-pair multiplier.

    abar_weld 等于 1 时，中间分支变为
    c0 + k*(dd - thr_lo) + mx*mx, so val is continuous at thr_lo by
    construction.  Legacy (0): c1 + k*dd + mx*mx, which jumps by
    c1 + k*thr_lo - c0 at the seam.  The thr_hi seam is welded in both
    variants by the min(., 1) saturation.
    """
    lo = np.minimum(p["abar_c0"] + p["abar_mx"] * mx, 1.0)
    if p.get("abar_weld", 0.0):
        mid = np.minimum(p["abar_c0"] + p["abar_k"] * (dd - p["abar_thr_lo"])
                         + p["abar_mx"] * mx, 1.0)
    else:
        mid = np.minimum(p["abar_c1"] + p["abar_k"] * dd +
                         p["abar_mx"] * mx, 1.0)
    return np.where(dd < p["abar_thr_lo"], lo,
                    np.where(dd < p["abar_thr_hi"], mid, 1.0))


# ------------------------------------------------------------------- struct
def build_struct(b, p: dict | None = None) -> Struct:
    s = Struct()
    s.batch = b  # keep a reference for components that need raw batch data
    n = b.n
    starts = b.starts
    counts = np.diff(starts)
    s.n = n
    s.n_charts = b.n_charts
    s.starts = starts
    s.counts = counts
    s.chart = b.chart
    s.t = b.t
    s.tg = b.tg
    s.tlo = np.repeat(b.tg[starts[:-1]], counts)
    s.thi = np.repeat(b.tg[starts[1:] - 1], counts)
    dtd = np.diff(b.tg)
    s.chart_same = (np.diff(b.chart) == 0)
    s.dt_diff = np.where(s.chart_same, dtd, 0.0)
    dt = np.zeros(n)
    dt[:-1] = s.dt_diff
    s.dt_ms = np.maximum(dt, 0.0)
    s.dt = np.where(dt > 0, dt / 1000.0, 1.0)
    s.mask = b.mask
    s.size = b.size
    s.x = b.x
    s.is_ln = b.is_ln
    s.ln_end = b.ln_end
    s.is_4k = b.is_4k
    s.n_notes = b.n_notes
    s.duration = b.duration
    s.cumsize = np.concatenate(([0.0], np.cumsum(b.size.astype(np.float64))))
    s.jlo = np.repeat(starts[:-1], counts)
    s.jhi = np.repeat(np.maximum(starts[1:] - 1, 0), counts)

    # ---- column chains
    chain = []
    for k in range(K):
        pos = b.col_pos[k]
        if pos.size:
            cid = b.chart[pos]
            same = np.zeros(pos.size, dtype=bool)
            same[:-1] = (cid[:-1] == cid[1:])
            g = np.zeros(max(pos.size - 1, 0))
            if pos.size >= 2:
                g = np.where(same[:-1], np.diff(b.tg[pos]) / 1000.0, np.inf)
        else:
            cid = np.zeros(0, np.int32)
            same = np.zeros(0, bool)
            g = np.zeros(0)
        chain.append(dict(pos=pos, g=g, same=same, cid=cid,
                          tc=b.t[pos] if pos.size else np.zeros(0),
                          tgc=b.tg[pos] if pos.size else np.zeros(0),
                          seg=b.col_seg[k]))
    s.col_chain = chain

    bchain = []
    for bd in range(N_BOUND):
        pos = b.bnd_pos[bd]
        if pos.size:
            cid = b.chart[pos]
            same = np.zeros(pos.size, dtype=bool)
            same[:-1] = (cid[:-1] == cid[1:])
            g = np.where(same[:-1], np.diff(b.tg[pos]) / 1000.0, np.inf) \
                if pos.size >= 2 else np.zeros(0)
        else:
            g = np.zeros(0)
        # effective column of each chain element within the boundary's
        # column pair {bd-1, bd}: left-only -> bd-1, right-only -> bd,
        # chord on both -> midpoint bd-0.5.  Drives the inward/outward
        # (内外切) direction split of the Xbar pairwise term.
        if pos.size and 0 < bd < K:
            m = b.mask[pos].astype(np.int64)
            has_l = ((m >> (bd - 1)) & 1).astype(np.float64)
            has_r = ((m >> bd) & 1).astype(np.float64)
            e = (has_l * (bd - 1) + has_r * bd) / np.maximum(has_l + has_r, 1.0)
        else:
            e = np.full(pos.size, float(min(max(bd, 0), K - 1)))
        bchain.append(dict(pos=pos, g=g, e=e,
                           same=(np.zeros(pos.size, bool) if pos.size == 0
                                 else bchain_same(b, pos))))
    s.bnd_chain = bchain

    # ---- per-row column gap
    dcol = np.full((K, n), 1e9, dtype=np.float32)
    for k in range(K):
        c = chain[k]
        if c["pos"].size >= 2:
            dcol[k] = _chain_scatter(c["pos"], np.where(np.isfinite(c["g"]),
                                                        c["g"], 1e9),
                                     n, c["same"][:-1]).astype(np.float32)
    s.dcol = dcol

    # ---- active columns + column usage
    act = np.zeros((K, n), dtype=bool)
    usage = np.zeros((K, n), dtype=np.float32)
    for k in range(K):
        imp = np.zeros(n)
        if chain[k]["pos"].size:
            imp[chain[k]["pos"]] = 1.0
        act[k] = _wcount(s, imp, 300.0) > 0
        usage[k] = _wcount(s, imp, 800.0).astype(np.float32)
    s.act = act
    s.usage = usage

    u = np.sort(usage, axis=0)[::-1, :]
    nz = u > 1e-6
    ratios = np.where(nz[1:], u[1:] / np.maximum(u[:-1], EPS), 0.0)
    walk = np.sum(np.where(nz[:-1], u[:-1] *
                           np.maximum(1.0 - 4.0 * (0.5 - ratios) ** 2, 0.0), 0.0),
                  axis=0)
    mx = np.sum(np.where(nz[:-1], u[:-1], 0.0), axis=0)
    s.anchor_raw = np.where(mx > EPS, walk / np.maximum(mx, EPS), 0.0)

    # ---- LN held count / held column mask
    # ln_eff_tail_ms: LN effective tail reduction.  For density/lock
    # calculations the player is already preparing to release near the tail,
    # so the LN occupies less time than its paper length.  We shorten the
    # effective end by this amount (ms) for held/density purposes only.
    # Release calculations (Rbar) still use the true ln_end.
    ln_eff = float((p or {}).get("ln_eff_tail_ms", 0.0))
    # ln_note_level 注记：旧的行级做法
    # is_ln/ln_end aggregates mislabel every tap sharing a row with an LN
    # head as an LN head carrying the row's MAX tail, inflating held on
    # chord-LN charts (~-37% body coverage when a 1 ms perturbation
    # separates the chords).  1 = build held/heldmask from the per-note
    # tail arrays (true head/tail/col per LN).  0 = legacy row-level.
    note_level = float((p or {}).get("ln_note_level", 0.0)) != 0.0
    held = np.zeros(n)
    heldmask = np.zeros(n, dtype=np.int32)
    if note_level and b.tail_t.size:
        for k in range(K):
            tk = np.flatnonzero(b.tail_col == k)
            if tk.size == 0:
                continue
            h = b.tail_h[tk]
            e = b.tail_t[tk]
            ch = b.tail_chart[tk].astype(np.int64)
            e_eff = np.maximum(e - ln_eff, h + 1.0)  # never shorter than 1ms
            j0 = np.searchsorted(b.tg, h + ch * (1 << 30) + 1.0, side="left")
            j1 = np.searchsorted(b.tg, e_eff + ch * (1 << 30), side="left")
            d = np.zeros(n + 1)
            np.add.at(d, np.minimum(j0, n), 1.0)
            np.add.at(d, np.minimum(j1, n), -1.0)
            hk = np.maximum(np.cumsum(d)[:n], 0.0)
            held += hk
            heldmask |= np.where(hk > 0, (1 << k), 0).astype(np.int32)
    else:
        for k in range(K):
            c = chain[k]
            if c["pos"].size == 0:
                continue
            li = c["pos"][b.is_ln[c["pos"]]]
            if li.size == 0:
                continue
            h = b.t[li]
            e = b.ln_end[li]
            e_eff = np.maximum(e - ln_eff, h + 1.0)  # never shorter than 1ms
            j0 = np.searchsorted(b.tg, b.tg[li] + 1.0, side="left")
            j1 = np.searchsorted(b.tg, e_eff + b.chart[li].astype(np.int64) * (1 << 30),
                                 side="left")
            d = np.zeros(n + 1)
            np.add.at(d, np.minimum(j0, n), 1.0)
            np.add.at(d, np.minimum(j1, n), -1.0)
            hk = np.maximum(np.cumsum(d)[:n], 0.0)
            held += hk
            heldmask |= np.where(hk > 0, (1 << k), 0).astype(np.int32)
    s.held = held
    s.heldmask = heldmask
    s.ln_body_ms = held * s.dt_ms

    # ---- C (local note count, +-500 ms) and Ks
    note_row = b.note_row + np.repeat(starts[:-1], b.n_notes)
    s.note_row = note_row
    imp = np.zeros(n)
    np.add.at(imp, note_row, 1.0)
    s.note_impulse = imp
    s.C = _wcount(s, imp, 1000.0)
    s.Ks = np.maximum(act.sum(axis=0), 1).astype(np.float64)

    # ---- tails
    s.tail = None
    if b.tail_t.size:
        cid = b.tail_chart.astype(np.int64)
        ch32 = b.tail_chart
        tg_tail = b.tail_t + cid * (1 << 30)
        # clip into the owning chart's row range (a tail can land past the
        # last note head, which would otherwise spill into the next chart)
        lo_r = b.starts[:-1][ch32]
        hi_r = np.maximum(b.starts[1:][ch32] - 1, lo_r)
        j0 = np.clip(np.searchsorted(b.tg, tg_tail, side="left"), lo_r, hi_r)
        # per-chart row bounds reused by _rbar
        s._tail_lo = lo_r
        s._tail_hi = hi_r
        # next event after the tail: next tail in the chart, or next head
        m = b.tail_t.size
        nxt_tail = np.full(m, np.inf)
        nxt_tail_col = np.full(m, -1, np.int64)
        idx = np.arange(m - 1)
        ok = b.tail_chart[idx] == b.tail_chart[idx + 1]
        nxt_tail[idx[ok]] = b.tail_t[idx[ok] + 1]
        nxt_tail_col[idx[ok]] = b.tail_col[idx[ok] + 1]
        is_tail = nxt_tail <= b.tail_next_head - 1e-9
        nxt = np.where(is_tail, nxt_tail, b.tail_next_head)
        nxt_col = np.where(is_tail, nxt_tail_col,
                           b.tail_next_head_col.astype(np.int64))
        nxt_col = np.where(np.isfinite(nxt), nxt_col, b.tail_col.astype(np.int64))
        s.tail = dict(t=b.tail_t, h=b.tail_h, col=b.tail_col,
                      chart=b.tail_chart, seg=b.tail_seg, j0=j0,
                      nh=b.tail_next_head, nhc=b.tail_next_head_col,
                      nt=nxt_tail,
                      nxt=nxt, nxt_col=nxt_col, is_tail=is_tail, tg=tg_tail)
    return s


def bchain_same(b, pos):
    cid = b.chart[pos]
    same = np.zeros(pos.size, dtype=bool)
    same[:-1] = (cid[:-1] == cid[1:])
    return same


# ------------------------------------------------------------------- curves
def _jack_kernel(g, xrow, p, od_mult):
    xk = np.power(xrow, od_mult)
    g = np.maximum(g, 1e-4)
    base = (1.0 / g) * (1.0 / (g + p["j_c1"] * np.power(np.maximum(xk, 1e-6), 0.25)))
    if p.get("j_nerf_a", 0.0):
        base = base * (1.0 - p["j_nerf_a"] *
                       np.power(p["j_nerf_off"] + np.abs(g - p["j_nerf_c"]), -4.0))
    return base


def build_jack_stats(b, s: Struct, p: dict) -> list:
    """Per-column jack-gap arrays.

    Extracted verbatim from `compute_curves` so the Cbar v2 component
    (`core.components.lncoord_v2`) can build per-column lock-modulated
    curves without duplicating this loop.  Pure move -- no arithmetic change.
    """
    xrow = s.x[s.chart]
    gap_thr = float(p["ja_gap_thr"]) / 1000.0
    h3_tau = max(float(p["j_triple_tau"]), 1.0) / 1000.0
    # chord_tau_c 大于 0 时，按攻击组计数行数，而不是原始行数。
    # 攻击组之间没有耦合：被拆成 1 毫秒多行的和弦不再让同列击打
    # 之间的行数翻倍，双押叠通道的负载也不再减半。
    tau_c = float(p.get("chord_tau_c", 0.0))
    att_cum = None
    if tau_c > 0.0:
        wf, wb = _chord_coupling(s, tau_c)
        att_cum = np.concatenate(
            ([0.0], np.cumsum((wb == 0.0).astype(np.float64))))
    # 同列三连增益按谱面键数分别取值：7 键与 4 键各一个系数，
    # 4 键系数小于 0 时表示沿用 7 键的值。
    g7 = float(p["j_triple_gain"])
    g4k_v = float(p.get("j_triple_gain_4k", -1.0))
    g4k = g7 if g4k_v < 0.0 else g4k_v
    is4k_chart = b.is_4k
    stats = []
    for k in range(K):
        c = s.col_chain[k]
        pos = c["pos"]
        if pos.size < 2:
            stats.append(None)
            continue
        g = c["g"]
        valid = np.isfinite(g) & (np.diff(c["cid"]) == 0)
        gs = np.maximum(np.where(valid, g, 1.0), 1e-4)
        if att_cum is not None:
            nrows = np.maximum(att_cum[pos[1:]] - att_cum[pos[:-1]], 1.0)
        else:
            nrows = np.maximum(pos[1:] - pos[:-1], 1).astype(np.float64)
        total = s.cumsize[pos[1:]] - s.cumsize[pos[:-1]]
        other = np.maximum(total - 1.0, 0.0)
        same_step = np.diff(c["cid"]) == 0
        runlen = _run_lengths((g <= gap_thr) & same_step)
        # same-column triple (h_jack3).  The three notes must be in ONE
        # chart -- using local times without that guard lets a chart
        # boundary produce an enormous negative span and blow the channel up.
        trip = np.zeros(max(pos.size - 1, 0))
        tg3 = s.t[pos]
        if tg3.size >= 3 and same_step.size >= 2:
            same3 = same_step[:-1] & same_step[1:]
            span_s = (tg3[2:] - tg3[:-2]) / 1000.0     # ms -> s
            trip[:-1] = np.where(
                same3, np.maximum(0.0, 1.0 - span_s / (2.0 * h3_tau)), 0.0)
        # fast same-column (fj): pairs with gap < 150 ms priced (1/g)(1-g/0.15)
        cid_gap = c["cid"][:-1]
        fj = np.where(same_step & (gs < 0.150),
                      (1.0 / gs) * np.maximum(1.0 - gs / 0.150, 0.0), 0.0)
        # per-gap triple gain by frame
        tgain = np.where(is4k_chart[cid_gap], g4k, g7)
        stats.append(dict(pos=pos, gs=gs, valid=valid, nrows=nrows,
                          other=other, load=total / nrows, runlen=runlen,
                          trip=trip, tgain=tgain, fj=fj, xr=xrow[pos[:-1]],
                          kern=_jack_kernel(gs, xrow[pos[:-1]], p,
                                            float(p["od_mult_jm"]))))
    return stats


def compute_curves(b, s: Struct, p: dict) -> dict:
    n = s.n
    xrow = s.x[s.chart]

    # ============================================================ jack family
    agg_pow = max(float(p["j_agg_pow"]), 0.25)
    triple_gain = float(p["j_triple_gain"])
    mj_dilute = float(p["mj_dilute"])
    mj_exp = float(p["mj_dilute_exp"])
    cj_exp = float(p["cj_size_exp"])
    ja_len_exp = float(p["ja_len_exp"])
    ja_coord_exp = float(p["ja_coord_exp"])
    ja_floor = float(p["ja_len_floor"])
    cj_norm = max(float(p["cj_norm"]), 1e-3)
    ja_norm = max(float(p["ja_norm"]), 1e-3)
    w_jm = float(p["w_jm"])
    # Per-channel smoothing width / OD response for the chordjack and anchor
    # channels.  w<=0 / od_mult<0 means "inherit the minijack values".
    # The anchor channel is redundant, so its knobs stay parked; the live
    # chordjack channel may carry its own values.
    od_jm = float(p["od_mult_jm"])
    w_jc = float(p.get("w_jc", 0.0))
    w_jc = w_jm if w_jc <= 0.0 else w_jc
    w_ja = float(p.get("w_ja", 0.0))
    w_ja = w_jm if w_ja <= 0.0 else w_ja
    od_jc = float(p.get("od_mult_jc", -1.0))
    od_jc = od_jm if od_jc < 0.0 else od_jc
    od_ja = float(p.get("od_mult_ja", -1.0))
    od_ja = od_jm if od_ja < 0.0 else od_ja

    stats = build_jack_stats(b, s, p)

    def jack_channel(kind: str, w_ms: float, od_mult: float) -> np.ndarray:
        num = np.zeros(n)
        den = np.zeros(n)
        for k in range(K):
            st = stats[k]
            if st is None:
                continue
            if kind == "plain":
                mod = np.ones(st["pos"].size - 1)
            elif kind == "mj":
                mod = (1.0 / (1.0 + mj_dilute *
                              np.power(st["other"] / st["nrows"], mj_exp))
                       * (1.0 + st["tgain"] * st["trip"]))
            elif kind == "cj":
                mod = np.power(np.maximum(st["load"], 1e-6) / cj_norm, cj_exp)
            else:  # "ja"
                e = np.maximum(st["runlen"] + 2.0 - ja_floor, 0.0)
                runf = np.power(e / (e + ja_norm), ja_len_exp)
                mod = runf * np.power(1.0 + st["other"] / st["nrows"], ja_coord_exp)
            kern = st["kern"] if od_mult == od_jm else \
                _jack_kernel(st["gs"], st["xr"], p, od_mult)
            v = np.where(st["valid"], kern * mod, 0.0)
            cur = np.maximum(_wmean(s, _chain_scatter(st["pos"], v, n,
                                                      st["valid"]), w_ms), 0.0)
            w = _chain_scatter(st["pos"],
                               np.where(st["valid"], 1.0 / st["gs"], 0.0),
                               n, st["valid"])
            num += np.power(cur, agg_pow) * w
            den += w
        val = np.where(den > 1e-9, num / np.where(den > 1e-9, den, 1.0), 0.0)
        return np.power(np.maximum(val, 0.0), 1.0 / agg_pow)

    need = set()
    if p["c_jm"] or p["t_jack_mix"] or p["cb_lockjack"]:
        need.add("mj")
    if p["c_jc"]:
        need.add("cj")
    if p["c_ja"] or p["cb_lockanchor"]:
        need.add("ja")
    if p["c_s"] or p["a_p"]:
        need.add("plain") if False else None
    need.add("plain")            # Jbar is also the Cbar lock-jack carrier
    Jbar = jack_channel("plain", w_jm, od_jm)
    Jm = jack_channel("mj", w_jm, od_jm) if "mj" in need else Jbar

    # fast same-column (fj) channel: rapid same-column pairs are priced below
    # 同列快对通道。按分帧分别计价：
    # c_fj for 7K rows, c_fj_4k for 4K rows (0 = off for that frame).
    # NB: FJ is merged into Jm BEFORE the Jc/Ja fallbacks below, so a
    # 回退拿到的永远是并入后的最终 Jm。
    c_fj = float(p.get("c_fj", 0.0))
    c_fj_4k = float(p.get("c_fj_4k", 0.0))
    if c_fj != 0.0 or c_fj_4k != 0.0:
        FJ = np.zeros(n)
        for k in range(K):
            st = stats[k]
            if st is None:
                continue
            FJ += _chain_scatter(st["pos"], np.where(st["valid"], st["fj"],
                                                     0.0), n, st["valid"])
        FJ = _wmean(s, FJ, w_jm)
        c_fj_row = np.where(s.is_4k[s.chart], c_fj_4k, c_fj)
        Jm = Jm + c_fj_row * FJ

    Jc = jack_channel("cj", w_jc, od_jc) if "cj" in need else Jm
    Ja = jack_channel("ja", w_ja, od_ja) if "ja" in need else Jm

    # ================================================================== Pbar
    xp = np.power(xrow, float(p["od_mult_p"]))
    d = s.dt
    tau_c = float(p.get("chord_tau_c", 0.0))
    d_fp = d
    if tau_c > 0.0:
        wf, wb = _chord_coupling(s, tau_c)
        # the synchronous-note term uses the gap to the next ATTACK (rows
        # with no coupling from their predecessor), skipping rows absorbed
        # into this row's chord group -- otherwise a split chord's
        # carrying row reads an artificially tiny interval and loses fp.
        att_idx = np.flatnonzero(wb == 0.0)
        j = np.searchsorted(att_idx, np.arange(n), side="right")
        has = j < att_idx.size
        na = att_idx[np.minimum(j, att_idx.size - 1)]
        ok_na = has & (s.chart[na] == s.chart)
        d_fp = np.where(ok_na, (s.t[na] - s.t) / 1000.0, d)
        d_fp = np.where(d_fp > 0, d_fp, d)
    q = np.minimum(d_fp - xp / 2.0, xp / 6.0)
    fp = np.power(np.maximum(p["p_scale"] / xp *
                             (1.0 - (p["p_lam3"] / xp) * q * q), 0.0), 0.25)
    r = 7.5 / d
    lo, hi = float(p["p_boost_lo"]), float(p["p_boost_hi"])
    bst = np.where((r > lo) & (r < hi),
                   1.0 + p["p_boost"] * (r - lo) * np.power(r - hi, 2.0), 1.0)
    vv = 1.0 + p["p_lam2"] * s.ln_body_ms / \
        np.power(np.maximum(s.dt_ms, 1.0), float(p["p_v_norm"]))
    chw = np.ones(K + 1)
    chw[2:K + 1] = [p.get(f"p_chw{i}", 1.0) for i in range(2, K + 1)]
    if tau_c > 0.0:
        # 分级和弦定价：
        #  * u = triangular +-tau_c window note count = effective chord
        #    size (a chord split into rows a few ms apart keeps ~its full
        #    group size); the p_chw* table is linearly interpolated at u.
        #  * att = attack discount: a row dt ms after another is only
        #    dt/tau_c of a new attack (absorbed into the previous row's
        #    chord group) -- Pbar's 1/d intensity would otherwise price a
        #    split chord as k full attacks.  The discount applies to the
        #    RATE part of mix only: the LN-body occupation gain (vv - 1)
        #    follows each row's own interval (its sum over rows is
        #    grid-split invariant) and must not be discounted -- otherwise
        #    the body mass migrates onto discounted sub-rows.
        u = np.clip(_tri_window_count(s, s.size.astype(np.float64), tau_c),
                    1.0, float(K))
        i0 = np.minimum(u.astype(np.int64), K - 1)
        frac = u - i0
        chw_u = chw[i0] * (1.0 - frac) + chw[i0 + 1] * frac
        att = 1.0 - wb
        if p["p_bv_max"] > 0.5:
            mix = att * np.maximum(bst, 1.0) + np.maximum(vv - 1.0, 0.0)
        else:
            mix = att * bst * vv
    else:
        chw_u = chw[np.minimum(s.size, K)]
        mix = np.maximum(bst, vv) if p["p_bv_max"] > 0.5 else bst * vv
    inc = (1.0 / d) * fp * chw_u * mix
    anchor_mult = 1.0 + p["an_on_p"] * np.minimum(
        s.anchor_raw - p["an_a0"],
        p["an_cubic"] * np.power(s.anchor_raw - p["an_a1"], 3.0))
    inc = np.minimum(inc * anchor_mult,
                     np.maximum(inc, inc * float(p["p_sat_b"]) - float(p["p_sat_a"])))
    if p["p_burst3"]:
        inc = inc + p["p_burst3"] * _wcount(s, s.note_impulse, 100.0) * 10.0
    if p["p_burst4"]:
        inc = inc + p["p_burst4"] * _wcount(s, s.note_impulse, 150.0) * 6.6667
    Pbar = _wmean(s, inc, float(p["w_p"]))

    # ================================================================== Abar
    A = np.ones(n)
    dk = s.dcol
    act = s.act
    # abar_active=0 pairs frame-index-adjacent columns (what v0.5.0delta's
    # shipped code does); abar_active=1 pairs adjacent *ACTIVE* columns,
    # which for 4K-embedded charts additionally spans the thumb column.
    if p["abar_active"] >= 0.5:
        prev_d = np.full(n, -1.0)
        prev_a = np.full(n, -1.0)
        for k in range(K):
            d0, a0 = prev_d, prev_a
            on = (a0 >= 0.0) & act[k]
            d1 = dk[k]
            mx = np.maximum(np.where(on, d0, 0.0), d1)
            dd = np.abs(np.where(on, d0, d1) - d1) + p["abar_mx_w"] * np.maximum(
                0.0, mx - p["abar_mx_thr"])
            val = _abar_val(dd, mx, p)
            A *= np.where(on, val, 1.0)
            prev_d = np.where(act[k], d1, d0)
            prev_a = np.where(act[k], 1.0, a0)
    else:
        for k in range(K - 1):
            on = act[k] & act[k + 1]
            d0, d1 = dk[k], dk[k + 1]
            mx = np.maximum(d0, d1)
            dd = np.abs(d0 - d1) + p["abar_mx_w"] * np.maximum(
                0.0, mx - p["abar_mx_thr"])
            val = _abar_val(dd, mx, p)
            A *= np.where(on, val, 1.0)
    Abar = _wmean(s, A, float(p["w_a"])) * p["abar_scale"]

    # ================================================================== Xbar
    x_x = np.power(xrow, float(p["od_mult_x"]))
    cw = np.array([p["x_cw0"], p["x_cw1"], p["x_cw2"], p["x_cw3"]])
    # inward/outward (内外切) direction split of the pairwise term.
    # d > 0 = the pair moves toward the thumb column (inward/内切);
    # chord-involved pairs contribute half.  0/0 = legacy symmetric Xbar.
    x_din = float(p.get("x_din", 0.0))
    x_dout = float(p.get("x_dout", 0.0))
    Xsum = np.zeros(n)
    fc_rows = []
    for bd in range(N_BOUND):
        c = s.bnd_chain[bd]
        if c["pos"].size < 2:
            fc_rows.append(np.zeros(n))
            continue
        if tau_c > 0.0:
            # merge boundary events closer than tau_c into single attacks
            # 注记：被拆成几毫秒多行的和弦
            # is one boundary event with the group's full span, not k
            # clamp-max events.
            pos = c["pos"]
            cid = s.chart[pos]
            bg = np.full(pos.size, np.inf)
            bg[1:] = np.where(cid[1:] == cid[:-1],
                              (s.tg[pos[1:]] - s.tg[pos[:-1]]) / 1000.0,
                              np.inf)
            keep = bg >= tau_c / 1000.0
            e_arr = c["e"][keep]
            pos = pos[keep]
            cid = cid[keep]
            g = np.full(max(pos.size - 1, 0), np.inf)
            same = np.zeros(pos.size, dtype=bool)
            if pos.size >= 2:
                g = np.where(cid[1:] == cid[:-1],
                             (s.tg[pos[1:]] - s.tg[pos[:-1]]) / 1000.0,
                             np.inf)
                same[:-1] = cid[:-1] == cid[1:]
        else:
            pos = c["pos"]
            g = c["g"]
            same = c["same"]
            e_arr = c["e"]
        g = np.where(np.isfinite(g), g, 1.0)
        gx = np.maximum(g, x_x[pos[:-1]])
        Xb = p["x_amp"] * np.power(gx, -2.0)
        if x_din != 0.0 or x_dout != 0.0:
            dd = np.abs(e_arr[:-1] - 3.0) - np.abs(e_arr[1:] - 3.0)
            Xb = Xb * (1.0 + x_din * np.maximum(dd, 0.0)
                       + x_dout * np.maximum(-dd, 0.0))
        Xsum += cw[BOUND_GROUP[bd]] * _chain_scatter(
            pos, np.where(same[:-1], Xb, 0.0), n, same[:-1])
        base = np.maximum(np.maximum(g, p["x_fc_floor"]), 0.75 * x_x[pos[:-1]])
        fc = np.maximum(p["x_fc_a"] * np.power(base, -2.0) - p["x_fc_off"], 0.0)
        fc_rows.append(_chain_scatter(pos,
                                      np.where(same[:-1], fc, 0.0),
                                      n, same[:-1]))
    for bd in range(N_BOUND - 1):
        w = np.sqrt(cw[BOUND_GROUP[bd]] * cw[BOUND_GROUP[bd + 1]])
        Xsum += p["x_fc_w"] * w * np.sqrt(np.maximum(fc_rows[bd] *
                                                     fc_rows[bd + 1], 0.0))

    # ---- geometric cross matrix (jump channel)
    if p["x_jump_w"]:
        Xsum = Xsum + p["x_jump_w"] * _jump_channel(b, s, p, x_x)
    Xbar = _wmean(s, Xsum, float(p["w_x"]))

    # ================================================================== Rbar
    Rbar = _rbar(b, s, p, xrow)

    # 同手和弦密度：单手（拇指列除外）同行达到两个音符的行。
    # Computed unconditionally (cheap): d_sh is a combine-level coefficient,
    # so gating the curve on it would freeze it inside the curve cache.
    d_sh = float(p.get("d_sh", 0.0))
    mask = s.mask.astype(np.int32)
    left = _POPCOUNT7[mask & 0b0000111]
    right = _POPCOUNT7[mask & 0b1110000]
    if tau_c > 0.0:
        # graded same-hand chord count (discrete_audit R5): triangular
        # +-tau_c window count of same-hand notes, then u*clip(u-1,0,1)
        # which equals the legacy (u>=2)*u at integer u and interpolates.
        # (SHd is a per-row VALUE, time-integrated by _wmean, so no attack
        # discount is needed -- sub-intervals already sum to the same span.)
        le = _tri_window_count(s, left.astype(np.float64), tau_c)
        ri = _tri_window_count(s, right.astype(np.float64), tau_c)
        sh = (le * np.clip(le - 1.0, 0.0, 1.0)
              + ri * np.clip(ri - 1.0, 0.0, 1.0))
    else:
        sh = (np.where(left >= 2, left, 0.0)
              + np.where(right >= 2, right, 0.0))
    SHd = _wmean(s, sh.astype(np.float64), max(float(p.get("shd_w", 1000.0)), 50.0))

    # ================================================================== Cbar
    Cbar = _cbar(b, s, p, xrow, stats, Jbar, Ja)
    CbarV2 = _cbar_v2(b, s, p, stats)
    Cbar = Cbar + CbarV2

    # ============================================================ hybrid/LN
    # 混合与长键机制。churn 与 holdage 并入 Cbar 曲线。
    # coordination phenomena and inherit the a_cb/(C+a_cbc) pricing);  chord2
    # and recov are separate curves consumed by combine's stream branch.
    Chord2 = np.zeros(n)
    Recov = np.zeros(n)
    if p.get("use_hb", 0.0):
        from .components import hb as _hb
        is4k_row = s.is_4k[s.chart]
        churn_c = float(p.get("cbv_churn", 0.0))
        churn_4k = float(p.get("cbv_churn_4k", 0.0))
        holdage_c = float(p.get("cbv_holdage", 0.0))
        holdage_4k = float(p.get("cbv_holdage_4k", 0.0))
        if churn_c or churn_4k:
            # 0 = inherit the 7K coefficient (per-frame LN physics)
            cw = np.where(is4k_row, churn_4k if churn_4k != 0.0 else churn_c,
                          churn_c)
            Cbar = Cbar + cw * _hb.compute_churn(b, s, p)
        if holdage_c or holdage_4k:
            hw = np.where(is4k_row,
                          holdage_4k if holdage_4k != 0.0 else holdage_c,
                          holdage_c)
            Cbar = Cbar + hw * _hb.compute_holdage(b, s, p)
        if p.get("c_chord2", 0.0):
            Chord2 = _hb.compute_chord2(b, s, p)
        if p.get("d_rec", 0.0):
            Recov = _hb.compute_recov(b, s, p)

    # ================================================================== eye
    # E4 节拍对齐曲线。形状与参数无关，只有
    # combine coefficients are tuned.  Gated: use_eye == 0 -> exact zero.
    EyeR = np.zeros(n)
    if p.get("use_eye", 0.0):
        from .components.eye import compute_eye_curve
        EyeR = compute_eye_curve(b, s, p)

    return dict(Jbar=Jbar, Jm=Jm, Jc=Jc, Ja=Ja, Pbar=Pbar, Xbar=Xbar,
                Abar=Abar, Rbar=Rbar, Cbar=Cbar, CbarV2=CbarV2,
                anchor=anchor_mult, EyeR=EyeR, Chord2=Chord2, Recov=Recov,
                SHd=SHd)


def _cbar_v2(b, s: Struct, p: dict, stats) -> np.ndarray:
    """Cbar v2 channel (core/components/lncoord_v2.py).

    Imported lazily: that module imports helpers from this one, so a
    top-level import here would be circular.  Default `use_cbar_v2 = 0`
    makes this return zeros and leave every number untouched.
    """
    if not p.get("use_cbar_v2", 0.0):
        return np.zeros(s.n)
    from .components.lncoord_v2 import compute_cbar_v2
    return compute_cbar_v2(b, s, p, stats)


# --------------------------------------------------------------- jump / X
def _jump_channel(b, s: Struct, p: dict, x_x: np.ndarray) -> np.ndarray:
    """Rate-weighted geometric transition cost between consecutive notes."""
    n = s.n
    # 7x7 weight matrix indexed by (from_col, to_col) -- small and static.
    d = np.abs(_IDX7[:, None] - _IDX7[None, :])
    same_hand = (HAND[:, None] == HAND[None, :]) & (HAND[:, None] != 1)
    thumb = (HAND[:, None] == 1) | (HAND[None, :] == 1)
    cross = (HAND[:, None] != HAND[None, :]) & ~thumb
    W = np.ones((K, K))
    W += p["x_jd_p"] * np.power(np.maximum(d, 1e-9), p["x_jd_e"])
    W = np.where(same_hand & (d > 0),
                 1.0 + p["x_jh_p"] * np.power(np.maximum(d, 1e-9),
                                              -p["x_jh_e"]), W)
    W = np.where(thumb, 1.0 - p["x_jt"] / np.maximum(d, 1.0), W)
    W = np.where(cross, 1.0 - p["x_jh_p"] * np.minimum(d / K, 1.0), W)
    # direction: inward = the *to* note is closer to the thumb column 3
    din = (np.abs(_IDX7[None, :] - 3) < np.abs(_IDX7[:, None] - 3)).astype(np.float64)
    W = W * (1.0 + p["x_dir_in"] * din + p["x_dir_out"] * (1.0 - din))

    # transitions between consecutive notes (global note order)
    a = b.note_col[:-1]
    bb = b.note_col[1:]
    ok = (b.note_chart[:-1] == b.note_chart[1:])
    w = np.where(ok, W[a, bb] - 1.0, 0.0)
    rows = b.note_row[1:] + np.repeat(s.starts[:-1], b.n_notes)[1:]
    step = np.zeros(n)
    np.add.at(step, rows, w)
    counts = np.zeros(n)
    np.add.at(counts, rows, np.where(ok, 1.0, 0.0))
    mean_w = step / np.maximum(counts, 1.0)
    return mean_w / s.dt


def _row_of_time(s: Struct, t: dict, times: np.ndarray,
                 sel: np.ndarray | None = None) -> np.ndarray:
    """Row index of `times` clipped to the owning chart's row range."""
    if sel is None:
        lo, hi = s._tail_lo, s._tail_hi
        ch = t["chart"]
    else:
        lo, hi = s._tail_lo[sel], s._tail_hi[sel]
        ch = t["chart"][sel]
    z = times + ch.astype(np.int64) * (1 << 30)
    return np.clip(np.searchsorted(s.tg, z, side="left"), lo, hi)


# --------------------------------------------------------------------- Rbar
def _coord_weight(c1, c2, p):
    """delta's coordination weight: same col / same hand / thumb / cross."""
    cw = np.full(c1.shape, p["r_cw_cross"], dtype=np.float64)
    cw = np.where(c1 == c2, p["r_cw_same"], cw)
    same_hand = (HAND[c1] == HAND[c2]) & (HAND[c1] != 1)
    cw = np.where(same_hand, p["r_cw_hand"], cw)
    thumb = (HAND[c1] == 1) | (HAND[c2] == 1)
    return np.where(thumb, p["r_cw_thumb"], cw)


def _rbar(b, s: Struct, p: dict, xrow: np.ndarray) -> np.ndarray:
    n = s.n
    t = s.tail
    if t is None:
        return np.zeros(n)
    xr = np.power(xrow, float(p["od_mult_r"]))
    xs = xr[t["j0"]]
    dur = t["t"] - t["h"]
    I_h = 0.001 * np.abs(dur - 80.0) / xs
    nh = t["nh"]
    I_t = np.where(np.isfinite(nh), 0.001 * np.abs(nh - t["t"] - 80.0) / xs, 10.0)
    I = 2.0 / (2.0 + np.exp(-p["r_I_steep"] * (I_h - p["r_I_off"]))
               + np.exp(-p["r_I_steep"] * (I_t - p["r_I_off"])))

    R = np.zeros(n)
    # 平滑开关，默认全部保持旧行为：
    #   r_dtr_min   physical floor for the dtr^-0.5 gap pricing (B4)
    #   r_soft_edges raised-cosine cutoff ramps + time-floored step
    #               intervals for sub-floor gaps (B2/B3/B5)
    #   r_sim_tau   blend zone for the tail/head coord-exponent pick (B1)
    dtr_min = p.get("r_dtr_min", 1e-4)
    soft = p.get("r_soft_edges", 0.0)
    # interval width floor (ms): a release priced at dtr_eff = max(dtr,
    # dtr_min) is spread over at least dtr_eff of time, so a simultaneous
    # (zero-width) release pair contributes the same mass as a sub-floor
    # one instead of being silently dropped -- and a 1 ms nudge no longer
    # creates mass out of nothing.  0 = legacy (max(nxt, t) == nxt).
    wfloor = dtr_min * 1000.0 if soft else 0.0
    sim_tau = p.get("r_sim_tau", 0.0)
    # ---------- A. per-tail release
    nxt = t["nxt"]
    nxt_is_tail = t["is_tail"]
    nxt_col = t["nxt_col"]
    ok = np.isfinite(nxt) & ((nxt - t["t"]) <= p["r_dt_max"])
    if ok.any():
        dtr = np.where(ok, (nxt - t["t"]) / 1000.0, 1.0)
        rv = p["r_tail"] * np.power(np.maximum(dtr, dtr_min), -0.5) / xs * (1.0 + p["r_I_w"] * I)
        same_col = (nxt_col == t["col"]) & ~nxt_is_tail
        rv = rv * np.where(same_col, p["r_same_col"], 1.0)
        # v0.5.0delta uses 1 + (cw-1)*e*f_tt, which goes NEGATIVE for
        # cross-hand releases.  We keep the ordering (cross-hand releases are
        # easy) but express it as a positive power so the channel stays a
        # difficulty, not an anti-difficulty.
        cw = _coord_weight(t["col"], nxt_col, p)
        if sim_tau > 0.0:
            # near-simultaneous next events: blend the tail/head coord
            # exponents linearly over +-sim_tau ms instead of the 1e-9
            # hard is_tail pick.  d_ev < 0 = the next tail comes first.
            with np.errstate(invalid="ignore"):
                d_ev = t["nt"] - nh      # inf - inf when neither exists
            beta = np.clip(0.5 - d_ev / (2.0 * sim_tau), 0.0, 1.0)
            beta = np.where(np.isfinite(d_ev), beta,
                            np.where(nxt_is_tail, 1.0, 0.0))
            ex = p["r_coord_e"] * (beta + (1.0 - beta) * p["r_tt"])
        else:
            ex = p["r_coord_e"] * np.where(nxt_is_tail, 1.0, p["r_tt"])
        rv = rv * np.power(np.maximum(cw, 1e-3), ex)
        # short-LN reduction
        thr = max(p["r_short_thr"], 1.0)
        red = p["r_short_red"] + (1.0 - p["r_short_red"]) * np.minimum(dur, thr) / thr
        rv = rv * np.where(dur > 0, red, 1.0)
        # lock terms
        lock = np.zeros(t["col"].size)
        hm = s.heldmask[t["j0"]]
        col = t["col"]
        # 锁手加成
        if p["r_lock"]:
            for j in range(K):
                bit = ((hm >> j) & 1).astype(bool) & (j != col)
                if bit.any():
                    lock = lock + np.where(bit, _coord_weight(col, np.full(
                        col.size, j), p), 0.0)
        rv = rv * (1.0 + p["r_lock"] * lock)
        # 新增：跨指独立性的锁手矩阵
        if p["r_straddle"]:
            ain = np.array([p["r_ain0"], p["r_ain1"], p["r_ain2"], p["r_ain3"]])
            ag = AIN_GROUP
            S0 = np.zeros(t["col"].size)
            for j in range(K):
                bit = ((hm >> j) & 1).astype(bool)
                if bit.any():
                    S0 = S0 + np.where(bit, ain[ag[j]], 0.0)
            S0 = S0 - np.where(((hm >> col) & 1).astype(bool), ain[ag[col]], 0.0)
            rv = rv * (1.0 + p["r_straddle"] * S0)
        if soft:
            # raised-cosine fade over the outer 20% before r_dt_max
            # instead of the hard cutoff (B3)
            rv = rv * _edge_ramp(nxt - t["t"], p["r_dt_max"])
        rv = np.where(ok, rv, 0.0)
        end_t = np.where(ok, np.maximum(t["nxt"], t["t"] + wfloor), t["t"])
        j1 = _row_of_time(s, t, end_t)
        if soft:
            # mass-exact step scaling (B2): the step holds over the grid
            # span [t[j0], t[j1]), which on a sparse grid far exceeds the
            # intended span (row-ceil overextension); scale the value so
            # the time-integral matches the intended span on any grid.
            span_i = np.maximum(end_t - t["t"], 0.0)
            j1c = np.minimum(np.maximum(j1, t["j0"]), n - 1)
            grid_span = s.t[j1c] - s.t[t["j0"]]
            den = np.maximum(grid_span, np.maximum(span_i, 1e-9))
            scale = np.where(span_i > 0.0, span_i / den, 1.0)
            rv = rv * scale
        R = R + _step_scatter(t["j0"], np.maximum(j1, t["j0"]), rv, n)

    # ---------- B. tail-to-tail sequence
    m = t["col"].size - 1
    if m > 0:
        idx = np.arange(m)
        idx = idx[t["chart"][idx] == t["chart"][idx + 1]]
        ch = t["chart"][idx]
        dtr = (t["t"][idx + 1] - t["t"][idx]) / 1000.0
        dtr = np.maximum(dtr, dtr_min)
        cw = _coord_weight(t["col"][idx], t["col"][idx + 1], p)
        sv = (p["r_seq"] * np.power(dtr, -0.5) / xs[idx]
              * (1.0 + p["r_I_w"] * (I[idx] + I[idx + 1]))
              * np.power(np.maximum(cw, 1e-3), p["r_coord_e"]))
        j0 = t["j0"][idx]
        end_B = np.maximum(t["t"][idx + 1], t["t"][idx] + wfloor)
        j1 = np.maximum(_row_of_time(s, t, end_B, idx + 1), j0)
        if soft:
            span_i = np.maximum(end_B - t["t"][idx], 0.0)
            j1c = np.minimum(j1, n - 1)
            grid_span = s.t[j1c] - s.t[j0]
            sv = sv * (span_i / np.maximum(grid_span, span_i))
        R = R + _step_scatter(j0, j1, sv, n)

    # ---------- C. release-order triple (102/120 harder than 012/210)
    if p["r_order_pen"]:
        m = t["col"].size - 2
        if m > 0:
            i = np.arange(m)
            ch = t["chart"][i]
            ok3 = (ch == t["chart"][i + 2])
            span = t["t"][i + 2] - t["t"][i]
            if not soft:
                ok3 &= span <= p["r_order_tau"]
            c1, c2, c3 = t["col"][i], t["col"][i + 1], t["col"][i + 2]
            hh = HAND[c1]
            ok3 &= (hh == HAND[c2]) & (hh == HAND[c3]) & (hh != 1)
            ok3 &= (c1 != c2) & (c2 != c3) & (c1 != c3)
            jump = np.abs(c1.astype(np.int64) - c2) + np.abs(c2.astype(np.int64) - c3)
            lo = np.minimum(np.minimum(c1, c2), c3).astype(np.int64)
            hi_ = np.maximum(np.maximum(c1, c2), c3).astype(np.int64)
            excess = np.maximum(jump - (hi_ - lo), 0.0).astype(np.float64)
            val = np.where(ok3, p["r_order_pen"] * excess / xs[i], 0.0)
            if soft:
                # raised-cosine fade over the outer 20% of the span
                # window instead of the hard cutoff (B5)
                val = val * _edge_ramp(span, p["r_order_tau"])
            j0 = t["j0"][i]
            end_C = np.maximum(t["t"][i + 2], t["t"][i] + wfloor)
            j1 = np.maximum(_row_of_time(s, t, end_C, i + 2), j0)
            if soft:
                span_i = np.maximum(end_C - t["t"][i], 0.0)
                j1c = np.minimum(j1, n - 1)
                grid_span = s.t[j1c] - s.t[j0]
                val = val * (span_i / np.maximum(grid_span, span_i))
            R = R + _step_scatter(j0, j1, val, n)

    return np.maximum(_wmean(s, R, float(p["w_r"])), 0.0)


# --------------------------------------------------------------------- Cbar
def _cbar(b, s: Struct, p: dict, xrow: np.ndarray, stats, Jbar, Ja) -> np.ndarray:
    n = s.n
    out = np.zeros(n)
    held = s.held
    any_ln = b.ln_end.max() > 0 if n else False
    if not any_ln:
        return out

    # ---- shield: same-column note shortly before an LN head
    if p["cb_shield"]:
        tau = max(p["cb_shield_tau"], 1.0)
        sh = np.zeros(n)
        for k in range(K):
            c = s.col_chain[k]
            pos = c["pos"]
            if pos.size < 3:
                continue
            li = np.flatnonzero(s.is_ln[pos])
            li = li[(li >= 1) & (li < pos.size)]
            if li.size == 0:
                continue
            ip = li[:, None] - np.arange(1, SHIELD_LOOKBACK + 1)[None, :]
            okm = ip >= 0
            idx = np.clip(ip, 0, pos.size - 1)
            dtp = c["tc"][li][:, None] - c["tc"][idx]
            okm &= (dtp > 0) & (dtp < 500.0)
            okm &= (c["cid"][idx] == c["cid"][li][:, None])
            contrib = np.where(okm, np.exp(-np.maximum(dtp, 0.0) / tau), 0.0)
            lock = np.zeros(li.size)
            hm = s.heldmask[pos[li]]
            col = k
            if p["cb_shield_lock"]:
                ain = np.array([p["r_ain0"], p["r_ain1"], p["r_ain2"], p["r_ain3"]])
                ag = AIN_GROUP
                S0 = np.zeros(li.size)
                for j in range(K):
                    bit = ((hm >> j) & 1).astype(bool) & (j != col)
                    if bit.any():
                        S0 = S0 + np.where(bit, ain[ag[j]], 0.0)
                lock = p["cb_shield_lock"] * S0
            val = contrib.sum(axis=1) * (1.0 + lock)
            rows = pos[li]
            j1 = np.minimum(rows + 1, n)
            sh = sh + _step_scatter(rows, j1, val, n)
        out = out + p["cb_shield"] * _wmean(s, sh, float(p["w_cb"]))

    # ---- locked jack / anchor
    hp = np.power(np.maximum(held, 0.0), float(p["cb_lock_pow"]))
    if p["cb_lockjack"]:
        out = out + p["cb_lockjack"] * Jbar * hp
    if p["cb_lockanchor"]:
        out = out + p["cb_lockanchor"] * Ja * hp
    # ---- hold mass / hold-tap density
    if p["cb_par"]:
        out = out + p["cb_par"] * hp
    if p["cb_holdtap"]:
        out = out + p["cb_holdtap"] * (s.size / s.dt) * hp
    return np.maximum(out, 0.0)


# ------------------------------------------------------------------ combine
def combine(s: Struct, cur: dict, p: dict) -> np.ndarray:
    A = np.maximum(cur["Abar"], 1e-6)
    Ks = s.Ks
    C = np.maximum(s.C, 0.0)

    def branch(v, mult):
        return np.power(A, mult / Ks) * np.minimum(
            np.maximum(v, 0.0), p["cap_a"] + p["cap_b"] * np.maximum(v, 0.0))

    # Per-frame stream weights: 7K rows use a_p/a_r/a_cb; 4K-embedded rows
    # use the *_4k override when it is non-zero (0 = inherit, the legacy
    # single value).  7K and 4K diverge on how dense streams / LN / release
    # price out -- one shared coefficient is a forced compromise.
    is4k_row = s.is_4k[s.chart]
    a_p_4k = float(p.get("a_p_4k", 0.0))
    a_r_4k = float(p.get("a_r_4k", 0.0))
    a_cb_4k = float(p.get("a_cb_4k", 0.0))
    a_p_row = np.where(is4k_row, a_p_4k, p["a_p"]) if a_p_4k != 0.0 else p["a_p"]
    a_r_row = np.where(is4k_row, a_r_4k, p["a_r"]) if a_r_4k != 0.0 else p["a_r"]
    a_cb_row = (np.where(is4k_row, a_cb_4k, p["a_cb"]) if a_cb_4k != 0.0
                else p["a_cb"])

    Jm_b = branch(cur["Jm"], p["aj"])
    Jc_b = branch(cur["Jc"], p["aj"])
    Ja_b = branch(cur["Ja"], p["aj"])
    # 视觉规整度：流分支内的密度门控规整度交互。
    # 按键数分别取值（7 键与 4 键各一个系数）。
    d_eye = float(p.get("d_eye", 0.0))
    d_eye_4k = float(p.get("d_eye_4k", 0.0))
    eye_term = 0.0
    if p.get("use_eye", 0.0) and (d_eye != 0.0 or d_eye_4k != 0.0):
        d_row = np.where(is4k_row, d_eye_4k, d_eye) if d_eye_4k != 0.0 else d_eye
        eye_term = d_row * np.maximum(cur["Pbar"], 0.0) * cur["EyeR"]
    stream = np.power(A, p["ap"]) * (
        a_p_row * np.maximum(cur["Pbar"], 0.0)
        + a_r_row * np.maximum(cur["Rbar"], 0.0) / (C + p["a_c"])
        + a_cb_row * np.maximum(cur["Cbar"], 0.0) / (C + p["a_cbc"])
        + float(p.get("d_sh", 0.0)) * cur.get("SHd", 0.0)
        + eye_term)

    # hybrid/LN: chord2 density into stream; recovery burst discount.
    if p.get("use_hb", 0.0):
        c_chord2 = float(p.get("c_chord2", 0.0))
        if c_chord2 != 0.0:
            stream = stream + c_chord2 * cur["Chord2"]
        d_rec = float(p.get("d_rec", 0.0))
        if d_rec != 0.0:
            rec = np.maximum(cur["Recov"], 0.0)
            half = max(float(p.get("rec_half", 0.5)), 1e-3)
            stream = np.maximum(stream - d_rec * rec / (rec + half), 0.0)

    sp = float(p["s_p"])
    if abs(sp - 1.0) < 1e-6:
        S = (p["c_jm"] * Jm_b + p["c_jc"] * Jc_b + p["c_ja"] * Ja_b
             + p["c_s"] * stream)
    else:
        inner = (p["c_jm"] * np.power(Jm_b, sp) + p["c_jc"] * np.power(Jc_b, sp)
                 + p["c_ja"] * np.power(Ja_b, sp)
                 + p["c_s"] * np.power(np.maximum(stream, 0.0), sp))
        S = np.power(np.maximum(inner, 0.0), 1.0 / sp)
    S = np.maximum(S, 0.0)

    Xt = np.maximum(cur["Xbar"], 0.0) + p["t_jack_mix"] * Jm_b
    T = np.power(A, p["at"] / Ks) * Xt / (Xt + S + p["t_s_off"])
    D = (p["d_b1"] * np.power(S, p["d_ds"]) * np.power(np.maximum(T, 1e-9), p["d_dt"])
         + p["d_b2"] * S)

    # ---- 读谱修正
    if p.get("use_read", 0.0):
        from .components.read import compute_read, compute_read_gate
        read_curve = compute_read(s.batch, s, p)
        gate = compute_read_gate(s.batch, s, p)
        dr = p.get("d_read", 0.0)
        dr_4k = p.get("d_read_4k", 0.0)
        # per-frame: dr for 7K, dr_4k for 4K
        dr_eff = dr + (dr_4k - dr) * gate
        if np.any(dr_eff != 0.0):
            modifier = np.clip(1.0 + dr_eff * read_curve,
                               p.get("read_clip_lo", 0.5),
                               p.get("read_clip_hi", 1.5))
            D = D * modifier

    return np.maximum(np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


# ---------------------------------------------------------------- aggregate
def _percentiles(s: Struct, D: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-chart weighted percentile aggregate (Sunny style)."""
    n = s.n
    order = np.lexsort((D, s.chart))
    Ds = D[order]
    ws = w[order]
    cum = seg_cumsum(ws, s.starts)
    tot = np.repeat(cum[s.starts[1:] - 1], s.counts)
    f = cum / np.maximum(tot, EPS)
    out = np.zeros(s.n_charts)
    for base, targs in ((0.93, (0.945, 0.935, 0.925, 0.915)),
                        (0.83, (0.845, 0.835, 0.825, 0.815))):
        acc = np.zeros(s.n_charts)
        for t in targs:
            # first index in each chart where f >= t
            hit = (f >= t)
            # position of first True per chart via segmented trick
            idx = np.zeros(s.n_charts, dtype=np.int64)
            flat = np.flatnonzero(hit)
            cids = s.chart[flat]
            first_mask = np.ones(flat.size, dtype=bool)
            first_mask[1:] = cids[1:] != cids[:-1]
            idx[cids[first_mask]] = flat[first_mask] - s.starts[:-1][cids[first_mask]]
            # guard: charts with no hit -> last row
            acc += Ds[s.starts[:-1] + np.clip(idx, 0, s.counts - 1)]
        out += (base == 0.93 and 0.25 or 0.20) * (base == 0.93 and 0.88 or 0.94) * acc / 4.0
    # weighted power mean
    wsum = np.zeros(s.n_charts)
    np.add.at(wsum, s.chart, np.power(Ds, 5.0) * ws)
    wden = np.zeros(s.n_charts)
    np.add.at(wden, s.chart, ws)
    out += 0.55 * np.power(wsum / np.maximum(wden, EPS), 0.2)
    return out


def _peak_envelope(s: Struct, D: np.ndarray, ms: float) -> np.ndarray:
    """Rolling-max envelope of D over a +-ms/2 time window, per chart.

    The power-mean aggregate dilutes narrow difficulty peaks into the chart's
    而标注奖励峰值高度。展宽
    peaks into ~ms-wide plateaus restores their time share in any weighted
    mean.  The window is approximated in rows via each chart's median row
    spacing, which keeps this a C-speed per-chart maximum_filter1d.
    """
    from scipy.ndimage import maximum_filter1d
    out = np.empty_like(D)
    for c in range(s.n_charts):
        a, z = int(s.starts[c]), int(s.starts[c + 1])
        v = D[a:z]
        m = v.size
        if m < 3:
            out[a:z] = v
            continue
        dur = max(float(s.duration[c]), 1.0)
        med_dt = max(dur / m, 1.0)
        half = max(1, int(round(0.5 * ms / med_dt)))
        out[a:z] = maximum_filter1d(v, size=2 * half + 1, mode="nearest")
    return out


def aggregate(s: Struct, D: np.ndarray, p: dict,
              hold_w: bool = False) -> np.ndarray:
    """Per-chart SR from the difficulty curve D(t).

    agg_mode: 0 = Sunny percentiles, 1 = soft-max (exponential mean),
              2 = sigmoid accuracy model, 3 = weighted power mean.
    """
    n = s.n
    # 保留峰值的包络。加权平均会稀释窄峰，
    # peaks; the labels reward peak height).  env_w = 0 disables.
    env_w = float(p.get("env_w", 0.0))
    if env_w != 0.0:
        E = _peak_envelope(s, D, max(float(p.get("env_ms", 800.0)), 50.0))
        D = (1.0 - env_w) * D + env_w * E
    dnext = np.zeros(n)
    dnext[:-1] = s.dt_diff
    dprev = np.zeros(n)
    dprev[1:] = dnext[:-1]
    dprev[s.starts[:-1]] = 0.0
    gp = np.maximum(0.5 * (dnext + dprev), 0.0)
    w = np.power(gp, float(p["agg_gap_w"])) * \
        np.power(np.maximum(s.C, 0.0), float(p["agg_wc"]))
    if hold_w and p.get("hw_a", 0.0):
        h = s.held
        w = w * (1.0 + p["hw_a"] * h / (h + max(p.get("hw_half", 1.0), 1e-6)))
    w = np.maximum(np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    D = np.maximum(np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    mode = int(p["agg_mode"])
    if mode == 0:
        return _percentiles(s, D, w, p)
    if mode == 1:
        lam = max(float(p["agg_k"]), 1e-3)
        m = D.max() if n else 0.0
        sh = np.zeros(s.n_charts)
        np.add.at(sh, s.chart, w * np.exp(np.clip(lam * (D - m), -50, 50)))
        sw = np.zeros(s.n_charts)
        np.add.at(sw, s.chart, w)
        return m + np.log(np.maximum(sh, EPS) / np.maximum(sw, EPS)) / lam
    if mode == 3:
        pw = max(float(p["agg_k"]), 0.05)
        acc = np.zeros(s.n_charts)
        np.add.at(acc, s.chart, w * np.power(D, pw))
        sw = np.zeros(s.n_charts)
        np.add.at(sw, s.chart, w)
        return np.power(acc / np.maximum(sw, EPS), 1.0 / pw)
    return _sigmoid_agg(s, D, w, p)


def _percentiles(s: Struct, D: np.ndarray, w: np.ndarray, p: dict) -> np.ndarray:
    order = np.lexsort((D, s.chart))
    Ds = D[order]
    ws = w[order]
    cum = seg_cumsum(ws, s.starts)
    tot = np.maximum(np.repeat(cum[s.starts[1:] - 1], s.counts), EPS)
    f = cum / tot
    out = np.zeros(s.n_charts)
    for lo, hi, coef, scale in ((0.915, 0.945, p["p93_c"], p["w93"]),
                                (0.815, 0.845, p["p83_c"], p["w83"])):
        acc = np.zeros(s.n_charts)
        for t in np.linspace(lo, hi, 4):
            acc += _value_at_frac(s, Ds, f, float(t))
        out += scale * coef * acc / 4.0
    wsum = np.zeros(s.n_charts)
    np.add.at(wsum, s.chart, np.power(Ds, p["mean_pow"]) * ws)
    wden = np.zeros(s.n_charts)
    np.add.at(wden, s.chart, ws)
    out += p["wmean"] * np.power(wsum / np.maximum(wden, EPS),
                                 1.0 / max(p["mean_pow"], 1e-6))
    return out


def _value_at_frac(s: Struct, Ds: np.ndarray, f: np.ndarray,
                   t: float) -> np.ndarray:
    """First sorted-D value whose cumulative weight fraction reaches t."""
    hit = f >= t
    flat = np.flatnonzero(hit)
    out = np.zeros(s.n_charts)
    if flat.size == 0:
        return out
    cids = s.chart[flat]
    first = np.ones(flat.size, dtype=bool)
    first[1:] = cids[1:] != cids[:-1]
    sel = flat[first]
    out[cids[first]] = Ds[sel]
    return out


def _sigmoid_agg(s: Struct, D: np.ndarray, w: np.ndarray, p: dict) -> np.ndarray:
    nseg = int(max(p["agg_nseg"], 2))
    order = np.lexsort((D, s.chart))
    Ds = D[order]
    ws = w[order]
    cum = seg_cumsum(ws, s.starts)
    tot = np.maximum(np.repeat(cum[s.starts[1:] - 1], s.counts), EPS)
    f = np.minimum(cum / tot, 1.0 - 1e-12)
    binid = np.minimum((f * nseg).astype(np.int64), nseg - 1)
    key = s.chart[order] * nseg + binid
    size = s.n_charts * nseg
    sd = np.bincount(key, weights=Ds * ws, minlength=size)
    sw = np.bincount(key, weights=ws, minlength=size)
    sd = sd.reshape(s.n_charts, nseg)
    sw = sw.reshape(s.n_charts, nseg)
    dseg = sd / np.maximum(sw, EPS)

    k = max(float(p["agg_k"]), 1e-3)
    C0 = float(p["agg_C"])
    gamma = float(min(max(p["agg_gamma"], 1e-3), 0.999))
    W = sw.sum(axis=1)
    target = W * gamma
    lo = dseg.min(axis=1) - 5.0
    hi = dseg.max(axis=1) + 5.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        e = np.clip(k * (dseg - mid[:, None]), -50, 50)
        val = (sw / (C0 + np.exp(e))).sum(axis=1)
        greater = val > target
        lo = np.where(greater, mid, lo)
        hi = np.where(greater, hi, mid)
    return 0.5 * (lo + hi)


def postprocess(s: Struct, sr: np.ndarray, p: dict) -> np.ndarray:
    n_eff = s.n_notes.astype(np.float64)
    sr = sr * n_eff / (n_eff + max(p["pp_n0"], 1e-6))
    thr = p["pp_thr"]
    div = max(p["pp_div"], 1e-3)
    sr = np.where(sr > thr, thr + (sr - thr) / div, sr)
    return np.maximum(sr * p["pp_scale"] * p["calib_a"] + p["calib_b"], 0.0)


# ------------------------------------------------------------------- driver
def evaluate_batch(b, s: Struct, p: dict) -> np.ndarray:
    cur = compute_curves(b, s, p)
    D = combine(s, cur, p)
    sr = aggregate(s, D, p, hold_w=True)
    return postprocess(s, sr, p)
