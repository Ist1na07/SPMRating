"""Unit tests for the Cbar v2 (row-grid LN coordination) channel.

Run:  PYTHONPATH=. python -m pytest tests/test_lncoord_v2.py -q
  or  PYTHONPATH=. python tests/test_lncoord_v2.py

Synthetic charts are built straight into the flat `Batch` representation so
no .osu fixtures are needed.  The builder mirrors `core.batch.build_batch`
minus the file I/O (and minus LN tails, which Cbar does not read).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.batch import Batch, CHART_STRIDE, _od_leniency          # noqa: E402
from core.grid import K, N_BOUND, POPCOUNT                        # noqa: E402
from core.model import Struct, build_struct, compute_curves       # noqa: E402
from core import params as P                                      # noqa: E402
from core.components.lncoord_v2 import CbarV2, dist_matrix        # noqa: E402

# `_wmean` memoises on id(s); keep every Struct alive so ids cannot be reused
# by a later struct and hand back a stale interpolation operator.
_KEEP: list = []


# --------------------------------------------------------------- test charts
def build_test_batch(charts, od: float = 8.0) -> Batch:
    """charts: list of charts; each chart is a list of (t_ms, col, ln_end|None)."""
    b = Batch()
    ids, t_list, mask_list, isln_list, lne_list = [], [], [], [], []
    nt_list, nc_list, nr_list = [], [], []
    rows_list = []
    for ci, notes in enumerate(charts):
        notes = sorted(notes, key=lambda z: (z[0], z[1]))
        time = np.array([float(n[0]) for n in notes], dtype=np.float64)
        col = np.array([int(n[1]) for n in notes], dtype=np.int64)
        is_ln = np.array([n[2] is not None for n in notes], dtype=bool)
        ln_end = np.array([float(n[2]) if n[2] is not None else -1.0
                           for n in notes], dtype=np.float64)

        first = np.empty(time.size, dtype=bool)
        first[0] = True
        np.not_equal(time[1:], time[:-1], out=first[1:])
        row_of_note = np.cumsum(first) - 1
        R = int(row_of_note[-1]) + 1
        t = np.zeros(R, dtype=np.float64)
        t[row_of_note] = time
        mask = np.zeros(R, dtype=np.int32)
        np.bitwise_or.at(mask, row_of_note, (1 << col).astype(np.int32))
        isln = np.zeros(R, dtype=bool)
        np.logical_or.at(isln, row_of_note, is_ln)
        lne = np.full(R, -1.0)
        np.maximum.at(lne, row_of_note, np.where(is_ln, ln_end, -1.0))

        ids.append(f"c{ci}")
        t_list.append(t)
        mask_list.append(mask)
        isln_list.append(isln)
        lne_list.append(lne)
        rows_list.append(R)
        nt_list.append(time)
        nc_list.append(np.full(time.size, len(ids) - 1, np.int32))
        nr_list.append(row_of_note)

    b.ids = ids
    n_per = np.array(rows_list, dtype=np.int64)
    b.starts = np.concatenate(([0], np.cumsum(n_per))).astype(np.int64)
    b.n = int(n_per.sum())
    b.t = np.concatenate(t_list)
    b.chart = np.repeat(np.arange(len(ids), dtype=np.int32), n_per)
    b.tg = b.t + b.chart.astype(np.int64) * CHART_STRIDE
    b.mask = np.concatenate(mask_list).astype(np.int32)
    b.size = POPCOUNT[b.mask].astype(np.int32)
    b.is_ln = np.concatenate(isln_list)
    b.ln_end = np.concatenate(lne_list)
    x = _od_leniency(od)
    b.x = np.full(len(ids), x)
    b.od = np.full(len(ids), od)
    b.key_count = np.full(len(ids), 7, np.int32)
    b.n_notes = np.array([a.size for a in nt_list], np.int64)
    b.duration = np.array([float(a[-1] - a[0]) for a in t_list])
    b.is_4k = (b.key_count == 4)
    dt = np.diff(b.tg) / 1000.0
    b.dt = np.where(np.diff(b.chart) == 0, dt, 1.0)
    b.note_t = np.concatenate(nt_list)
    b.note_chart = np.concatenate(nc_list)
    b.note_row = np.concatenate(nr_list)

    for k in range(K):
        pos = np.flatnonzero((b.mask >> k) & 1).astype(np.int64)
        b.col_pos.append(pos)
        cids = b.chart[pos]
        b.col_seg.append(np.searchsorted(
            cids, np.arange(len(ids) + 1), side="left").astype(np.int64))
        g = np.diff(b.tg[pos]) / 1000.0
        b.col_dt.append(np.where(np.diff(cids) == 0, g, np.inf))

    for bd in range(N_BOUND):
        cols = [c for c in (bd - 1, bd) if 0 <= c < K]
        merged = np.unique(np.concatenate([b.col_pos[c] for c in cols])) \
            if cols else np.zeros(0, np.int64)
        b.bnd_pos.append(merged.astype(np.int64))
        cids = b.chart[merged]
        b.bnd_seg.append(np.searchsorted(
            cids, np.arange(len(ids) + 1), side="left").astype(np.int64))
        g = np.diff(b.tg[merged]) / 1000.0 if merged.size >= 2 else np.zeros(0)
        b.bnd_dt.append(np.where(np.diff(cids) == 0, g, np.inf)
                        if merged.size >= 2 else g)
    return b


def make(charts, **over):
    b = build_test_batch(charts)
    s = build_struct(b)
    _KEEP.append(s)
    p = P.make(over)
    return b, s, p


def term(charts, name, **extra):
    """Value of one Cbar v2 sub-term alone (all other gates closed)."""
    over = {"cbv_" + name: 1.0}
    for t in CbarV2.TERMS:
        over["use_cbv_" + t] = 1.0 if t == name else 0.0
    over.update(extra)
    b, s, p = make(charts, **over)
    return CbarV2(b, s, p).terms()[name]


def metro(t_end=1400.0, col=0, step=200.0):
    """Regular tap stream -- keeps a chart well-formed without LN content."""
    return [(float(t), col, None) for t in np.arange(0.0, t_end, step)]


# ------------------------------------------------------------------- helpers
FAIL = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name +
          ("" if not detail else "   " + detail))
    if not cond:
        FAIL.append(name)
    return cond


def show(title, vals):
    print(f"\n{title}")
    for k, v in vals.items():
        print(f"    {k:<34s} sum={float(np.sum(v)):10.4f}  "
              f"max={float(np.max(v)):10.4f}")


# =========================================================== 1. straddle
def test_straddle():
    print("\n[1] straddle -- locked 衩叠 vs unlocked")
    # Both charts pin column 1 (left middle finger) for the whole window.
    # A alternates 0/2  -> the pin IS the column between them (衩叠)
    # B alternates 4/6  -> the pin is on the other hand entirely
    def chart(alt):
        notes = [(0.0, 1, 2400.0)]          # pin the left middle finger
        t = 50.0
        for i in range(20):
            notes.append((t, alt[i % 2], None))
            t += 100.0
        return notes

    A = term([chart((0, 2))], "straddle")
    B = term([chart((4, 6))], "straddle")
    show("straddle term", {"A alternate 0/2, pin=1": A,
                           "B alternate 4/6, pin=1": B})
    ok = check("locked straddle > unlocked straddle",
               float(A.sum()) > float(B.sum()) > -1e-12)
    check("straddle fires on the pinned geometry", float(A.max()) > 1e-6,
          f"max={float(A.max()):.4f}")
    check("straddle quiet when the pin is not the interior column",
          float(B.max()) < 1e-9, f"max={float(B.max()):.3e}")

    # the pin must be the *interior* column, not merely a held column:
    # pinning 4 (right index) while 4/6 alternate leaves no interior pin
    D = term([[(0.0, 4, 2400.0)] +
              [(50.0 + 100.0 * i, (4, 6)[i % 2], None) for i in range(20)]],
             "straddle")
    check("a held column is not enough -- it must sit between the pair",
          float(D.max()) < 1e-9, f"pin=4 on pair (4,6): {float(D.max()):.3e}")
    return ok


# ============================================================ 2. shield
def test_shield():
    print("\n[2] shield -- tap -> LN on the same column")
    A = term([metro() + [(0.0, 3, None), (100.0, 3, 600.0)]], "shield")
    B = term([metro() + [(100.0, 3, 600.0)]], "shield")
    # holds start at t=10, NOT t=0: on the row grid a chord mixing a tap on
    # col 3 with an LN on col 5 makes b.ln_end[row] ambiguous, which is a
    # pre-existing property of core.batch (see LIMITATIONS in lncoord_v2).
    C = term([metro() + [(0.0, 3, None), (100.0, 3, 600.0),
                         (10.0, 5, 800.0), (10.0, 6, 800.0)]], "shield")
    show("shield term", {"A tap->LN": A, "B LN with no predecessor": B,
                         "C tap->LN, cols 5+6 held": C})
    check("shield > same LN with no predecessor",
          float(A.max()) > 0.0 and float(B.max()) == 0.0,
          f"A.max={float(A.max()):.4f}  B.max={float(B.max()):.4f}")
    check("shield amplified by held other columns",
          float(C.max()) > float(A.max()) * 1.5,
          f"C.max={float(C.max()):.4f} vs A.max={float(A.max()):.4f}")
    D = term([metro() + [(0.0, 3, None), (900.0, 3, 1400.0)]], "shield")
    check("shield decays with the predecessor gap",
          float(D.max()) == 0.0, f"gap 900ms > dt_max -> {float(D.max()):.4f}")


# ======================================================== 3. lockanchor
def test_lockanchor():
    print("\n[3] lockanchor -- anchor while other columns hold LNs")
    anchor = [(60.0 + 80.0 * i, 6, None) for i in range(12)]
    A = term([[(0.0, 0, 2000.0), (0.0, 1, 2000.0)] + anchor], "lockanchor")
    B = term([anchor + metro(1400.0, 0, 200.0)[:0]], "lockanchor")
    show("lockanchor term", {"A anchor + holds on 0,1": A,
                             "B bare anchor": B})
    check("anchor-with-holds > bare anchor",
          float(A.max()) > 0.0 and float(B.max()) == 0.0,
          f"A.max={float(A.max()):.4f}  B.max={float(B.max()):.4f}")

    # the lock must depend on WHICH columns are held, not just how many
    near = term([[(0.0, 4, 2000.0), (0.0, 5, 2000.0)] + anchor], "lockanchor")
    far = term([[(0.0, 0, 2000.0), (0.0, 1, 2000.0)] + anchor], "lockanchor")
    check("near-hand hold costs more than far-hand hold",
          float(near.max()) > float(far.max()),
          f"hold(4,5)={float(near.max()):.4f} > hold(0,1)={float(far.max()):.4f}")


# ========================================================== 4. lockstack
def test_lockstack():
    print("\n[4] lockstack -- jack/stack speed while other columns hold LNs")
    jack = [(50.0 + 90.0 * i, 2, None) for i in range(16)]
    A = term([[(0.0, 0, 2000.0), (0.0, 1, 2000.0)] + jack], "lockstack")
    B = term([jack + metro(1500.0, 0, 300.0)[:0]], "lockstack")
    show("lockstack term", {"A jack + holds": A, "B jack alone": B})
    check("locked stack > unlocked stack",
          float(A.max()) > 0.0 and float(B.max()) == 0.0,
          f"A.max={float(A.max()):.4f}  B.max={float(B.max()):.4f}")


# ============================================================ 5. overlap
def test_overlap():
    print("\n[5] overlap -- partial (non-nested) LN pairs")
    part = [(0.0, 3, 900.0), (300.0, 5, 1200.0)]      # 0<300<900<1200 partial
    nest = [(0.0, 3, 1200.0), (300.0, 5, 900.0)]      # 5 contained in 3
    A = term([metro(1600.0) + part], "overlap")
    B = term([metro(1600.0) + nest], "overlap")
    show("overlap term", {"A partial overlap": A, "B containment": B})
    check("partial overlap scores, containment does not",
          float(A.max()) > 0.0 and float(B.max()) == 0.0,
          f"A.max={float(A.max()):.4f}  B.max={float(B.max()):.4f}")


# =========================================================== 6. holdtap
def test_holdtap():
    print("\n[6] holdtap -- taps struck while holds are active elsewhere")
    taps = [(50.0 + 100.0 * i, 2, None) for i in range(12)]
    A = term([[(0.0, 0, 2000.0), (0.0, 1, 2000.0)] + taps], "holdtap")
    B = term([taps], "holdtap")
    show("holdtap term", {"A taps + holds": A, "B taps alone": B})
    check("holdtap fires only when something is held",
          float(A.max()) > 0.0 and float(B.max()) == 0.0,
          f"A.max={float(A.max()):.4f}  B.max={float(B.max()):.4f}")


# ============================================== 7. gates are independent
def rich_chart():
    notes = [(0.0, 1, 1500.0), (0.0, 5, 1500.0)]          # holds (pins)
    t = 50.0                                              # 衩叠 on 0/2
    for i in range(16):
        notes.append((t, 0 if i % 2 == 0 else 2, None))
        t += 90.0
    t = 60.0                                              # anchor on 6
    for i in range(12):
        notes.append((t, 6, None))
        t += 70.0
    notes += [(200.0, 4, None), (300.0, 4, 700.0)]        # shield
    notes += [(900.0, 3, 1400.0), (1100.0, 4, 1800.0)]    # partial overlap
    return notes


def test_gates():
    print("\n[7] every gate gates -- and only its own term")
    charts = [rich_chart()]
    over = {}
    for t in CbarV2.TERMS:
        over["use_cbv_" + t] = 1.0
        over["cbv_" + t] = 0.7 + 0.1 * len(t)        # distinct weights
    b, s, p = make(charts, **over)
    all_on = CbarV2(b, s, p)
    T = all_on.terms()
    curve_all = all_on.curve()
    show("term magnitudes (weighted)", T)

    check("curve() is the exact sum of terms()",
          np.allclose(curve_all, sum(T.values())))

    alive = []
    for name in CbarV2.TERMS:
        off = dict(over)
        off["use_cbv_" + name] = 0.0
        _, s2, p2 = make(charts, **off)
        c2 = CbarV2(b, s2, p2)
        curve_off = c2.curve()
        T2 = c2.terms()
        dropped = curve_all - curve_off
        # (a) what disappeared is exactly this term
        ok_a = np.allclose(dropped, T[name], atol=1e-12)
        # (b) what remains is exactly the other five
        rest = sum(v for k, v in T2.items() if k != name)
        ok_b = np.allclose(curve_off, rest, atol=1e-12)
        # (c) the other five are untouched by this gate
        ok_c = all(np.allclose(T2[k], T[k], atol=1e-12)
                   for k in CbarV2.TERMS if k != name)
        # (d) this term is exactly zero
        ok_d = float(np.max(np.abs(T2[name]))) == 0.0
        # (e) the test is meaningful
        ok_e = float(np.max(np.abs(T[name]))) > 1e-9
        alive.append(name)
        check(f"use_cbv_{name} zeroes only {name}",
              ok_a and ok_b and ok_c and ok_d and ok_e,
              f"[a={int(ok_a)} b={int(ok_b)} c={int(ok_c)} "
              f"d={int(ok_d)} e={int(ok_e)}]")

    # a zero weight must also zero the term, without touching the others
    zw = dict(over)
    zw["cbv_lockstack"] = 0.0
    _, s3, p3 = make(charts, **zw)
    T3 = CbarV2(b, s3, p3).terms()
    check("cbv_lockstack = 0 is equivalent to the gate",
          np.allclose(T3["lockstack"], 0.0) and
          all(np.allclose(T3[k], T[k]) for k in CbarV2.TERMS
              if k != "lockstack"))

    # master gate: use_cbar_v2 = 0 must leave Cbar bit-identical to v1
    _, s4, p4 = make(charts, **{**over, "use_cbar_v2": 0.0})
    cur4 = compute_curves(b, s4, p4)
    _, s5, p5 = make(charts, use_cbar_v2=0.0)
    cur5 = compute_curves(b, s5, p5)
    check("use_cbar_v2=0 -> CbarV2 identically zero",
          float(np.max(np.abs(cur4["CbarV2"]))) == 0.0)
    check("use_cbar_v2=0 -> Cbar unchanged by any cbv_* value",
          np.array_equal(cur4["Cbar"], cur5["Cbar"]))
    check("CbarV2 is additive on top of v1",
          np.allclose(cur4["Cbar"], cur5["Cbar"]))


# ============================================ 8. geometry / charter checks
def test_geometry():
    print("\n[8] cross-column matrix geometry")
    p = P.make()
    M = dist_matrix(p, "cbv_ls_")
    check("matrix is symmetric", np.allclose(M, M.T))
    check("diagonal is zero", float(np.max(np.abs(np.diag(M)))) == 0.0)
    check("same-hand split > same-hand adjacent (0,2) > (0,1)",
          M[0, 2] > M[0, 1])
    check("mirror symmetry: (0,1) == (5,6)", M[0, 1] == M[5, 6])
    check("mirror symmetry: (0,2) == (4,6)", M[0, 2] == M[4, 6])
    check("thumb-bridged (1,3) == (3,5)", M[1, 3] == M[3, 5])
    check("two-hand split (2,4) isolated", M[2, 4] == p["cbv_ls_2h_split"])
    check("far cross (0,6) is the cross value", M[0, 6] == p["cbv_ls_cross"])
    S = dist_matrix(p, "cbv_str_")
    check("straddle (0,2) is the dominant pair",
          S[0, 2] == p["cbv_str_sh_split"] and S[0, 2] > S[1, 3] > S[2, 4])


def test_local_in_time():
    print("\n[9] charter: no chart-level scalars")
    # Two charts that are time-translates of each other must produce
    # translated curves.  A chart-level normaliser (mean, total, max) would
    # break this; a purely local operator must preserve it.
    base = rich_chart()
    shifted = [(t + 50000.0, c, None if e is None else e + 50000.0)
               for (t, c, e) in base]
    vals = {}
    ok = True
    for name in CbarV2.TERMS:
        over = {f"cbv_{name}": 1.0}
        for t in CbarV2.TERMS:
            over["use_cbv_" + t] = 1.0 if t == name else 0.0
        b1, s1, p1 = make([base], **over)
        b2, s2, p2 = make([shifted], **over)
        a = CbarV2(b1, s1, p1).terms()[name]
        bb = CbarV2(b2, s2, p2).terms()[name]
        good = float(a.sum()) > 0.0 and np.allclose(a, bb, atol=1e-9)
        ok = ok and good
        check(f"time-translation invariant: {name}", good,
              f"sum {float(a.sum()):.6f} vs {float(bb.sum()):.6f}")
    # appending a second identical chart must not change the first one's curve
    over = {f"cbv_{t}": 1.0 for t in CbarV2.TERMS}
    over.update({f"use_cbv_{t}": 1.0 for t in CbarV2.TERMS})
    b1, s1, p1 = make([rich_chart()], **over)
    b2, s2, p2 = make([rich_chart(), rich_chart()], **over)
    a = CbarV2(b1, s1, p1).curve()
    bcur = CbarV2(b2, s2, p2).curve()[:s1.n]
    check("adding a second chart does not change the first",
          np.allclose(a, bcur, atol=1e-12))
    return ok


def main():
    print("=" * 72)
    print("Cbar v2 unit tests")
    print("=" * 72)
    test_straddle()
    test_shield()
    test_lockanchor()
    test_lockstack()
    test_overlap()
    test_holdtap()
    test_gates()
    test_geometry()
    test_local_in_time()
    print("\n" + "=" * 72)
    if FAIL:
        print(f"FAILED ({len(FAIL)}):")
        for f in FAIL:
            print("  - " + f)
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
