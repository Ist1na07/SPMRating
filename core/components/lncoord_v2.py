"""Cbar v2 -- LN coordination on the row grid.

Why this file exists
--------------------
Cbar v1 (`core/model.py::_cbar`) models LN coordination as

    shield (per-LN-head lookback)
    + cb_lockjack  * Jbar * held^p
    + cb_lockanchor* Ja   * held^p
    + cb_par       * held^p
    + cb_holdtap   * (size/dt) * held^p

Three things are wrong with that:

1. The lock terms are *scalar* modulations of the global Jbar / Ja power
   means.  Jbar is already an aggregate over columns, so `Jbar * held^p`
   cannot tell *which* column is locked -- a jack in column 0 while column 6
   is held scores the same as while column 1 is held.  That is the single
   biggest modelling loss: 锁手叠 is entirely about *which* fingers are
   pinned.
2. `held^p` is a bare count.  It has no notion of hand geometry, so the
   衩叠 pattern (alternate 0/2 while 1 is held) is invisible.
3. `cb_par` never existed as a real term -- it is just `held^p` again, so
   "overlapping LNs" and "many holds" are the same number.

Cbar v2 fixes all three with a **cross-column cost matrix** applied
per column, per row.

Architecture: the row grid is the interval grid
-----------------------------------------------
This repo has no `Interval` object.  The equivalent -- and the thing every
other channel in `core/model.py` uses -- is the **row grid**: row r is one
distinct timestamp and carries a *step function* value that holds on the
half-open interval ``[t[r], t[r+1])``.  So "interval-based" here means
"step function on the row grid, smoothed by a centred windowed mean".

Every sub-term below is built that way:

    per-event value  --scatter-->  step function on rows  --_wmean-->  curve

Concretely each term returns an array of length ``s.n`` (total rows over
all charts in the batch) and every operation is a segmented / scatter numpy
op.  There is **no chart-level scalar anywhere** -- not a total, not a mean,
not a normaliser.  The only reductions are (a) the centred windowed mean
with a fixed ms window, and (b) per-row sums over the 7 columns.  Both are
local in time.

Gates
-----
Each sub-term has an *independent* switch ``use_cbv_<term>`` and an
*independent* weight ``cbv_<term>``.  ``CbarV2.terms()`` returns the six
weighted curves and ``curve()`` is their elementwise sum, so

    curve(all on) - curve(use_cbv_x = 0) == terms()["x"]

holds exactly (up to float rounding).  That is what the gate test asserts.

Hand geometry
-------------
    HAND      = [0,0,0,1,2,2,2]   cols 0,1,2 = left | 3 = thumb | 4,5,6 = right
    AIN_GROUP = [0,1,2,3,2,1,0]   ring / middle / index / thumb

Both are imported from `core.model` so Cbar v2 can never drift from the
rest of the model.
"""
from __future__ import annotations

import numpy as np

from ..model import (AIN_GROUP, HAND, K, _chain_scatter, _run_lengths,
                     _step_scatter, _wmean, build_jack_stats)
from ..params import DEFAULTS

# column-pair groups of the 7x7 cross-column matrices (mirror-symmetric by
# construction):  same hand adjacent | same hand split | one hand split
# (thumb) | two hand split | far cross
_SUFFIXES = ("sh_adj", "sh_split", "1h_split", "2h_split", "cross")

_IDX = np.arange(K, dtype=np.int64)
_STRIDE = 1 << 30          # must match core.batch.CHART_STRIDE

# default matrix shapes, used only when a key is absent from `p`
_DEF_SHAPE = {"sh_adj": 0.6, "sh_split": 1.0, "1h_split": 0.8,
              "2h_split": 0.5, "cross": 0.3}

TERMS = ("shield", "straddle", "lockstack", "lockanchor", "overlap", "holdtap")


def dist_matrix(p, prefix: str) -> np.ndarray:
    """7x7 mirror-symmetric cross-column cost matrix.

    Partition of unordered pairs (a, b), a != b -- disjoint and complete:
        both in {0,1,2} or both in {4,5,6}   -> same hand
        exactly one of them is 3             -> thumb ("one hand split")
        one in {0,1,2}, other in {4,5,6}     -> cross hand ("two hand split")
    within same hand, d == 1 -> sh_adj, d == 2 -> sh_split.  d >= 3 -> cross.

    Diagonal is 0: acting on a column is not locked by that same column.
    """
    d = np.abs(_IDX[:, None] - _IDX[None, :])
    h = HAND[:, None]
    same_hand = (h == HAND[None, :]) & (h != 1)
    thumb = (h == 1) | (HAND[None, :] == 1)
    cross = (h != HAND[None, :]) & ~thumb

    def g(suf):
        v = p.get(prefix + suf)
        return _DEF_SHAPE[suf] if v is None else float(v)

    M = np.full((K, K), g("cross"), dtype=np.float64)
    M = np.where(cross & (d <= 2) & (d > 0), g("2h_split"), M)
    M = np.where(thumb & (d <= 2) & (d > 0), g("1h_split"), M)
    M = np.where(same_hand & (d == 1), g("sh_adj"), M)
    M = np.where(same_hand & (d == 2), g("sh_split"), M)
    M = np.where(d == 0, 0.0, M)
    return M


class CbarV2:
    """Interval(row)-based LN coordination channel.

    Parameters
    ----------
    b, s : Batch / Struct from `core.model`
    p    : flat parameter dict (missing keys fall back to `core.params`)
    stats: optional per-column jack stats from `build_jack_stats` (rebuilt
           if absent)
    """

    TERMS = TERMS

    def __init__(self, b, s, p, stats=None):
        self.b = b
        self.s = s
        self.p = p
        self.n = s.n
        self._stats = stats
        self._mats: dict = {}

    # ---------------------------------------------------------------- utils
    def _f(self, key: str) -> float:
        v = self.p.get(key)
        if v is None:
            v = DEFAULTS.get(key, 0.0)
        return float(v)

    def _gate(self, name: str) -> bool:
        return self._f("use_cbv_" + name) != 0.0

    def _mat(self, prefix: str) -> np.ndarray:
        M = self._mats.get(prefix)
        if M is None:
            M = dist_matrix(self.p, prefix)
            self._mats[prefix] = M
        return M

    @property
    def stats(self):
        if self._stats is None:
            self._stats = build_jack_stats(self.b, self.s, self.p)
        return self._stats

    def _lockmat(self) -> np.ndarray:
        return self._mat("cbv_ls_")

    def _bits(self, rows: np.ndarray) -> np.ndarray:
        """(m, K) float matrix: 1 where column j is held at `rows`."""
        hm = self.s.heldmask[rows]
        return ((hm[:, None] >> _IDX[None, :]) & 1).astype(np.float64)

    def _lockmass(self, rows: np.ndarray, col: int) -> np.ndarray:
        """sum_j M[col, j] * held_j(row), j != col  --  per entry of `rows`.

        M[col, col] == 0 so the column itself is excluded automatically.
        This is the whole point of v2: the *identity* of the held columns
        matters, not just their number.
        """
        return self._bits(rows) @ self._lockmat()[col]

    # ------------------------------------------------------------- 1. shield
    def _t_shield(self) -> np.ndarray:
        """盾: a tap (or very short LN) struck shortly before an LN head on
        the *same* column, amplified by what else is held.

        v1 summed exp(-dt/tau) over the 16 previous same-column notes, which
        double counts long chains.  v2 uses the **immediate** predecessor --
        that is what "rapidly striking another LN on the same column" means.

        The predecessor's own duration matters: a tap leaves the finger free
        and is the classic 盾 setup; a long LN already occupies the finger,
        so re-striking is a different (and cheaper) action.  Hence
        `cbv_sh_ln_w * exp(-dur / cbv_sh_ln_tau)` for an LN predecessor.
        """
        s, b, n = self.s, self.b, self.n
        tau = max(self._f("cbv_sh_tau"), 1.0)
        dtmax = max(self._f("cbv_sh_dt_max"), 1.0)
        ln_w = self._f("cbv_sh_ln_w")
        ln_tau = max(self._f("cbv_sh_ln_tau"), 1.0)
        amp = self._f("cbv_sh_lock")
        out = np.zeros(n)
        for k in range(K):
            c = s.col_chain[k]
            pos = c["pos"]
            if pos.size < 2:
                continue
            li = np.flatnonzero(s.is_ln[pos])
            li = li[li >= 1]
            if li.size == 0:
                continue
            pv = li - 1
            dtp = c["tc"][li] - c["tc"][pv]
            ok = (c["cid"][pv] == c["cid"][li]) & (dtp > 0.0) & (dtp < dtmax)
            # predecessor type weight.
            # A mania column cannot hold two overlapping notes, so for two
            # consecutive notes in column k we always have
            #     t_prev + dur_prev <= t_now   ->   dur_prev <= dtp
            # Clamping the predecessor's duration to `dtp` therefore costs
            # nothing on clean data and bounds the damage on mixed chords,
            # where `b.ln_end[row]` is the row max and may belong to a
            # *different* column (see LIMITATIONS).  Without the clamp that
            # case reads as an arbitrarily long LN and the term collapses.
            prow = pos[pv]
            is_ln_prev = s.is_ln[prow]
            dur = np.clip(b.ln_end[prow] - s.t[prow], 0.0, dtp)
            wpred = np.where(is_ln_prev,
                             ln_w * np.exp(-np.maximum(dur, 0.0) / ln_tau), 1.0)
            val = np.where(ok, wpred * np.exp(-np.maximum(dtp, 0.0) / tau), 0.0)
            rows = pos[li]
            val = val * (1.0 + amp * self._lockmass(rows, k))
            out = out + _step_scatter(rows, np.minimum(rows + 1, n), val, n)
        return _wmean(s, out, self._f("cbv_w"))

    # ----------------------------------------------------------- 2. straddle
    def _t_straddle(self) -> np.ndarray:
        """衩叠: alternation between columns a and b while a column *between*
        them is held.  The canonical case is 0/2 alternating while 1 (the
        left middle finger) is pinned, or 4/6 while 5 is pinned.

        This is the term v1 structurally cannot express: `held^p` is a count,
        so it cannot see that the pinned finger is the one *in the middle*.

        Construction
        ------------
        For every pair (a, b) with b - a == d, 2 <= d <= cbv_str_maxd:

            act   = sqrt(rate_a * rate_b) * balance ** cbv_str_bal
                    (geometric mean = both sides busy; balance = alternation
                     rather than one side dominating)
            pin   = sum over interior columns m of held_m * hw[AIN_GROUP[m]]
            term  = M[a, b] * act * pin

        `M` is the mirror-symmetric 7x7 pair matrix (`dist_matrix`) whose
        d == 2 entries are exactly the four groups named in the brief:
            (0,2) (4,6)  same_hand_split  <- true 衩叠, middle finger pinned
            (1,3) (3,5)  one_hand_split   <- thumb-bridged
            (2,4)        two_hand_split
        `hw` (cbv_str_h0..h3, indexed by AIN_GROUP) adds a second,
        finger-identity weighting on the pinned column on top of the pair
        group, so "middle finger pinned" can be stressed independently.

        Everything is evaluated per row against `s.usage` (a +-400 ms local
        note count) and `s.heldmask` (per-row), so the term is local in time.
        """
        s, n = self.s, self.n
        M = self._mat("cbv_str_")
        hw = np.array([self._f(f"cbv_str_h{i}") for i in range(4)])
        bal_p = self._f("cbv_str_bal")
        maxd = int(min(max(self._f("cbv_str_maxd"), 2.0), K - 1))
        r = s.usage.astype(np.float64)                     # (K, n)
        hmb = ((self.s.heldmask[:, None] >> _IDX[None, :]) & 1).astype(np.float64)
        out = np.zeros(n)
        # A pair can only be "locked straddle" if some column strictly
        # between a and b can actually be held in this frame.  In the 4K
        # embedding (frame cols {1,2,4,5}) the interior of pairs like (1,3)
        # is column 3 (thumb) or empty -- never held, pin==0, so the term
        # silently died on the whole 4K dataset.  Gate on the interior
        # columns being *ever active*.
        for d in range(2, maxd + 1):
            inter = np.arange(1, d)                        # offsets a+1..b-1
            for a in range(K - d):
                bcol = a + d
                interior = a + inter
                if not s.act[interior].any(axis=1).any():
                    continue
                ra, rb = r[a], r[bcol]
                co = np.sqrt(np.maximum(ra * rb, 0.0))
                bal = 2.0 * np.minimum(ra, rb) / np.maximum(ra + rb, 1e-9)
                act = co * np.power(bal, bal_p)
                pin = (hmb[:, interior] * hw[AIN_GROUP[interior]]).sum(axis=1)
                out = out + M[a, bcol] * act * pin
        return _wmean(s, out, self._f("cbv_w"))

    # -------------------------------------------------- 3/4. locked jack/anchor
    def _col_locked(self, kind: str) -> np.ndarray:
        """Power-mean over columns of (per-column jack curve * its lock mass).

        Unlike v1's `Jbar * held^p`, the lock mass is evaluated **per
        column** before the aggregation, so a jack in column 0 is amplified
        by what is held relative to column 0 specifically.

        kind == "plain"     -> stack speed            (lockstack)
        kind == "anchor"    -> stack speed gated by same-column run length
                               (lockanchor), using the same run-length shape
                               as the Ja channel so the two agree.
        """
        s, p, n = self.s, self.p, self.n
        agg_pow = max(self._f("j_agg_pow"), 0.25)
        ja_floor = self._f("ja_len_floor")
        ja_norm = max(self._f("ja_norm"), 1e-3)
        ja_len_exp = self._f("ja_len_exp")
        ja_coord_exp = self._f("ja_coord_exp")
        # cbv_lock_pow: exponent on the per-column lock mass.  1.0 = linear
        # (legacy).  <1 makes heavy locking saturate -- the high-end 7K LN
        # charts (lockanchor z>+12 vs peers) suggest the linear form
        # over-prices extreme lock states.
        lock_pow = self._f("cbv_lock_pow")
        num = np.zeros(n)
        den = np.zeros(n)
        for k in range(K):
            st = self.stats[k]
            if st is None:
                continue
            if kind == "anchor":
                e = np.maximum(st["runlen"] + 2.0 - ja_floor, 0.0)
                runf = np.power(e / (e + ja_norm), ja_len_exp)
                mod = runf * np.power(1.0 + st["other"] / st["nrows"],
                                      ja_coord_exp)
            else:
                mod = np.ones(st["pos"].size - 1)
            v = np.where(st["valid"], st["kern"] * mod, 0.0)
            cur = _wmean(s, _chain_scatter(st["pos"], v, n, st["valid"]),
                         self._f("cbv_w"))
            # lock mass is a step function over the *gap* [pos[j], pos[j+1])
            lm = _chain_scatter(st["pos"],
                                self._lockmass(st["pos"][:-1], k),
                                n, st["valid"])
            if lock_pow != 1.0:
                lm = np.power(np.maximum(lm, 0.0), lock_pow)
            w = _chain_scatter(st["pos"],
                               np.where(st["valid"], 1.0 / st["gs"], 0.0),
                               n, st["valid"])
            num = num + np.power(np.maximum(cur * lm, 0.0), agg_pow) * w
            den = den + w
        val = np.where(den > 1e-9, num / np.where(den > 1e-9, den, 1.0), 0.0)
        return np.power(np.maximum(val, 0.0), 1.0 / agg_pow)

    def _t_lockstack(self) -> np.ndarray:
        return self._col_locked("plain")

    def _t_lockanchor(self) -> np.ndarray:
        return self._col_locked("anchor")

    # ------------------------------------------------------------ 5. overlap
    def _t_overlap(self) -> np.ndarray:
        """Genuinely *partial* LN overlap (not containment).

        Two LNs i (earlier) and j (later) on different columns:
            containment  t_i < t_j and e_j <= e_i   -> the classic
                         "hold + something inside"; already priced by
                         holdtap / lockstack, and cheaper.
            partial      t_i < t_j < e_i < e_j      -> the two releases are
                         interleaved with a re-press: this is real 面叠.
        Contribution = min(e_i - t_j, cap)/1000 * M[c_i, c_j], held as a step
        function across the overlap window [t_j, e_i).
        """
        s, b, n = self.s, self.b, self.n
        M = self._mat("cbv_ov_")
        dtmax = max(self._f("cbv_ov_dt_max"), 1.0)
        cap = max(self._f("cbv_ov_cap"), 1.0)

        rows_l, cols_l = [], []
        for k in range(K):
            pos = s.col_chain[k]["pos"]
            if pos.size == 0:
                continue
            m = s.is_ln[pos]
            if not m.any():
                continue
            rows_l.append(pos[m])
            cols_l.append(np.full(int(m.sum()), k, np.int64))
        if not rows_l:
            return np.zeros(n)
        rows = np.concatenate(rows_l)
        cols = np.concatenate(cols_l)
        o = np.argsort(s.tg[rows], kind="stable")
        rows, cols = rows[o], cols[o]
        N = rows.size
        if N < 2:
            return np.zeros(n)

        tg = s.tg[rows]
        while True:
            hi = np.searchsorted(tg, tg + dtmax, side="right")
            cnt = np.maximum(hi - np.arange(N) - 1, 0)
            if cnt.sum() <= 4_000_000 or dtmax <= 50.0:
                break
            dtmax *= 0.5
        total = int(cnt.sum())
        if total == 0:
            return np.zeros(n)
        i_rep = np.repeat(np.arange(N), cnt)
        offs = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt) + 1
        j_rep = i_rep + offs

        ti, tj = s.t[rows[i_rep]], s.t[rows[j_rep]]
        ei, ej = b.ln_end[rows[i_rep]], b.ln_end[rows[j_rep]]
        ok = (tj > ti) & (ei > tj) & (ej > ei)
        if not ok.any():
            return np.zeros(n)
        ov = np.minimum(ei - tj, cap)
        val = np.where(ok, (ov / 1000.0) * M[cols[i_rep], cols[j_rep]], 0.0)
        j0 = rows[j_rep]
        z = (s.t[j0] + ov) + s.chart[j0].astype(np.int64) * _STRIDE
        j1 = np.clip(np.searchsorted(s.tg, z, side="left"), s.jlo[j0], s.jhi[j0])
        return _wmean(s, _step_scatter(j0, np.maximum(j1, j0), val, n),
                      self._f("cbv_w"))

    # ------------------------------------------------------------ 6. holdtap
    def _t_holdtap(self) -> np.ndarray:
        """Taps struck while holds are active elsewhere.

        v1: `(size/dt) * held^p` -- note rate times a bare hold count.
        v2 keeps the rate but replaces the count by the *mean, over the notes
        actually struck at this row*, of that note's own lock mass.  So a tap
        in column 0 while column 1 is held costs more than while column 6 is
        held, which is the same geometry the Rbar `r_lock` / `r_straddle`
        terms use on the release side.
        """
        s, n = self.s, self.n
        tot = np.zeros(n)
        for k in range(K):
            pos = s.col_chain[k]["pos"]
            if pos.size == 0:
                continue
            np.add.at(tot, pos, self._lockmass(pos, k))
        mean_lm = tot / np.maximum(s.size, 1)
        raw = (s.size / s.dt) * mean_lm
        return _wmean(s, np.maximum(raw, 0.0), self._f("cbv_w"))

    # --------------------------------------------------------------- assembly
    def terms(self) -> dict:
        """Six weighted, gated, smoothed curves; sums exactly to `curve()`."""
        out = {}
        for name in TERMS:
            w = self._f("cbv_" + name)
            if not self._gate(name) or w == 0.0:
                out[name] = np.zeros(self.n)
                continue
            raw = getattr(self, "_t_" + name)()
            out[name] = w * np.maximum(raw, 0.0)
        return out

    def curve(self) -> np.ndarray:
        t = self.terms()
        acc = np.zeros(self.n)
        for v in t.values():
            acc = acc + v
        return np.maximum(np.nan_to_num(acc, nan=0.0, posinf=0.0,
                                        neginf=0.0), 0.0)


def compute_cbar_v2(b, s, p, stats=None) -> np.ndarray:
    """Entry point used by `core.model.compute_curves` when use_cbar_v2 != 0."""
    return CbarV2(b, s, p, stats).curve()


# --------------------------------------------------------------------------
# LIMITATIONS (documented, not silently ignored)
#
# * s.is_ln / b.ln_end are *row*-level aggregates (any note at the row is an
#   LN / max ln_end at the row).  A chord that puts a tap on column k and an
#   LN on column j makes column k look like an LN head at that row.  This
#   affects `shield` and `overlap` only, and only inside mixed chords.  The
#   row grid in core/batch.py has no per-note column array (b.col is
#   referenced by _jump_channel but never defined), so fixing it means
#   changing the Batch layout -- out of scope here.
#
# * `cbv_str_*` / `cbv_ls_*` / `cbv_ov_*` share the same pair-group
#   partition; only their weights differ.  Deliberate: the geometry is one
#   fact, the three channels price it differently.
