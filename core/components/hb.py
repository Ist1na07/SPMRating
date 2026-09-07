"""混合长键补充项。

四个机制，每条都是局部时间序列曲线（不含整谱标量）：

1. **churn** (hb main): rate-concave LN-release churn.  Hybrid charts with
   frequent releases are under-priced; the LN "wall" charts (22-31 releases/s)
   are already saturated, so the curve is concave to leave them alone.
       imp[tail row] += 1;  r = wmean(imp, churn_w)   # releases per second
       curve = r / (1 + r / churn_sat)                # concave saturation

2. **holdage** (ln main): LN occupation duration.  Long holds (>= ~0.25 s)
   cost the holding finger continuously; this channel is exactly what the
   negative-weighted holdtap term cancels on sustained-occupation charts.
       per column, active LN: age(t) = max(min(t - head, cap)/1000 - th, 0)
       curve = sum over columns, smoothed.

3. **chord2**: double-note density rate (size == 2 rows per second).
   Our base prices 2-chords below singles via p_chw2 < 1; the labels reward
   double-note density in hybrid charts.

4. **recov**: recovery-aware burst discount.  relu(local rate / 20 s baseline
   - 1), smoothed -- bursts followed by rest get a discount (the labels
   reward recovery; uniform streams are untouched because C_s == C_l).

All gates default to 0 (exact no-op).
"""
from __future__ import annotations

import numpy as np

from ..model import _wmean


def compute_churn(b, s, p: dict) -> np.ndarray:
    """Concave LN-release churn curve (releases/s, saturated)."""
    n = s.n
    if s.tail is None:
        return np.zeros(n)
    imp = np.zeros(n)
    np.add.at(imp, s.tail["j0"], 1.0)
    r = _wmean(s, imp, max(float(p.get("churn_w", 1000.0)), 50.0))
    sat = max(float(p.get("churn_sat", 15.0)), 1.0)
    return r / (1.0 + r / sat)


def compute_holdage(b, s, p: dict) -> np.ndarray:
    """LN occupation duration curve (seconds past threshold, per column sum).

    Vectorised: within one column LNs never overlap, so the head time of the
    currently-held LN is a step function on the row grid (step_scatter of the
    head over its [head, end) span).  age(t) = clamp(min(t-head, cap) - th).
    """
    from ..model import _step_scatter
    n = s.n
    th = float(p.get("holdage_th", 0.25))
    cap = float(p.get("holdage_cap", 1.0))
    out = np.zeros(n)
    tg = s.tg
    t_loc = s.t.astype(np.float64)
    for k in range(7):
        c = s.col_chain[k]
        pos = c["pos"]
        if pos.size == 0:
            continue
        ln_rows = pos[s.is_ln[pos]]
        if ln_rows.size == 0:
            continue
        h = t_loc[ln_rows]                          # local ms
        e = np.minimum(b.ln_end[ln_rows], h + cap * 1000.0)
        cid = b.chart[ln_rows].astype(np.int64)
        j0 = ln_rows
        z1 = e + cid * (1 << 30)
        j1 = np.clip(np.searchsorted(tg, z1, side="left"),
                     s.jlo[j0], s.jhi[j0] + 1)
        # step functions over each LN's span (disjoint within a column)
        H = _step_scatter(j0, j1, h, n)             # active LN's head (ms)
        M = _step_scatter(j0, j1, np.ones_like(h), n)  # held indicator
        age_s = np.clip(np.minimum(t_loc - H, cap * 1000.0) / 1000.0 - th,
                        0.0, None)
        out += np.where(M > 0, age_s, 0.0)
    return out


def compute_chord2(b, s, p: dict) -> np.ndarray:
    """Double-note row rate (per second)."""
    w = max(float(p.get("chord2_w", 1000.0)), 50.0)
    return _wmean(s, (s.size == 2).astype(np.float64), w)


def compute_recov(b, s, p: dict) -> np.ndarray:
    """Burst-over-baseline excess (recovery discount driver)."""
    w = max(float(p.get("recov_w", 1000.0)), 50.0)
    wl = max(float(p.get("recov_wl", 20000.0)), 1000.0)
    C_s = _wmean(s, s.C, w)
    C_l = np.maximum(_wmean(s, s.C, wl), 1e-6)
    burst = np.maximum(C_s / C_l - 1.0, 0.0)
    return _wmean(s, burst, w)
