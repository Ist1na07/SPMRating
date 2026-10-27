"""读谱分支：间隔规整度曲线（默认关闭）。

核心想法：规整的节奏（楼梯、匀速连打）比不规整的节奏好读，与手部
执行难度无关。

计算公式：
  d_i = t_{i+1} - t_i
  med_i = median(d[i-16 .. i+16])       # 局部中位，随曲速自适应
  reg_i = [ |d_i - med_i| <= tol * med_i ]   # 容差 18%
  # reg_i 赋给区间 [t_i, t_{i+1})，再平滑并按谱面中心化
"""
from __future__ import annotations

import numpy as np

from ..batch import seg_cumsum


def compute_read_curve(b, s, p: dict) -> np.ndarray:
    """Return the read-regularity curve on the flat row grid.

    Shape: (s.n,), values in [0, 1] where 1 = regular (easy to read).
    """
    n = s.n
    reg = np.zeros(n)
    tol = float(p.get("read_tol", 0.18))
    win = int(p.get("read_n_win", 16))

    # per-chart: compute interval regularity
    for ci in range(b.n_charts):
        a, z = int(s.starts[ci]), int(s.starts[ci + 1])
        if z - a < 3:
            continue
        t = b.t[a:z]  # local times, ms
        d = np.diff(t)  # interval lengths
        if d.size == 0:
            continue
        # local median with window
        nd = d.size
        med = np.zeros(nd)
        for i in range(nd):
            lo = max(0, i - win)
            hi = min(nd, i + win + 1)
            med[i] = np.median(d[lo:hi])
        med = np.maximum(med, 1.0)  # avoid div by zero
        # regularity flag
        reg_i = (np.abs(d - med) <= tol * med).astype(np.float64)
        # assign to rows: reg_i[i] applies to interval [t_i, t_{i+1})
        # which corresponds to row a+i (the row at time t_i)
        reg[a:a + nd] = reg_i

    return reg


def compute_read_gate(b, s, p: dict) -> np.ndarray:
    """门控：4 键谱面取 1.0，7 键谱面取 0.0。"""
    # key_count is per-chart; broadcast to rows
    kc = b.key_count[s.chart]  # (n,)
    return (kc == 4).astype(np.float64)


def compute_read(b, s, p: dict) -> np.ndarray:
    """Compute the smoothed, mean-centered read curve.

    Returns a curve that can be used as a multiplicative modifier:
      D = D * (1 + d_read * read_curve)
    """
    reg = compute_read_curve(b, s, p)
    # smooth with the same window as Pbar
    from ..model import _wmean
    W = float(p.get("pbar_smooth_window", 1000.0))
    reg_s = _wmean(s, reg, W)
    # mean-center per chart (make it a relative tilt, not a level shift)
    out = reg_s.copy()
    for ci in range(b.n_charts):
        a, z = int(s.starts[ci]), int(s.starts[ci + 1])
        if z - a > 0:
            out[a:z] -= np.mean(reg_s[a:z])
    return out
