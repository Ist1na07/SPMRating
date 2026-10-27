"""视觉规整度通道：拍点与间隔对齐度。

八度折叠的对数间隔对齐度，计算在行网格上：

    logdt = log(max(dt_ms, 1e-6))
    mu    = wmean(logdt, eye_w_ref)        # 局部几何平均速率（4 秒窗）
    r     = (logdt - mu) mod log 2
    dist  = min(r, log2 - r)               # 到最近八度比例的距离
    R(t)  = exp(-wmean(dist, eye_w_s) / eye_tau)   # 取值 (0, 1]，1 表示对齐

接线形式是曲线级的密度门控交互，加进流分支：
        stream += d_eye_row * Pbar * R(t)
（等价于 Pbar * (1 + (d_eye_row/a_p) * R)，即密度通道上的规整度倾斜，
随时间局部作用，不含整谱标量）。
"""
from __future__ import annotations

import numpy as np

from ..model import _wmean


def compute_eye_curve(b, s, p: dict) -> np.ndarray:
    """Row-grid alignment curve R(t) in (0, 1]; 1 = locally on-grid."""
    w_ref = max(float(p.get("eye_w_ref", 4000.0)), 100.0)
    w_s = max(float(p.get("eye_w_s", 500.0)), 50.0)
    tau = max(float(p.get("eye_tau", 0.2)), 1e-3)

    dt_ms = np.maximum(s.dt_ms, 0.0)
    # rows with no successor (chart-final) have dt_ms == 0: exclude them by
    # substituting the local mean below (dist = 0 -> neutral).
    valid = dt_ms > 0.0
    logdt = np.log(np.maximum(dt_ms, 1e-6))
    mu = _wmean(s, np.where(valid, logdt, 0.0), w_ref)
    cnt = _wmean(s, valid.astype(np.float64), w_ref)
    # _wmean is a plain windowed mean; approximate the valid-only mean:
    mu = np.where(cnt > 1e-6, mu / np.maximum(cnt, 1e-9), 0.0)
    r = np.mod(logdt - mu, np.log(2.0))
    dist = np.minimum(r, np.log(2.0) - r)
    dist = np.where(valid, dist, 0.0)
    return np.exp(-_wmean(s, dist, w_s) / tau)
