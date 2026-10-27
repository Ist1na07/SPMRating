"""Cross-column transition budgets and temporal window integration."""
from types import FunctionType

import numpy as np
from numba import njit
from ._core import model as M
from ._core.batch import seg_cumsum
SYNC = 30.0
MEMORY = 180.0
NULL = 0.1
RADIUS = 500.0

def weights(p):
    idx = np.arange(7)
    d = np.abs(idx[:, None] - idx[None, :])
    hand = np.array([0, 0, 0, 1, 2, 2, 2])
    same = (hand[:, None] == hand[None, :]) & (hand[:, None] != 1)
    thumb = (hand[:, None] == 1) | (hand[None, :] == 1)
    cross = (hand[:, None] != hand[None, :]) & ~thumb
    w = 1.0 + p['x_jd_p'] * np.maximum(d, 1e-09) ** p['x_jd_e']
    w = np.where(same & (d > 0), 1.0 + p['x_jh_p'] * np.maximum(d, 1e-09) ** (-p['x_jh_e']), w)
    w = np.where(thumb, 1.0 - p['x_jt'] / np.maximum(d, 1.0), w)
    w = np.where(cross, 1.0 - p['x_jh_p'] * np.minimum(d / 7.0, 1.0), w)
    inward = np.abs(idx[None, :] - 3) < np.abs(idx[:, None] - 3)
    w *= 1.0 + p['x_dir_in'] * inward + p['x_dir_out'] * (1 - inward)
    return w - 1.0

@njit(cache=True)
def atoms(t, chart, mask, coactivation):
    out = np.zeros((len(t), 49))
    a = np.zeros((2, 7))
    b = np.zeros((2, 7))
    c = np.zeros((2, 7))
    for row in range(len(t)):
        if row == 0 or chart[row] != chart[row - 1]:
            a[:] = 0.0
            b[:] = 0.0
            c[:] = 0.0
            dt = 0.0
        else:
            dt = t[row] - t[row - 1]
        a *= np.exp(-2 * dt / MEMORY)
        b *= np.exp(-dt * (1 / MEMORY + 1 / SYNC))
        c *= np.exp(-2 * dt / SYNC)
        source = np.maximum(a - 2 * b + c, 0.0)
        norm0 = NULL + source[0].sum()
        norm1 = NULL + source[1].sum()
        occupancy = np.minimum(np.maximum(coactivation[row], 0.0), 1.0)
        support = max(coactivation[row].sum(), 1.0)
        low = np.ones(7)
        high = np.ones(7)
        for k in range(1, 7):
            low[k] = low[k - 1] * (1 - occupancy[k - 1])
        for k in range(5, -1, -1):
            high[k] = high[k + 1] * (1 - occupancy[k + 1])
        for target in range(7):
            if mask[row] & 1 << target:
                for origin in range(7):
                    internal = 0.0
                    if origin != target:
                        absent = 1.0
                        for k in range(min(origin, target) + 1, max(origin, target)):
                            absent *= 1 - occupancy[k]
                        internal = occupancy[origin] * absent
                    incoming = low[target] * source[0, origin] / norm0 + high[target] * source[1, origin] / norm1
                    out[row, origin * 7 + target] = 0.5 * (internal + incoming) / support
        for k in range(7):
            if mask[row] & 1 << k:
                a[0, k] += high[k]
                b[0, k] += high[k]
                c[0, k] += high[k]
                a[1, k] += low[k]
                b[1, k] += low[k]
                c[1, k] += low[k]
    return out

def budgets(b, s):
    cached = getattr(b, '_adjacency_atom_budgets', None)
    if cached is not None:
        return cached
    co = np.column_stack([M._tri_window_count(s, (s.mask >> k & 1).astype(float), SYNC) for k in range(7)])
    cached = atoms(s.t, s.chart, s.mask, co)
    cached.setflags(write=False)
    b._adjacency_atom_budgets = cached
    return cached

@njit(cache=True)
def cumulative(t, budget, query, radius):
    """CDF of the raised-cosine density, including signed atom budgets."""
    out = np.zeros(len(query))
    n = len(t)
    if n == 0:
        return out
    prefix = np.zeros((n + 1, 4))
    origin = t[0]
    omega = np.pi / radius
    for i in range(n):
        x = t[i] - origin
        phase = omega * x
        prefix[i + 1, 0] = prefix[i, 0] + budget[i]
        prefix[i + 1, 1] = prefix[i, 1] + budget[i] * x
        prefix[i + 1, 2] = prefix[i, 2] + budget[i] * np.cos(phase)
        prefix[i + 1, 3] = prefix[i, 3] + budget[i] * np.sin(phase)
    for j in range(len(query)):
        q = query[j]
        lo = np.searchsorted(t, q - radius, side='left')
        hi = np.searchsorted(t, q + radius, side='right')
        amount = prefix[hi, 0] - prefix[lo, 0]
        first = prefix[hi, 1] - prefix[lo, 1]
        co = prefix[hi, 2] - prefix[lo, 2]
        si = prefix[hi, 3] - prefix[lo, 3]
        phase = omega * (q - origin)
        out[j] = prefix[lo, 0] + 0.5 * amount + ((q - origin) * amount - first) / (2 * radius) + (np.sin(phase) * co - np.cos(phase) * si) / (2 * np.pi)
    return out

def linear_wmean(s, val, width):
    """Same step integral as original _wmean, without its final clamp."""
    inc = np.zeros(s.n)
    inc[:-1] = val[:-1] * s.dt_diff
    integral = seg_cumsum(inc, s.starts) - inc
    jhi, dhi, jlo, dlo = M._wmean_index(s, width)
    return (integral[jhi] + val[jhi] * dhi - (integral[jlo] + val[jlo] * dlo)) / width

def clipped(value):
    return np.maximum(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

def window_curve(b, s, p):
    width = float(p['w_x'])
    signed = budgets(b, s) @ weights(p).reshape(49)
    out = np.zeros(s.n)
    for a, z in zip(s.starts[:-1], s.starts[1:]):
        t = s.t[a:z]
        qlo = np.maximum(t - width / 2, t[0])
        qhi = np.minimum(t + width / 2, t[-1])
        out[a:z] = (cumulative(t, signed[a:z], qhi, RADIUS) - cumulative(t, signed[a:z], qlo, RADIUS)) / (width / 1000.0)
    return out

def compute_curves(b, s, p):
    if p.get('group_jump', 0.0) == 0.0 or p.get('x_adj_w', 0.0) == 0.0:
        return M.compute_curves(b, s, p)
    if p.get('group_jump', 0.0) != 6.0:
        raise ValueError('Only frozen flag0/flag6 candidates are supported.')
    state = {'pending': None, 'consumed': False}

    def jump_hook(b, s, p, x_x):
        state['pending'] = (s, float(p['w_x']), float(p['x_adj_w']), window_curve(b, s, p))
        return M._jump_channel(b, s, p, x_x) if p['x_jump_w'] else np.zeros(s.n)

    def mean_hook(s, value, width):
        pending = state['pending']
        if pending is None:
            return M._wmean(s, value, width)
        state['pending'] = None
        owner, expected, alpha, adjacency = pending
        if owner is not s or expected != float(width):
            raise RuntimeError('Unexpected X window order.')
        base = linear_wmean(s, value, width)
        state['consumed'] = True
        return clipped(base + alpha * adjacency)
    namespace = {**M.__dict__, '_jump_channel': jump_hook, '_wmean': mean_hook}
    fn = FunctionType(M.compute_curves.__code__, namespace, M.compute_curves.__name__, M.compute_curves.__defaults__)
    result = fn(b, s, p)
    if state['pending'] is not None or not state['consumed']:
        raise RuntimeError('X addition was not consumed.')
    return result
