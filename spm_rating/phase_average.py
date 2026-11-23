"""Exact integral over all origins of the two-level window aggregate."""
from numba import njit
import numpy as np


@njit(cache=True)
def integrate(phase, src, dst, mass, weighted, n_bins, width, pw1, pw2):
    """Between split points the two-level scalar is constant."""
    m, x = np.zeros(n_bins), np.zeros(n_bins)
    count = np.zeros(n_bins, np.int64)
    for i in range(len(phase)):
        slot = src[i]
        m[slot] += mass[i]
        x[slot] += weighted[i]
        count[slot] += 1
    exponent = pw2 / pw1

    def term(mass_value, numerator):
        return mass_value * (max(numerator, 0.) / max(mass_value, 1e-9))**exponent

    total = m.sum()
    if total == 0.:
        return 0.
    value = 0.
    for k in range(n_bins):
        value += term(m[k], x[k])
    integral, left = 0., 0.
    order = np.argsort(phase)
    for i in order:
        right = phase[i]
        integral += (right - left) * (max(value, 0.) / total)**(1 / pw2)
        a, b = src[i], dst[i]
        value -= term(m[a], x[a]) + term(m[b], x[b])
        count[a] -= 1
        count[b] += 1
        m[a] -= mass[i]
        x[a] -= weighted[i]
        if count[a] == 0:
            m[a] = x[a] = 0.
        m[b] += mass[i]
        x[b] += weighted[i]
        value += term(m[a], x[a]) + term(m[b], x[b])
        left = right
    return (integral + (width - left) * (max(value, 0.) / total)**(1 / pw2)) / width
