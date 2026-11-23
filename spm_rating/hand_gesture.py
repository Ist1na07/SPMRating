"""Bounded no-LN gesture hypothesis for 7K development probes.

Atomic rows contain simultaneous fingers. A gesture is a contiguous run of
rows within a synchronization window, with no finger repeated in the run.
Use a weighted partition, not a hard threshold or one greedy grouping.
"""
from dataclasses import dataclass

import numpy as np
from numba import njit

from . import actions


@dataclass(frozen=True)
class GestureTrack:
    time: np.ndarray
    masks: np.ndarray
    mass: np.ndarray
    fixed_mass: np.ndarray
    changed_mass: np.ndarray
    groups: np.ndarray  # columns: end row, span in rows (minus one)


@njit(cache=True)
def _smoother(x):
    u = min(max(x, 0.), 1.)
    return u * u * u * (10. - 15. * u + 6. * u * u)


@njit(cache=True)
def _gesture_partitions(time, mask, sync):
    n = len(time)
    width = 4  # A thumb profile contains at most four fingers per hand.
    weights = np.full((n, width), -np.inf)
    masks = np.zeros((n, width), np.uint8)
    fwd = np.full(n + 1, -np.inf)
    bwd = np.full(n + 1, -np.inf)
    fwd[0] = bwd[n] = 0.
    for end in range(n):
        used = 0
        for d in range(min(width, end + 1)):
            start = end - d
            if used & mask[start] or time[end] - time[start] >= sync:
                break
            used |= mask[start]
            # A cut between two rows with the same finger is mandatory.
            # Remaining cuts fade out at simultaneity, as the rows coalesce.
            cut = 1.
            if start and not (mask[start - 1] & mask[start]):
                cut = _smoother((time[start] - time[start - 1]) / sync)
            join = 1. - _smoother((time[end] - time[start]) / sync)
            w = np.log(cut * join)
            weights[end, d] = w
            masks[end, d] = used
            fwd[end + 1] = np.logaddexp(fwd[end + 1], fwd[start] + w)
    for end in range(n - 1, -1, -1):
        for d in range(min(width, end + 1)):
            start = end - d
            bwd[start] = np.logaddexp(bwd[start], weights[end, d] + bwd[end + 1])

    mass = np.zeros(n)
    fixed = np.zeros(n)
    changed = np.zeros(n)
    groups = np.zeros((n, width))
    for end in range(n):
        suffix = bwd[end + 1] - fwd[n]
        for d in range(min(width, end + 1)):
            start = end - d
            cur = masks[end, d]
            if not cur:
                continue
            w = weights[end, d] + suffix
            groups[end, d] = np.exp(w + fwd[start])
            if not start or not (cur & (cur - 1)):
                continue
            prev_end = start - 1
            gate = _smoother((time[end] - time[prev_end]) / sync)
            for dd in range(min(width, start)):
                prev = masks[prev_end, dd]
                if not (prev & cur and prev & (prev - 1)):
                    continue
                before = prev_end - dd
                amount = gate * np.exp(w + weights[prev_end, dd] + fwd[before])
                mass[end] += amount
                if prev == cur:
                    fixed[end] += amount
                else:
                    changed[end] += amount
    return mass, fixed, changed, groups


def extract(time, col, sync=30.):
    """Return [thumb-left, thumb-right] × [left, right] gesture tracks."""
    time, col = np.asarray(time, dtype=float), np.asarray(col, dtype=np.int64)
    if (time.ndim != 1 or col.shape != time.shape or not np.isfinite(time).all()
            or np.any((col < 0) | (col > 6)) or not np.isfinite(sync) or sync <= 0):
        raise ValueError("Expected finite 7K contacts and positive synchronization window")
    profiles = []
    for cut in (4, 3):
        left = col < cut
        profile = []
        for take in (left, ~left):
            rows, index = np.unique(time[take], return_inverse=True)
            masks = np.zeros(len(rows), dtype=np.uint8)
            np.bitwise_or.at(masks, index, 1 << col[take])
            mass, fixed, changed, groups = _gesture_partitions(rows, masks, sync)
            profile.append(GestureTrack(rows, masks, mass, fixed, changed, groups))
        profiles.append(tuple(profile))
    return tuple(profiles)


def field(profiles, query, width, sync=30.):
    """Project expected coherent repeat atoms using RC3's rate kernel."""
    out = []
    for profile in profiles:
        hand = []
        for track in profile:
            take = track.mass > 0
            hand.append(actions.field(track.time[take], track.mass[take], query, width, sync))
        out.append(hand)
    return np.asarray(out)
