"""Shared repeat-contact accounting and parallel hand-rate fields."""
from types import SimpleNamespace
import numpy as np
from numba import njit
from . import actions as A

@njit(cache=True)
def _raw_eligibility(time, col, hands, scale):
    last = np.full(7, -1, np.int64)
    lane = np.zeros(len(time))
    e = np.zeros(len(time))
    visits = 0

    def smooth(x):
        u = min(max(x, 0.0), 1.0)
        return u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)
    for j in range(len(time)):
        k = col[j]
        i = last[k]
        if i >= 0:
            lane[j] = smooth((time[j] - time[i]) / scale)
            count = 0.0
            a = np.searchsorted(time, time[i], side='right')
            b = np.searchsorted(time, time[j], side='left')
            for m in range(a, b):
                visits += 1
                if hands[m] == hands[j]:
                    count += smooth((time[m] - time[i]) / scale) * smooth((time[j] - time[m]) / scale)
                    if count >= 1.0:
                        break
            e[j] = lane[j] * (1.0 - smooth(count))
        last[k] = j
    return (lane, e, visits)

@njit(cache=True)
def maximum_envelope(time, weight, query, scale):
    """Linear upper hull of weighted compact cosine-squared pulses.

    Ordered pulse centers cross at most once on their common support:
    log(cos²(a*(q-ti))/cos²(a*(q-tj))) is monotone for ti<tj.
    Taking positive square roots gives an analytic atan2 intersection.
    Hull construction and nondecreasing query evaluation are O(N+Q).
    """
    n = len(time)
    centers = np.empty(n)
    weights = np.empty(n)
    starts = np.empty(n)
    size = 0
    a = np.pi / (2.0 * scale)
    for j in range(n):
        if weight[j] <= 0.0:
            continue
        t = time[j]
        w = weight[j]
        if size and t == centers[size - 1]:
            if w <= weights[size - 1]:
                continue
            size -= 1
        crossing = t - scale
        while size:
            old = centers[size - 1]
            distance = (t - old) * a
            if distance >= np.pi:
                crossing = t - scale
            else:
                ra = np.sqrt(weights[size - 1])
                rb = np.sqrt(w)
                numerator = ra - rb + 2.0 * rb * np.sin(distance / 2.0) ** 2
                crossing = old + np.arctan2(numerator, rb * np.sin(distance)) / a
                crossing = min(max(crossing, t - scale), old + scale)
            if crossing <= starts[size - 1]:
                size -= 1
            else:
                break
        if not size:
            crossing = t - scale
        centers[size] = t
        weights[size] = w
        starts[size] = crossing
        size += 1
    out = np.zeros(len(query))
    cursor = 0
    for j in range(len(query)):
        if j and query[j] < query[j - 1]:
            raise ValueError('Sorted envelope queries required.')
        if not size:
            continue
        while cursor + 1 < size and starts[cursor + 1] <= query[j]:
            cursor += 1
        delta = query[j] - centers[cursor]
        if abs(delta) < scale:
            out[j] = weights[cursor] * np.cos(a * delta) ** 2
    return out

@njit(cache=True)
def moments(time, col, e0, e1, scale):
    n = len(time)
    mass = np.zeros((2, n))
    coactive = np.zeros((2, n))
    maps = np.array([[0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1]])
    maximum = np.zeros((2, n))
    if n == 0:
        return (mass, coactive)
    for profile in range(2):
        ev = e0 if profile == 0 else e1
        hands = maps[profile][col]
        for hand in range(2):
            indices = np.flatnonzero(hands == hand)
            maximum[profile, indices] = maximum_envelope(time[indices], ev[indices], time[indices], scale)
    phase = (time - time[0]) * (np.pi / scale)
    cs = np.cos(phase)
    sn = np.sin(phase)
    total = np.zeros((7, 3, 3))
    correction = np.zeros((7, 3, 3))
    left = 0
    right = 0

    def update(index, sign):
        k = col[index]
        for channel in range(3):
            weight = 1.0 if channel == 0 else e0[index] if channel == 1 else e1[index]
            for power in range(3):
                value = sign * weight * (1.0 if power == 0 else cs[index] if power == 1 else sn[index])
                y = value - correction[k, channel, power]
                t = total[k, channel, power] + y
                correction[k, channel, power] = t - total[k, channel, power] - y
                total[k, channel, power] = t
    for j in range(n):
        while right < n and time[right] <= time[j] + scale:
            update(right, 1.0)
            right += 1
        while left < right and time[left] < time[j] - scale:
            update(left, -1.0)
            left += 1
        for profile in range(2):
            ev = e0 if profile == 0 else e1
            hand = maps[profile, col[j]]
            u = 0.0
            denominator = 0.0
            for k in range(7):
                if maps[profile, k] == hand:
                    den = max(0.5 * (total[k, 0, 0] + cs[j] * total[k, 0, 1] + sn[j] * total[k, 0, 2]), 0.0)
                    num = max(0.5 * (total[k, profile + 1, 0] + cs[j] * total[k, profile + 1, 1] + sn[j] * total[k, profile + 1, 2]), 0.0)
                    u += min(den, 1.0)
                    denominator += num
            denominator = max(denominator, ev[j])
            z = min(max(u - 1.0, 0.0), 1.0)
            gate = z * z * z * (10.0 - 15.0 * z + 6.0 * z * z)
            coactive[profile, j] = u
            if denominator > 0.0:
                mass[profile, j] = ev[j] * (min(maximum[profile, j], denominator) / denominator) * gate
    return (mass, coactive)

def _extract(notes, scale=30.0):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Positive existing synchronization scale required.')
    pairs = np.unique(np.column_stack([notes.time, notes.col]), axis=0)
    time = pairs[:, 0]
    col = pairs[:, 1].astype(np.int64)
    mapping = [np.array([0, 0, 0, 0, 1, 1, 1]), np.array([0, 0, 0, 1, 1, 1, 1])]
    profiles = []
    for m in mapping:
        hands = m[col]
        lane, e, visits = F.eligibility(time, col, hands, scale)
        profiles.append(dict(hand=hands, lane_mass=lane, eligible=e, interval_visits=int(visits)))
    mass, u = moments(time, col, profiles[0]['eligible'], profiles[1]['eligible'], scale)
    for i, p in enumerate(profiles):
        p.update(hand_mass=mass[i], coactive=u[i])
    assert np.all(np.isfinite(mass)) and np.all(mass >= -1e-12) and np.all(mass <= 1.0 + 1e-12)
    return dict(time=time, column=col, profiles=profiles, scale=scale, physical_heads=len(time), model='Fused normalized per-finger hand transfer; same bounded detector semantics.')

def _general_field(data, query, width=1000.0):
    t = data['time']
    col = data['column']
    scale = data['scale']
    outputs = []
    details = []
    for p in data['profiles']:
        extra = []
        states = []
        for hand in [0, 1]:
            take = p['hand'] == hand
            h = A.field(t[take], p['hand_mass'][take], query, width, scale)
            independent = []
            for k in np.unique(col[take]):
                lane = col == k
                independent.append(A.field(t[lane], p['lane_mass'][lane], query, width, scale))
            own = np.maximum.reduce(independent) if independent else np.zeros_like(h)
            increment = np.maximum(h - own, 0.0)
            extra.append(increment)
            states.append(dict(hand_rate=h, independent_max=own, extra=increment))
        outputs.append(np.maximum(extra[0], extra[1]))
        details.append(states)
    return dict(extra=np.sqrt(np.mean(np.square(outputs), axis=0)), profiles=details, scope='Additional rate of coherent repeated hand contacts above independent-finger repeats. Not wrist speed, difficulty/SR, or generic superset monotonicity.')

def field(data, query, width=1000.0):
    t = data['time']
    col = data['column']
    scale = data['scale']
    profiles = data['profiles']
    outputs = []
    details = []
    if any((not np.array_equal(p['lane_mass'], profiles[0]['lane_mass']) for p in profiles[1:])):
        return _general_field(data, query, width)
    lanes = {int(k): A.field(t[col == k], profiles[0]['lane_mass'][col == k], query, width, scale) for k in np.unique(col)}
    previous = []
    for p in profiles:
        extra = []
        states = []
        reuse = None
        for old, oldfields in previous:
            if np.array_equal(p['hand'], old['hand']) and np.array_equal(p['hand_mass'], old['hand_mass']):
                reuse = oldfields
                break
        hands = []
        for hand in [0, 1]:
            take = p['hand'] == hand
            h = reuse[hand] if reuse is not None else A.field(t[take], p['hand_mass'][take], query, width, scale)
            hands.append(h)
            independent = [lanes[int(k)] for k in np.unique(col[take])]
            own = np.maximum.reduce(independent) if independent else np.zeros_like(h)
            increment = np.maximum(h - own, 0.0)
            extra.append(increment)
            states.append(dict(hand_rate=h, independent_max=own, extra=increment))
        previous.append((p, hands))
        outputs.append(np.maximum(extra[0], extra[1]))
        details.append(states)
    return dict(extra=np.sqrt(np.mean(np.square(outputs), axis=0)), profiles=details, scope='Bitwise-identical reused physical-finger/profile projection; same detector prior and limitations.')

def eligibility(*args):
    lane, eligible, visits = _raw_eligibility(*args)
    assert np.all(np.isfinite(lane)) and np.all(np.isfinite(eligible))
    assert np.all(lane >= -1e-12) and np.all(lane <= 1.0 + 1e-12)
    assert np.all(eligible >= -1e-12) and np.all(eligible <= lane + 1e-12)
    lane = np.clip(lane, 0.0, 1.0)
    return (lane, np.clip(eligible, 0.0, lane), visits)
F = SimpleNamespace(eligibility=eligibility)

def extract(notes, scale=30.0):
    data = _extract(notes, scale)
    for profile in data['profiles']:
        assert np.all(profile['hand_mass'] <= profile['eligible'] + 1e-12)
        profile['hand_mass'] = np.clip(profile['hand_mass'], 0.0, profile['eligible'])
    return data
