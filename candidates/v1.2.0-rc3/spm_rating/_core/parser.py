"""Minimal, fast .osu parser for osu!mania difficulty analysis.

Only what the difficulty model needs is extracted:
  * key count (CircleSize for mania)
  * overall difficulty (OD) -- with the standard rate conversion
  * the note list: (start_ms, column, end_ms) in chronological order

Everything downstream works in a fixed 7-column frame (7K native).  4K
charts are embedded as columns 0,1,2,3 -> 1,2,4,5 (tracks 0, 3, 6 empty),
which keeps physical column distances meaningful (2 -> 4 jumps over the
thumb track) and lets one geometry serve both keyspaces.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

MANIA_MODE = 3

# 4K original column -> 7K frame column
EMBED_4K_TO_7K = (1, 2, 4, 5)

# 6K original column -> 7K frame column.  Default is the historical generic
# embedding round(c*6/(K-1)) = (0, 1, 2, 4, 5, 6), i.e. the two hands sit in
# frame {0,1,2} and {4,5,6} with the thumb column 3 empty.  Overriding this
# table (or the SPM_EMBED_6K environment variable, same format) selects a
# different physical embedding for 6K without touching 4K/7K parsing.
EMBED_6K_TO_7K = (0, 1, 2, 4, 5, 6)


def embed_6k() -> Tuple[int, ...]:
    """Active 6K -> 7K embedding (env override wins over the module table)."""
    raw = os.environ.get("SPM_EMBED_6K")
    if not raw:
        return EMBED_6K_TO_7K
    try:
        vals = tuple(int(v) for v in raw.replace(" ", "").split(","))
    except ValueError:
        return EMBED_6K_TO_7K
    if len(vals) != 6 or len(set(vals)) != 6 or any(v < 0 or v > 6 for v in vals):
        return EMBED_6K_TO_7K
    return vals

# Default OD -> hit window mapping is not used directly; we only need a
# normalised "timing pressure" scalar.  Standard osu!mania windows (ms):
OD_WINDOWS = {0: (22.0, 64.0, 97.0, 151.0, 188.0),
              5: (19.5, 49.0, 82.0, 136.0, 172.0),
              10: (16.0, 34.0, 67.0, 121.0, 151.0)}


def od_windows(od: float) -> Tuple[float, float, float, float, float]:
    """Interpolate the osu!mania hit windows (300g, 300, 200, 100, 50) for OD."""
    od = float(min(max(od, 0.0), 10.0))
    lo_i = int(od // 5) * 5
    hi_i = min(lo_i + 5, 10)
    t = (od - lo_i) / 5.0
    a, b = OD_WINDOWS[lo_i], OD_WINDOWS[hi_i]
    return tuple(a[k] + (b[k] - a[k]) * t for k in range(5))  # type: ignore[return-value]


def rate_od(od: float, rate: float) -> float:
    """Convert OD to the given speed rate (DT raises effective OD)."""
    # standard: OD_ms = rate-scaled; new_od such that window/rate matches
    if rate <= 0:
        return od
    w = od_windows(od)[0]
    target = w / rate
    lo, hi = 0.0, 10.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if od_windows(mid)[0] > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@dataclass
class Notes:
    """Note arrays in the 7K frame.  All arrays share the leading axis."""
    time: np.ndarray        # float64, ms, non-decreasing
    col: np.ndarray         # int32, 0..6
    ln_end: np.ndarray      # float64, ms; == time for non-LN
    is_ln: np.ndarray       # bool
    key_count: int          # original key count (4 or 7)
    od: float               # effective OD after rate conversion
    rate: float
    n_notes: int
    # per-column index lists (list of 7 int64 arrays)
    col_idx: Tuple[np.ndarray, ...] = ()
    # derived
    duration: float = 0.0
    title: str = ""

    @property
    def n_ln(self) -> int:
        return int(self.is_ln.sum())

    @property
    def ln_ratio(self) -> float:
        if self.n_notes == 0:
            return 0.0
        return float(self.is_ln.sum()) / float(self.n_notes)


def col_from_x(x: float, key_count: int) -> int:
    """osu!mania encodes the column as x = round(512*(c+0.5)/K)."""
    c = int(float(x) * key_count / 512.0)
    if c < 0:
        c = 0
    if c > key_count - 1:
        c = key_count - 1
    return c


def _col_from_x(x: float, key_count: int) -> int:
    c = col_from_x(x, key_count)
    if key_count == 7:
        return c
    if key_count == 4:
        return EMBED_4K_TO_7K[c]
    if key_count == 6:
        # explicit embedding table (default == the generic round() below)
        return embed_6k()[c]
    # generic: spread evenly across the 7 tracks
    if key_count <= 1:
        return 3
    return int(round(float(c) * 6.0 / (key_count - 1)))


def parse_content(text: str, rate: float = 1.0,
                  title: str = "") -> Optional[Notes]:
    lines = text.splitlines()
    key_count = 0
    od = 5.0
    section = ""
    hit_objects: List[Tuple[int, float, int, int]] = []   # (type, t, x, end)
    in_objects = False
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("["):
            section = line.strip("[]").strip()
            in_objects = section == "HitObjects"
            continue
        if ":" in line and not in_objects:
            k, _, v = line.partition(":")
            k = k.strip()
            v = v.strip()
            if k == "Mode" and v != str(MANIA_MODE):
                return None
            if k == "CircleSize":
                try:
                    key_count = int(float(v))
                except ValueError:
                    pass
            elif k == "OverallDifficulty":
                try:
                    od = float(v)
                except ValueError:
                    pass
            continue
        if in_objects:
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                x = int(float(parts[0]))
                t = float(parts[2])
                typ = int(parts[3])
            except ValueError:
                continue
            end = t
            if typ & 128:
                end = float(parts[5].split(":")[0]) if len(parts) > 5 else t
            hit_objects.append((typ, t, x, end))
    if not hit_objects or key_count <= 0:
        return None

    eff_od = rate_od(od, rate)
    hit_objects.sort(key=lambda o: (o[1], o[2]))
    n = len(hit_objects)
    time = np.empty(n, dtype=np.float64)
    col = np.empty(n, dtype=np.int32)
    ln_end = np.empty(n, dtype=np.float64)
    is_ln = np.zeros(n, dtype=bool)
    for i, (typ, t, x, end) in enumerate(hit_objects):
        time[i] = t / rate
        col[i] = _col_from_x(x, key_count)
        if typ & 128 and end > t:
            ln_end[i] = end / rate
            is_ln[i] = True
        else:
            ln_end[i] = time[i]
    order = np.argsort(time, kind="stable")
    time = time[order]
    col = col[order]
    ln_end = ln_end[order]
    is_ln = is_ln[order]

    col_idx = tuple(np.flatnonzero(col == c) for c in range(7))
    return Notes(time=time, col=col, ln_end=ln_end, is_ln=is_ln,
                 key_count=key_count, od=eff_od, rate=rate, n_notes=n,
                 col_idx=col_idx,
                 duration=float(time[-1] - time[0]) if n else 0.0,
                 title=title)


def parse_file(path: str, rate: float = 1.0) -> Optional[Notes]:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()
    return parse_content(text, rate=rate, title=os.path.basename(path))
