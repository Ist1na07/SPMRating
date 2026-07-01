"""
SPM Rating — Playtest data loader.

Reads Dan and Tournament playtest Excel files, parses the difficulty
encoding (including special values g/a/z/s), and computes reference
SR with error bounds.
"""

import os
import re
import pandas as pd
import numpy as np


# Difficulty encoding map
DIFF_MAP = {
    "g": 11.5,
    "a": 13.0,
    "z": 14.5,
    "s": 16.0,
}

# Difficulty to Star Rating conversion
def diff_to_sr(d, accurate=0, error=0.5):
    """
    Convert difficulty + accurate_difficulty + error to SR scale.

    SR = 3.8 + 0.435 * (difficulty + accurate_difficulty ± error)
    """
    base = 3.8 + 0.435 * d
    acc_mod = 0.435 * accurate
    sr_ref = base + acc_mod
    sr_error = 0.435 * error
    return sr_ref, sr_error


def parse_difficulty(value):
    """
    Parse a difficulty value which may be numeric, a letter code, or unknown.

    Returns:
        float or None (if '?' or empty/NaN)
    """
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if value == "?" or value == "":
            return None
        if value.lower() in DIFF_MAP:
            return DIFF_MAP[value.lower()]
        try:
            return float(value)
        except ValueError:
            return None
    return None


def parse_float(value, default=0.5):
    """Parse a numeric value that may be a float, int, string with comma decimal,
    NaN, or missing. Returns default if unparseable/missing."""
    if pd.isna(value):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s == "" or s == "?":
        return default
    # handle comma decimal: '1,5' -> '1.5'
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return default


def parse_accurate(value, difficulty_value):
    """
    Parse accurate difficulty notation.

    Args:
        value: "+" / "-" / "0" or numeric
        difficulty_value: the base difficulty (to determine tier)

    Returns:
        float modifier
    """
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip()
    if value == "+":
        return 0.33 if (difficulty_value and difficulty_value <= 10) else 0.5
    elif value == "-":
        return -0.33 if (difficulty_value and difficulty_value <= 10) else -0.5
    elif value == "0":
        return 0.0
    return 0.0


def load_playtest_data(maps_root=None):
    """
    Load all playtest data from Dan and Tournament Excel files.

    Args:
        maps_root: root directory containing maps/ folder

    Returns:
        list of dicts, each containing:
            - mapfile: filename (with extension)
            - sr_ref: reference SR
            - sr_error: error bound in SR scale
            - d_rc: RC sub-difficulty (or None)
            - d_ln: LN sub-difficulty (or None)
            - tags: category tags string
            - sort: "rc" or "ln"
            - source: "dan" or "tournament"
            - osu_path: full path to .osu file (or None if not found)
    """
    if maps_root is None:
        maps_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    entries = []

    # === Dan Playtest ===
    dan_xlsx = os.path.join(maps_root, "maps", "Dan", "Dan_Playtest.xlsx")
    if os.path.exists(dan_xlsx):
        df = pd.read_excel(dan_xlsx)
        entries += _parse_playtest_df(df, "dan")

    # === Tournament Playtest ===
    tour_xlsx = os.path.join(maps_root, "maps", "Tournaments", "TournamentsPlaytest.xlsx")
    if os.path.exists(tour_xlsx):
        df = pd.read_excel(tour_xlsx)
        entries += _parse_playtest_df(df, "tournament")

    # === Graveyard Playtest ===
    grave_xlsx = os.path.join(maps_root, "maps", "Graveyard", "GraveyardPlaytest.xlsx")
    if os.path.exists(grave_xlsx):
        df = pd.read_excel(grave_xlsx)
        entries += _parse_playtest_df(df, "graveyard")

    # === Ranked Playtest ===
    ranked_xlsx = os.path.join(maps_root, "maps", "Ranked", "RankedPlaytest.xlsx")
    if os.path.exists(ranked_xlsx):
        df = pd.read_excel(ranked_xlsx)
        entries += _parse_playtest_df(df, "ranked")

    # === Map .osu file paths to entries ===
    _match_osu_paths(entries, maps_root)

    # === Filter: only entries where we have both the .osu file and valid reference ===
    valid = [e for e in entries if e["osu_path"] is not None and e["sr_ref"] is not None]

    return valid


def _parse_playtest_df(df, source):
    """Parse a playtest DataFrame into entry dicts."""
    entries = []
    for _, row in df.iterrows():
        mapfile = row.get("mapfile", "")
        if pd.isna(mapfile) or not str(mapfile).strip():
            continue

        mapfile = str(mapfile).strip()
        if not mapfile.endswith(".osu"):
            mapfile += ".osu"

        # Total difficulty
        diff_raw = parse_difficulty(row.get("difficulty"))
        if diff_raw is None:
            continue

        acc_raw = parse_accurate(row.get("accurate difficulty"), diff_raw)
        error_raw = parse_float(row.get("error", 0.5), 0.5)

        sr_ref, sr_error = diff_to_sr(diff_raw, acc_raw, error_raw)

        # RC sub-difficulty
        d_rc = parse_difficulty(row.get("d(rc)"))
        ad_rc = parse_accurate(row.get("ad(rc)"), d_rc)
        e_rc = parse_float(row.get("e(rc)", 0.5), 0.5)

        # LN sub-difficulty
        d_ln = parse_difficulty(row.get("d(ln)"))
        ad_ln = parse_accurate(row.get("ad(ln)"), d_ln)
        e_ln = parse_float(row.get("e(ln)", 0.5), 0.5)

        tags = str(row.get("tags", "")) if pd.notna(row.get("tags")) else ""
        sort = str(row.get("sort", "")) if pd.notna(row.get("sort")) else ""

        entries.append({
            "mapfile": mapfile,
            "sr_ref": sr_ref,
            "sr_error": max(sr_error, 0.05),  # minimum error bound
            "d_rc": d_rc,
            "d_ln": d_ln,
            "sr_ref_rc": diff_to_sr(d_rc, ad_rc, e_rc)[0] if d_rc is not None else None,
            "sr_error_rc": max(diff_to_sr(d_rc, ad_rc, e_rc)[1], 0.05) if d_rc is not None else None,
            "sr_ref_ln": diff_to_sr(d_ln, ad_ln, e_ln)[0] if d_ln is not None else None,
            "sr_error_ln": max(diff_to_sr(d_ln, ad_ln, e_ln)[1], 0.05) if d_ln is not None else None,
            "tags": tags,
            "sort": sort,
            "source": source,
            "osu_path": None,  # filled by _match_osu_paths
        })

    return entries


def _match_osu_paths(entries, maps_root):
    """Find the actual .osu file for each playtest entry."""
    maps_dir = os.path.join(maps_root, "maps")

    # Build index of all .osu files
    osu_files = {}
    for root, dirs, files in os.walk(maps_dir):
        for f in files:
            if f.endswith(".osu") and not f.startswith("."):
                osu_files[f] = os.path.join(root, f)

    for entry in entries:
        fname = entry["mapfile"]
        # Try exact match first
        if fname in osu_files:
            entry["osu_path"] = osu_files[fname]
        else:
            # Try fuzzy match
            for k, v in osu_files.items():
                # Check if one contains the other
                if fname.replace(".osu", "") in k or k.replace(".osu", "") in fname:
                    entry["osu_path"] = v
                    break

    return entries
