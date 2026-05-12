#!/usr/bin/env python
"""
Rebuild enhanced cache to v8 (adds cross_data for RC/LN param blending).
Uses the current rating.py (which must have cross_data support).

Usage: python scripts/rebuild_enhanced_cache.py
"""
import os, sys, time, pickle

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)

from tuning.data_loader import load_playtest_data
from spm_rating import rating

CACHE_DIR = os.path.join(_project_root, "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "precomputed_enhanced.pkl")

def main():
    print("=" * 70)
    print("Rebuilding Enhanced Cache v8 (with cross_data + ln_ratio)")
    print("=" * 70)

    # Load all entries
    entries = load_playtest_data(maps_root=_project_root)
    print(f"Entries: {len(entries)}")

    # Precompute
    cache_map = {}
    t0 = time.time()
    for i, e in enumerate(entries):
        path = e["osu_path"]
        fname = e["mapfile"]
        try:
            cache = rating.precompute(path, use_enhanced=True)
            cache_map[fname] = cache
        except Exception as ex:
            print(f"  FAIL [{fname}]: {ex}")
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(entries)} ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nDone: {len(cache_map)}/{len(entries)} maps in {elapsed:.0f}s")

    # Verify cross_data present
    sample = next(iter(cache_map.values()))
    has_cross = "cross_data" in sample
    has_ln = "ln_ratio" in sample
    print(f"cross_data: {'YES' if has_cross else 'NO'}, ln_ratio: {'YES' if has_ln else 'NO'}")

    if not has_cross:
        print("ERROR: cross_data missing! Check rating.py precompute().")
        return

    # Backup old cache
    if os.path.exists(CACHE_FILE):
        backup = CACHE_FILE + ".v7_backup"
        import shutil
        shutil.copy2(CACHE_FILE, backup)
        print(f"Old cache backed up to {backup}")

    # Save
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({
            "version": 8,
            "use_enhanced": True,
            "caches": cache_map,
        }, f)
    print(f"Cache saved: {CACHE_FILE} ({os.path.getsize(CACHE_FILE)/1024/1024:.0f} MB)")

if __name__ == "__main__":
    main()
