"""
Fit separate RC and LN Dan regression lines by running the actual algorithm
on Dan marathon maps.

Replaces the hardcoded SR = 3.8 + 0.435 * Dan mapping with measured values.

Output: tosustatic/spm-ratingV2pro-sigmoid/dan_constants.json
"""
import sys, os, re, json
import numpy as np
from scipy.stats import linregress

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spm_rating.rating import precompute, combine as combine_total
from spm_rating.combine_rc_ln import compute_rc_sr
from tuning.data_loader import load_playtest_data


def extract_dan_level(filename):
    """Extract numeric Dan level from .osu filename."""
    m = re.search(r'\[(\d+)(?:st|nd|rd|th)\s+Dan', filename)
    if m:
        return float(m.group(1))
    m = re.search(r'\[(Gamma|Azimuth|Zenith)\s+Dan\]', filename)
    if m:
        return {"Gamma": 11.5, "Azimuth": 13, "Zenith": 14.5}.get(m.group(1))
    return None


def find_dan_maps(entries):
    """Find Dan marathon maps (from RegularDan and LNDan folders, sorted by Dan level)."""
    dan_maps = []
    for entry in entries:
        path = entry.get("osu_path", "")
        # Only marathon Dan maps (not course practice maps)
        if "RegularDan" not in path and "LNDan" not in path:
            continue
        # Skip sub-course maps (they're in RegularDanCourses/LNDanCourses subfolders)
        if "Courses" in path or os.path.sep + "release" in path or \
           os.path.sep + "inverse" in path or os.path.sep + "technical" in path or \
           os.path.sep + "general" in path:
            continue
        if "RegularDanCourses" in path or "LNDanCourses" in path:
            continue

        level = extract_dan_level(os.path.basename(path))
        if level is not None:
            # Determine if RC or LN from folder
            map_type = "LN" if "LNDan" in path else "RC"
            dan_maps.append({
                "path": path,
                "level": level,
                "type": map_type,
                "filename": os.path.basename(path),
            })

    dan_maps.sort(key=lambda m: (m["type"], m["level"]))
    return dan_maps


def main():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load params
    with open(os.path.join(ROOT, "tuned_params_sigmoid.json")) as f:
        total_params = dict(json.load(f)["params"])

    # Find Dan maps
    entries = load_playtest_data()
    dan_maps = find_dan_maps(entries)

    print(f"Found {len(dan_maps)} Dan marathon maps:")
    rc_maps = [m for m in dan_maps if m["type"] == "RC"]
    ln_maps = [m for m in dan_maps if m["type"] == "LN"]
    print(f"  RC: {len(rc_maps)} maps")
    for m in rc_maps:
        print(f"    Dan {m['level']:5.1f}  {m['filename'][:70]}")
    print(f"  LN: {len(ln_maps)} maps")
    for m in ln_maps:
        print(f"    Dan {m['level']:5.1f}  {m['filename'][:70]}")

    # Compute SR for each map
    print(f"\nComputing SR for {len(dan_maps)} maps...")
    import time
    t0 = time.time()
    results = []

    for i, dm in enumerate(dan_maps):
        try:
            cache = precompute(dm["path"], use_enhanced=True, params=total_params)
            sr_total, _ = combine_total(cache, total_params)
            results.append({
                "level": dm["level"],
                "type": dm["type"],
                "sr_total": float(sr_total),
                "filename": dm["filename"],
            })
            print(f"  [{i+1}/{len(dan_maps)}] {dm['type']} Dan {dm['level']:5.1f} → SR={sr_total:.3f}")
        except Exception as e:
            print(f"  [{i+1}/{len(dan_maps)}] ERROR {dm['filename'][:60]}: {e}")

    print(f"  Done in {time.time() - t0:.0f}s. {len(results)} valid.")

    # Separate RC and LN results
    rc_results = [r for r in results if r["type"] == "RC"]
    ln_results = [r for r in results if r["type"] == "LN"]

    # Linear regression
    print(f"\n{'='*60}")
    print("LINEAR REGRESSION")
    print(f"{'='*60}")

    # RC regression
    rc_levels = np.array([r["level"] for r in rc_results])
    rc_srs = np.array([r["sr_total"] for r in rc_results])
    rc_fit = linregress(rc_levels, rc_srs)

    print(f"\n  RC regression ({len(rc_results)} points):")
    print(f"    SR_rc = {rc_fit.intercept:.4f} + {rc_fit.slope:.4f} * Dan")
    print(f"    R^2 = {rc_fit.rvalue**2:.4f}")
    print(f"    Residuals:")
    for r in rc_results:
        pred = rc_fit.intercept + rc_fit.slope * r["level"]
        res = r["sr_total"] - pred
        print(f"      Dan {r['level']:5.1f}: measured={r['sr_total']:.3f} "
              f"pred={pred:.3f} res={res:+.3f}")

    # LN regression
    ln_levels = np.array([r["level"] for r in ln_results])
    ln_srs = np.array([r["sr_total"] for r in ln_results])
    ln_fit = linregress(ln_levels, ln_srs)

    print(f"\n  LN regression ({len(ln_results)} points):")
    print(f"    SR_ln = {ln_fit.intercept:.4f} + {ln_fit.slope:.4f} * Dan")
    print(f"    R^2 = {ln_fit.rvalue**2:.4f}")
    print(f"    Residuals:")
    for r in ln_results:
        pred = ln_fit.intercept + ln_fit.slope * r["level"]
        res = r["sr_total"] - pred
        print(f"      Dan {r['level']:5.1f}: measured={r['sr_total']:.3f} "
              f"pred={pred:.3f} res={res:+.3f}")

    # Compare with hardcoded formula
    print(f"\n  Comparison with hardcoded SR=3.8+0.435*Dan:")
    for r in results:
        hardcoded = 3.8 + 0.435 * r["level"]
        if r["type"] == "RC":
            fitted = rc_fit.intercept + rc_fit.slope * r["level"]
        else:
            fitted = ln_fit.intercept + ln_fit.slope * r["level"]
        print(f"    Dan {r['level']:5.1f} ({r['type']}): hardcoded={hardcoded:.3f} "
              f"fitted={fitted:.3f} measured={r['sr_total']:.3f}")

    # Generate Dan threshold arrays
    dan_levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11.5, 13, 14.5, 16]
    dan_names = ['0th', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th',
                 '8th', '9th', '10th', 'Gamma', 'Azimuth', 'Zenith', 'Stellium']

    thresholds_rc = [rc_fit.intercept + rc_fit.slope * L for L in dan_levels]
    thresholds_ln = [ln_fit.intercept + ln_fit.slope * L for L in dan_levels]

    # Piecewise interpolation data (more accurate than single regression line)
    rc_measured_sr = [float(r["sr_total"]) for r in sorted(rc_results, key=lambda r: r["level"])]
    rc_measured_levels = sorted([float(r["level"]) for r in rc_results])
    ln_measured_sr = [float(r["sr_total"]) for r in sorted(ln_results, key=lambda r: r["level"])]
    ln_measured_levels = sorted([float(r["level"]) for r in ln_results])

    # Check monotonicity
    rc_mono = all(rc_measured_sr[i] <= rc_measured_sr[i + 1] for i in range(len(rc_measured_sr) - 1))
    ln_mono = all(ln_measured_sr[i] <= ln_measured_sr[i + 1] for i in range(len(ln_measured_sr) - 1))
    print(f"\n  RC monotonic: {rc_mono}")
    print(f"  LN monotonic: {ln_mono}")

    # Output
    out = {
        "rc": {
            "a": float(rc_fit.intercept),
            "b": float(rc_fit.slope),
            "r2": float(rc_fit.rvalue**2),
            "n_samples": len(rc_results),
            "formula": f"SR_rc = {rc_fit.intercept:.4f} + {rc_fit.slope:.4f} * Dan",
        },
        "ln": {
            "a": float(ln_fit.intercept),
            "b": float(ln_fit.slope),
            "r2": float(ln_fit.rvalue**2),
            "n_samples": len(ln_results),
            "formula": f"SR_ln = {ln_fit.intercept:.4f} + {ln_fit.slope:.4f} * Dan",
        },
        "old_formula": "SR = 3.8 + 0.435 * Dan",
        "dan_levels": dan_levels,
        "dan_names": dan_names,
        # Piecewise interpolation data (primary method)
        "rc_measured_sr": [round(s, 4) for s in rc_measured_sr],
        "rc_measured_levels": [round(L, 1) for L in rc_measured_levels],
        "ln_measured_sr": [round(s, 4) for s in ln_measured_sr],
        "ln_measured_levels": [round(L, 1) for L in ln_measured_levels],
        # Legacy: regression-based thresholds
        "thresholds_rc": thresholds_rc,
        "thresholds_ln": thresholds_ln,
        "javascript": generate_js(rc_fit, ln_fit, rc_measured_sr, rc_measured_levels,
                                  ln_measured_sr, ln_measured_levels,
                                  dan_levels, dan_names, thresholds_rc, thresholds_ln),
    }

    out_path = os.path.join(ROOT, "tosustatic", "spm-ratingV2pro-sigmoid",
                            "dan_constants.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {out_path}")
    print(f"\n=== JavaScript Code ===")
    print(out["javascript"])


def generate_js(rc_fit, ln_fit, rc_sr, rc_lv, ln_sr, ln_lv,
                 dan_levels, dan_names, thresholds_rc, thresholds_ln):
    """Generate JS code for piecewise interpolation Dan mapping."""
    js = "/** Auto-generated Dan mapping constants. */\n\n"

    # === PIECEWISE INTERPOLATION DATA (primary) ===
    js += "// Measured SR at each Dan level (piecewise interpolation)\n"
    js += f"const RC_MEASURED_SR = {json.dumps([round(s, 4) for s in rc_sr])};\n"
    js += f"const RC_MEASURED_LEVELS = {json.dumps([round(lv, 1) for lv in rc_lv])};\n"
    js += f"const LN_MEASURED_SR = {json.dumps([round(s, 4) for s in ln_sr])};\n"
    js += f"const LN_MEASURED_LEVELS = {json.dumps([round(lv, 1) for lv in ln_lv])};\n\n"

    # === DAN LABEL BINS ===
    js += "// Dan label bins\n"
    js += f"const DAN_THRESHOLDS = {json.dumps(dan_levels)};\n"
    js += f"const DAN_NAMES = {json.dumps(dan_names)};\n\n"

    js += """// Piecewise linear interpolation: SR -> Dan level
function interpDan(sr, measuredSR, measuredLevels) {
    const n = measuredSR.length;
    if (n === 0) return 0;
    // Below lowest measured point: extrapolate using first segment slope
    if (sr <= measuredSR[0]) {
        if (n < 2) return measuredLevels[0];
        const slope = (measuredLevels[1] - measuredLevels[0]) / Math.max(measuredSR[1] - measuredSR[0], 0.001);
        return Math.max(0, measuredLevels[0] + slope * (sr - measuredSR[0]));
    }
    // Above highest measured point: extrapolate using last segment slope
    if (sr >= measuredSR[n - 1]) {
        if (n < 2) return measuredLevels[n - 1];
        const slope = (measuredLevels[n - 1] - measuredLevels[n - 2]) / Math.max(measuredSR[n - 1] - measuredSR[n - 2], 0.001);
        return measuredLevels[n - 1] + slope * (sr - measuredSR[n - 1]);
    }
    // Find interval [i, i+1] where sr lies
    let i = 0;
    for (; i < n - 1; i++) {
        if (sr < measuredSR[i + 1]) break;
    }
    const t = (sr - measuredSR[i]) / Math.max(measuredSR[i + 1] - measuredSR[i], 0.0001);
    return measuredLevels[i] + t * (measuredLevels[i + 1] - measuredLevels[i]);
}

function ratingToDanRC(sr) { return interpDan(sr, RC_MEASURED_SR, RC_MEASURED_LEVELS); }
function ratingToDanLN(sr) { return interpDan(sr, LN_MEASURED_SR, LN_MEASURED_LEVELS); }

"""
    # === DAN LABEL FUNCTIONS ===
    js += "function danToLabelRC(danLevel) {\n"
    js += "  return danToLabelGeneric(danLevel, DAN_THRESHOLDS);\n"
    js += "}\n\n"
    js += "function danToLabelLN(danLevel) {\n"
    js += "  return danToLabelGeneric(danLevel, DAN_THRESHOLDS);\n"
    js += "}\n\n"

    js += "function danToLabelGeneric(danLevel, thresholds) {\n"
    js += "  if (danLevel < 0) return '<0th';\n"
    js += "  let idx = 0;\n"
    js += "  for (let i = thresholds.length - 1; i >= 0; i--) {\n"
    js += "    if (danLevel >= thresholds[i]) { idx = i; break; }\n"
    js += "  }\n"
    js += "  if (idx >= DAN_NAMES.length) return '>' + DAN_NAMES[DAN_NAMES.length - 1];\n"
    js += "  const name = DAN_NAMES[idx];\n"
    js += "  const binStart = thresholds[idx];\n"
    js += "  const binEnd = idx < thresholds.length - 1 ? thresholds[idx + 1] : binStart + 1;\n"
    js += "  const binWidth = binEnd - binStart;\n"
    js += "  const frac = binWidth > 0 ? (danLevel - binStart) / binWidth : 0;\n"
    js += "  if (frac < 0.25) return name + ' low';\n"
    js += "  else if (frac < 0.75) return name;\n"
    js += "  else return name + ' high';\n"
    js += "}\n\n"

    # Backward compat
    js += "// Deprecated: use ratingToDanRC/ratingToDanLN\n"
    js += "function ratingToDan(r) { return ratingToDanRC(r); }\n"
    js += "function danToLabel(d) { return danToLabelRC(d); }\n"

    return js


if __name__ == "__main__":
    main()
