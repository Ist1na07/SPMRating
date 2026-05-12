#!/usr/bin/env python
"""
SPM Rating — Nelder-Mead for Enhanced Mode (v3: 31 params, separated U-curve).

Optimizes ~31 key params (combine + aggregate + Inverse spike/guide + Release + Shield)
using scipy's Nelder-Mead.

Changes from v2:
  - Inverse U-curve split into independent spike + guide dip (7 params)
  - Release seq_coeff now tunable
  - Shield anchor_mod + coord_factor unlocked
  - Cross column distance params fixed (non-tunable)
  - Total: 31 params (was 25)
"""

import os, sys, time, json, pickle, numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)

from scipy.optimize import minimize
from tuning.data_loader import load_playtest_data
from tuning.scorer import score_batch
from spm_rating import rating

CACHE_FILE = os.path.join(_project_root, "cache", "precomputed_enhanced.pkl")
OUTPUT_FILE = os.path.join(_project_root, "tuned_params_enhanced.json")

# ============================================================
# Parameter definitions — 31 tunable params
# ============================================================
# (name, start, lower, upper)
PARAM_DEFS = [
    # Combine: S formula (2)
    ("S_w1",    0.572, 0.1,  0.9),
    ("S_p",     1.109, 0.5,  4.0),
    # Combine: stream branch (5)
    ("alpha_P", 0.725, 0.1,  3.0),
    ("alpha_R", 28.36, 5.0,  100.0),
    ("alpha_C", 9.595, 2.0,  30.0),
    ("alpha_S", 0.480, 0.0,  5.0),
    ("alpha_V", 0.438, 0.0,  5.0),
    # Combine: D formula (2)
    ("D_beta1", 1.122, 0.5,  10.0),
    ("D_beta2", 0.389, 0.05, 1.0),
    # Aggregate (6)
    ("w_93",          0.182, 0.05, 0.5),
    ("w_83",          0.237, 0.05, 0.5),
    ("coeff_93",      0.960, 0.5,  1.5),
    ("coeff_83",      0.605, 0.5,  1.5),
    ("mean_power",    2.103, 1.0,  10.0),
    ("note_norm_N0",  10.06, 10,   200),
    ("global_scale",  1.052, 0.9,  1.1),
    # Inverse spike: very close same-col → harder (3)
    ("inv_amplitude",         3.009, 0.5, 15.0),
    ("inv_tau",               31.05, 5,   80),
    ("inv_power",             1.0,   0.5, 3.0),
    # Guide dip: medium distance → easier (3)
    ("guide_depth",           0.90,  0.1, 5.0),
    ("guide_center",          78,    30,  200),
    ("guide_width",           31,    10,  100),
    # Cross-column guide (1)
    ("cross_guide_scale",     0.67,  0.1, 2.0),
    # Same-column bonus (1)
    ("inverse_same_col_bonus", 3.624, 1.0, 8.0),
    # Shield (3)
    ("shield_tau_ms",          83.9,  20,  500),
    ("shield_anchor_mod",      1.0,   0.1, 5.0),
    ("shield_coord_factor",    1.0,   0.1, 3.0),
    # Release (6)
    ("release_tail_coeff",     0.082, 0.01, 0.5),
    ("release_tail_to_tap",    1.182, 0.3,  3.0),
    ("release_same_col_bonus", 1.0,   0.3,  4.0),
    ("release_coord_exponent", 0.876, 0.3,  2.5),
    ("release_seq_coeff",      0.03,  0.005, 0.2),
    ("lock_interaction_coeff", 0.252, 0.0,  1.0),
]

# Fixed params (not tuned here)
FIXED_PARAMS = {
    "use_enhanced": True,
    "use_enhanced_release": 1,
    "use_column_distance": 1,
    "use_shield": 1,
    "use_inverse": 1,
    "use_stamina": 0,
    "use_comprehensiveness": 0,
    "D_gamma_e": 0.0,
    "w_mean": 0.572,
    "rescale_threshold": 9.54,
    "rescale_divisor": 2.0,
    "jack_aggregation_power": 3.98,
    "multi_jack_boost": 0.003,
    "Abar_scale": 1.016,
    # Precompute-level (fixed — need recache to change)
    "stream_booster_scale": 1.75e-7,
    # Cross column distance (fixed — low impact, keep at defaults)
    "cross_dist_exponent": 1.0,
    "cross_same_hand_penalty": 0.3,
    "cross_thumb_bridge_factor": 0.5,
    # Release — low sensitivity, keep fixed
    "short_ln_threshold": 200,
    "short_ln_reduction": 0.5,
}


def load_cache():
    print("Loading enhanced cache...", end=" ", flush=True)
    with open(CACHE_FILE, "rb") as f:
        caches = pickle.load(f)["caches"]
    print(f"{len(caches)} maps")
    return caches


# Keep original params loaded from file (preserves all ES-tuned values)
_ORIG_START_PARAMS = {}

def x_to_params(x):
    """Convert optimization vector to parameter dict, preserving ES-tuned params."""
    params = dict(_ORIG_START_PARAMS)
    for i, (name, _, _, _) in enumerate(PARAM_DEFS):
        params[name] = float(x[i])
    return params


def params_to_x(params_dict):
    """Convert parameter dict to optimization vector."""
    x = []
    for name, start, _, _ in PARAM_DEFS:
        x.append(params_dict.get(name, start))
    return np.array(x, dtype=float)


def clip_x(x):
    """Clip x to bounds."""
    for i, (_, _, lb, ub) in enumerate(PARAM_DEFS):
        x[i] = float(np.clip(x[i], lb, ub))
    return x


def objective(x, entries, cache_map, evals_counter):
    """NM objective: piecewise-linear loss with dead zone."""
    x = clip_x(x)
    params = x_to_params(x)
    evals_counter[0] += 1

    preds, refs, errs = [], [], []
    for e in entries:
        try:
            sr, _ = rating.combine(cache_map[e["mapfile"]], params=params)
            preds.append(sr)
            refs.append(e["sr_ref"])
            errs.append(e["sr_error"])
        except Exception:
            return 999.0

    loss, _, _ = score_batch(preds, refs, errs)

    if evals_counter[0] % 5 == 0:
        print(f"  Eval {evals_counter[0]}: loss={loss:.4f}", flush=True)

    return loss


def callback(xk, evals_counter, t_start):
    """NM callback: save checkpoint periodically."""
    elapsed = time.time() - t_start
    xk = clip_x(xk)
    params = dict(_ORIG_START_PARAMS)
    for i, (name, _, _, _) in enumerate(PARAM_DEFS):
        params[name] = float(xk[i])

    # Quick evaluation for MAE
    # (We could skip this to save time, but it's useful for monitoring)
    print(f"  NM iter {evals_counter[0]}: simplex best point saved ({elapsed:.0f}s)", flush=True)

    # Save checkpoint
    output = {
        "mae": None,  # Don't compute here to save time
        "params": params,
        "nm_evals": evals_counter[0],
        "elapsed_s": elapsed,
    }
    with open(OUTPUT_FILE + ".nm_checkpoint", "w") as f:
        json.dump(output, f, indent=2)


def main():
    print("=" * 70)
    print("SPM Rating — Focused Nelder-Mead for Enhanced Mode")
    print(f"{len(PARAM_DEFS)} parameters")
    print("=" * 70)
    sys.stdout.flush()

    # Load data
    entries = load_playtest_data(maps_root=_project_root)
    cache_map = load_cache()

    # Starting point
    with open(OUTPUT_FILE) as f:
        best = json.load(f)
    start_params = best.get("params", {})

    # Preserve all ES-tuned params as base (not just FIXED_PARAMS)
    global _ORIG_START_PARAMS
    _ORIG_START_PARAMS = dict(start_params)

    # Clone MAE for comparison
    clone_file = os.path.join(_project_root, "tuned_params.json")
    with open(clone_file) as f:
        clone_best = json.load(f)
    clone_mae = clone_best.get("mae", 0.1836)

    print(f"Starting from ES-tuned params (MAE={best.get('mae', 'N/A')})")
    sys.stdout.flush()

    x0 = params_to_x(start_params)
    x0 = clip_x(x0)

    # Compute starting MAE
    evals_counter = [0]
    t_start = time.time()
    mae_start = objective(x0, entries, cache_map, evals_counter)
    evals_counter[0] = 0  # Reset
    print(f"Starting MAE: {mae_start:.4f}")

    # Clone MAE for comparison
    with open(clone_file) as f:
        clone_mae = json.load(f).get("mae", 0.1836)
    print(f"Clone best MAE: {clone_mae:.4f} (target to beat)")
    print(f"Gap to close: {mae_start - clone_mae:.4f}")
    sys.stdout.flush()

    # Nelder-Mead
    print("\nStarting Nelder-Mead...")
    print(f"Estimated ~500-800 evaluations, ~1.5-2.5 hours")
    sys.stdout.flush()

    result = minimize(
        objective,
        x0,
        args=(entries, cache_map, evals_counter),
        method="Nelder-Mead",
        options={
            "maxiter": 800,
            "xatol": 1e-6,
            "fatol": 1e-7,
            "adaptive": True,
        },
        callback=lambda xk: callback(xk, evals_counter, t_start),
    )

    elapsed = time.time() - t_start

    # Final evaluation
    x_final = clip_x(result.x)
    params_final = x_to_params(x_final)
    mae_final = objective(x_final, entries, cache_map, [0])

    print(f"\n{'=' * 70}")
    print(f"NM Complete: {result.nit} iters, {result.nfev} evals, {elapsed:.0f}s")
    print(f"Start MAE: {mae_start:.4f}")
    print(f"Final MAE: {mae_final:.4f}")
    impr = (1 - mae_final / mae_start) * 100
    print(f"Improvement: {impr:+.2f}%")
    if mae_final < clone_mae:
        print(f"BEATS Clone ({clone_mae:.4f}) by {(1 - mae_final/clone_mae)*100:.2f}%!")
    else:
        print(f"Clone lead: {(1 - clone_mae/mae_final)*100:.2f}%")
    print(f"{'=' * 70}")

    # Save
    output = {
        "mae": mae_final,
        "params": params_final,
        "method": "Nelder-Mead",
        "nm_evals": result.nfev,
        "nm_iters": result.nit,
        "start_mae": mae_start,
        "clone_mae": clone_mae,
        "improvement_pct": impr,
        "elapsed_s": elapsed,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
