#!/usr/bin/env python
"""
SPM Rating — Comprehensive Enhanced ES Tuning.

~30 runtime-tunable parameters, (1+λ)-ES with adaptive sigma.
Uses structured precomputation (cache v7) for fast evaluation (~8s per 186 maps).

Can be interrupted/resumed — loads best from OUTPUT_FILE and continues.
"""

import os, sys, time, json, pickle, numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)

from tuning.data_loader import load_playtest_data
from spm_rating import rating

CACHE_FILE = os.path.join(_project_root, "cache", "precomputed_enhanced.pkl")
OUTPUT_FILE = os.path.join(_project_root, "tuned_params_enhanced.json")

rng = np.random.RandomState(42)


def load_enhanced_cache():
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)["caches"]


def evaluate(params_dict, entries, cache_map):
    preds, refs, errs = [], [], []
    for e in entries:
        try:
            sr, _ = rating.combine(cache_map[e["mapfile"]], params=params_dict)
            preds.append(sr)
            refs.append(e["sr_ref"])
            errs.append(e["sr_error"])
        except Exception:
            return 999, 999, 0, 0
    preds = np.array(preds)
    refs = np.array(refs)
    errs = np.array(errs)
    mae = float(np.mean(np.abs(preds - refs)))
    rmse = float(np.sqrt(np.mean((preds - refs) ** 2)))
    with np.errstate(invalid='ignore'):
        corr = float(np.corrcoef(preds, refs)[0, 1])
    if np.isnan(corr):
        corr = 0.0
    in_range = float(np.mean(np.abs(preds - refs) <= errs)) * 100
    return mae, rmse, corr, in_range


def build_full(params_dict):
    """Build full parameter dict with enhanced features enabled."""
    d = {
        # Precompute-level (fixed)
        "use_enhanced": True,
        "use_enhanced_release": 1,
        "use_column_distance": 1,
        "use_shield": 1,
        "use_inverse": 1,
        "use_stamina": 0,
        "use_comprehensiveness": 0,
        # Stream (fixed — requires recache to change)
        "stream_booster_scale": params_dict.get("stream_booster_scale", 1.7e-7),
        # Cross enhanced (fixed — requires recache)
        "cross_dist_exponent": params_dict.get("cross_dist_exponent", 1.0),
        "cross_same_hand_penalty": params_dict.get("cross_same_hand_penalty", 0.3),
        "cross_thumb_bridge_factor": params_dict.get("cross_thumb_bridge_factor", 0.5),
        # Release enhanced (fixed — requires recache)
        "release_tail_coeff": params_dict.get("release_tail_coeff", 1.0),
        "release_tail_to_tap": params_dict.get("release_tail_to_tap", 0.8),
        "release_same_col_bonus": params_dict.get("release_same_col_bonus", 1.5),
        "release_coord_exponent": params_dict.get("release_coord_exponent", 1.0),
    }
    # Copy all provided params
    d.update(params_dict)
    return d


# ============================================================
# Parameter definitions: (name, start_value, lower, upper)
# ============================================================
def get_es_params(best_p):
    """Return list of (name, start, lower, upper) for ES optimization."""
    return [
        # --- Combine: S formula ---
        ("S_w1",    best_p.get("S_w1", 0.4),    0.1,  0.9),
        ("S_p",     best_p.get("S_p", 1.5),     0.5,  4.0),

        # --- Combine: stream branch weights ---
        ("alpha_P", best_p.get("alpha_P", 0.8),   0.1,  3.0),
        ("alpha_R", best_p.get("alpha_R", 35.0),  5.0,  100.0),
        ("alpha_C", best_p.get("alpha_C", 8.0),   2.0,  30.0),
        ("alpha_S", best_p.get("alpha_S", 1.0),   0.0,  5.0),
        ("alpha_V", best_p.get("alpha_V", 1.0),   0.0,  5.0),

        # --- Combine: D formula ---
        ("D_beta1", best_p.get("D_beta1", 2.7),   0.5,  10.0),
        ("D_beta2", best_p.get("D_beta2", 0.27),  0.05, 1.0),
        ("D_gamma_e", best_p.get("D_gamma_e", 0.0), 0.0, 2.0),

        # --- Aggregate: SR formula ---
        ("w_93",          best_p.get("w_93", 0.25),   0.05, 0.5),
        ("w_83",          best_p.get("w_83", 0.20),   0.05, 0.5),
        ("w_mean",        best_p.get("w_mean", 0.55), 0.2,  0.8),
        ("mean_power",    best_p.get("mean_power", 5), 1.0,  10.0),
        ("coeff_93",      best_p.get("coeff_93", 0.88), 0.5, 1.5),
        ("coeff_83",      best_p.get("coeff_83", 0.94), 0.5, 1.5),
        ("note_norm_N0",  best_p.get("note_norm_N0", 60), 10, 200),
        ("rescale_threshold", best_p.get("rescale_threshold", 9), 7, 12),
        ("rescale_divisor",   best_p.get("rescale_divisor", 1.2), 1.05, 2.0),
        ("global_scale",      best_p.get("global_scale", 0.975), 0.9, 1.1),

        # --- Shield ---
        ("shield_tau_ms",      best_p.get("shield_tau_ms", 100), 20, 500),
        ("shield_anchor_mod", best_p.get("shield_anchor_mod", 1.0), 0.1, 5.0),
        ("shield_coord_factor", best_p.get("shield_coord_factor", 1.0), 0.1, 3.0),

        # --- Inverse ---
        ("inverse_max_amplitude",  best_p.get("inverse_max_amplitude", 2.0), 0.5, 10.0),
        ("inverse_peak_time",      best_p.get("inverse_peak_time", 40), 10, 120),
        ("inverse_peak_width",     best_p.get("inverse_peak_width", 2.0), 0.5, 5.0),
        ("inverse_same_col_bonus", best_p.get("inverse_same_col_bonus", 2.0), 1.0, 5.0),
        # Note: inverse_window_ms and inverse_long_decay are precompute-only (NOT runtime-tunable)

        # --- Runtime-tunable jack/anchor (carried from Clone best) ---
        ("jack_aggregation_power", best_p.get("jack_aggregation_power", 5), 2, 10),
        ("multi_jack_boost",       best_p.get("multi_jack_boost", 0.0), 0.0, 0.05),
        ("Abar_scale",             best_p.get("Abar_scale", 1.0), 0.8, 1.3),
    ]


def select_subset(entries, n=30):
    sorted_e = sorted(entries, key=lambda e: e["sr_ref"])
    idx = np.linspace(0, len(sorted_e) - 1, n, dtype=int)
    return [sorted_e[i] for i in idx]


def main():
    print("=" * 70)
    print("SPM Rating — Comprehensive Enhanced ES Tuning")
    print("=" * 70)
    sys.stdout.flush()

    # Load data
    entries = load_playtest_data(maps_root=_project_root)
    cache_map = load_enhanced_cache()
    subset = select_subset(entries, n=30)
    print(f"{len(entries)} maps total, {len(subset)} in subset")
    sys.stdout.flush()

    # Load previous best if exists
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            prev = json.load(f)
        best_p = prev["params"]
        mae_baseline = prev.get("baseline_mae", prev["mae"])
        print(f"Resuming from {OUTPUT_FILE}: MAE={prev['mae']:.4f}")
    else:
        # Use default enhanced params
        from spm_rating.config import get_default_params
        best_p = {k: v[0] for k, v in get_default_params().items()}
        best_p["use_enhanced"] = True
        best_p["use_enhanced_release"] = 1
        best_p["use_column_distance"] = 1
        best_p["use_shield"] = 1
        best_p["use_inverse"] = 1
        mae_baseline = None
        print("Starting from defaults")
    sys.stdout.flush()

    # Get ES params
    es_params = get_es_params(best_p)
    names = [t[0] for t in es_params]
    x_best = np.array([t[1] for t in es_params], dtype=float)
    lb = np.array([t[2] for t in es_params], dtype=float)
    ub = np.array([t[3] for t in es_params], dtype=float)
    n = len(names)

    # Compute baseline MAE
    d_baseline = build_full(best_p)
    mae_start, rmse_start, corr_start, ir_start = evaluate(d_baseline, entries, cache_map)
    if mae_baseline is None:
        mae_baseline = mae_start
    print(f"Baseline MAE={mae_baseline:.4f}, Current MAE={mae_start:.4f}")
    sys.stdout.flush()

    best_mae = mae_start

    # ES config
    popsize = 12
    generations = 80
    sigma = 0.05

    print(f"ES: {n} params, popsize={popsize}, {generations} gens = {popsize * generations} evals")
    print(f"~{(popsize * generations * 9) / 60:.0f} min estimated")
    print(f"Params: {', '.join(names)}")
    sys.stdout.flush()

    t_start = time.time()
    success_streak = 0
    best_gen = 0

    for gen in range(1, generations + 1):
        improved = False
        for i in range(popsize):
            # Perturb in normalized space
            noise = rng.normal(0, sigma, n) * (ub - lb) * 0.25
            trial = np.clip(x_best + noise, lb, ub)
            overrides = {names[j]: float(trial[j]) for j in range(n)}
            d_test = build_full({**best_p, **overrides})
            mae_test, _, _, _ = evaluate(d_test, entries, cache_map)

            if mae_test < best_mae:
                best_mae = mae_test
                x_best = trial.copy()
                improved = True
                best_gen = gen
                best_p.update(overrides)

        # Adaptive sigma
        if improved:
            success_streak += 1
            if success_streak >= 2:
                sigma = min(sigma * 1.15, 0.2)
                success_streak = 0
        else:
            success_streak = 0
            sigma = max(sigma * 0.93, 0.005)

        elapsed = time.time() - t_start
        impr = (1 - best_mae / mae_baseline) * 100
        impr_gen = (1 - best_mae / mae_start) * 100
        marker = " *" if improved else ""
        print(f"  Gen {gen:3d}: MAE={best_mae:.4f} ({impr:+.1f}% vs baseline, {impr_gen:+.1f}% this run) "
              f"σ={sigma:.4f} {elapsed:.0f}s{marker}")
        sys.stdout.flush()

        # Save checkpoint every 10 gens or on improvement
        if gen % 10 == 0 or (improved and gen - best_gen < 1):
            d_save = build_full(best_p)
            mae_ck, rmse_ck, corr_ck, ir_ck = evaluate(d_save, entries, cache_map)
            output = {
                "mae": mae_ck, "rmse": rmse_ck, "correlation": corr_ck,
                "in_range_pct": ir_ck, "params": d_save,
                "baseline_mae": mae_baseline,
                "improvement_pct": (1 - mae_ck / mae_baseline) * 100,
                "gen": gen, "best_gen": best_gen,
                "elapsed_s": elapsed,
                "sigma": sigma,
                "n_params": n,
                "param_names": names,
            }
            with open(OUTPUT_FILE, "w") as f:
                json.dump(output, f, indent=2)

    # Final evaluation
    d_final = build_full(best_p)
    mae_f, rmse_f, corr_f, ir_f = evaluate(d_final, entries, cache_map)
    impr = (1 - mae_f / mae_baseline) * 100

    print()
    print("=" * 70)
    print(f"FINAL: MAE={mae_f:.4f} ({impr:+.1f}% vs baseline) "
          f"r={corr_f:.4f} RMSE={rmse_f:.4f} in_range={ir_f:.1f}%")
    print(f"Best found at gen {best_gen}")
    print(f"Total time: {time.time() - t_start:.0f}s")
    print("=" * 70)
    sys.stdout.flush()

    output = {
        "mae": mae_f, "rmse": rmse_f, "correlation": corr_f,
        "in_range_pct": ir_f, "params": d_final,
        "baseline_mae": mae_baseline,
        "improvement_pct": impr,
        "gen": best_gen,
        "elapsed_s": time.time() - t_start,
        "n_params": n,
        "param_names": names,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
