"""
Fine k sweep using B1 fast path.
Pre-extracts D_seg per map once, then only runs sigmoid bisection.
Each k value: ~3 seconds (including B1 NM).
"""
import sys, os, json, time, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tuning.data_loader import load_playtest_data
from tuning.scorer import score_single
from spm_rating.rating import precompute, combine
from spm_rating.aggregate_sigmoid import _compute_effective_weights, segment_by_difficulty, solve_D_bisection

# Load best feature params from k=1.5 optimization
with open("tuned_params_sigmoid_k15.json") as f:
    best = json.load(f)
base_params = dict(best["params"])

entries = load_playtest_data()
print(f"{len(entries)} entries")

# 1. Precompute + pre-extract D_seg
print("Precomputing & extracting D_seg...")
seg_data = {}
caches_ok = []
for i, e in enumerate(entries):
    try:
        c = precompute(e["osu_path"], use_enhanced=True, params=base_params)
        _, details = combine(c, {**base_params, "use_sigmoid_aggregation": 0})
        eff_w = _compute_effective_weights(c["all_corners"], details["C_arr"])
        D_seg, w_seg = segment_by_difficulty(details["D_all"], eff_w, 30)
        caches_ok.append((D_seg, w_seg, details["total_notes"], e))
    except Exception as ex:
        pass
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(entries)}")
print(f"  {len(caches_ok)} valid maps")

def eval_loss_b1(params, k, data=caches_ok):
    """Fast eval: only sigmoid bisection, no combine()."""
    losses = []
    for D_seg, w_seg, total_notes, entry in data:
        try:
            D_cal = params["calib_a"] * D_seg + params["calib_b"]
            D_solved, _ = solve_D_bisection(D_cal, w_seg, k=k,
                                             C=params["agg_sigmoid_C"],
                                             gamma=params["agg_sigmoid_ref_gamma"])
            SR = float(D_solved)
            SR *= total_notes / (total_notes + params["note_norm_N0"])
            if SR > params["rescale_threshold"]:
                SR = params["rescale_threshold"] + (SR - params["rescale_threshold"]) / params["rescale_divisor"]
            SR *= params["global_scale"]
            losses.append(score_single(SR, entry["sr_ref"], entry["sr_error"]))
        except Exception:
            losses.append(100.0)
    return float(np.mean(losses))

# 2. Sweep k with B1 NM per value
B1_DEFS = [
    # (name, init, lo, hi) — normalized to [0,1]
]
BOUNDS = [
    ("calib_a", 0.6, 1.2),
    ("calib_b", -0.8, 0.8),
    ("agg_sigmoid_C", 2.0, 10.0),
    ("agg_sigmoid_ref_gamma", 0.08, 0.35),
    ("note_norm_N0", 0.0, 80.0),
    ("rescale_threshold", 6.0, 14.0),
    ("rescale_divisor", 1.2, 4.0),
    ("global_scale", 0.92, 1.15),
]

def extract_u(params):
    """params → u vector"""
    u = []
    for name, lo, hi in BOUNDS:
        v = params.get(name, lo)
        u.append((v - lo) / (hi - lo) if hi > lo else 0.5)
    return u

def apply_u(u, base):
    """u vector → params dict"""
    p = dict(base)
    for i, (name, lo, hi) in enumerate(BOUNDS):
        p[name] = float(np.clip(lo + u[i] * (hi - lo), lo, hi))
    return p

results = []
for k in np.arange(0.5, 3.1, 0.1):
    k = round(k, 1)
    t0 = time.time()

    # Start from current best B1 params
    u0 = extract_u(base_params)
    init_params = apply_u(u0, base_params)
    init_loss = eval_loss_b1(init_params, k)

    def obj(u):
        return eval_loss_b1(apply_u(u, base_params), k)

    r = minimize(obj, u0, method="Nelder-Mead",
                 options={"maxiter": 30, "adaptive": True, "xatol": 0.0005, "fatol": 0.0005})

    opt_params = apply_u(r.x, base_params)
    final_loss = obj(r.x)

    t = time.time() - t0
    results.append((k, final_loss, init_loss, opt_params, r.nfev, t))

    print(f"k={k:.1f}: {init_loss:.4f} → {final_loss:.4f} (Δ={final_loss-init_loss:+.4f}), "
          f"calib=({opt_params['calib_a']:.3f},{opt_params['calib_b']:.3f}), "
          f"C={opt_params['agg_sigmoid_C']:.1f}, γ={opt_params['agg_sigmoid_ref_gamma']:.3f}, "
          f"evals={r.nfev}, time={t:.0f}s", flush=True)


# Also compute MAE for each k
print("\nComputing MAE for top candidates...")
for k, loss, _, opt_params, _, _ in sorted(results, key=lambda x: x[1])[:5]:
    mae_vals = []
    for D_seg, w_seg, total_notes, entry in caches_ok:
        try:
            D_cal = opt_params["calib_a"] * D_seg + opt_params["calib_b"]
            D_solved, _ = solve_D_bisection(D_cal, w_seg, k=k,
                                             C=opt_params["agg_sigmoid_C"],
                                             gamma=opt_params["agg_sigmoid_ref_gamma"])
            SR = float(D_solved)
            SR *= total_notes / (total_notes + opt_params["note_norm_N0"])
            if SR > opt_params["rescale_threshold"]:
                SR = opt_params["rescale_threshold"] + (SR - opt_params["rescale_threshold"]) / opt_params["rescale_divisor"]
            SR *= opt_params["global_scale"]
            mae_vals.append(abs(SR - entry["sr_ref"]))
        except:
            pass
    mae = float(np.mean(mae_vals)) if mae_vals else 0
    print(f"  k={k:.1f}: Loss={loss:.4f}, MAE={mae:.4f}")


# ============================================================
# Results
# ============================================================
print("\n" + "=" * 70)
print("ALL RESULTS (sorted by Loss)")
print("=" * 70)
results.sort(key=lambda x: x[1])
print(f"{'k':>5}  {'Loss':>8}  {'ΔLoss':>8}  {'MAE':>8}  {'calib_a':>7}  {'calib_b':>7}  {'C':>5}  {'gamma':>6}  {'evals':>5}  {'time':>5}")
print("-" * 80)
# Pre-compute MAE for display
mae_cache = {}
for k, loss, init_loss, opt_params, nfev, t in sorted(results, key=lambda x: x[1]):
    mae_vals = []
    for D_seg, w_seg, total_notes, entry in caches_ok:
        try:
            D_cal = opt_params["calib_a"] * D_seg + opt_params["calib_b"]
            D_solved, _ = solve_D_bisection(D_cal, w_seg, k=k,
                                             C=opt_params["agg_sigmoid_C"],
                                             gamma=opt_params["agg_sigmoid_ref_gamma"])
            SR = float(D_solved)
            SR *= total_notes / (total_notes + opt_params["note_norm_N0"])
            if SR > opt_params["rescale_threshold"]:
                SR = opt_params["rescale_threshold"] + (SR - opt_params["rescale_threshold"]) / opt_params["rescale_divisor"]
            SR *= opt_params["global_scale"]
            mae_vals.append(abs(SR - entry["sr_ref"]))
        except:
            pass
    mae_cache[k] = float(np.mean(mae_vals)) if mae_vals else 0

for k, loss, init_loss, opt_params, nfev, t in sorted(results, key=lambda x: x[1]):
    mae = mae_cache[k]
    print(f"{k:5.1f}  {loss:8.4f}  {loss-init_loss:8.4f}  {mae:8.4f}  "
          f"{opt_params['calib_a']:7.4f}  {opt_params['calib_b']:7.4f}  "
          f"{opt_params['agg_sigmoid_C']:5.2f}  {opt_params['agg_sigmoid_ref_gamma']:6.4f}  "
          f"{nfev:5}  {t:4.0f}s")

# Best
best_k, best_loss, _, best_params, _, _ = min(results, key=lambda x: x[1])
best_mae = mae_cache[best_k]
print(f"\nBEST: k={best_k:.1f}, Loss={best_loss:.4f}, MAE={best_mae:.4f}")
print(f"(Percentile baseline: Loss=1.0280, MAE=0.2273)")

# Save best
out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tuned_params_sigmoid_bestk.json")
json.dump({"k": best_k, "loss": best_loss, "mae": best_mae,
           "pct_loss": 1.0280, "pct_mae": 0.2273,
           "params": best_params}, open(out_path, "w"), ensure_ascii=False, indent=2)
print(f"Saved: {out_path}")
