"""
RC-only model tuning via block-wise alternating Nelder-Mead.
"""
import sys, os, json, time, numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tuning.data_loader import load_playtest_data
from tuning.scorer import score_single
from spm_rating.rating import precompute, combine as combine_total
from spm_rating.combine_rc_ln import compute_rc_sr
from spm_rating.aggregate_sigmoid import _compute_effective_weights, segment_by_difficulty, solve_D_bisection

# Load base params (full sigmoid model best)
params_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "tuned_params_sigmoid.json")
with open(params_path) as f:
    base_full = json.load(f)
base_params = dict(base_full["params"])

# Initialize RC params from full model params
rc_base = dict(base_params)
# RC-specific overrides
rc_base.update({
    "S_w1_rc": base_params.get("S_w1", 0.514),
    "S_p_rc": base_params.get("S_p", 1.117),
    "alpha_P_rc": base_params.get("alpha_P", 0.724),
    "D_beta1_rc": base_params.get("D_beta1", 1.170),
    "D_beta2_rc": base_params.get("D_beta2", 0.389),
    "Abar_scale_rc": base_params.get("Abar_scale", 1.016),
    "calib_a_rc": 1.0,
    "calib_b_rc": 0.0,
    "agg_sigmoid_k_rc": 1.5,
    "agg_sigmoid_C_rc": 4.0,
    "agg_sigmoid_gamma_rc": 0.20,
    "note_norm_N0_rc": 10.0,
    "rescale_threshold_rc": 9.54,
    "rescale_divisor_rc": 2.00,
    "global_scale_rc": 1.055,
})

# Block definitions
BLOCK_RC_D_FORMULA = [
    ("S_w1_rc",       0.514, 0.15, 0.85, "jack branch weight RC"),
    ("S_p_rc",        1.117, 0.70, 2.50, "p-norm exponent RC"),
    ("alpha_P_rc",    0.724, 0.30, 1.50, "Pbar weight RC"),
    ("D_beta1_rc",    1.170, 0.50, 2.50, "S*T coefficient RC"),
    ("D_beta2_rc",    0.389, 0.15, 0.80, "S linear coefficient RC"),
    ("Abar_scale_rc", 1.016, 0.85, 1.20, "Abar scale RC"),
]

BLOCK_RC_SIGMOID = [
    ("calib_a_rc",         1.00, 0.60, 1.50, "D calibration scale RC"),
    ("calib_b_rc",         0.00, -0.50, 0.80, "D calibration offset RC"),
    ("agg_sigmoid_k_rc",   1.50, 0.50, 3.00, "sigmoid k RC"),
    ("agg_sigmoid_C_rc",   4.00, 2.00, 10.0, "sigmoid C RC"),
    ("agg_sigmoid_gamma_rc", 0.20, 0.08, 0.35, "sigmoid gamma RC"),
]

BLOCK_RC_POST = [
    ("note_norm_N0_rc",      10.0,  0.0, 80.0, "N0 RC"),
    ("rescale_threshold_rc", 9.54,  6.0, 14.0, "threshold RC"),
    ("rescale_divisor_rc",   2.00,  1.2, 4.0,  "divisor RC"),
    ("global_scale_rc",      1.055, 0.80, 1.30, "global scale RC"),
]


def make_block(block_def, params):
    p0 = [params.get(name, init) for name, init, *rest in block_def]
    bounds = [(rest[0], rest[1]) for _, _, *rest in block_def]
    names = [name for name, _, *rest in block_def]
    return p0, bounds, names


def apply_block(names, values, params):
    for name, val in zip(names, values):
        params[name] = float(val)
    return params


# ============================================================
def extract_rc_segments(caches, params):
    """Pre-extract D_rc segments for fast path."""
    result = []
    for cache in caches:
        if cache is None:
            result.append(None)
            continue
        try:
            _, details = compute_rc_sr(cache, params)
            eff_w = _compute_effective_weights(cache["all_corners"], details["C_arr"])
            D_seg, w_seg = segment_by_difficulty(details["D_all"], eff_w, 30)
            result.append({
                "D_seg": D_seg, "w_seg": w_seg,
                "total_notes": details["total_notes"],
            })
        except Exception:
            result.append(None)
    return result


def rc_sr_fast(pre_data, params):
    """Fast eval: only sigmoid bisection, no D recomputation."""
    D_cal = params["calib_a_rc"] * pre_data["D_seg"] + params["calib_b_rc"]
    D_solved, _ = solve_D_bisection(
        D_cal, pre_data["w_seg"],
        k=params["agg_sigmoid_k_rc"],
        C=params["agg_sigmoid_C_rc"],
        gamma=params["agg_sigmoid_gamma_rc"],
        high_weight_power=0.0, delta=5.0, tol=0.0001)
    SR = float(D_solved)
    SR *= pre_data["total_notes"] / (pre_data["total_notes"] + params["note_norm_N0_rc"])
    if SR > params["rescale_threshold_rc"]:
        SR = params["rescale_threshold_rc"] + (SR - params["rescale_threshold_rc"]) / params["rescale_divisor_rc"]
    SR *= params["global_scale_rc"]
    return SR


def eval_rc_fast(pre_data_list, entries, params):
    losses = []
    for pre_data, entry in zip(pre_data_list, entries):
        if pre_data is None:
            losses.append(100.0)
            continue
        try:
            sr = rc_sr_fast(pre_data, params)
            losses.append(score_single(sr, entry["sr_ref_rc"], entry["sr_error_rc"]))
        except Exception:
            losses.append(100.0)
    return float(np.mean(losses)) if losses else 1e9


def eval_rc_full(caches, entries, params):
    losses = []
    for cache, entry in zip(caches, entries):
        if cache is None:
            losses.append(100.0)
            continue
        try:
            sr, _ = compute_rc_sr(cache, params)
            losses.append(score_single(sr, entry["sr_ref_rc"], entry["sr_error_rc"]))
        except Exception:
            losses.append(100.0)
    return float(np.mean(losses)) if losses else 1e9


def eval_rc_detailed(caches, entries, params):
    preds, refs, errs = [], [], []
    for cache, entry in zip(caches, entries):
        if cache is None:
            continue
        try:
            sr, _ = compute_rc_sr(cache, params)
            preds.append(sr)
            refs.append(entry["sr_ref_rc"])
            errs.append(entry["sr_error_rc"])
        except Exception:
            pass
    preds_arr = np.array(preds)
    refs_arr = np.array(refs)
    res = preds_arr - refs_arr
    loss = float(np.mean([score_single(p, r, e) for p, r, e in zip(preds, refs, errs)]))
    mae = float(np.mean(np.abs(res)))
    rmse = float(np.sqrt(np.mean(res ** 2)))
    r = float(np.corrcoef(preds_arr, refs_arr)[0, 1]) if len(preds) > 1 else 0
    return {"loss": loss, "mae": mae, "rmse": rmse, "r": r, "n": len(preds)}


def run_nm(block_def, params, obj_fn, label, maxiter):
    p0, bounds, names = make_block(block_def, params)
    u0 = [(v - lo) / max(hi - lo, 1e-9) for v, (lo, hi) in zip(p0, bounds)]

    def objective(u):
        vals = [np.clip(lo + ui * (hi - lo), lo, hi) for ui, (lo, hi) in zip(u, bounds)]
        return obj_fn(apply_block(names, vals, dict(params)))

    init_loss = objective(u0)
    print(f"\n  [{label}] Loss={init_loss:.4f}  ({len(names)}p, maxiter={maxiter})")
    t0 = time.time()

    result = minimize(objective, u0, method="Nelder-Mead",
                      options={"maxiter": maxiter, "xatol": 0.0005, "fatol": 0.0005,
                               "adaptive": True})

    vals_opt = [np.clip(lo + ui * (hi - lo), lo, hi) for ui, (lo, hi) in zip(result.x, bounds)]
    final_loss = objective(result.x)
    elapsed = time.time() - t0

    print(f"  [{label}] Done: Loss={final_loss:.4f} (Δ={final_loss - init_loss:+.4f}), "
          f"evals={result.nfev}, iters={result.nit}, time={elapsed:.0f}s")
    for i, name in enumerate(names):
        delta = vals_opt[i] - p0[i]
        if abs(delta) > 0.001:
            print(f"    {name:<28}: {p0[i]:.4f} → {vals_opt[i]:.4f}  ({delta:+.4f})")

    return apply_block(names, vals_opt, dict(params)), final_loss


# ============================================================
def main():
    print("=" * 72)
    print("RC Model Tuning — Nelder-Mead")
    print("=" * 72)

    entries = load_playtest_data()
    print(f"[0] {len(entries)} total entries")

    # Filter to entries with RC labels
    rc_entries = [(i, e) for i, e in enumerate(entries) if e.get("sr_ref_rc") is not None]
    print(f"[0] {len(rc_entries)} entries with RC labels")
    rc_indices = [i for i, _ in rc_entries]
    rc_entries_list = [e for _, e in rc_entries]

    # Precompute caches
    params = dict(rc_base)
    print(f"[1] Precomputing {len(rc_indices)} maps...")
    t0 = time.time()
    caches = []
    for idx in rc_indices:
        e = entries[idx]
        try:
            c = precompute(e["osu_path"], use_enhanced=True, params=params)
            caches.append(c)
        except Exception:
            caches.append(None)
        if (len(caches)) % 50 == 0:
            print(f"  {len(caches)}/{len(rc_indices)}...")
    valid_c = [c for c in caches if c is not None]
    valid_e = [e for c, e in zip(caches, rc_entries_list) if c is not None]
    n = len(valid_c)
    print(f"  {n}/{len(rc_indices)} OK in {time.time() - t0:.0f}s")

    # Initial evaluation: use total SR as baseline for RC labels
    print(f"\n[2] Baseline evaluation...")
    total_preds = []
    for cache, entry in zip(valid_c, valid_e):
        try:
            sr_total, _ = combine_total(cache, dict(base_params))
            total_preds.append(sr_total)
        except Exception:
            total_preds.append(0)
    total_refs = [e["sr_ref_rc"] for e in valid_e]
    total_errs = [e.get("sr_error_rc", 0.5) for e in valid_e]
    total_loss = float(np.mean([score_single(p, r, e) for p, r, e in zip(total_preds, total_refs, total_errs)]))
    total_mae = float(np.mean(np.abs(np.array(total_preds) - np.array(total_refs))))
    print(f"  Total SR vs RC label: Loss={total_loss:.4f}, MAE={total_mae:.4f}")

    # Initial RC model eval
    rc_init = eval_rc_detailed(valid_c, valid_e, params)
    print(f"  RC model (initial):  Loss={rc_init['loss']:.4f}, MAE={rc_init['mae']:.4f}, r={rc_init['r']:.4f}")

    # Pre-extract D_rc segments for fast path
    print(f"\n[3] Extracting RC D segments for fast path...")
    rc_pre = extract_rc_segments(valid_c, params)
    print(f"  Done in {time.time() - t0:.0f}s")

    history = []
    best_loss = rc_init["loss"]
    best_params = dict(params)

    # ================ Tuning blocks ================
    blocks = [
        (BLOCK_RC_SIGMOID,  "B1: Sigmoid+Calib",  "fast"),
        (BLOCK_RC_D_FORMULA, "B2: D Formula",      "full"),
        (BLOCK_RC_POST,      "B3: Post-processing", "fast"),
    ]

    for round_idx in range(2):
        print(f"\n{'─' * 60}")
        print(f"ROUND {round_idx + 1}/2")
        print(f"{'─' * 60}")
        maxiter = 50 if round_idx == 0 else 25

        for block_def, label, mode in blocks:
            if mode == "fast":
                obj_fn = lambda p: eval_rc_fast(rc_pre, valid_e, p)
            else:
                obj_fn = lambda p: eval_rc_full(valid_c, valid_e, p)

            params, loss = run_nm(block_def, params, obj_fn, label, maxiter)
            history.append((label, loss))
            if loss < best_loss:
                best_loss = loss
                best_params = dict(params)
                print(f"  >> NEW BEST: Loss={loss:.4f}")

            if mode == "full":
                # Re-extract D_rc segments after D formula change
                rc_pre = extract_rc_segments(valid_c, params)

    # ================ Final evaluation ================
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    final = eval_rc_detailed(valid_c, valid_e, best_params)
    print(f"  Total SR as RC:    Loss={total_loss:.4f}, MAE={total_mae:.4f}")
    print(f"  RC model (final):  Loss={final['loss']:.4f}, MAE={final['mae']:.4f}, r={final['r']:.4f}")
    print(f"  Improvement:       ΔLoss={final['loss'] - total_loss:+.4f}, ΔMAE={final['mae'] - total_mae:+.4f}")

    # Per-source breakdown
    for src in ["dan", "tournament", "graveyard"]:
        idxs = [i for i, e in enumerate(valid_e) if e.get("source") == src]
        if idxs:
            preds_sub = [compute_rc_sr(valid_c[i], best_params)[0] for i in idxs]
            refs_sub = [valid_e[i]["sr_ref_rc"] for i in idxs]
            mae_sub = float(np.mean(np.abs(np.array(preds_sub) - np.array(refs_sub))))
            r_sub = float(np.corrcoef(preds_sub, refs_sub)[0, 1]) if len(preds_sub) > 1 else 0
            print(f"  {src:>12} ({len(idxs)}): MAE={mae_sub:.4f}, r={r_sub:.4f}")

    # Save
    out = {
        "type": "rc_model",
        "loss": final["loss"], "mae": final["mae"], "r": final["r"],
        "baseline_total_loss": total_loss, "baseline_total_mae": total_mae,
        "method": "NM 2 rounds (B1 sigmoid+calib, B2 D formula, B3 post)",
        "params": best_params,
        "history": history,
    }
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "tuned_params_rc.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
