"""
LN-only model tuning via Nelder-Mead (optimized: pre-extract component arrays).
"""
import sys, os, json, time, numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tuning.data_loader import load_playtest_data
from tuning.scorer import score_single
from spm_rating.rating import precompute, combine as combine_total
from spm_rating.combine_rc_ln import compute_ln_sr, compute_D_ln
from spm_rating.aggregate_sigmoid import _compute_effective_weights, segment_by_difficulty, solve_D_bisection
from spm_rating.utils import interp_values, step_interp

# Load base params
params_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "tuned_params_sigmoid.json")
with open(params_path) as f:
    base_full = json.load(f)
base_params = dict(base_full["params"])

# Initialize LN params
ln_base = dict(base_params)
ln_base.update({
    "alpha_R_ln": 0.5,
    "alpha_S_ln": 0.1,
    "alpha_V_ln": 0.5,
    "calib_a_ln": 3.0,
    "calib_b_ln": 0.0,
    "agg_sigmoid_k_ln": 1.5,
    "agg_sigmoid_C_ln": 4.0,
    "agg_sigmoid_gamma_ln": 0.20,
    "note_norm_N0_ln": 10.0,
    "rescale_threshold_ln": 9.54,
    "rescale_divisor_ln": 2.00,
    "global_scale_ln": 1.00,
})

BLOCK_LN_WEIGHTS = [
    ("alpha_R_ln", 0.5, 0.01, 5.0, "Rbar weight LN"),
    ("alpha_S_ln", 0.1, 0.0,  3.0, "Sbar weight LN"),
    ("alpha_V_ln", 0.5, 0.0,  5.0, "Vbar weight LN"),
]

BLOCK_LN_SIGMOID = [
    ("calib_a_ln",         3.00, 0.50, 10.0, "D calib scale LN"),
    ("calib_b_ln",         0.00, -2.00, 2.00, "D calib offset LN"),
    ("agg_sigmoid_k_ln",   1.50, 0.50, 3.00, "sigmoid k LN"),
    ("agg_sigmoid_C_ln",   4.00, 2.00, 10.0, "sigmoid C LN"),
    ("agg_sigmoid_gamma_ln", 0.20, 0.08, 0.35, "sigmoid gamma LN"),
]

BLOCK_LN_POST = [
    ("note_norm_N0_ln",      10.0, 0.0,  80.0, "N0 LN"),
    ("rescale_threshold_ln", 9.54, 5.0,  14.0, "threshold LN"),
    ("rescale_divisor_ln",   2.00, 1.2,  4.0,  "divisor LN"),
    ("global_scale_ln",      1.00, 0.50, 2.50, "global scale LN"),
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


# ============ Medium path: pre-extract Rbar/Sbar/Vbar arrays ============

def extract_ln_arrays(caches, params):
    """
    Compute Rbar_all, Sbar_all, Vbar_all, C_arr for each map ONCE.
    Returns list of dicts: {Rbar, Sbar, Vbar, C_arr, total_notes_ln, all_corners}
    """
    from spm_rating.components import release_enhanced as _release_enh
    from spm_rating.components import shield as _shield
    from spm_rating.components import inverse as _inverse

    result = []
    for cache in caches:
        if cache is None or len(cache.get("LN_seq", [])) == 0:
            result.append(None)
            continue
        try:
            all_corners = cache["all_corners"]
            base_corners = cache["base_corners"]

            # Rbar
            release_data = cache.get("release_data")
            if release_data is not None:
                Rbar_base = _release_enh.compute_Rbar_enhanced_fast(
                    release_data, base_corners,
                    release_tail_coeff=params.get("release_tail_coeff", 0.08),
                    release_tail_to_tap_factor=params.get("release_tail_to_tap", 1.0),
                    release_same_col_bonus=params.get("release_same_col_bonus", 1.5),
                    release_coord_exponent=params.get("release_coord_exponent", 1.0),
                    short_ln_threshold=params.get("short_ln_threshold", 200),
                    short_ln_reduction=params.get("short_ln_reduction", 0.5),
                    lock_interaction_coeff=params.get("lock_interaction_coeff", 0.3),
                    release_seq_coeff=params.get("release_seq_coeff", 0.03),
                    smooth_window=params.get("release_smooth_window", 500),
                    smooth_scale=params.get("release_scale", 0.001),
                )
            else:
                Rbar_base = cache.get("Rbar_base_clone", np.zeros(len(base_corners)))

            # Sbar
            shield_data = cache.get("shield_data")
            if shield_data is not None:
                Sbar_base = _shield.compute_Sbar_fast(
                    shield_data, base_corners,
                    shield_tau_ms=params.get("shield_tau_ms", 100),
                    shield_anchor_mod=params.get("shield_anchor_mod", 1.0),
                    shield_coord_factor=params.get("shield_coord_factor", 1.0),
                    smooth_window=params.get("shield_smooth_window", 500),
                    smooth_scale=params.get("shield_scale", 0.001),
                )
            else:
                Sbar_base = np.zeros(len(base_corners))

            # Vbar
            inverse_data = cache.get("inverse_data")
            if inverse_data is not None:
                Vbar_base = _inverse.compute_Vbar_fast(
                    inverse_data, base_corners,
                    inv_amplitude=params.get("inv_amplitude", 3.0),
                    inv_tau=params.get("inv_tau", 31),
                    inv_power=params.get("inv_power", 1.0),
                    guide_depth=params.get("guide_depth", 0.9),
                    guide_center=params.get("guide_center", 78),
                    guide_width=params.get("guide_width", 31),
                    cross_guide_scale=params.get("cross_guide_scale", 0.67),
                    same_col_bonus=params.get("inverse_same_col_bonus", 3.6),
                    window_ms=params.get("inverse_window_ms", 200),
                )
            else:
                Vbar_base = np.zeros(len(base_corners))

            # Interpolate to all_corners
            Rbar_all = interp_values(all_corners, base_corners, Rbar_base)
            Sbar_all = interp_values(all_corners, base_corners, Sbar_base)
            Vbar_all = interp_values(all_corners, base_corners, Vbar_base)

            # C_arr (for segment weights)
            C_step = cache["C_step"]
            C_arr = step_interp(all_corners, base_corners, C_step)

            total_notes_ln = len(cache["LN_seq"])

            result.append({
                "Rbar": Rbar_all, "Sbar": Sbar_all, "Vbar": Vbar_all,
                "C_arr": C_arr, "total_notes_ln": total_notes_ln,
                "all_corners": all_corners,
            })
        except Exception:
            result.append(None)
    return result


def ln_sr_from_arrays(ln_data, params):
    """Compute LN SR from pre-extracted arrays (medium path)."""
    D_ln_raw = compute_D_ln(
        ln_data["all_corners"], None,
        ln_data["Rbar"], ln_data["Sbar"], ln_data["Vbar"],
        alpha_R=params["alpha_R_ln"],
        alpha_S=params["alpha_S_ln"],
        alpha_V=params["alpha_V_ln"],
    )

    # Calibration
    calib_a = params["calib_a_ln"]
    calib_b = params["calib_b_ln"]
    D_calib = calib_a * D_ln_raw + calib_b if abs(calib_a - 1.0) > 1e-12 or abs(calib_b) > 1e-12 else D_ln_raw

    # Segment
    gaps = np.empty(len(ln_data["all_corners"]))
    ac = ln_data["all_corners"]
    gaps[0] = (ac[1] - ac[0]) / 2.0
    gaps[-1] = (ac[-1] - ac[-2]) / 2.0
    gaps[1:-1] = (ac[2:] - ac[:-2]) / 2.0
    efw = ln_data["C_arr"] * gaps
    D_seg, w_seg = segment_by_difficulty(D_calib, efw, 30)

    # Sigmoid bisection
    D_solved, _ = solve_D_bisection(
        D_seg, w_seg,
        k=params["agg_sigmoid_k_ln"],
        C=params["agg_sigmoid_C_ln"],
        gamma=params["agg_sigmoid_gamma_ln"],
        high_weight_power=0.0, delta=5.0, tol=0.0001)

    SR = float(D_solved)
    tn = ln_data["total_notes_ln"]
    SR *= tn / (tn + params["note_norm_N0_ln"])
    if SR > params["rescale_threshold_ln"]:
        SR = params["rescale_threshold_ln"] + (SR - params["rescale_threshold_ln"]) / params["rescale_divisor_ln"]
    SR *= params["global_scale_ln"]
    return SR


def eval_ln_medium(ln_arrays, entries, params):
    """Medium eval: linear combination of pre-extracted arrays + segment + sigmoid."""
    losses = []
    for ln_data, entry in zip(ln_arrays, entries):
        if ln_data is None:
            losses.append(100.0)
            continue
        try:
            sr = ln_sr_from_arrays(ln_data, params)
            losses.append(score_single(sr, entry["sr_ref_ln"], entry["sr_error_ln"]))
        except Exception:
            losses.append(100.0)
    return float(np.mean(losses)) if losses else 1e9


def eval_ln_detailed_from_arrays(ln_arrays, entries, params):
    """Detailed eval from pre-extracted arrays."""
    preds, refs, errs = [], [], []
    for ln_data, entry in zip(ln_arrays, entries):
        if ln_data is None:
            continue
        try:
            sr = ln_sr_from_arrays(ln_data, params)
            if sr > 0.01:
                preds.append(sr)
                refs.append(entry["sr_ref_ln"])
                errs.append(entry["sr_error_ln"])
        except Exception:
            pass
    if len(preds) < 2:
        return {"loss": 1e9, "mae": 1e9, "rmse": 1e9, "r": 0, "n": 0}
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


def main():
    print("=" * 72)
    print("LN Model Tuning — Nelder-Mead (optimized)")
    print("=" * 72)

    entries = load_playtest_data()
    print(f"[0] {len(entries)} total entries")

    ln_entries = [(i, e) for i, e in enumerate(entries) if e.get("sr_ref_ln") is not None]
    print(f"[0] {len(ln_entries)} entries with LN labels")
    ln_indices = [i for i, _ in ln_entries]
    ln_entries_list = [e for _, e in ln_entries]

    params = dict(ln_base)
    print(f"[1] Precomputing {len(ln_indices)} maps...")
    t0 = time.time()
    caches = []
    for idx in ln_indices:
        e = entries[idx]
        try:
            c = precompute(e["osu_path"], use_enhanced=True, params=params)
            caches.append(c)
        except Exception:
            caches.append(None)
        if len(caches) % 50 == 0:
            print(f"  {len(caches)}/{len(ln_indices)}...")
    valid_c = [c for c in caches if c is not None]
    valid_e = [e for c, e in zip(caches, ln_entries_list) if c is not None]
    n = len(valid_c)
    print(f"  {n}/{len(ln_indices)} OK in {time.time() - t0:.0f}s")

    # Baseline
    print(f"\n[2] Baseline evaluation...")
    total_preds = []
    for cache, entry in zip(valid_c, valid_e):
        try:
            sr_total, _ = combine_total(cache, dict(base_params))
            total_preds.append(sr_total)
        except Exception:
            total_preds.append(0)
    total_refs = [e["sr_ref_ln"] for e in valid_e]
    total_errs = [e.get("sr_error_ln", 0.5) for e in valid_e]
    total_loss = float(np.mean([score_single(p, r, e) for p, r, e in zip(total_preds, total_refs, total_errs)]))
    total_mae = float(np.mean(np.abs(np.array(total_preds) - np.array(total_refs))))
    print(f"  Total SR vs LN label: Loss={total_loss:.4f}, MAE={total_mae:.4f}")

    # Pre-extract LN component arrays (one-time cost)
    print(f"\n[3] Extracting Rbar/Sbar/Vbar arrays for all maps...")
    ln_arrays = extract_ln_arrays(valid_c, params)
    n_ok = sum(1 for x in ln_arrays if x is not None)
    print(f"  {n_ok}/{n} OK in {time.time() - t0:.0f}s")

    # Initial LN eval from pre-extracted arrays
    ln_init = eval_ln_detailed_from_arrays(ln_arrays, valid_e, params)
    print(f"  LN model (initial):   Loss={ln_init['loss']:.4f}, MAE={ln_init['mae']:.4f},"
          f" r={ln_init['r']:.4f}, n={ln_init['n']}")

    history = []
    best_loss = ln_init["loss"]
    best_params = dict(params)

    # B1 uses "medium" path (fast linear combination of pre-extracted arrays)
    # B2/B3 use same medium path since params affect D formula, sigmoid, and post equally
    blocks = [
        (BLOCK_LN_WEIGHTS, "B1: D Weights",  "medium"),
        (BLOCK_LN_SIGMOID, "B2: Sigmoid+Calib", "medium"),
        (BLOCK_LN_POST,    "B3: Post-processing", "medium"),
    ]

    for round_idx in range(2):
        print(f"\n{'─' * 60}")
        print(f"ROUND {round_idx + 1}/2")
        print(f"{'─' * 60}")
        maxiter = 50 if round_idx == 0 else 25

        for block_def, label, mode in blocks:
            obj_fn = lambda p: eval_ln_medium(ln_arrays, valid_e, p)
            params, loss = run_nm(block_def, params, obj_fn, label, maxiter)
            history.append((label, loss))
            if loss < best_loss:
                best_loss = loss
                best_params = dict(params)
                print(f"  >> NEW BEST: Loss={loss:.4f}")

    # Final eval
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    final = eval_ln_detailed_from_arrays(ln_arrays, valid_e, best_params)
    print(f"  Total SR as LN:   Loss={total_loss:.4f}, MAE={total_mae:.4f}")
    print(f"  LN model (final): Loss={final['loss']:.4f}, MAE={final['mae']:.4f},"
          f" r={final['r']:.4f}, n={final['n']}")
    if final['n'] > 0:
        print(f"  Improvement:      ΔLoss={final['loss'] - total_loss:+.4f}, ΔMAE={final['mae'] - total_mae:+.4f}")

    # Per-source
    for src in ["dan", "tournament", "graveyard"]:
        idxs = [i for i, e in enumerate(valid_e) if e.get("source") == src]
        if idxs:
            preds_sub = []
            refs_sub = []
            errs_sub = []
            for i in idxs:
                if ln_arrays[i] is None: continue
                try:
                    sr = ln_sr_from_arrays(ln_arrays[i], best_params)
                    if sr > 0.01:
                        preds_sub.append(sr)
                        refs_sub.append(valid_e[i]["sr_ref_ln"])
                        errs_sub.append(valid_e[i].get("sr_error_ln", 0.5))
                except: pass
            if len(preds_sub) > 1:
                psub = np.array(preds_sub)
                rsub = np.array(refs_sub)
                mae_sub = float(np.mean(np.abs(psub - rsub)))
                r_sub = float(np.corrcoef(psub, rsub)[0, 1])
            else:
                mae_sub = 0; r_sub = 0
            print(f"  {src:>12} ({len(idxs)}): MAE={mae_sub:.4f}, r={r_sub:.4f}, n={len(preds_sub)}")

    out = {
        "type": "ln_model",
        "loss": final["loss"], "mae": final["mae"], "r": final["r"], "n_valid": final["n"],
        "baseline_total_loss": total_loss, "baseline_total_mae": total_mae,
        "method": "NM 2 rounds (medium path: pre-extracted Rbar/Sbar/Vbar arrays)",
        "params": best_params,
        "history": history,
    }
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "tuned_params_ln.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
