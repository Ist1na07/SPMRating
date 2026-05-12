"""
Block-wise alternating NM with k=1.5 (optimal from sweep).
Focused: B1 fast path maxiter=50, B2-B3c maxiter=20, 2 rounds.
"""
import sys, os, json, time
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tuning.data_loader import load_playtest_data
from tuning.scorer import score_single
from spm_rating.rating import precompute, combine
from spm_rating.aggregate_sigmoid import _compute_effective_weights, segment_by_difficulty, solve_D_bisection

# ============================================================
# Blocks (k fixed at 1.5, not in any block)
# ============================================================

BLOCK1_SIGMOID = [
    ("calib_a",        0.89,   0.60, 1.20, "D pre-calibration scale"),
    ("calib_b",        0.06,   -0.80, 0.80, "D pre-calibration offset"),
    ("agg_sigmoid_C",  4.0,    2.00, 10.0, "sigmoid C shape"),
    ("agg_sigmoid_ref_gamma", 0.20, 0.08, 0.35, "reference gamma"),
    ("note_norm_N0",   10.0,   0.0,  80.0, "note count N0"),
    ("rescale_threshold", 9.54, 6.0, 14.0, "rescale threshold"),
    ("rescale_divisor",   2.00, 1.2,  4.0,  "rescale divisor"),
    ("global_scale",   1.055,  0.92, 1.15, "global scale"),
]

BLOCK2_D_FORMULA = [
    ("S_w1",      0.514,  0.15, 0.85),
    ("S_p",       1.117,  0.70, 2.50),
    ("alpha_P",   0.724,  0.30, 1.50),
    ("alpha_R",   28.47,  12.0, 50.0),
    ("alpha_C",   9.64,   3.00, 20.0),
    ("alpha_S",   0.479,  0.05, 2.50),
    ("alpha_V",   0.435,  0.10, 2.00),
    ("D_beta1",   1.170,  0.50, 2.50),
    ("D_beta2",   0.389,  0.15, 0.80),
    ("Abar_scale", 1.016, 0.85, 1.20),
]

BLOCK3A_CROSS = [
    ("cross_dist_exponent_rc",  1.010, 0.50, 2.00),
    ("cross_dist_exponent_ln",  0.988, 0.50, 2.00),
    ("cross_same_hand_penalty_rc", 0.337, 0.05, 0.80),
    ("cross_same_hand_penalty_ln", 0.294, 0.05, 0.80),
    ("cross_thumb_bridge_factor",  0.496, 0.10, 0.90),
]

BLOCK3B_RELEASE = [
    ("release_tail_coeff",      0.123, 0.03, 0.30),
    ("release_tail_to_tap",     2.099, 0.80, 4.00),
    ("release_same_col_bonus",  0.300, 0.10, 1.50),
    ("release_coord_exponent",  0.630, 0.20, 1.50),
    ("release_seq_coeff",       0.047, 0.01, 0.15),
    ("lock_interaction_coeff",  0.235, 0.05, 0.60),
]

BLOCK3C_INVERSE = [
    ("inv_amplitude",       3.465, 1.50, 6.00),
    ("inv_tau",             30.82, 15.0, 60.0),
    ("inv_power",           0.927, 0.50, 1.50),
    ("guide_depth",         0.797, 0.30, 1.50),
    ("guide_center",        80.70, 50.0, 120.0),
    ("guide_width",         36.86, 15.0, 70.0),
    ("cross_guide_scale",   0.553, 0.20, 1.20),
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
# B1 fast path
# ============================================================
def extract_b1_segments(caches, params):
    result = []
    for cache in caches:
        p = dict(params)
        p["use_sigmoid_aggregation"] = 0
        _, details = combine(cache, p)
        eff_w = _compute_effective_weights(cache["all_corners"], details["C_arr"])
        D_seg, w_seg = segment_by_difficulty(details["D_all"], eff_w, 30)
        result.append({"D_seg": D_seg, "w_seg": w_seg, "total_notes": details["total_notes"]})
    return result


def sigmoid_sr_fast(pre_data, params, k):
    D_cal = params["calib_a"] * pre_data["D_seg"] + params["calib_b"]
    D_solved, _ = solve_D_bisection(
        D_cal, pre_data["w_seg"], k=k,
        C=params["agg_sigmoid_C"], gamma=params["agg_sigmoid_ref_gamma"],
        high_weight_power=0.0, delta=5.0, tol=0.0001)
    SR = float(D_solved)
    SR *= pre_data["total_notes"] / (pre_data["total_notes"] + params["note_norm_N0"])
    if SR > params["rescale_threshold"]:
        SR = params["rescale_threshold"] + (SR - params["rescale_threshold"]) / params["rescale_divisor"]
    SR *= params["global_scale"]
    return SR


def eval_b1(pre_data_list, entries, params, k):
    losses = []
    for pre_data, entry in zip(pre_data_list, entries):
        try:
            sr = sigmoid_sr_fast(pre_data, params, k)
            losses.append(score_single(sr, entry["sr_ref"], entry["sr_error"]))
        except:
            losses.append(100.0)
    return float(np.mean(losses)) if losses else 1e9


def eval_full(params, caches, entries, k):
    losses = []
    for cache, entry in zip(caches, entries):
        try:
            p = dict(params)
            p["use_sigmoid_aggregation"] = 1
            p["agg_sigmoid_k"] = k
            sr, _ = combine(cache, p)
            losses.append(score_single(sr, entry["sr_ref"], entry["sr_error"]))
        except:
            losses.append(100.0)
    return float(np.mean(losses)) if losses else 1e9


def eval_detailed(params, caches, entries, k, use_sigmoid=True):
    preds, refs, errs = [], [], []
    for cache, entry in zip(caches, entries):
        try:
            p = dict(params)
            p["use_sigmoid_aggregation"] = 1 if use_sigmoid else 0
            p["agg_sigmoid_k"] = k
            sr, _ = combine(cache, p)
            preds.append(sr); refs.append(entry["sr_ref"]); errs.append(entry["sr_error"])
        except:
            preds.append(0); refs.append(entry["sr_ref"]); errs.append(entry["sr_error"])
    preds_arr = np.array(preds); refs_arr = np.array(refs)
    res = preds_arr - refs_arr
    loss = float(np.mean([score_single(p, r, e) for p, r, e in zip(preds, refs, errs)]))
    mae = float(np.mean(np.abs(res)))
    bands = {}
    for lo, hi in [(0, 4.5), (4.5, 6.5), (6.5, 8.5), (8.5, 10.5), (10.5, 20)]:
        m = (refs_arr >= lo) & (refs_arr < hi)
        if m.sum() > 0:
            bands[f"SR{lo}-{hi}"] = float(np.mean(res[m]))
    return {"loss": loss, "mae": mae, "bands": bands}


def run_nm(block_def, params, obj_fn, label, maxiter):
    p0, bounds, names = make_block(block_def, params)
    u0 = [(v - lo) / (hi - lo) if hi > lo else 0.5 for v, (lo, hi) in zip(p0, bounds)]

    def objective(u):
        vals = [np.clip(lo + ui*(hi-lo), lo, hi) for ui, (lo, hi) in zip(u, bounds)]
        return obj_fn(apply_block(names, vals, dict(params)))

    init_loss = objective(u0)
    print(f"\n  [{label}] Loss={init_loss:.4f}  ({len(names)}p, maxiter={maxiter})")
    t0 = time.time()

    result = minimize(objective, u0, method="Nelder-Mead",
                      options={"maxiter": maxiter, "xatol": 0.0005, "fatol": 0.0005,
                               "adaptive": True})

    vals_opt = [np.clip(lo + ui*(hi-lo), lo, hi) for ui, (lo, hi) in zip(result.x, bounds)]
    final_loss = objective(result.x)
    elapsed = time.time() - t0

    print(f"  [{label}] Done: Loss={final_loss:.4f} (Δ={final_loss-init_loss:+.4f}), "
          f"evals={result.nfev}, iters={result.nit}, time={elapsed:.0f}s")
    for i, name in enumerate(names):
        delta = vals_opt[i] - p0[i]
        if abs(delta) > 0.001:
            print(f"    {name:<28}: {p0[i]:.4f} → {vals_opt[i]:.4f}  ({delta:+.4f})")

    return apply_block(names, vals_opt, dict(params)), final_loss


# ============================================================
def main():
    K = 1.5
    print(f"=" * 72)
    print(f"Sigmoid Alternating NM — k={K} (fixed)")
    print(f"=" * 72)

    entries = load_playtest_data()
    print(f"[0] {len(entries)} entries")

    params_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "tuned_params_enhanced.json")
    with open(params_path) as f:
        p5 = json.load(f)
    params = dict(p5["params"])
    params.update({
        "use_sigmoid_aggregation": 1,
        "agg_sigmoid_k": K, "agg_sigmoid_C": 4.0, "agg_sigmoid_ref_gamma": 0.2,
        "calib_a": 0.89, "calib_b": 0.06,
        "note_norm_N0": 10.0, "rescale_threshold": 9.54,
        "rescale_divisor": 2.0, "global_scale": 1.055,
    })

    print(f"[1] Precomputing...")
    t0 = time.time()
    caches = []
    for i, e in enumerate(entries):
        try:
            c = precompute(e["osu_path"], use_enhanced=True, params=params)
            caches.append(c)
        except:
            caches.append(None)
        if (i+1) % 50 == 0: print(f"  {i+1}/{len(entries)}...")
    print(f"  {sum(1 for c in caches if c)}/{len(entries)} OK in {time.time()-t0:.0f}s")

    valid_c = [c for c in caches if c is not None]
    valid_e = [e for c, e in zip(caches, entries) if c is not None]
    n = len(valid_c)

    print(f"[2] Extracting B1 segments...")
    b1_pre = extract_b1_segments(valid_c, params)
    print(f"  Done in {time.time()-t0:.0f}s")

    print(f"[3] Initial evaluation...")
    sig_metrics = eval_detailed(params, valid_c, valid_e, K, use_sigmoid=True)
    pct_metrics = eval_detailed(params, valid_c, valid_e, K, use_sigmoid=False)
    print(f"  Percentile: Loss={pct_metrics['loss']:.4f}, MAE={pct_metrics['mae']:.4f}")
    print(f"  Sigmoid:    Loss={sig_metrics['loss']:.4f}, MAE={sig_metrics['mae']:.4f}")
    print(f"  Gap:        Loss={sig_metrics['loss']-pct_metrics['loss']:+.4f}, MAE={sig_metrics['mae']-pct_metrics['mae']:+.4f}")

    history = []
    best_loss = sig_metrics["loss"]
    best_params = dict(params)

    blocks = [
        (BLOCK1_SIGMOID,  "B1: Sigmoid+Calib+Post", "fast"),
        (BLOCK2_D_FORMULA, "B2: D Formula Core", "full"),
        (BLOCK3A_CROSS,   "B3a: Cross Distance", "full"),
        (BLOCK3B_RELEASE, "B3b: Release LN Tail", "full"),
        (BLOCK3C_INVERSE, "B3c: Inverse/Guide", "full"),
    ]

    for round_idx in range(2):
        print(f"\n{'─'*60}")
        print(f"ROUND {round_idx+1}/2")
        print(f"{'─'*60}")
        maxiter = 50 if round_idx == 0 else 25

        for block_def, label, mode in blocks:
            if mode == "fast":
                obj_fn = lambda p: eval_b1(b1_pre, valid_e, p, K)
            else:
                obj_fn = lambda p: eval_full(p, valid_c, valid_e, K)

            params, loss = run_nm(block_def, params, obj_fn, label, maxiter)
            history.append((label, loss))
            if loss < best_loss:
                best_loss = loss
                best_params = dict(params)
                print(f"  >> NEW BEST: Loss={loss:.4f}")

            if mode == "full":
                b1_pre = extract_b1_segments(valid_c, params)

    print(f"\n[5] Final evaluation...")
    final_metrics = eval_detailed(best_params, valid_c, valid_e, K, use_sigmoid=True)

    print(f"\n{'='*60}")
    print(f"RESULTS (k={K})")
    print(f"{'='*60}")
    print(f"  Percentile:  Loss={pct_metrics['loss']:.4f}, MAE={pct_metrics['mae']:.4f}")
    print(f"  Sigmoid:     Loss={final_metrics['loss']:.4f}, MAE={final_metrics['mae']:.4f}")
    print(f"  vs Pct:      Loss={final_metrics['loss']-pct_metrics['loss']:+.4f}, MAE={final_metrics['mae']-pct_metrics['mae']:+.4f}")
    print(f"\n  Sigmoid per-band:")
    for band, res in final_metrics["bands"].items():
        print(f"    {band}: {res:+.4f}")

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "tuned_params_sigmoid_k15.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"k": K, "loss": final_metrics["loss"], "mae": final_metrics["mae"],
                   "pct_loss": pct_metrics["loss"], "pct_mae": pct_metrics["mae"],
                   "method": f"alternating NM k={K}, 2 rounds", "history": history,
                   "params": best_params}, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()
