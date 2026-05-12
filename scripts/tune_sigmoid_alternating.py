"""
Block-wise alternating Nelder-Mead optimization for sigmoid aggregation.

Strategy: partition ~35 params into blocks of 5-10, cycle NM on each block.

Blocks:
  B1: Sigmoid aggregation + D calibration + post-processing (9 params)
  B2: D formula core (10 params)
  B3a: Cross distance weights (5 params)
  B3b: Release LN tail (6 params)
  B3c: Inverse/Guide (7 params)

Cycle: B1 → B2 → B3a → B3b → B3c → B1 → B2 → ... until convergence.
"""

import sys, os, json, time, copy
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tuning.data_loader import load_playtest_data
from tuning.scorer import score_single
from spm_rating.rating import precompute, combine
from spm_rating.aggregate_sigmoid import compute_SR_sigmoid
from spm_rating.aggregate import compute_SR as compute_SR_pct
from spm_rating.aggregate import compute_total_notes
from spm_rating.components import cross_enhanced, release_enhanced, shield, inverse
from spm_rating.combine import compute_D as _compute_D
from spm_rating.utils import interp_values


# ============================================================
# Parameter block definitions
# ============================================================
# (name, initial, lower, upper, description)

BLOCK1_SIGMOID = [
    ("calib_a",        0.8902, 0.60, 1.20, "D pre-calibration: D' = a*D + b"),
    ("calib_b",        0.2880, -0.80, 0.80, "D pre-calibration offset"),
    ("agg_sigmoid_k",  0.5,    0.10, 3.00, "sigmoid steepness"),
    ("agg_sigmoid_C",  4.0,    1.50, 12.0, "sigmoid shape C"),
    ("agg_sigmoid_ref_gamma", 0.20, 0.03, 0.40, "reference accuracy gamma"),
    ("note_norm_N0",   10.0,   0.0,  80.0, "note count normalization offset"),
    ("rescale_threshold", 9.54, 6.0, 14.0, "high-SR rescale threshold"),
    ("rescale_divisor",   2.00, 1.2,  4.0,  "high-SR rescale divisor"),
    ("global_scale",   1.055,  0.92, 1.15, "global output scale"),
]

BLOCK2_D_FORMULA = [
    ("S_w1",      0.514,  0.15, 0.85, "S: jack branch weight"),
    ("S_p",       1.117,  0.70, 2.50, "S: p-norm exponent"),
    ("alpha_P",   0.724,  0.30, 1.50, "stream: Pbar weight"),
    ("alpha_R",   28.47,  12.0, 50.0, "stream: Rbar numerator"),
    ("alpha_C",   9.64,   3.00, 20.0, "stream: Rbar denominator offset"),
    ("alpha_S",   0.479,  0.05, 2.50, "stream: Shield weight"),
    ("alpha_V",   0.435,  0.10, 2.00, "stream: Inverse/Vbar weight"),
    ("D_beta1",   1.170,  0.50, 2.50, "D: S^0.5*T^1.5 coefficient"),
    ("D_beta2",   0.389,  0.15, 0.80, "D: linear S coefficient"),
    ("Abar_scale", 1.016, 0.85, 1.20, "Abar: anchor sensitivity"),
]

BLOCK3A_CROSS = [
    ("cross_dist_exponent_rc",  1.010, 0.50, 2.00, "RC: column distance exponent"),
    ("cross_dist_exponent_ln",  0.988, 0.50, 2.00, "LN: column distance exponent"),
    ("cross_same_hand_penalty_rc", 0.337, 0.05, 0.80, "RC: same-hand extra penalty"),
    ("cross_same_hand_penalty_ln", 0.294, 0.05, 0.80, "LN: same-hand extra penalty"),
    ("cross_thumb_bridge_factor",  0.496, 0.10, 0.90, "thumb bridge factor"),
]

BLOCK3B_RELEASE = [
    ("release_tail_coeff",      0.123, 0.03, 0.30, "LN tail difficulty base"),
    ("release_tail_to_tap",     2.099, 0.80, 4.00, "LN tail→tap vs tail→LN weight"),
    ("release_same_col_bonus",  0.300, 0.10, 1.50, "same-column LN tail bonus"),
    ("release_coord_exponent",  0.630, 0.20, 1.50, "cross-column coord weight exponent"),
    ("release_seq_coeff",       0.047, 0.01, 0.15, "LN tail sequence difficulty"),
    ("lock_interaction_coeff",  0.235, 0.05, 0.60, "lock (other fingers holding LN) interaction"),
]

BLOCK3C_INVERSE = [
    ("inv_amplitude",       3.465, 1.50, 6.00, "inverse spike amplitude"),
    ("inv_tau",             30.82, 15.0, 60.0, "inverse spike decay time (ms)"),
    ("inv_power",           0.927, 0.50, 1.50, "inverse spike decay shape"),
    ("guide_depth",         0.797, 0.30, 1.50, "guide dip depth"),
    ("guide_center",        80.70, 50.0, 120.0,"guide dip center time (ms)"),
    ("guide_width",         36.86, 15.0, 70.0, "guide dip width (ms)"),
    ("cross_guide_scale",   0.553, 0.20, 1.20, "cross-column guide scaling"),
]


def make_block(block_def, params):
    """Extract a param dict and bounds list from block definition and current params."""
    p0 = [params.get(name, init) for name, init, _, _, _ in block_def]
    bounds = [(lo, hi) for _, _, lo, hi, _ in block_def]
    names = [name for name, _, _, _, _ in block_def]
    return p0, bounds, names


def apply_block(names, values, params):
    """Write optimized values back into params dict."""
    for name, val in zip(names, values):
        params[name] = float(val)
    return params


# ============================================================
# Loss function
# ============================================================
def evaluate_loss(params, caches, entries, use_sigmoid=True):
    """Compute mean loss across all maps with given params."""
    losses = []
    for cache, entry in zip(caches, entries):
        try:
            p = dict(params)
            p["use_sigmoid_aggregation"] = 1 if use_sigmoid else 0
            sr, _ = combine(cache, p)
            losses.append(score_single(sr, entry["sr_ref"], entry["sr_error"]))
        except Exception:
            losses.append(100.0)  # heavy penalty for failures
    return float(np.mean(losses)) if losses else 1e9


def evaluate_mae(params, caches, entries, use_sigmoid=True):
    """Compute MAE across all maps."""
    errors = []
    for cache, entry in zip(caches, entries):
        try:
            p = dict(params)
            p["use_sigmoid_aggregation"] = 1 if use_sigmoid else 0
            sr, _ = combine(cache, p)
            errors.append(abs(sr - entry["sr_ref"]))
        except Exception:
            errors.append(10.0)
    return float(np.mean(errors)) if errors else 1e9


def evaluate_detailed(params, caches, entries):
    """Return loss, mae, per-band residuals."""
    preds, refs, errs = [], [], []
    for cache, entry in zip(caches, entries):
        try:
            p = dict(params)
            p["use_sigmoid_aggregation"] = 1
            sr, _ = combine(cache, p)
            preds.append(sr)
            refs.append(entry["sr_ref"])
            errs.append(entry["sr_error"])
        except Exception:
            preds.append(0)
            refs.append(entry["sr_ref"])
            errs.append(entry["sr_error"])
    preds_arr = np.array(preds)
    refs_arr = np.array(refs)
    res = preds_arr - refs_arr
    loss = float(np.mean([score_single(p, r, e) for p, r, e in zip(preds, refs, errs)]))
    mae = float(np.mean(np.abs(res)))

    bands = {}
    for lo, hi in [(0, 4.5), (4.5, 6.5), (6.5, 8.5), (8.5, 10.5), (10.5, 20)]:
        m = (refs_arr >= lo) & (refs_arr < hi)
        if m.sum() > 0:
            bands[f"SR{lo}-{hi}"] = float(np.mean(res[m]))

    return {"loss": loss, "mae": mae, "bands": bands}


# ============================================================
# Main optimization
# ============================================================
def run_nm_block(block_def, params, caches, entries, label, maxiter=200):
    """Run Nelder-Mead on one block. Returns updated params + results."""
    p0, bounds, names = make_block(block_def, params)

    # Normalize to [0,1] per param
    u0 = []
    for val, (lo, hi) in zip(p0, bounds):
        u0.append((val - lo) / (hi - lo) if hi > lo else 0.5)

    def objective(u):
        # Denormalize
        vals = []
        for ui, (lo, hi) in zip(u, bounds):
            v = lo + ui * (hi - lo)
            vals.append(np.clip(v, lo, hi))
        temp = apply_block(names, vals, dict(params))
        return evaluate_loss(temp, caches, entries, use_sigmoid=True)

    print(f"\n  [{label}] NM start: Loss={objective(u0):.4f}")
    t0 = time.time()

    result = minimize(objective, u0, method="Nelder-Mead",
                      options={"maxiter": maxiter, "xatol": 0.0001, "fatol": 0.0001,
                               "adaptive": True})

    elapsed = time.time() - t0

    # Denormalize result
    u_opt = result.x
    vals_opt = []
    for ui, (lo, hi) in zip(u_opt, bounds):
        v = lo + ui * (hi - lo)
        vals_opt.append(np.clip(v, lo, hi))
    final_loss = objective(u_opt)

    print(f"  [{label}] Done: Loss={final_loss:.4f}, evals={result.nfev}, "
          f"iters={result.nit}, time={elapsed:.0f}s")

    # Report param changes
    for i, name in enumerate(names):
        delta = vals_opt[i] - p0[i]
        if abs(delta) > 0.001:
            print(f"    {name:<28}: {p0[i]:.4f} → {vals_opt[i]:.4f}  ({delta:+.4f})")

    updated = apply_block(names, vals_opt, dict(params))
    return updated, final_loss


def main():
    print("=" * 72)
    print("Sigmoid Alternating Block-wise Optimization")
    print("=" * 72)

    # 0. Load data
    print("\n[0/5] Loading playtest data...")
    entries = load_playtest_data()
    print(f"  {len(entries)} entries")

    # Load Phase 5 base params
    params_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "tuned_params_enhanced.json")
    with open(params_path) as f:
        phase5_data = json.load(f)
    params = dict(phase5_data["params"])
    p5_loss = phase5_data["loss"]
    print(f"  Phase 5 baseline: Loss={p5_loss:.4f}")

    # Set sigmoid defaults + calibration
    params["use_sigmoid_aggregation"] = 1
    params["calib_a"] = 0.8902
    params["calib_b"] = 0.2880
    params["agg_sigmoid_k"] = 0.5
    params["agg_sigmoid_C"] = 4.0
    params["agg_sigmoid_ref_gamma"] = 0.2

    # 1. Precompute all maps
    print(f"\n[1/5] Precomputing {len(entries)} maps (enhanced mode)...")
    caches = []
    for i, entry in enumerate(entries):
        try:
            c = precompute(entry["osu_path"], use_enhanced=True, params=params)
            caches.append(c)
        except Exception as e:
            print(f"  FAIL: {entry['mapfile']}: {e}")
            caches.append(None)
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(entries)}...")
    n_ok = sum(1 for c in caches if c is not None)
    print(f"  OK: {n_ok}/{len(entries)}")

    # Filter to valid pairs
    valid_caches = [c for c in caches if c is not None]
    valid_entries = [e for c, e in zip(caches, entries) if c is not None]
    print(f"  Using {len(valid_caches)} valid maps")

    # 2. Evaluate initial state
    print(f"\n[2/5] Evaluating initial state...")
    pct_start = evaluate_detailed(params, valid_caches, valid_entries)
    # Also percentile baseline
    pct_params = dict(params)
    pct_params["use_sigmoid_aggregation"] = 0
    pct_metrics = evaluate_detailed(pct_params, valid_caches, valid_entries)
    print(f"  Percentile baseline: Loss={pct_metrics['loss']:.4f}, MAE={pct_metrics['mae']:.4f}")
    print(f"  Sigmoid initial:     Loss={pct_start['loss']:.4f}, MAE={pct_start['mae']:.4f}")
    print(f"  Initial vs pct:      {pct_start['loss'] - pct_metrics['loss']:+.4f}")

    # 3. Optimization rounds
    print(f"\n[3/5] Starting alternating optimization...")
    print(f"{'=' * 60}")

    history = []
    best_loss = pct_start["loss"]
    best_params = dict(params)

    blocks = [
        (BLOCK1_SIGMOID,  "B1: Sigmoid+Calib+Post"),
        (BLOCK2_D_FORMULA, "B2: D Formula Core"),
        (BLOCK3A_CROSS,   "B3a: Cross Distance"),
        (BLOCK3B_RELEASE, "B3b: Release LN Tail"),
        (BLOCK3C_INVERSE, "B3c: Inverse/Guide"),
    ]

    max_rounds = 3
    for round_idx in range(max_rounds):
        print(f"\n{'─' * 60}")
        print(f"ROUND {round_idx + 1}/{max_rounds}")
        print(f"{'─' * 60}")

        for block_def, label in blocks:
            params, loss = run_nm_block(block_def, params, valid_caches, valid_entries,
                                        label, maxiter=150 if round_idx == 0 else 100)
            history.append((label, loss))

            if loss < best_loss:
                best_loss = loss
                best_params = dict(params)
                print(f"  >> NEW BEST: Loss={loss:.4f}")

            # Show current vs percentile
            _, mae = evaluate_mae(params, valid_caches, valid_entries, use_sigmoid=True), \
                     evaluate_mae(params, valid_caches, valid_entries, use_sigmoid=False)
            # Quick MAE check
            mae_sig = evaluate_mae(params, valid_caches, valid_entries, use_sigmoid=True)
            print(f"  [MAE sigmoid={mae_sig:.4f}]")

    # 4. Final evaluation
    print(f"\n[4/5] Final evaluation...")
    final_metrics = evaluate_detailed(best_params, valid_caches, valid_entries)
    pct_final = pct_metrics  # unchanged since percentile params fixed

    print(f"\n{'=' * 60}")
    print(f"FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"  Percentile (Phase 5):   Loss={pct_final['loss']:.4f}, MAE={pct_final['mae']:.4f}")
    print(f"  Sigmoid (optimized):    Loss={final_metrics['loss']:.4f}, MAE={final_metrics['mae']:.4f}")
    print(f"  Improvement:            {final_metrics['loss'] - pct_final['loss']:+.4f} Loss")
    print(f"\n  Per-band residuals:")
    for band, res in final_metrics["bands"].items():
        print(f"    {band}: {res:+.4f}")

    # 5. Save best params
    print(f"\n[5/5] Saving best params...")
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "tuned_params_sigmoid.json")
    save_data = {
        "loss": final_metrics["loss"],
        "mae": final_metrics["mae"],
        "pct_loss": pct_final["loss"],
        "method": "alternating block NM (sigmoid aggregation)",
        "history": history,
        "params": best_params,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved to: {out_path}")

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
