"""
Fit calibration params for LN-masked SR model.

The LN-masked model computes SR using only LN-dense sections of the map
(D_all masked to regions where LN_rep density > 0, dilated by 500ms).
This prevents hard RC sections from inflating the LN difficulty rating
on Hybrid (HB) maps.

Calibration fits: sr_ref_ln ≈ calib_a * ln_masked_sr + calib_b
using LN-labeled maps where the ground truth is the LN sub-rating.
"""
import sys, os, json, time
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tuning.data_loader import load_playtest_data
from spm_rating.rating import precompute, combine
from spm_rating.aggregate_sigmoid import (
    compute_SR_sigmoid, _compute_effective_weights,
    segment_by_difficulty, solve_D_bisection,
)
from spm_rating.utils import step_interp


def compute_ln_mask(all_corners, LN_rep):
    """Compute dilated LN-section mask (matches computeHBSectionMasks in JS)."""
    # Interpolate LN_rep density to all_corners grid
    ln_pts, ln_cum, ln_vals = LN_rep
    ln_density = step_interp(all_corners, ln_pts, np.array(ln_vals, dtype=float))

    # LN-dense: density > 0.01
    ln_mask = ln_density > 0.01

    # Dilate by ~500ms
    step_ms = 50
    n_steps = 10  # 500ms / 50ms
    # Approximate index step
    time_span = all_corners[-1] - all_corners[0] if len(all_corners) > 1 else 1
    idx_step = max(1, int(len(all_corners) * step_ms / time_span))
    for _ in range(n_steps):
        dilated = ln_mask.copy()
        for i in range(1, len(all_corners) - 1):
            if ln_mask[i - 1] or ln_mask[i + 1]:
                dilated[i] = True
        ln_mask = dilated

    return ln_mask


def compute_ln_masked_sr(all_corners, C_arr, D_all, ln_mask, total_notes, params):
    """
    Compute LN-masked SR: sigmoid aggregation over LN sections only.

    Only LN-masked time points contribute weight (C_arr zeroed elsewhere).
    """
    n = len(all_corners)
    C_ln = np.where(ln_mask, C_arr, 0.0)
    eff_w = _compute_effective_weights(all_corners, C_ln)

    calib_a = params.get("calib_a_ln_masked", 1.0)
    calib_b = params.get("calib_b_ln_masked", 0.0)
    D_calib = D_all * calib_a + calib_b

    n_seg = params.get("agg_n_segments", 30)
    D_seg, w_seg = segment_by_difficulty(D_calib, eff_w, n_seg)

    total_w = float(np.sum(w_seg))
    if len(D_seg) == 0 or total_w <= 0:
        return 0.0

    D_solved, _ = solve_D_bisection(
        D_seg, w_seg,
        k=params.get("agg_sigmoid_k", 1.56),
        C=params.get("agg_sigmoid_C", 4.0),
        gamma=params.get("agg_sigmoid_ref_gamma", 0.20),
        high_weight_power=0.0, delta=5.0, tol=0.0001,
    )

    SR = float(D_solved)
    SR *= total_notes / (total_notes + params.get("note_norm_N0", 10.0))

    threshold = params.get("rescale_threshold", 9.54)
    divisor = params.get("rescale_divisor", 2.0)
    if SR > threshold:
        SR = threshold + (SR - threshold) / divisor

    SR *= params.get("global_scale", 1.055)
    return SR


def load_params():
    """Load sigmoid tuned params."""
    params_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tuned_params_sigmoid.json"
    )
    # Fall back to tuned_params.json if sigmoid not found
    if not os.path.exists(params_path):
        params_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tuned_params.json"
        )
    with open(params_path) as f:
        data = json.load(f)
    return dict(data.get("params", data))


def main():
    print("=" * 60)
    print("LN-Masked SR Calibration")
    print("=" * 60)

    # Load params
    params = load_params()
    print(f"\nLoaded params from tuned_params_sigmoid.json")
    print(f"  sigmoid_k={params.get('agg_sigmoid_k', 1.56):.4f}")
    print(f"  sigmoid_C={params.get('agg_sigmoid_C', 4.0):.4f}")
    print(f"  sigmoid_gamma={params.get('agg_sigmoid_ref_gamma', 0.2):.4f}")

    # Load maps
    print("\nLoading playtest data...")
    entries = load_playtest_data()
    print(f"  Total valid entries: {len(entries)}")

    # Separate by sort
    ln_entries = [e for e in entries if e["sort"] == "ln" and e["sr_ref_ln"] is not None]
    rc_entries = [e for e in entries if e["sort"] == "rc" and e["d_ln"] is not None]
    hb_entries = [e for e in entries if e["sort"] == "hb" and e["sr_ref_ln"] is not None]
    print(f"  LN-labeled: {len(ln_entries)}, RC-labeled: {len(rc_entries)}, HB-labeled: {len(hb_entries)}")

    # Collect data points
    data_points = []  # [(ln_masked_sr, sr_ref_ln, mapfile, sort)]

    print("\nComputing LN-masked SR for all maps...")
    t0 = time.time()
    n_computed = 0

    # Precompute all maps first (cache-friendly)
    for entry in entries:
        osu_path = entry["osu_path"]
        if not osu_path or not os.path.exists(osu_path):
            continue

        try:
            # Precompute
            cache = precompute(osu_path, use_enhanced=True, params=params)

            # Combine to get D_all, C_arr
            SR_total, details = combine(cache, params)
            D_all = details.get("D_all")
            if D_all is None:
                continue

            all_corners = cache["all_corners"]
            C_arr = details["C_arr"]
            total_notes = details["total_notes"]
            LN_rep = cache["LN_rep"]

            # Compute LN mask and LN-masked SR
            ln_mask = compute_ln_mask(all_corners, LN_rep)
            ln_masked_sr = compute_ln_masked_sr(all_corners, C_arr, D_all, ln_mask,
                                                 total_notes, params)

            sort = entry["sort"]
            sr_ref_ln = entry.get("sr_ref_ln")

            if sr_ref_ln is not None and ln_masked_sr > 0.01:
                data_points.append((ln_masked_sr, sr_ref_ln, entry["mapfile"], sort))

            n_computed += 1

        except Exception as e:
            print(f"  Error on {entry.get('mapfile', '?')}: {e}")
            continue

    elapsed = time.time() - t0
    print(f"  Computed {n_computed} maps in {elapsed:.1f}s")
    print(f"  Valid data points: {len(data_points)}")

    if len(data_points) < 5:
        print("\n  ERROR: Not enough data points for calibration.")
        return

    # Separate by sort
    ln_pts = [(s, r) for s, r, _, sort in data_points if sort == "ln"]
    hb_pts = [(s, r) for s, r, _, sort in data_points if sort == "hb"]
    rc_pts = [(s, r) for s, r, _, sort in data_points if sort == "rc"]

    # ============================================================
    # Fit on LN-labeled maps only (ground truth = sr_ref_ln)
    # ============================================================
    if len(ln_pts) >= 3:
        print(f"\n{'='*40}")
        print("Fitting on LN-labeled maps (n={})".format(len(ln_pts)))
        print(f"{'='*40}")

        X_ln = np.array([s for s, r in ln_pts])
        Y_ln = np.array([r for s, r in ln_pts])

        # Linear fit: Y ≈ a * X + b
        A = np.column_stack([X_ln, np.ones_like(X_ln)])
        coeffs, residuals, rank, singular = np.linalg.lstsq(A, Y_ln, rcond=None)
        calib_a, calib_b = coeffs[0], coeffs[1]

        Y_pred = calib_a * X_ln + calib_b
        mae = np.mean(np.abs(Y_ln - Y_pred))
        rmse = np.sqrt(np.mean((Y_ln - Y_pred) ** 2))
        r2 = 1 - np.sum((Y_ln - Y_pred) ** 2) / np.sum((Y_ln - np.mean(Y_ln)) ** 2)

        print(f"\n  Linear fit: sr_ref_ln = {calib_a:.6f} * ln_masked_sr + {calib_b:.6f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")

        # Check stability across data points
        errors = np.abs(Y_ln - Y_pred)
        worst_idx = np.argmax(errors)
        print(f"\n  Worst fit: {ln_pts[worst_idx][1]:.2f} vs pred {Y_pred[worst_idx]:.2f} "
              f"(error={errors[worst_idx]:.4f}, file={ln_pts[worst_idx]})")

        # Check ratio of ln_masked_sr / total_sr
        ratios = np.array([s / r for s, r in ln_pts])
        print(f"\n  ln_masked / sr_ref ratio: mean={np.mean(ratios):.4f}, "
              f"median={np.median(ratios):.4f}, std={np.std(ratios):.4f}")
        print(f"  (Should be ~1.0 for pure LN maps where all sections are LN)")
    else:
        print(f"\n  Not enough LN-labeled maps for fitting (need >=3, have {len(ln_pts)})")
        calib_a, calib_b = 1.0, 0.0

    # ============================================================
    # Check on HB-labeled maps (if available)
    # ============================================================
    if len(hb_pts) >= 3:
        print(f"\n{'='*40}")
        print(f"Validation on HB-labeled maps (n={len(hb_pts)})")
        print(f"{'='*40}")

        X_hb = np.array([s for s, r in hb_pts])
        Y_hb = np.array([r for s, r in hb_pts])

        # Apply calibration
        Y_pred_hb = calib_a * X_hb + calib_b
        mae_hb = np.mean(np.abs(Y_hb - Y_pred_hb))
        rmse_hb = np.sqrt(np.mean((Y_hb - Y_pred_hb) ** 2))

        # Also compare: ratio of ln_masked vs total (should be <1 for HB maps)
        ln_ratios = np.array([s / r for s, r in hb_pts])
        print(f"\n  Calibrated MAE: {mae_hb:.4f}, RMSE: {rmse_hb:.4f}")
        print(f"  ln_masked / sr_ref ratio: mean={np.mean(ln_ratios):.4f}, "
              f"median={np.median(ln_ratios):.4f}")
        print(f"  (Should be <1.0 — LN-masked < total because RC sections excluded)")

        # Show top differences
        errors_hb = np.abs(Y_hb - Y_pred_hb)
        for i in np.argsort(errors_hb)[-3:][::-1]:
            print(f"  {hb_pts[i][1]:.2f} vs pred {Y_pred_hb[i]:.2f} "
                  f"(error={errors_hb[i]:.4f}, ln_masked={X_hb[i]:.2f}, "
                  f"file={hb_pts[i]})")

    # ============================================================
    # Print final recommended params
    # ============================================================
    print(f"\n{'='*60}")
    print("RECOMMENDED JS PARAMS")
    print(f"{'='*60}")
    print(f'    calib_a_ln_masked: {calib_a:.6f},')
    print(f'    calib_b_ln_masked: {calib_b:.6f},')
    print(f"{'='*60}")

    # Save to JSON
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "ln_masked_calibration.json")
    with open(output_path, "w") as f:
        json.dump({
            "description": "LN-masked SR calibration (linear fit on LN-labeled maps)",
            "calib_a_ln_masked": round(float(calib_a), 6),
            "calib_b_ln_masked": round(float(calib_b), 6),
            "n_ln_maps": len(ln_pts),
            "mae_ln": round(float(mae) if len(ln_pts) >= 3 else 0, 4),
            "rmse_ln": round(float(rmse) if len(ln_pts) >= 3 else 0, 4),
            "r2_ln": round(float(r2) if len(ln_pts) >= 3 else 0, 4),
        }, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
