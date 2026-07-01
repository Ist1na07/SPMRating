"""Retrain the correction layer with the v0.4.0 9-feature model (chord2 + nps_std).

This is the canonical retraining for the release: v1 D-formula base (no z_level
modifications), 9 features = v1's 7 + nps_std + chord2.

Uses the displacement-invariance fast path (same as retrain_correction_z.py):
  - Precompute D_solved_base per map ONCE with v1 tuned params.
  - NM optimizes 9 weights + 4 postprocess on the fast path.
  - 5-fold CV + 5 restarts, L2 λ=0.01.

Outputs tuned_correction.json (overwriting the v1 7-feature version).

Usage:
  python scripts/retrain_correction_zver.py
"""
import sys, os, json, time, pickle, argparse
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ZROOT = os.path.dirname(HERE)              # SPMRating-Z-Release
PROJROOT = os.path.dirname(ZROOT)          # SPMRating_Upgrade_Z
# Caches live in the Z research repo (precomputed). The release repo's own
# tuning/data_loader is now synced (parse_float + Ranked support).
ZRESEARCH = os.path.join(PROJROOT, "SPMRating-Z")
sys.path.insert(0, ZROOT)
sys.path.insert(0, os.path.join(ZROOT, "tuning"))
os.chdir(ZROOT)

from tuning.data_loader import load_playtest_data
from tuning.scorer import score_single
from spm_calc import (FEATURE_NAMES, FEATURE_PARAMS, compute_features)
from spm_rating import rating
from spm_rating.combine_rc_ln import compute_total_notes
from spm_rating.aggregate_sigmoid import (
    segment_by_difficulty, solve_D_bisection, _compute_effective_weights
)

# Load v1 tuned params + v1 correction (for init / postprocess init)
with open(os.path.join(ZROOT, "tuned_params_sigmoid.json"), encoding="utf-8") as f:
    TUNED_PARAMS = json.load(f)["params"]
TUNED_PARAMS["use_sigmoid_aggregation"] = 1
with open(os.path.join(ZROOT, "tuned_correction.json"), encoding="utf-8") as f:
    _tc = json.load(f)
CORRECTION_WEIGHTS = _tc["correction_weights"]
CORRECTION_POSTPROCESS = _tc["postprocess"]

LAMBDA = 0.01  # L2 regularization (optimal per TUNING_CORRECTION_LAYER.md §7)


def cache_path_for(entry):
    base = os.path.basename(entry["mapfile"]).replace(".osu", "")
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in base)
    return os.path.join(ZRESEARCH, "cache", safe + ".pkl")


def precompute_records(caches, entries):
    """Precompute D_solved_base + 9 features for all maps (v1 base, z_level=0)."""
    p = dict(TUNED_PARAMS)
    records = []
    t0 = time.time()
    for i, (c, e) in enumerate(zip(caches, entries)):
        try:
            sr_base, d = rating.combine(c, params=p)
        except Exception as ex:
            print(f"  [{i}] combine fail: {ex}", flush=True)
            continue
        D_full = d["D_all"]
        C_arr = d["C_arr"]
        cal_a = p.get("calib_a", 0.893)
        cal_b = p.get("calib_b", 0.031)
        D_calib = cal_a * D_full + cal_b
        feats = compute_features(c)
        total_notes = compute_total_notes(c["note_seq"], c["LN_seq"])
        eff_w = _compute_effective_weights(c["all_corners"], C_arr)
        D_seg, w_seg = segment_by_difficulty(D_calib, eff_w, 30)
        D_solved, _ = solve_D_bisection(
            D_seg, w_seg,
            k=p.get("agg_sigmoid_k", 2.09),
            C=p.get("agg_sigmoid_C", 3.969),
            gamma=p.get("agg_sigmoid_ref_gamma", 0.196),
            delta=5.0, tol=0.0001,
        )
        records.append({
            "D_solved_base": D_solved, "n_eff": total_notes, "features": feats,
            "sr_ref": e["sr_ref"], "sr_error": e["sr_error"], "sort": e["sort"],
        })
        if (i + 1) % 100 == 0:
            print(f"  precompute {i+1}/{len(caches)} ({time.time()-t0:.1f}s)", flush=True)
    print(f"  precompute done: {len(records)} records in {time.time()-t0:.1f}s", flush=True)
    return records


def build_block():
    """9 weights + 4 postprocess = 13 params. Init from v1 (new feats init 0)."""
    cw = CORRECTION_WEIGHTS
    pp = CORRECTION_POSTPROCESS
    return [
        ("w_speed",  cw.get("speed", -0.038), -1.0, 1.0),
        ("w_burst",  cw.get("burst", -0.025), -1.0, 1.0),
        ("w_chord",  cw.get("chord", -0.714), -3.0, 3.0),
        ("w_pj",     cw.get("pj",   -0.005), -1.0, 1.0),
        ("w_hs",     cw.get("hs",    0.043), -1.0, 1.0),
        ("w_lb",     cw.get("lb",    0.020), -1.0, 1.0),
        ("w_fj",     cw.get("fj",    0.265), -1.0, 1.0),
        ("w_nps_std", 0.0, -1.0, 1.0),   # new
        ("w_chord2",  0.0, -1.0, 1.0),   # new
        ("note_norm_N0",      float(pp["N0"]),       0.0,  80.0),
        ("rescale_threshold", float(pp["threshold"]), 6.0,  14.0),
        ("rescale_divisor",   float(pp["divisor"]),   1.2,  4.0),
        ("global_scale",      float(pp["scale"]),     0.85, 1.20),
    ]

# Map block param names -> feature names
W_NAMES = ["w_speed","w_burst","w_chord","w_pj","w_hs","w_lb","w_fj","w_nps_std","w_chord2"]
F_NAMES = ["speed","burst","chord","pj","hs","lb","fj","nps_std","chord2"]


def fast_eval(x, records, block_def, indices=None):
    vals = [lo + ui*(hi-lo) for ui, (_,_,lo,hi) in zip(x, block_def)]
    p = {name: v for (name,_,_,_), v in zip(block_def, vals)}
    if p["rescale_divisor"] <= 1.0 or p["global_scale"] <= 0 or p["note_norm_N0"] < 0:
        return 1e6
    ws = [p[wn] for wn in W_NAMES]
    N0, thr, div, gs = p["note_norm_N0"], p["rescale_threshold"], p["rescale_divisor"], p["global_scale"]
    reg = LAMBDA * sum(w*w for w in ws)
    if indices is None:
        indices = range(len(records))
    total = 0.0; n_ok = 0
    for i in indices:
        rec = records[i]
        feats = rec["features"]
        correction = sum(w * feats.get(fn, 0.0) for w, fn in zip(ws, F_NAMES))
        SR = rec["D_solved_base"] + correction
        n_eff = rec["n_eff"]
        SR *= n_eff / (n_eff + max(N0, 0.01))
        if SR > thr:
            SR = thr + (SR - thr) / div
        SR *= gs
        total += score_single(SR, rec["sr_ref"], rec["sr_error"])
        n_ok += 1
    if n_ok == 0:
        return 1e6
    return total / n_ok + reg


def cross_validate(records, block_def, n_folds=5, seed=42, restarts=3):
    N = len(records)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    fold_size = N // n_folds
    folds = [perm[k*fold_size:(k+1)*fold_size] for k in range(n_folds)]
    if N % n_folds != 0:
        folds[-1] = perm[(n_folds-1)*fold_size:]
    train_losses, test_losses = [], []
    for fold_idx in range(n_folds):
        test_idx = folds[fold_idx]
        train_idx = np.array([i for i in range(N) if i not in set(test_idx.tolist())])
        x0 = np.array([(init-lo)/(hi-lo) for (_,init,lo,hi) in block_def])
        x0 = np.clip(x0, 0.01, 0.99)
        best_x = x0.copy(); best_loss = fast_eval(x0, records, block_def, indices=train_idx)
        for restart in range(restarts):
            if restart == 0:
                x_start = x0.copy()
            else:
                rng2 = np.random.RandomState(seed + restart + fold_idx*100)
                x_start = np.clip(x0 + rng2.normal(0, 0.15, len(x0)), 0.01, 0.99)
            res = minimize(fast_eval, x_start, args=(records, block_def, train_idx),
                          method="Nelder-Mead",
                          options={"maxiter": 5000, "xatol": 1e-7, "fatol": 1e-7, "adaptive": True})
            if res.fun < best_loss:
                best_loss = res.fun; best_x = res.x.copy()
        train_losses.append(best_loss)
        test_loss = fast_eval(best_x, records, block_def, indices=test_idx)
        test_losses.append(test_loss)
        print(f"    Fold {fold_idx+1}: train={best_loss:.4f} test={test_loss:.4f} gap={test_loss-best_loss:.4f}", flush=True)
    print(f"  CV: train={np.mean(train_losses):.4f}±{np.std(train_losses):.4f} "
          f"test={np.mean(test_losses):.4f}±{np.std(test_losses):.4f} "
          f"gap={np.mean(test_losses)-np.mean(train_losses):.4f}", flush=True)
    return float(np.mean(train_losses)), float(np.mean(test_losses)), float(np.std(test_losses))


def main():
    print("="*70, flush=True)
    print("RETRAIN CORRECTION LAYER — v0.4.0 (9 features: +nps_std +chord2)", flush=True)
    print("="*70, flush=True)

    entries = load_playtest_data(maps_root=PROJROOT)
    print(f"entries: {len(entries)}", flush=True)

    caches, used = [], []
    for e in entries:
        cp = cache_path_for(e)
        if not os.path.exists(cp):
            continue
        try:
            with open(cp, "rb") as f:
                c = pickle.load(f)
            caches.append(c); used.append(e)
        except Exception:
            pass
    print(f"loaded {len(caches)} caches", flush=True)

    records = precompute_records(caches, used)
    block_def = build_block()

    x0 = np.array([(init-lo)/(hi-lo) for (_,init,lo,hi) in block_def])
    x0 = np.clip(x0, 0.01, 0.99)
    base_loss = fast_eval(x0, records, block_def)
    print(f"  Baseline (v1 7-feature weights, new feats=0): {base_loss:.4f}", flush=True)

    best_x = x0.copy(); best_loss = base_loss
    t0 = time.time()
    for restart in range(5):
        if restart == 0:
            x_start = x0.copy()
        else:
            rng = np.random.RandomState(42 + restart)
            x_start = np.clip(x0 + rng.normal(0, 0.15, len(x0)), 0.01, 0.99)
        print(f"  Restart {restart+1}/5 ...", flush=True, end=" ")
        res = minimize(fast_eval, x_start, args=(records, block_def),
                      method="Nelder-Mead",
                      options={"maxiter": 10000, "xatol": 1e-7, "fatol": 1e-7, "adaptive": True})
        print(f"loss={res.fun:.4f}", flush=True)
        if res.fun < best_loss:
            best_loss = res.fun; best_x = res.x.copy()
    print(f"  Retrain done in {time.time()-t0:.1f}s, best loss={best_loss:.4f}", flush=True)

    print(f"\n  5-fold cross-validation:", flush=True)
    cv_train, cv_test, cv_std = cross_validate(records, block_def, n_folds=5, seed=42, restarts=3)

    vals = [lo + ui*(hi-lo) for ui, (_,_,lo,hi) in zip(best_x, block_def)]
    best_params = {name: float(v) for (name,_,_,_), v in zip(block_def, vals)}

    # Per-sort breakdown
    by_sort = {}
    ws = [best_params[wn] for wn in W_NAMES]
    all_preds, all_refs, all_errs = [], [], []
    for rec in records:
        feats = rec["features"]
        correction = sum(w * feats.get(fn, 0.0) for w, fn in zip(ws, F_NAMES))
        SR = rec["D_solved_base"] + correction
        n_eff = rec["n_eff"]
        SR *= n_eff / (n_eff + max(best_params["note_norm_N0"], 0.01))
        if SR > best_params["rescale_threshold"]:
            SR = best_params["rescale_threshold"] + (SR - best_params["rescale_threshold"]) / best_params["rescale_divisor"]
        SR *= best_params["global_scale"]
        by_sort.setdefault(rec["sort"], []).append((SR, rec["sr_ref"], rec["sr_error"]))
        all_preds.append(SR); all_refs.append(rec["sr_ref"]); all_errs.append(rec["sr_error"])

    print(f"\n  Per-sort (retrained, full set):", flush=True)
    for s in sorted(by_sort):
        lst = by_sort[s]
        preds = np.array([x[0] for x in lst]); refs = np.array([x[1] for x in lst]); errs = np.array([x[2] for x in lst])
        dd = np.abs(preds-refs); l = np.array([score_single(p,r,e) for p,r,e in zip(preds,refs,errs)])
        print(f"    {s:<5s} n={len(lst):3d} loss={np.mean(l):.4f} MAE={np.mean(dd):.4f} "
              f"pass={np.mean(dd<=errs):.3f} bias={np.mean(preds-refs):+.3f}", flush=True)
    all_preds=np.array(all_preds); all_refs=np.array(all_refs); all_errs=np.array(all_errs)
    dd=np.abs(all_preds-all_refs); l=np.array([score_single(p,r,e) for p,r,e in zip(all_preds,all_refs,all_errs)])
    print(f"  overall: n={len(all_preds)} loss={np.mean(l):.4f} MAE={np.mean(dd):.4f} "
          f"pass={np.mean(dd<=all_errs):.3f} bias={np.mean(all_preds-all_refs):+.3f}", flush=True)

    print(f"\n  Retrained weights (9 features):", flush=True)
    for wn, fn in zip(W_NAMES, F_NAMES):
        w = best_params[wn]
        marker = " (NEW)" if fn in ("nps_std","chord2") else ""
        print(f"    {fn:<12} {w:+.4f}{marker}", flush=True)

    # Save — overwrite tuned_correction.json with the 9-feature model
    out = {
        "model_type": "single_weight",
        "features": F_NAMES,
        "correction_weights": {fn: best_params[wn] for wn, fn in zip(W_NAMES, F_NAMES)},
        "postprocess": {
            "N0": best_params["note_norm_N0"],
            "threshold": best_params["rescale_threshold"],
            "divisor": best_params["rescale_divisor"],
            "scale": best_params["global_scale"],
        },
        "regularization_lambda": LAMBDA,
        "in_sample_loss": {"total": float(np.mean(l))},
        "cv_test_loss": cv_test,
        "cv_test_std": cv_std,
        "cv_train_loss": cv_train,
    }
    out_path = os.path.join(ZROOT, "tuned_correction.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
