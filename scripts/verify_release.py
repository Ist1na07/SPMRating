"""Verify the v0.4.0 release: full-set loss + Speed Practice distribution.

Compares:
  - v1 baseline (SPMRating-main, 7 features)
  - v0.4.0 (SPMRating-Z-Release, 9 features +nps_std +chord2)

Reports overall loss, per-sort, and Speed Practice 0th->Stellium distribution.
"""
import sys, os, json, pickle, re, time, glob
import numpy as np

PROJROOT = r"E:\Files\Developing\SPMRating_Upgrade_Z"
MAIN = os.path.join(PROJROOT, "SPMRating-main")
ZREL = os.path.join(PROJROOT, "SPMRating-Z-Release")
ZRES = os.path.join(PROJROOT, "SPMRating-Z")

# Use Z research tuning (has Ranked + parse_float) for data loading consistency
sys.path.insert(0, os.path.join(ZRES, "tuning"))
# tuning is a package dir, need its parent too
sys.path.insert(0, ZRES)
from tuning.data_loader import load_playtest_data
from tuning.scorer import score_single


def cache_path_for(entry):
    base = os.path.basename(entry["mapfile"]).replace(".osu", "")
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in base)
    return os.path.join(ZRES, "cache", safe + ".pkl")


def eval_repo(repo_path, entries, caches, label):
    """Evaluate a repo's spm_calc pipeline on all caches."""
    sys.path.insert(0, repo_path)
    # Force reimport
    for mod in list(sys.modules):
        if mod.startswith("spm_calc") or mod.startswith("spm_rating"):
            del sys.modules[mod]
    import spm_calc
    from spm_rating import rating
    from spm_rating.combine_rc_ln import compute_total_notes
    from spm_rating.aggregate_sigmoid import compute_SR_sigmoid

    params, weights, postprocess = spm_calc.load_params()
    params["use_sigmoid_aggregation"] = 1
    print(f"\n[{label}] FEATURE_NAMES={spm_calc.FEATURE_NAMES}")
    print(f"[{label}] {len(weights)} weights")

    recs = []
    t0 = time.time()
    for c, e in zip(caches, entries):
        sr_base, d = rating.combine(c, params=params)
        D_full = d["D_all"]; C_arr = d["C_arr"]
        cal_a = params.get("calib_a", 0.893); cal_b = params.get("calib_b", 0.031)
        D_calib = cal_a * D_full + cal_b
        feats = spm_calc.compute_features(c)
        correction = sum(weights.get(fn, 0.0) * feats.get(fn, 0.0) for fn in spm_calc.FEATURE_NAMES)
        D_new = np.maximum(D_calib + correction, 0.01)
        tn = compute_total_notes(c["note_seq"], c["LN_seq"])
        sr, _ = compute_SR_sigmoid(
            c["all_corners"], C_arr, D_new, tn, c["LN_seq"],
            sigmoid_k=params.get("agg_sigmoid_k", 2.09),
            sigmoid_C=params.get("agg_sigmoid_C", 3.969),
            sigmoid_gamma=params.get("agg_sigmoid_ref_gamma", 0.196),
            note_norm_N0=postprocess["N0"],
            rescale_threshold=postprocess["threshold"],
            rescale_divisor=postprocess["divisor"],
            global_scale=postprocess["scale"],
        )
        recs.append({"sr": float(sr), "sr_ref": e["sr_ref"], "sr_error": e["sr_error"],
                     "sort": e["sort"], "source": e["source"], "mapfile": e["mapfile"],
                     "loss": score_single(sr, e["sr_ref"], e["sr_error"])})
    print(f"  evaluated {len(recs)} in {time.time()-t0:.0f}s")

    # Remove repo from path to avoid contamination
    sys.path.remove(repo_path)
    for mod in list(sys.modules):
        if mod.startswith("spm_calc") or mod.startswith("spm_rating"):
            del sys.modules[mod]
    return recs


def agg(recs, label):
    preds = np.array([r["sr"] for r in recs]); refs = np.array([r["sr_ref"] for r in recs])
    errs = np.array([r["sr_error"] for r in recs]); losses = np.array([r["loss"] for r in recs])
    deltas = np.abs(preds - refs)
    print(f"  {label:<28} n={len(recs):3d} loss={np.mean(losses):.4f} MAE={np.mean(deltas):.4f} "
          f"pass={np.mean(deltas<=errs):.3f} bias={np.mean(preds-refs):+.3f}")


TIER_ORDER = ["0th","1st","2nd","3rd","4th","5th","6th","7th","8th","9th","10th",
              "Azimuth","Gamma","Stellium","Zenith"]


def get_tier(name):
    m = re.search(r"~\s*(\w+)\s*~", name)
    return m.group(1) if m else None


def speed_table(recs, label):
    print(f"\n  Speed Practice 0th->Stellium ({label}):")
    print(f"    {'tier':<10} {'sr_ref':>6} {'pred':>6} {'err':>6} {'within':>6}")
    by_mf = {r["mapfile"]: i for i, r in enumerate(recs)}
    for tier in TIER_ORDER:
        for mf, idx in by_mf.items():
            if "Speed Practice" in mf and get_tier(mf) == tier:
                rec = recs[idx]
                sr = rec["sr"]; ref = rec["sr_ref"]; err = rec["sr_error"]
                within = "Y" if abs(sr-ref) <= err else "n"
                print(f"    {tier:<10} {ref:>6.2f} {sr:>6.2f} {abs(sr-ref):>6.2f} {within:>6}")
                break


def main():
    entries = load_playtest_data(maps_root=PROJROOT)
    caches = []
    used = []
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
    print(f"loaded {len(caches)} caches")

    recs_v1 = eval_repo(MAIN, used, caches, "v1 (SPMRating-main)")
    recs_z = eval_repo(ZREL, used, caches, "v0.4.0 (SPMRating-Z-Release)")

    print(f"\n{'='*72}")
    print("OVERALL + PER-SORT")
    print(f"{'='*72}")
    agg(recs_v1, "v1 overall")
    agg(recs_z, "v0.4.0 overall")
    for s in ["rc", "ln", "hb", "mix"]:
        agg([r for r in recs_v1 if r["sort"]==s], f"v1  {s}")
        agg([r for r in recs_z if r["sort"]==s], f"Z   {s}")

    # Paired test
    from scipy import stats
    lv = np.array([r["loss"] for r in recs_v1])
    lz = np.array([r["loss"] for r in recs_z])
    diffs = lv - lz
    t_stat, p_val = stats.ttest_rel(lv, lz)
    try:
        _, w_p = stats.wilcoxon(lv, lz)
    except Exception:
        w_p = float("nan")
    rng = np.random.RandomState(42)
    boot = [np.mean(diffs[rng.choice(len(diffs), len(diffs), replace=True)]) for _ in range(10000)]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n  Paired test (v1 vs v0.4.0):")
    print(f"    mean improvement: {np.mean(diffs):+.4f}")
    print(f"    improved: {np.sum(diffs>0)}/{len(diffs)} ({100*np.sum(diffs>0)/len(diffs):.1f}%)")
    print(f"    paired t p={p_val:.4f}  Wilcoxon p={w_p:.4f}")
    print(f"    bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    print(f"\n{'='*72}")
    print("SPEED PRACTICE 0th -> Stellium")
    print(f"{'='*72}")
    speed_table(recs_v1, "v1")
    speed_table(recs_z, "v0.4.0")

    out = {
        "v1_loss": float(np.mean(lv)), "z_loss": float(np.mean(lz)),
        "improvement": float(np.mean(diffs)), "p_value": float(p_val),
        "ci": [float(ci_lo), float(ci_hi)],
        "speed_practice_z": {get_tier(r["mapfile"]): r["sr"] for r in recs_z if "Speed Practice" in r["mapfile"]},
    }
    with open(os.path.join(ZREL, "release_verification.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to release_verification.json")


if __name__ == "__main__":
    main()
