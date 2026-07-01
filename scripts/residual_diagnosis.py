"""Residual diagnosis for v0.4.0 (9-feature model).

Two diagnostic passes:
  PASS 1 — Simple residual diagnosis:
    - residual = pred - ref, per-map loss
    - correlation of residual/loss with attributes (D_solved, features, shape, duration, NPS)
    - per-sort / per-source / per-tag stratified residual means
    - top-30 worst maps qualitative

  PASS 2 — Structured framework:
    - Decompose error: (D_solved_base - ref) vs correction (coverage check)
    - Conditional/partial information: does attribute A explain residual
      BEYOND the 9 features? (partial R²)
    - Stratified by sort/tag (global R² may be ~0 but RC-internal significant)
    - Error budget: R²(D_solved_base→ref) vs R²(+9feat→ref) vs noise ceiling
"""
import sys, os, json, pickle, re, time, glob
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ZROOT = os.path.dirname(HERE)
PROJROOT = os.path.dirname(ZROOT)
ZRES = os.path.join(PROJROOT, "SPMRating-Z")
# CRITICAL: release repo must come FIRST so its spm_calc/spm_rating/tuning
# shadow the Z research repo's (which has different FEATURE_NAMES and rating_z).
# The release repo's data_loader is now synced (parse_float + Ranked support).
sys.path.insert(0, ZROOT)
sys.path.insert(0, os.path.join(ZROOT, "tuning"))
os.chdir(ZROOT)

from tuning.data_loader import load_playtest_data
from tuning.scorer import score_single
from spm_calc import FEATURE_NAMES, FEATURE_PARAMS, compute_features
from spm_rating import rating
from spm_rating.combine_rc_ln import compute_total_notes
from spm_rating.aggregate_sigmoid import (
    segment_by_difficulty, solve_D_bisection, _compute_effective_weights, compute_SR_sigmoid
)


def cache_path_for(entry):
    base = os.path.basename(entry["mapfile"]).replace(".osu", "")
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in base)
    return os.path.join(ZRES, "cache", safe + ".pkl")


def read_bpm(osu_path):
    try:
        in_tp = False
        with open(osu_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if line.startswith("["):
                    in_tp = (line.strip() == "[TimingPoints]")
                    continue
                if in_tp:
                    parts = line.split(",")
                    if len(parts) >= 2 and float(parts[1]) > 0:
                        return 60000.0 / float(parts[1])
    except Exception:
        pass
    return None


def main():
    with open(os.path.join(ZROOT, "tuned_params_sigmoid.json"), encoding="utf-8") as f:
        TUNED_PARAMS = json.load(f)["params"]
    TUNED_PARAMS["use_sigmoid_aggregation"] = 1
    with open(os.path.join(ZROOT, "tuned_correction.json"), encoding="utf-8") as f:
        _tc = json.load(f)
    WEIGHTS = _tc["correction_weights"]
    POST = _tc["postprocess"]

    entries = load_playtest_data(maps_root=PROJROOT)
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

    # Precompute everything per map
    recs = []
    t0 = time.time()
    for c, e in zip(caches, used):
        p = dict(TUNED_PARAMS)
        sr_base, d = rating.combine(c, params=p)
        D_full = d["D_all"]
        C_arr = d["C_arr"]
        cal_a = p.get("calib_a", 0.893); cal_b = p.get("calib_b", 0.031)
        D_calib = cal_a * D_full + cal_b
        feats = compute_features(c)
        correction = sum(WEIGHTS.get(fn, 0.0) * feats.get(fn, 0.0) for fn in FEATURE_NAMES)
        D_new = np.maximum(D_calib + correction, 0.01)
        tn = compute_total_notes(c["note_seq"], c["LN_seq"])
        sr, _ = compute_SR_sigmoid(
            c["all_corners"], C_arr, D_new, tn, c["LN_seq"],
            sigmoid_k=p.get("agg_sigmoid_k", 2.09),
            sigmoid_C=p.get("agg_sigmoid_C", 3.969),
            sigmoid_gamma=p.get("agg_sigmoid_ref_gamma", 0.196),
            note_norm_N0=POST["N0"], rescale_threshold=POST["threshold"],
            rescale_divisor=POST["divisor"], global_scale=POST["scale"],
        )
        # D_solved_base (pre-correction, pre-postprocess): re-solve without correction
        # to isolate D formula's raw output
        eff_w = _compute_effective_weights(c["all_corners"], C_arr)
        D_seg, w_seg = segment_by_difficulty(D_calib, eff_w, 30)
        D_solved_base, _ = solve_D_bisection(
            D_seg, w_seg, k=p["agg_sigmoid_k"], C=p["agg_sigmoid_C"],
            gamma=p["agg_sigmoid_ref_gamma"], delta=5.0, tol=0.0001)
        # D(t) shape metrics (on D_calib, the calibrated instantaneous difficulty)
        Dp = D_calib[D_calib > 0.01]
        times = np.array([n[1] for n in c["note_seq"]], dtype=np.float64)
        duration_s = max((times[-1] - times[0]) / 1000.0, 1.0)
        rec = {
            "sr": float(sr), "sr_ref": e["sr_ref"], "sr_error": e["sr_error"],
            "sort": e["sort"], "source": e["source"], "tags": e.get("tags", ""),
            "mapfile": e["mapfile"],
            "loss": score_single(sr, e["sr_ref"], e["sr_error"]),
            "residual": float(sr - e["sr_ref"]),
            "abs_resid": abs(float(sr - e["sr_ref"])),
            "D_solved_base": float(D_solved_base),
            "correction": float(correction),
            # features
            **{fn: float(feats.get(fn, 0.0)) for fn in FEATURE_NAMES},
            # D(t) shape
            "D_mean": float(np.mean(Dp)) if len(Dp) else 0.0,
            "D_std": float(np.std(Dp)) if len(Dp) else 0.0,
            "D_max": float(np.max(Dp)) if len(Dp) else 0.0,
            "D_cv": float(np.std(Dp) / max(np.mean(Dp), 0.01)) if len(Dp) else 0.0,
            "D_p70": float(np.percentile(Dp, 70)) if len(Dp) else 0.0,
            "D_p90": float(np.percentile(Dp, 90)) if len(Dp) else 0.0,
            "D_skew": float(stats.skew(Dp)) if len(Dp) > 2 else 0.0,
            "D_peak_ratio": float(np.max(Dp) / max(np.mean(Dp), 0.01)) if len(Dp) else 0.0,
            # chart meta
            "duration_s": float(duration_s),
            "n_notes": int(len(c["note_seq"])),
            "n_LN": int(len(c["LN_seq"])),
            "avg_nps": float(len(c["note_seq"]) / duration_s),
            "ln_ratio": float(c.get("ln_ratio", 0.0)),
            "bpm": read_bpm(e["osu_path"]) or 0.0,
        }
        recs.append(rec)
    print(f"precompute done in {time.time()-t0:.0f}s", flush=True)

    # ==================== PASS 1: SIMPLE RESIDUAL ====================
    print(f"\n{'='*72}")
    print("PASS 1: SIMPLE RESIDUAL DIAGNOSIS")
    print(f"{'='*72}")

    resids = np.array([r["residual"] for r in recs])
    losses = np.array([r["loss"] for r in recs])
    abs_resids = np.array([r["abs_resid"] for r in recs])

    print(f"\noverall: loss={np.mean(losses):.4f} MAE={np.mean(abs_resids):.4f} "
          f"bias={np.mean(resids):+.4f} resid_std={np.std(resids):.4f}")

    # 1a. Correlation of residual/loss with attributes
    attrs = ["D_solved_base","correction","D_mean","D_std","D_max","D_cv","D_p70","D_p90",
             "D_skew","D_peak_ratio","duration_s","n_notes","avg_nps","ln_ratio","bpm"] + FEATURE_NAMES
    print(f"\n  Correlation of |residual| with attributes:")
    print(f"  {'attr':<18} {'r(resid)':>9} {'r(|resid|)':>10} {'r(loss)':>8}")
    rows = []
    for a in attrs:
        vals = np.array([r[a] for r in recs])
        if np.std(vals) < 1e-9:
            continue
        r_resid = np.corrcoef(vals, resids)[0,1]
        r_abs = np.corrcoef(vals, abs_resids)[0,1]
        r_loss = np.corrcoef(vals, losses)[0,1]
        rows.append((a, r_resid, r_abs, r_loss))
    rows.sort(key=lambda x: -abs(x[3]))
    for a, rr, ra, rl in rows:
        print(f"  {a:<18} {rr:>+9.3f} {ra:>+10.3f} {rl:>+8.3f}")

    # 1b. Stratified by sort
    print(f"\n  By sort:")
    print(f"  {'sort':<6} {'n':>4} {'loss':>7} {'MAE':>7} {'bias':>7} {'resid_std':>9}")
    for s in ["rc","ln","hb","mix"]:
        idx = [i for i,r in enumerate(recs) if r["sort"]==s]
        if not idx: continue
        print(f"  {s:<6} {len(idx):>4} {np.mean(losses[idx]):>7.4f} {np.mean(abs_resids[idx]):>7.4f} "
              f"{np.mean(resids[idx]):>+7.3f} {np.std(resids[idx]):>9.3f}")

    # 1c. Stratified by source
    print(f"\n  By source:")
    for s in ["dan","tournament","ranked","graveyard"]:
        idx = [i for i,r in enumerate(recs) if r["source"]==s]
        if not idx: continue
        print(f"  {s:<12} {len(idx):>4} loss={np.mean(losses[idx]):.4f} bias={np.mean(resids[idx]):+.3f}")

    # 1d. Stratified by tag
    print(f"\n  By tag (top tags):")
    print(f"  {'tag':<22} {'n':>4} {'loss':>7} {'bias':>7}")
    tag_recs = {}
    for r in recs:
        for t in (r["tags"] or "").split(","):
            t = t.strip()
            if t:
                tag_recs.setdefault(t, []).append(r)
    tag_rows = []
    for t, lst in tag_recs.items():
        if len(lst) >= 8:
            tl = np.mean([x["loss"] for x in lst])
            tb = np.mean([x["residual"] for x in lst])
            tag_rows.append((t, len(lst), tl, tb))
    tag_rows.sort(key=lambda x: -x[2])
    for t, n, tl, tb in tag_rows:
        print(f"  {t:<22} {n:>4} {tl:>7.4f} {tb:>+7.3f}")

    # 1e. Top-20 worst maps
    print(f"\n  Top-20 worst maps (by loss):")
    print(f"  {'mapfile':<48} {'sort':>4} {'ref':>5} {'pred':>5} {'err':>5} {'loss':>6} {'tags'}")
    order = sorted(range(len(recs)), key=lambda i: -recs[i]["loss"])
    for i in order[:20]:
        r = recs[i]
        print(f"  {r['mapfile'][:48]:<48} {r['sort']:>4} {r['sr_ref']:>5.2f} {r['sr']:>5.2f} "
              f"{r['sr_error']:>5.2f} {r['loss']:>6.2f} {r['tags'][:25]}")

    # ==================== PASS 2: STRUCTURED FRAMEWORK ====================
    print(f"\n{'='*72}")
    print("PASS 2: STRUCTURED FRAMEWORK")
    print(f"{'='*72}")

    # 2a. Error decomposition: does correction match (D_solved_base - ref)?
    print(f"\n  2a. Correction coverage:")
    print(f"      Does correction track D_solved_base's bias?")
    refs = np.array([r["sr_ref"] for r in recs])
    D_base = np.array([r["D_solved_base"] for r in recs])
    corr = np.array([r["correction"] for r in recs])
    # Note: sr = postprocess(D_solved_base + correction). The postprocess is nonlinear,
    # but for a first-order check: does corr oppose D_base's bias?
    base_bias = D_base - refs  # raw D formula bias (ignoring postprocess nonlinearity)
    print(f"      corr(D_base - ref, correction) = {np.corrcoef(base_bias, corr)[0,1]:+.3f}")
    print(f"      (−1 = perfect compensation, 0 = no relation, +1 = wrong direction)")
    # Ratio: how much of the bias does correction cover on average?
    print(f"      mean(D_base - ref) = {np.mean(base_bias):+.3f}")
    print(f"      mean(correction)   = {np.mean(corr):+.3f}")
    # Per sort
    print(f"\n      Per-sort correction coverage:")
    print(f"      {'sort':<6} {'mean(D_base-ref)':>16} {'mean(corr)':>11} {'ratio':>7}")
    for s in ["rc","ln","hb","mix"]:
        idx = [i for i,r in enumerate(recs) if r["sort"]==s]
        if not idx: continue
        bb = np.mean(base_bias[idx]); cc = np.mean(corr[idx])
        ratio = cc/bb if abs(bb)>0.01 else float('nan')
        print(f"      {s:<6} {bb:>+16.3f} {cc:>+11.3f} {ratio:>7.2f}")

    # 2b. Conditional/partial information: does attribute A explain residual
    #     BEYOND the 9 features? Use partial R² via OLS residuals.
    print(f"\n  2b. Conditional information (partial R² of residual ~ A | 9 features):")
    # Fit sr_ref ~ 9 features (OLS), get residual r9
    X9 = np.array([[r[fn] for fn in FEATURE_NAMES] for r in recs])
    # Add intercept
    X9i = np.hstack([np.ones((len(recs),1)), X9])
    beta, _, _, _ = np.linalg.lstsq(X9i, refs, rcond=None)
    pred9 = X9i @ beta
    resid9 = refs - pred9  # what 9 features can't explain
    ss9 = np.sum(resid9**2)
    print(f"      R²(9 features -> ref) = {1 - ss9/np.sum((refs-np.mean(refs))**2):.4f}")

    # Noise ceiling: R² upper bound from sr_error
    # If pred = ref + noise where noise ~ N(0, sr_error²), then the best possible
    # R² is 1 - Var(noise)/Var(ref). Approximate Var(noise) = mean(sr_error²).
    serr = np.array([r["sr_error"] for r in recs])
    noise_var = np.mean(serr**2)
    ref_var = np.var(refs)
    r2_ceiling = max(0, 1 - noise_var/ref_var)
    print(f"      R² noise ceiling (from sr_error) = {r2_ceiling:.4f}")
    print(f"      gap (ceiling - 9feat) = {r2_ceiling - (1 - ss9/np.sum((refs-np.mean(refs))**2)):.4f}")

    # For each candidate attribute A: partial R² = 1 - SS(resid9 ~ A)/SS(resid9)
    candidates = ["D_solved_base","D_mean","D_std","D_cv","D_max","D_p70","D_p90","D_skew",
                  "D_peak_ratio","duration_s","n_notes","avg_nps","ln_ratio","bpm"]
    print(f"\n      {'attr':<18} {'partialR²':>9} {'p(val)':>8}   (global)")
    cand_rows = []
    for a in candidates:
        vals = np.array([r[a] for r in recs])
        if np.std(vals) < 1e-9: continue
        # regress resid9 on vals
        Xa = np.hstack([np.ones((len(recs),1)), vals.reshape(-1,1)])
        ba, _, _, _ = np.linalg.lstsq(Xa, resid9, rcond=None)
        pa = Xa @ ba
        ss_a = np.sum((resid9 - pa)**2)
        pr2 = 1 - ss_a/ss9
        # t-test on slope
        n = len(recs); k = 2
        residuals = resid9 - pa
        sse = np.sum(residuals**2)
        if sse > 0 and np.var(vals) > 0:
            se = np.sqrt(sse/(n-k)) / (np.sqrt(np.sum((vals-vals.mean())**2)))
            t_stat = ba[1]/se if se > 0 else 0
            p_val = 2*(1-stats.t.cdf(abs(t_stat), n-k))
        else:
            p_val = 1.0
        cand_rows.append((a, pr2, p_val))
    cand_rows.sort(key=lambda x: -x[1])
    for a, pr2, pv in cand_rows:
        sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
        print(f"      {a:<18} {pr2:>9.4f} {pv:>8.4f} {sig}")

    # 2c. Stratified partial R² (within RC sort only — where the loss is)
    print(f"\n      Stratified within RC sort (n={sum(1 for r in recs if r['sort']=='rc')}):")
    rc_idx = [i for i,r in enumerate(recs) if r["sort"]=="rc"]
    X9rc = X9i[rc_idx]; refsrc = refs[rc_idx]
    beta_rc, _, _, _ = np.linalg.lstsq(X9rc, refsrc, rcond=None)
    pred9rc = X9rc @ beta_rc
    resid9rc = refsrc - pred9rc
    ss9rc = np.sum(resid9rc**2)
    print(f"      R²(9feat -> ref) within RC = {1 - ss9rc/np.sum((refsrc-refsrc.mean())**2):.4f}")
    print(f"      {'attr':<18} {'partialR²':>9} {'p(val)':>8}   (RC only)")
    rc_rows = []
    for a in candidates:
        vals = np.array([r[a] for r in recs])[rc_idx]
        if np.std(vals) < 1e-9: continue
        Xa = np.hstack([np.ones((len(rc_idx),1)), vals.reshape(-1,1)])
        ba, _, _, _ = np.linalg.lstsq(Xa, resid9rc, rcond=None)
        pa = Xa @ ba
        ss_a = np.sum((resid9rc - pa)**2)
        pr2 = 1 - ss_a/ss9rc
        n=len(rc_idx); k=2
        residuals = resid9rc - pa; sse = np.sum(residuals**2)
        if sse>0 and np.var(vals)>0:
            se = np.sqrt(sse/(n-k))/np.sqrt(np.sum((vals-vals.mean())**2))
            t_stat = ba[1]/se if se>0 else 0
            pv = 2*(1-stats.t.cdf(abs(t_stat), n-k))
        else: pv = 1.0
        rc_rows.append((a, pr2, pv))
    rc_rows.sort(key=lambda x: -x[1])
    for a, pr2, pv in rc_rows:
        sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
        print(f"      {a:<18} {pr2:>9.4f} {pv:>8.4f} {sig}")

    # 2d. Stratified by tag (within each major tag, partial R² of top attrs)
    print(f"\n      Stratified by tag (partial R² of D_solved_base, within tag):")
    for t in ["speed","chordjack","tech","dense chordstream","inverse","release"]:
        tidx = [i for i,r in enumerate(recs) if t in (r["tags"] or "")]
        if len(tidx) < 10: continue
        X9t = X9i[tidx]; refst = refs[tidx]
        if len(tidx) < 12: continue
        bt, _, _, _ = np.linalg.lstsq(X9t, refst, rcond=None)
        pt = X9t @ bt; rt = refst - pt; sst = np.sum(rt**2)
        if sst < 1e-6: continue
        vals = np.array([r["D_solved_base"] for r in recs])[tidx]
        Xa = np.hstack([np.ones((len(tidx),1)), vals.reshape(-1,1)])
        ba, _, _, _ = np.linalg.lstsq(Xa, rt, rcond=None)
        pa = Xa @ ba
        pr2 = 1 - np.sum((rt-pa)**2)/sst
        tl = np.mean([recs[i]["loss"] for i in tidx])
        print(f"      {t:<22} n={len(tidx):>3} loss={tl:.3f} partialR²(D_base)={pr2:+.4f}")

    # Save
    out = {
        "overall_loss": float(np.mean(losses)),
        "r2_9feat": float(1 - ss9/np.sum((refs-np.mean(refs))**2)),
        "r2_ceiling": float(r2_ceiling),
        "worst_maps": [{"mapfile": recs[i]["mapfile"], "loss": float(recs[i]["loss"]),
                        "sr_ref": recs[i]["sr_ref"], "sr": recs[i]["sr"],
                        "sort": recs[i]["sort"], "tags": recs[i]["tags"]}
                       for i in order[:20]],
    }
    with open(os.path.join(ZROOT, "residual_diagnosis.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary to residual_diagnosis.json")


if __name__ == "__main__":
    main()
