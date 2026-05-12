"""
Train a multi-label tag classifier for the tosu plugin.

Uses labeled data from Excel (tags column) to build per-tag decision trees
that can be ported to pure JavaScript. Replaces the old rule-based
computePatternTags() with ML-based prediction.

Features:
- Skill fractions (stream, jack, tech, chordjack, release)
  — aggregated from SR component arrays, matching JS computeSkillRatings()
- Structural features (chord overlap, NPS, ln_ratio, avg_ln_dur)
- Component distribution stats (percentiles of Pbar, Jbar, Xbar, Rbar, etc.)
- Derived SR ratios (rc_sr_ratio, ln_contrib)

Asymmetric loss: FN (missing tag) penalized more than FP (extra tag).

Output: tosustatic/spm-ratingV2pro-sigmoid/tag_classifier.json
"""
import sys, os, json, re, pickle, time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tuning.data_loader import load_playtest_data
from spm_rating.rating import precompute, combine as combine_total

# ============================================================
# JS-EQUIVALENT SKILL AGGREGATION
# ============================================================
DECAY_WEIGHT = 0.88
RATING_MULTIPLIER = 0.090


def skill_aggregate(arr):
    """Exactly match JS computeSkillRatings.aggregate()."""
    if arr is None or len(arr) == 0:
        return 0.0
    arr = np.asarray(arr, dtype=float)
    positive = arr[arr > 0]
    n = len(arr)
    avg_val = float(np.sum(positive) / max(n, 1))
    peaks = np.sort(positive)[::-1][:20]
    weighted = 0.0
    w = 1.0
    for v in peaks:
        weighted += float(v) * w
        w *= DECAY_WEIGHT
    return weighted * RATING_MULTIPLIER * 0.5 + avg_val * 0.1


def compute_chord_overlap(note_seq, ln_seq):
    """Compute avg chord overlap (Jaccard), matching JS computePatternTags.
    JS noteSeq includes BOTH taps and LN heads, so we combine them here.
    note_seq format: (col, time, end_time_or_-1) — 3-tuple."""
    # Combine tap notes and LN heads, sort by time
    all_notes = [(n[0], n[1]) for n in note_seq]  # taps
    if ln_seq:
        all_notes += [(ln[0], ln[1]) for ln in ln_seq]  # LN heads
    all_notes.sort(key=lambda x: x[1])

    if len(all_notes) == 0:
        return 0.0

    # Group notes into chords (< 30ms apart)
    chords = []
    current_cols = set()
    current_time = None
    for col, t in all_notes:
        if current_time is None or t - current_time > 30:
            if current_cols:
                chords.append(current_cols)
            current_cols = {col}
            current_time = t
        else:
            current_cols.add(col)
    if current_cols:
        chords.append(current_cols)

    if len(chords) < 2:
        return 0.0

    overlaps = []
    for i in range(1, len(chords)):
        intersection = len(chords[i - 1] & chords[i])
        union = len(chords[i - 1] | chords[i])
        if union > 0:
            overlaps.append(intersection / union)
    return float(np.mean(overlaps)) if overlaps else 0.0


# ============================================================
# FEATURE EXTRACTION
# ============================================================
def extract_features(entry, cache, total_params):
    """Extract tag-classification features from a cached map.
    entry can be None when using precomputed cache (all info in cache)."""
    """Extract tag-classification features from a cached map."""
    # Combine to get intermediate values
    try:
        sr_total, details = combine_total(cache, total_params)
    except Exception:
        sr_total = 5.0
        details = {}

    # === Basic counts ===
    note_seq = cache["note_seq"]
    ln_seq = cache["LN_seq"]
    n_total = len(note_seq)
    n_ln = len(ln_seq)
    ln_ratio = n_ln / max(n_total, 1)

    # === LN duration ===
    if n_ln > 0:
        avg_ln_dur = float(np.mean([max(ln[2] - ln[1], 0) for ln in ln_seq]))
    else:
        avg_ln_dur = 0.0

    # === NPS (using all notes: taps + LN heads) ===
    # note_seq: (col, time, -1), LN_seq: (col, start, end)
    all_times = [n[1] for n in note_seq] + [ln[1] for ln in ln_seq]
    all_times.sort()
    n_all = len(all_times)
    if n_all > 1:
        duration_s = (all_times[-1] - all_times[0]) / 1000.0
        nps = n_all / max(duration_s, 1.0)
    else:
        nps = 0.0

    # === Chord overlap (all notes) ===
    avg_chord_overlap = compute_chord_overlap(note_seq, ln_seq)

    # === Skill fractions (exact JS equivalent) ===
    Jbar = details.get("Jbar", np.zeros(1))
    Xbar = details.get("Xbar", np.zeros(1))
    Pbar = details.get("Pbar", np.zeros(1))
    Rbar = details.get("Rbar", np.zeros(1))

    stream_val = skill_aggregate(Pbar)
    jack_val = skill_aggregate(Jbar)
    tech_val = skill_aggregate(Xbar)
    chordjack_arr = np.asarray(Jbar) * (1 - np.exp(-np.asarray(Pbar) / 5.0))
    chordjack_val = skill_aggregate(chordjack_arr)
    release_val = skill_aggregate(Rbar)

    total_skill = stream_val + jack_val + tech_val + chordjack_val + release_val
    if total_skill > 0:
        stream_frac = stream_val / total_skill
        jack_frac = jack_val / total_skill
        tech_frac = tech_val / total_skill
        chordjack_frac = chordjack_val / total_skill
        release_frac = release_val / total_skill
    else:
        stream_frac = jack_frac = tech_frac = chordjack_frac = release_frac = 0.0

    # === Component distribution stats ===
    def dist_stats(arr):
        if arr is None or len(arr) == 0:
            return {"mean": 0, "std": 0, "p50": 0, "p75": 0, "p90": 0, "max": 0}
        arr = np.asarray(arr)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "max": float(np.max(arr)),
        }

    pbar_stats = dist_stats(Pbar)
    jbar_stats = dist_stats(Jbar)
    xbar_stats = dist_stats(Xbar)
    rbar_stats = dist_stats(Rbar)

    # === Derived SR ratios ===
    rc_sr_ratio = 1.0
    ln_contrib = 0.0
    if details:
        # We don't compute RC SR here (expensive), approximate from components
        # Use release_frac and ln_ratio as proxy for LN contribution
        ln_contrib = release_frac  # simplified
        rc_sr_ratio = 1.0 - ln_contrib

    # === Build feature dict ===
    features = {
        # Core skill fractions (from JS computeSkillRatings)
        "stream_frac": stream_frac,
        "jack_frac": jack_frac,
        "tech_frac": tech_frac,
        "chordjack_frac": chordjack_frac,
        "release_frac": release_frac,
        # Structural
        "avg_chord_overlap": avg_chord_overlap,
        "nps": nps,
        "ln_ratio": ln_ratio,
        "avg_ln_dur": avg_ln_dur,
        # Pbar distribution
        "pbar_mean": pbar_stats["mean"],
        "pbar_std": pbar_stats["std"],
        "pbar_p50": pbar_stats["p50"],
        "pbar_p90": pbar_stats["p90"],
        "pbar_max": pbar_stats["max"],
        # Jbar distribution
        "jbar_mean": jbar_stats["mean"],
        "jbar_std": jbar_stats["std"],
        "jbar_p90": jbar_stats["p90"],
        "jbar_max": jbar_stats["max"],
        # Xbar distribution
        "xbar_mean": xbar_stats["mean"],
        "xbar_std": xbar_stats["std"],
        "xbar_p90": xbar_stats["p90"],
        # Rbar distribution
        "rbar_mean": rbar_stats["mean"],
        "rbar_std": rbar_stats["std"],
        "rbar_p90": rbar_stats["p90"],
        # Derived
        "rc_sr_ratio": rc_sr_ratio,
        "ln_contrib": ln_contrib,
    }

    return features


# ============================================================
# LABEL PARSING
# ============================================================
# Mapping from Excel-lowercase tags to display tags (matching JS naming convention)
TAG_MAP = {
    # RC individual tags
    "speed": "Speed",
    "dense chordstream": "Dense Chordstream",
    "tech": "Tech",
    "chordjack": "Chordjack",
    "fast chordstream": "Fast Chordstream",
    "minijack": "Minijack",
    "vibro": "Vibro",
    # LN individual tags
    "release": "Release",
    "coordination": "Coordination",
    "density": "Density",
    "inverse": "Inverse",
    "technical": "Technical",
}
# Synthesis tags — NOT trained; generated from ≥3 individual tags
SYNTHESIS_TAGS = {"RC Mix", "LN Mix", "Hybrid"}
# Deprecated/removed tags — ignored during parsing
DEPRECATED_TAGS = {"anchor", "chordstream"}


def parse_tags(tags_str):
    """Parse comma-separated tags string into list of normalized tag names.
    Filters out synthesis tags (RC Mix, LN Mix, Hybrid) and deprecated tags."""
    if not tags_str or not tags_str.strip():
        return []
    tags = [t.strip().lower() for t in tags_str.split(",") if t.strip()]
    result = []
    for t in tags:
        mapped = TAG_MAP.get(t)
        if mapped is not None:
            result.append(mapped)
        # t not in TAG_MAP → synthesis or deprecated → skip silently
    return result


# ============================================================
# DECISION TREE → JS EXPORT
# ============================================================
def tree_to_js(root, feature_ids, indent=0):
    """Recursively convert sklearn tree node to JS if-else code."""
    prefix = "  " * indent
    if root["is_leaf"]:
        return f"{prefix}return {root['prob']:.6f};\n"
    feat = feature_ids[root["feature"]]
    thresh = root["threshold"]
    return (
        f"{prefix}if ({feat} <= {thresh:.6f}) {{\n"
        f"{tree_to_js(root['left'], feature_ids, indent + 1)}"
        f"{prefix}}} else {{\n"
        f"{tree_to_js(root['right'], feature_ids, indent + 1)}"
        f"{prefix}}}\n"
    )


def extract_tree_root(tree, node_id=0):
    """Extract sklearn tree node as nested dict."""
    if tree.children_left[node_id] == -1:
        counts = tree.value[node_id][0]
        total = counts.sum()
        prob_positive = counts[1] / total if total > 0 else 0.0
        return {"is_leaf": True, "prob": float(prob_positive), "counts": counts.tolist()}
    return {
        "is_leaf": False,
        "feature": int(tree.feature[node_id]),
        "threshold": float(tree.threshold[node_id]),
        "left": extract_tree_root(tree, tree.children_left[node_id]),
        "right": extract_tree_root(tree, tree.children_right[node_id]),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    # Load params
    sigmoid_params_path = os.path.join(ROOT, "tuned_params_sigmoid.json")
    if os.path.exists(sigmoid_params_path):
        with open(sigmoid_params_path) as f:
            total_params = dict(json.load(f)["params"])
    else:
        print("WARNING: tuned_params_sigmoid.json not found, using defaults")
        total_params = {}

    # Load entries
    entries = load_playtest_data()
    print(f"Loaded {len(entries)} entries")

    # Filter entries with tags
    tagged = [(i, e) for i, e in enumerate(entries) if e.get("tags") and e["tags"].strip()]
    print(f"Entries with tags: {len(tagged)}")

    # Collect all unique tags
    all_tags_set = set()
    tag_lists = []
    for _, e in tagged:
        tags = parse_tags(e["tags"])
        tag_lists.append(tags)
        all_tags_set.update(tags)
    all_tags = sorted(all_tags_set)
    print(f"Unique tags ({len(all_tags)}): {all_tags}")

    # Count synthesis-only maps (excluded from individual tag training)
    synthesis_only = sum(1 for tl in tag_lists if len(tl) == 0)
    print(f"Synthesis-only maps (excluded from training): {synthesis_only}")

    # Filter tags with too few samples (< 4)
    tag_counts = {t: sum(1 for tl in tag_lists if t in tl) for t in all_tags}
    active_tags = [t for t in all_tags if tag_counts[t] >= 4]
    skipped_tags = [t for t in all_tags if tag_counts[t] < 4]
    if skipped_tags:
        print(f"Skipping rare tags (< 4 samples): {skipped_tags}")

    print(f"\nTraining tags ({len(active_tags)}): {active_tags}")
    for t in active_tags:
        print(f"  {t:>25s}: {tag_counts[t]:>3d} samples")

    # ============================================================
    # Extract features (with caching)
    # ============================================================
    cache_dir = os.path.join(ROOT, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    features_cache_path = os.path.join(cache_dir, "tag_classifier_features.pkl")

    if os.path.exists(features_cache_path):
        print(f"\nLoading cached features from {features_cache_path}...")
        with open(features_cache_path, "rb") as f:
            data = pickle.load(f)
        features_list = data["features"]
        tag_labels = data["tag_labels"]
        entries_used = data["entries_used"]
        print(f"  Loaded {len(features_list)} samples")
    else:
        # Load the enhanced precomputed cache
        enhanced_cache_path = os.path.join(cache_dir, "precomputed_enhanced.pkl")
        if not os.path.exists(enhanced_cache_path):
            print(f"ERROR: Enhanced cache not found at {enhanced_cache_path}")
            print("Run rebuild_enhanced_cache.py first")
            return

        with open(enhanced_cache_path, "rb") as f:
            enhanced_data = pickle.load(f)
        caches = enhanced_data["caches"]
        print(f"Loaded enhanced cache with {len(caches)} entries")

        total_params["use_sigmoid_aggregation"] = 1

        print(f"\nExtracting features for {len(tagged)} maps...")
        t0 = time.time()
        features_list = []
        tag_labels = []
        entries_used = []
        errors = 0

        for idx, (i, entry) in enumerate(tagged):
            mapfile = entry["mapfile"]
            try:
                if mapfile not in caches:
                    errors += 1
                    continue
                cache = caches[mapfile]
                feat = extract_features(None, cache, total_params)
                features_list.append(feat)
                tag_labels.append(parse_tags(entry["tags"]))
                entries_used.append(mapfile)
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  ERROR [{mapfile[:60]}]: {e}")
            if (idx + 1) % 30 == 0:
                print(f"  {idx + 1}/{len(tagged)}... ({time.time() - t0:.0f}s)")

        print(f"  Done in {time.time() - t0:.0f}s. {len(features_list)} valid, {errors} errors.")

        with open(features_cache_path, "wb") as f:
            pickle.dump({
                "features": features_list, "tag_labels": tag_labels,
                "entries_used": entries_used
            }, f)
        print(f"  Cached to {features_cache_path}")

    # ============================================================
    # Prepare feature matrix
    # ============================================================
    # Filter out synthesis-only maps (empty tag list — no individual tag signal)
    n_before_filter = len(tag_labels)
    valid_indices = [i for i, tl in enumerate(tag_labels) if len(tl) > 0]
    features_list = [features_list[i] for i in valid_indices]
    tag_labels = [tag_labels[i] for i in valid_indices]
    print(f"\nFiltered to {len(valid_indices)} maps with individual tags "
          f"(excluded {n_before_filter - len(valid_indices)} synthesis-only)")

    feature_names = sorted(features_list[0].keys())
    X = np.array([[f[n] for n in feature_names] for f in features_list])
    n_samples, n_features = X.shape
    print(f"\nFeature matrix: {n_samples} x {n_features}")
    print(f"Feature names: {feature_names}")

    # Build binary label matrix
    y = np.zeros((n_samples, len(active_tags)), dtype=int)
    for i, tl in enumerate(tag_labels):
        for j, tag in enumerate(active_tags):
            if tag in tl:
                y[i, j] = 1

    print(f"\nLabel distribution:")
    for j, tag in enumerate(active_tags):
        print(f"  {tag:>25s}: pos={y[:, j].sum():>3d}, neg={n_samples - y[:, j].sum():>3d}")

    # ============================================================
    # Train per-tag decision trees with asymmetric loss
    # ============================================================
    print(f"\n{'='*60}")
    print("Training per-tag decision trees (class_weight=balanced, max_depth=4)")
    print(f"{'='*60}")

    FN_WEIGHT = 3  # FN is this many times worse than FP
    trees = {}
    results = {}

    for j, tag in enumerate(active_tags):
        y_tag = y[:, j]
        n_pos = y_tag.sum()

        if n_pos < 4:
            print(f"  [{tag}] Skipped: only {n_pos} positives")
            continue

        # Train without class_weight — let decision tree learn natural boundaries
        # Asymmetry comes from threshold tuning, not training bias
        clf = DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=3, min_samples_split=6,
            random_state=42
        )
        clf.fit(X, y_tag)

        # Predict probabilities for threshold tuning
        y_prob = clf.predict_proba(X)[:, 1]

        # Find best threshold: minimize FN*weight + FP, with minimum recall
        best_thresh = 0.5
        best_score = -1e9
        best_metrics = None

        for thresh in np.arange(0.28, 0.65, 0.02):
            y_pred = (y_prob >= thresh).astype(int)
            fn = np.sum((y_tag == 1) & (y_pred == 0))
            fp = np.sum((y_tag == 0) & (y_pred == 1))
            tp = np.sum((y_tag == 1) & (y_pred == 1))
            tn = np.sum((y_tag == 0) & (y_pred == 0))

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)

            # Hard floor: precision must be >= 0.20, recall >= 0.50
            if precision < 0.20 or recall < 0.50:
                continue

            # Score: reward recall, penalize FP quadratically
            # (1-precision)² * 2 means: a few FP is OK, many FP is heavily penalized
            # This prevents over-prediction while still favoring recall > precision
            score = recall - (1 - precision) ** 2 * 2
            if score > best_score:
                best_score = score
                best_thresh = thresh
                best_metrics = {
                    "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                    "precision": float(precision), "recall": float(recall),
                }

        # Final prediction at best threshold
        y_pred_final = (y_prob >= best_thresh).astype(int)

        # Feature importance
        importances = dict(zip(feature_names, clf.feature_importances_))
        top_feats = sorted(importances.items(), key=lambda x: -x[1])[:5]

        tree_root = extract_tree_root(clf.tree_)

        trees[tag] = {
            "tree": tree_root,
            "threshold": float(best_thresh),
            "n_samples_pos": int(n_pos),
        }
        results[tag] = {
            **best_metrics,
            "f1": float(f1_score(y_tag, y_pred_final, zero_division=0)),
            "threshold": float(best_thresh),
            "top_features": [(n, round(v, 4)) for n, v in top_feats],
        }

        print(f"  [{tag:>25s}] prec={best_metrics['precision']:.3f} "
              f"rec={best_metrics['recall']:.3f} "
              f"F1={results[tag]['f1']:.3f} "
              f"thresh={best_thresh:.3f} "
              f"FN={best_metrics['fn']} FP={best_metrics['fp']} "
              f"top: {top_feats[0][0]}={top_feats[0][1]:.3f}")

    # ============================================================
    # Overall evaluation
    # ============================================================
    print(f"\n{'='*60}")
    print("Overall Evaluation")
    print(f"{'='*60}")

    # Build predictions at tuned thresholds
    y_pred_all = np.zeros_like(y)
    for j, tag in enumerate(active_tags):
        if tag in trees:
            # Re-predict using the saved tree
            clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=3, random_state=42)
            # We need to re-fit — use stored tree structure for prediction
            y_prob_j = predict_prob_from_tree(trees[tag]["tree"], X, feature_names)
            y_pred_all[:, j] = (y_prob_j >= trees[tag]["threshold"]).astype(int)

    # Per-map evaluation
    exact_matches = 0
    total_extra = 0
    total_missing = 0
    total_pred_tags = 0
    total_true_tags = 0

    for i in range(n_samples):
        true_tags_i = set(active_tags[j] for j in range(len(active_tags)) if y[i, j] == 1)
        pred_tags_i = set(active_tags[j] for j in range(len(active_tags)) if y_pred_all[i, j] == 1)
        total_true_tags += len(true_tags_i)
        total_pred_tags += len(pred_tags_i)

        if true_tags_i == pred_tags_i:
            exact_matches += 1

        extra = len(pred_tags_i - true_tags_i)
        missing = len(true_tags_i - pred_tags_i)
        total_extra += extra
        total_missing += missing

    exact_match_rate = exact_matches / n_samples
    avg_extra = total_extra / n_samples
    avg_missing = total_missing / n_samples

    print(f"Exact match rate:  {exact_matches}/{n_samples} = {exact_match_rate:.1%}")
    print(f"Avg extra tags:    {avg_extra:.2f} per map")
    print(f"Avg missing tags:  {avg_missing:.2f} per map")
    print(f"True tags/map:     {total_true_tags / n_samples:.2f}")
    print(f"Pred tags/map:     {total_pred_tags / n_samples:.2f}")

    # ============================================================
    # Per-tag detailed report
    # ============================================================
    print(f"\n{'='*60}")
    print("Per-Tag Report")
    print(f"{'='*60}")
    print(f"{'Tag':>25s} | {'Prec':>6s} {'Rec':>6s} {'F1':>6s} | {'FN':>4s} {'FP':>4s} | {'Thresh':>6s}")
    print("-" * 75)

    for tag in active_tags:
        if tag in results:
            r = results[tag]
            print(f"{tag:>25s} | {r['precision']:6.3f} {r['recall']:6.3f} {r['f1']:6.3f} | "
                  f"{r['fn']:>4d} {r['fp']:>4d} | {r['threshold']:6.3f}")

    # ============================================================
    # Output: JSON + JS
    # ============================================================
    print(f"\n{'='*60}")
    print("Exporting for tosu plugin")
    print(f"{'='*60}")

    # Generate JS function
    js_code = generate_js_classifier(trees, active_tags, feature_names, FN_WEIGHT)

    # Output JSON
    out = {
        "version": "1.0",
        "n_samples": n_samples,
        "n_tags": len(active_tags),
        "tags": active_tags,
        "feature_names": feature_names,
        "fn_weight": FN_WEIGHT,
        "exact_match_rate": float(exact_match_rate),
        "avg_extra_tags": float(avg_extra),
        "avg_missing_tags": float(avg_missing),
        "per_tag_results": {t: results[t] for t in active_tags if t in results},
        "trees": {t: trees[t] for t in active_tags if t in trees},
        "javascript": js_code,
    }

    out_path = os.path.join(ROOT, "tosustatic", "spm-ratingV2pro-sigmoid", "tag_classifier.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")

    print(f"\n=== Generated JavaScript ===")
    print(js_code[:500] + "\n... (truncated)")


# ============================================================
# TREE PROBABILITY PREDICTION (Python-side, for evaluation)
# ============================================================
def predict_prob_from_tree(tree_root, X, feature_names):
    """Predict probabilities using extracted tree structure."""
    n = X.shape[0]
    probs = np.zeros(n)
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    for i in range(n):
        node = tree_root
        while not node["is_leaf"]:
            feat_idx = name_to_idx[feature_names[node["feature"]]]
            if X[i, feat_idx] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        probs[i] = node["prob"]
    return probs


# ============================================================
# JS CODE GENERATION
# ============================================================
def generate_js_classifier(trees, active_tags, feature_names, fn_weight):
    """Generate JavaScript computePatternTagsML function."""
    js = "/**\n"
    js += f" * ML-based pattern tag classifier (auto-generated).\n"
    js += f" * {len(active_tags)} tags, {len(feature_names)} features, per-tag decision trees.\n"
    js += f" * Asymmetric loss: FN > FP (weight={fn_weight}x).\n"
    js += f" * Replaces the old rule-based computePatternTags().\n"
    js += f" */\n"

    # Feature extraction helper
    js += """
const DECAY_WEIGHT_TAG = 0.88;
const RATING_MULTIPLIER_TAG = 0.090;

function _tagAggregate(arr) {
    if (!arr || arr.length === 0) return 0;
    let sum = 0; const n = arr.length;
    for (let i = 0; i < n; i++) if (arr[i] > 0) sum += arr[i];
    const avg = sum / Math.max(n, 1);
    const peaks = [...arr].filter(v => v > 0).sort((a, b) => b - a);
    let weighted = 0, weight = 1;
    for (let i = 0; i < Math.min(peaks.length, 20); i++) {
        weighted += peaks[i] * weight;
        weight *= DECAY_WEIGHT_TAG;
    }
    return weighted * RATING_MULTIPLIER_TAG * 0.5 + avg * 0.1;
}

function _tagChordOverlap(noteSeq) {
    const chords = [];
    let currentCols = new Set(), currentTime = null;
    for (const note of noteSeq) {
        if (currentTime === null || note.start - currentTime > 30) {
            if (currentCols.size > 0) chords.push(new Set(currentCols));
            currentCols = new Set([note.col || note[0]]);
            currentTime = note.start || note[1];
        } else {
            currentCols.add(note.col || note[0]);
        }
    }
    if (currentCols.size > 0) chords.push(new Set(currentCols));
    if (chords.length < 2) return 0;
    let total = 0;
    for (let i = 1; i < chords.length; i++) {
        let intersect = 0;
        for (const col of chords[i]) if (chords[i - 1].has(col)) intersect++;
        const union = new Set([...chords[i], ...chords[i - 1]]).size;
        if (union > 0) total += intersect / union;
    }
    return total / (chords.length - 1);
}

function _tagDistStats(arr) {
    if (!arr || arr.length === 0) return { mean: 0, std: 0, p50: 0, p75: 0, p90: 0, max: 0 };
    const n = arr.length;
    const sorted = [...arr].sort((a, b) => a - b);
    let sum = 0, sumSq = 0;
    for (const v of arr) { sum += v; sumSq += v * v; }
    const mean = sum / n;
    const std = Math.sqrt(sumSq / n - mean * mean);
    const idx = (p) => Math.max(0, Math.min(n - 1, Math.round(p * (n - 1) / 100)));
    return {
        mean, std,
        p50: sorted[idx(50)],
        p75: sorted[idx(75)],
        p90: sorted[idx(90)],
        max: sorted[n - 1],
    };
}

function extractTagFeatures(Jbar, Xbar, Pbar, Rbar, noteSeq, LNSeq) {
    const nTotal = noteSeq.length;
    const nLn = LNSeq.length;
    const lnRatio = nLn / Math.max(nTotal, 1);

    let avgLnDur = 0;
    if (nLn > 0) {
        let sum = 0;
        for (const ln of LNSeq) sum += Math.max((ln.end - ln.start), 0);
        avgLnDur = sum / nLn;
    }

    let nps = 0;
    if (nTotal > 1) {
        const dur = (noteSeq[nTotal - 1].start - noteSeq[0].start) / 1000;
        nps = nTotal / Math.max(dur, 1);
    }

    const avgChordOverlap = _tagChordOverlap(noteSeq);

    // Skill fractions (exact match with computeSkillRatings)
    const streamRaw = _tagAggregate(Pbar);
    const jackRaw = _tagAggregate(Jbar);
    const techRaw = _tagAggregate(Xbar);
    const cjArr = Jbar.map((v, i) => v * (1 - Math.exp(-Pbar[i] / 5)));
    const chordjackRaw = _tagAggregate(cjArr);
    const releaseRaw = _tagAggregate(Rbar);

    const totalSkill = streamRaw + jackRaw + techRaw + chordjackRaw + releaseRaw;
    const streamFrac = totalSkill > 0 ? streamRaw / totalSkill : 0;
    const jackFrac = totalSkill > 0 ? jackRaw / totalSkill : 0;
    const techFrac = totalSkill > 0 ? techRaw / totalSkill : 0;
    const chordjackFrac = totalSkill > 0 ? chordjackRaw / totalSkill : 0;
    const releaseFrac = totalSkill > 0 ? releaseRaw / totalSkill : 0;

    // Distribution stats
    const pStats = _tagDistStats(Pbar);
    const jStats = _tagDistStats(Jbar);
    const xStats = _tagDistStats(Xbar);
    const rStats = _tagDistStats(Rbar);

    const rcSrRatio = 1.0 - releaseFrac;
    const lnContrib = releaseFrac;

    return {
        streamFrac, jackFrac, techFrac, chordjackFrac, releaseFrac,
        avgChordOverlap, nps, lnRatio, avgLnDur,
        pbarMean: pStats.mean, pbarStd: pStats.std,
        pbarP50: pStats.p50, pbarP90: pStats.p90, pbarMax: pStats.max,
        jbarMean: jStats.mean, jbarStd: jStats.std,
        jbarP90: jStats.p90, jbarMax: jStats.max,
        xbarMean: xStats.mean, xbarStd: xStats.std, xbarP90: xStats.p90,
        rbarMean: rStats.mean, rbarStd: rStats.std, rbarP90: rStats.p90,
        rcSrRatio, lnContrib,
    };
}
"""

    # Feature name mapping: index → JS expression (tree nodes use integer indices)
    js_feat_by_index = []
    for name in feature_names:
        parts = name.split("_")
        js_name = parts[0] + "".join(p.capitalize() for p in parts[1:])
        js_feat_by_index.append(f"feat.{js_name}")

    # Per-tag decision tree functions
    for tag in active_tags:
        if tag not in trees:
            continue
        tree_data = trees[tag]
        root = tree_data["tree"]
        thresh = tree_data["threshold"]

        func_name = f"_predictTag_{tag.replace(' ', '_').replace('-', '_')}"

        js += f"\nfunction {func_name}(feat) {{\n"
        js += tree_to_js(root, js_feat_by_index, 1)
        js += "}\n"

    # Note: computePatternTagsML() is hand-maintained in spm_algorithm.js
    # It calls these individual _predictTag_* functions with mapType filtering
    # and Mix/Hybrid synthesis logic (≥3 tags → suppression)

    return js


if __name__ == "__main__":
    main()
