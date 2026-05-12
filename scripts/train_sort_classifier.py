"""
Train a map sort classifier (RC/LN/HB/Mix) for the tosu plugin.

Uses labeled data from Excel (sort column) to build a simple decision tree
that can be ported to pure JavaScript.

Features:
- lnRatio: LN notes / total notes
- avgLNDuration: mean LN duration in ms
- maxLNDensity: peak LN_rep value (proxy for LN section intensity)
- rcSrRatio: RC model SR / Total SR

Output: tosustatic/spm-ratingV2pro-sigmoid/classifier_constants.json
"""
import sys, os, json, re
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tuning.data_loader import load_playtest_data
from spm_rating.rating import precompute, combine as combine_total
from spm_rating.combine_rc_ln import compute_rc_sr


def extract_dan_level(filename):
    """Extract numeric Dan level from filename. Returns None if not a Dan map."""
    m = re.search(r'\[(\d+)(?:st|nd|rd|th)\s+Dan\]', filename)
    if m:
        return float(m.group(1))
    m = re.search(r'\[(Gamma|Azimuth|Zenith|Stellium)\s+Dan\]', filename)
    if m:
        return {"Gamma": 11.5, "Azimuth": 13, "Zenith": 14.5, "Stellium": 16}.get(m.group(1))
    return None


def extract_features(entry, cache, total_params, rc_params):
    """Extract classification features from a cached map."""
    # Basic counts
    note_seq = cache["note_seq"]
    ln_seq = cache["LN_seq"]
    n_total = len(note_seq)
    n_ln = len(ln_seq)
    ln_ratio = n_ln / max(n_total, 1)

    # Average LN duration: LN_seq entries are (col, head_time, tail_time) tuples
    if n_ln > 0:
        avg_ln_dur = np.mean([max(ln[2] - ln[1], 0) for ln in ln_seq])
    else:
        avg_ln_dur = 0

    # Peak LN density from LN_rep
    ln_rep = cache.get("LN_rep", None)
    if ln_rep is not None and len(ln_rep[2]) > 0:
        max_ln_density = float(np.max(ln_rep[2]))
    else:
        max_ln_density = 0.0

    # RC SR and Total SR
    try:
        sr_total, _ = combine_total(cache, total_params)
        sr_rc, _ = compute_rc_sr(cache, rc_params)
        rc_sr_ratio = sr_rc / max(sr_total, 0.01)
    except Exception:
        sr_total = 5.0
        sr_rc = 5.0
        rc_sr_ratio = 1.0

    # LN contribution: how much LN-only components contribute
    # (total SR - RC SR) / total SR
    ln_contrib = max(0, sr_total - sr_rc) / max(sr_total, 0.01)

    return {
        "ln_ratio": ln_ratio,
        "avg_ln_dur": avg_ln_dur,
        "max_ln_density": max_ln_density,
        "rc_sr_ratio": rc_sr_ratio,
        "ln_contrib": ln_contrib,
        "sr_total": sr_total,
        "sr_rc": sr_rc,
    }


def tree_to_js(node, feature_ids, class_names, indent=0):
    """Recursively convert sklearn tree node to JS if-else code."""
    prefix = "  " * indent

    # Leaf node
    if node["is_leaf"]:
        class_idx = node["class"]
        return f"{prefix}return '{class_names[class_idx].upper()}';\n"

    # Internal node
    feat = feature_ids[node["feature"]]
    thresh = node["threshold"]
    return (
        f"{prefix}if ({feat} <= {thresh:.6f}) {{\n"
        f"{tree_to_js(node['left'], feature_ids, class_names, indent + 1)}"
        f"{prefix}}} else {{\n"
        f"{tree_to_js(node['right'], feature_ids, class_names, indent + 1)}"
        f"{prefix}}}\n"
    )


def build_js_classifier(clf, feature_ids, mix_thresholds):
    """Generate JavaScript classifier code."""
    class_names = list(clf.classes_)  # sklearn's alphabetical order
    print(f"  Class order from sklearn: {class_names}")
    # Convert tree to nested dict
    def extract_node(tree, node_id):
        if tree.children_left[node_id] == -1:  # leaf
            counts = tree.value[node_id][0]
            return {"is_leaf": True, "class": int(np.argmax(counts))}
        return {
            "is_leaf": False,
            "feature": int(tree.feature[node_id]),
            "threshold": float(tree.threshold[node_id]),
            "left": extract_node(tree, tree.children_left[node_id]),
            "right": extract_node(tree, tree.children_right[node_id]),
        }

    root = extract_node(clf.tree_, 0)

    js = "/** Auto-generated map sort classifier. */\n"
    js += "function classifyMapSort(lnRatio, avgLNDur, maxLNDensity, rcSrRatio, lnContrib) {\n"

    # Feature expression mapping
    feat_exprs = {
        0: "lnRatio",
        1: "avgLNDur",
        2: "maxLNDensity",
        3: "rcSrRatio",
        4: "lnContrib",
    }
    js += tree_to_js(root, feat_exprs, class_names, 1)
    js += "}\n\n"

    # Mix heuristic
    js += "/** Mix heuristic: low LN count but high LN impact. */\n"
    js += "function isMapMix(lnRatio, lnContrib) {\n"
    js += f"  return (lnRatio < {mix_thresholds['ln_ratio_max']:.4f}"
    js += f" && lnContrib > {mix_thresholds['ln_contrib']:.4f});\n"
    js += "}\n"

    return js


def main():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load params
    with open(os.path.join(ROOT, "tuned_params_sigmoid.json")) as f:
        total_params = dict(json.load(f)["params"])
    with open(os.path.join(ROOT, "tuned_params_rc.json")) as f:
        rc_params = dict(json.load(f)["params"])

    # Load entries
    entries = load_playtest_data()
    print(f"Loaded {len(entries)} entries")

    # Filter entries with sort labels
    labeled = [(i, e) for i, e in enumerate(entries) if e.get("sort") in ("rc", "ln", "hb")]
    print(f"Labeled entries: {len(labeled)}")
    print(f"  RC: {sum(1 for _, e in labeled if e['sort']=='rc')}")
    print(f"  LN: {sum(1 for _, e in labeled if e['sort']=='ln')}")
    print(f"  HB: {sum(1 for _, e in labeled if e['sort']=='hb')}")

    # Check for cached features
    import pickle
    features_cache = os.path.join(ROOT, "cache", "classifier_features.pkl")
    if os.path.exists(features_cache):
        print(f"\nLoading cached features from {features_cache}...")
        with open(features_cache, "rb") as f:
            data = pickle.load(f)
        features_list = data["features"]
        labels_list = data["labels"]
        sources = data["sources"]
        print(f"  Loaded {len(features_list)} samples")
    else:
        print(f"\nPrecomputing {len(labeled)} maps...")
        import time
        t0 = time.time()
        features_list = []
        labels_list = []
        sources = []

        for idx, (i, entry) in enumerate(labeled):
            try:
                cache = precompute(entry["osu_path"], use_enhanced=True, params=total_params)
                feat = extract_features(entry, cache, total_params, rc_params)
                features_list.append(feat)
                labels_list.append(entry["sort"])
                sources.append(entry.get("source", "unknown"))
            except Exception as e:
                print(f"  ERROR [{entry['mapfile'][:60]}]: {e}")
            if (idx + 1) % 50 == 0:
                print(f"  {idx + 1}/{len(labeled)}...")
        print(f"  Done in {time.time() - t0:.0f}s. {len(features_list)} valid.")

        # Cache features
        os.makedirs(os.path.join(ROOT, "cache"), exist_ok=True)
        with open(features_cache, "wb") as f:
            pickle.dump({"features": features_list, "labels": labels_list, "sources": sources}, f)
        print(f"  Cached features to {features_cache}")

    # Convert to numpy
    feature_names = ["ln_ratio", "avg_ln_dur", "max_ln_density", "rc_sr_ratio", "ln_contrib"]
    X = np.array([[f[n] for n in feature_names] for f in features_list])
    y = np.array(labels_list)

    # Print feature statistics by class
    print(f"\n{'='*60}")
    print("Feature statistics by class:")
    print(f"{'='*60}")
    for cls in ["rc", "ln", "hb"]:
        mask = y == cls
        if mask.sum() == 0:
            continue
        print(f"\n  {cls.upper()} ({mask.sum()} samples):")
        for j, name in enumerate(feature_names):
            vals = X[mask, j]
            print(f"    {name:>20s}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
                  f"min={vals.min():.4f}, max={vals.max():.4f}")

    # Train decision tree (3 classes only: rc, ln, hb)
    print(f"\n{'='*60}")
    print("Training Decision Tree (max_depth=3)")
    print(f"{'='*60}")

    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)

    print(f"Training accuracy: {acc:.1%}")
    print(f"\nClassification report:")
    print(classification_report(y, y_pred, labels=["rc", "ln", "hb"]))

    # Show misclassifications
    errors = [(i, y[i], y_pred[i], sources[i]) for i in range(len(y)) if y[i] != y_pred[i]]
    if errors:
        print(f"\nMisclassifications ({len(errors)}):")
        for i, true, pred, src in errors[:10]:
            print(f"  {true}→{pred} [{src}] lnRatio={X[i,0]:.3f} avgDur={X[i,1]:.0f} "
                  f"density={X[i,2]:.2f} rcSrRat={X[i,3]:.3f} lnContrib={X[i,4]:.3f}")

    # === Mix heuristic calibration ===
    # Mix = maps where LN note count is low but LN difficulty impact is high.
    # Find thresholds: ln_contrib > X AND max_ln_density > Y, but ln_ratio not too high
    # (otherwise it would be LN, not Mix)
    print(f"\n{'='*60}")
    print("Mix Heuristic Calibration")
    print(f"{'='*60}")

    # Look at HB-labeled maps for Mix candidates
    hb_mask = y == "hb"
    if hb_mask.sum() > 0:
        print(f"\n  HB maps:")
        for i in np.where(hb_mask)[0]:
            print(f"    lnR={X[i,0]:.3f} lnDur={X[i,1]:.0f}ms "
                  f"dens={X[i,2]:.2f} rcSRrat={X[i,3]:.3f} lnContrib={X[i,4]:.3f}")

    # Look at LN-labeled maps with low ln_ratio (potential Mix)
    ln_mask = y == "ln"
    low_ln = (X[:, 0] < 0.60) & ln_mask
    if low_ln.sum() > 0:
        print(f"\n  LN maps with lnRatio < 0.60 (potential Mix):")
        for i in np.where(low_ln)[0]:
            print(f"    lnR={X[i,0]:.3f} lnDur={X[i,1]:.0f}ms "
                  f"dens={X[i,2]:.2f} rcSRrat={X[i,3]:.3f} lnContrib={X[i,4]:.3f}")

    # Heuristic thresholds for Mix
    # Mix = LN note count is low, but LN difficulty impact is disproportionately high
    mix_thresholds = {
        "ln_contrib": 0.10,       # LN contributes >10% of total difficulty
        "ln_ratio_max": 0.45,      # LN note ratio < 45% (otherwise it's just LN)
    }

    print(f"\n  Mix thresholds:")
    for k, v in mix_thresholds.items():
        print(f"    {k}: {v}")

    # Output
    feature_ids = {name: f"features[{i}]" for i, name in enumerate(feature_names)}
    js_code = build_js_classifier(clf, feature_ids, mix_thresholds)

    out = {
        "classifier": "decision_tree",
        "max_depth": 3,
        "accuracy": float(acc),
        "feature_names": feature_names,
        "n_samples": int(len(y)),
        "class_distribution": {c: int(sum(y == c)) for c in ["rc", "ln", "hb"]},
        "class_order": list(clf.classes_),
        "mix_thresholds": mix_thresholds,
        "javascript": js_code,
        # Tree structure for reference
        "tree": {
            "n_nodes": int(clf.tree_.node_count),
            "children_left": clf.tree_.children_left.tolist(),
            "children_right": clf.tree_.children_right.tolist(),
            "feature": [feature_names[i] if i >= 0 else None for i in clf.tree_.feature],
            "threshold": clf.tree_.threshold.tolist(),
            "value": clf.tree_.value.tolist(),
        },
    }

    out_path = os.path.join(ROOT, "tosustatic", "spm-ratingV2pro-sigmoid",
                            "classifier_constants.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {out_path}")
    print(f"\n=== JavaScript Classifier ===")
    print(js_code)


if __name__ == "__main__":
    main()
