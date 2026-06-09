"""
Build standalone spm_calc_standalone.py from spm_rating/ source files.

Merges all module code into a single file, strips relative imports,
inlines tuned parameters and correction layer. Output is a self-contained .py file.
"""
import sys, os, re, json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPM_DIR = os.path.join(PROJECT_ROOT, "spm_rating")
COMP_DIR = os.path.join(SPM_DIR, "components")

# Files to include, in dependency order (no inter-module deps come first)
COMPONENT_FILES = [
    "anchor.py",       # no internal deps
    "jack.py",         # no internal deps
    "cross.py",        # no internal deps
    "cross_enhanced.py",  # no internal deps (but uses utils indirectly)
    "stream.py",       # no internal deps
    "release.py",      # no internal deps
    "release_enhanced.py",  # no internal deps (but uses utils)
    "shield.py",       # no internal deps
    "inverse.py",      # no internal deps
    "stamina.py",      # no internal deps (unused, but included for completeness)
]

CORE_FILES = [
    "utils.py",
    "config.py",
    "parser.py",
    "preprocessor.py",
    "combine.py",
    "aggregate.py",
    "aggregate_sigmoid.py",
    "rating.py",
]

# ============================================================
# Read and process source files
# ============================================================
REL_IMPORT_RE = re.compile(r'^from\s+\.', re.MULTILINE)

# Collect stdlib + third-party imports
stdlib_imports = set()
code_blocks = []

for fname in COMPONENT_FILES:
    fpath = os.path.join(COMP_DIR, fname)
    with open(fpath, encoding="utf-8") as f:
        src = f.read()
    # Extract stdlib/3rd-party imports
    for line in src.split("\n"):
        line = line.strip()
        if (line.startswith("import ") or line.startswith("from ")) and not line.startswith("from ."):
            stdlib_imports.add(line)
    # Strip all import lines, relative and absolute
    src = "\n".join(l for l in src.split("\n")
                    if not l.strip().startswith("import ") and not l.strip().startswith("from "))
    src = src.strip()
    if src:
        code_blocks.append((f"# === {fname} ===", src))

for fname in CORE_FILES:
    fpath = os.path.join(SPM_DIR, fname)
    with open(fpath, encoding="utf-8") as f:
        src = f.read()
    # Extract stdlib/3rd-party imports
    for line in src.split("\n"):
        line = line.strip()
        skip_prefixes = ["from .", "from spm_rating"]
        if (line.startswith("import ") or line.startswith("from ")):
            is_internal = any(line.startswith(p) for p in skip_prefixes)
            if not is_internal:
                stdlib_imports.add(line)
    # Strip ALL import lines
    src = "\n".join(l for l in src.split("\n")
                    if not l.strip().startswith("import ") and not l.strip().startswith("from "))
    src = src.strip()
    if src:
        code_blocks.append((f"# === {fname} ===", src))


# ============================================================
# Load tuned params and correction layer
# ============================================================
params_path = os.path.join(PROJECT_ROOT, "tuned_params_sigmoid.json")
with open(params_path, encoding="utf-8") as f:
    tuned_data = json.load(f)

correction_path = os.path.join(PROJECT_ROOT, "tuned_correction.json")
with open(correction_path, encoding="utf-8") as f:
    correction_data = json.load(f)

def json_to_py(obj, indent=0):
    """Convert Python dict to Python literal (not JSON — uses True/False/None)."""
    sp = " " * indent
    sp1 = " " * (indent + 4)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = ",\n".join(
            f"{sp1}{json.dumps(k)}: {json_to_py(v, indent+4)}"
            for k, v in obj.items()
        )
        return "{\n" + items + "\n" + sp + "}"
    elif isinstance(obj, list):
        if not obj:
            return "[]"
        items = ",\n".join(f"{sp1}{json_to_py(v, indent+4)}" for v in obj)
        return "[\n" + items + "\n" + sp + "]"
    elif isinstance(obj, bool):
        return "True" if obj else "False"
    elif isinstance(obj, float):
        return repr(obj)
    elif isinstance(obj, int):
        return str(obj)
    elif isinstance(obj, str):
        return json.dumps(obj)
    elif obj is None:
        return "None"
    return repr(obj)

tuned_params_block = json_to_py(tuned_data["params"], indent=4)
correction_weights_block = json_to_py(correction_data["correction_weights"], indent=4)
postprocess_block = json_to_py(correction_data["postprocess"], indent=4)


# ============================================================
# Build imports section
# ============================================================
# Build imports from all extracted stdlib_imports
IMPORT_LINES = []
SEEN_IMPORTS = set()

# Standard modules we always need
always_needed = [
    "import sys, os, json, time, glob, re, pickle, warnings",
    "import numpy as np",
    "import math",
    "import bisect",
    "import heapq",
    "from collections import defaultdict",
    "from dataclasses import dataclass, field",
]
for line in always_needed:
    IMPORT_LINES.append(line)
    SEEN_IMPORTS.add(line)
# Also mark individual imports as seen to prevent duplicates
for indiv in ["import pickle", "import warnings", "import math", "import bisect",
              "import heapq", "from collections import defaultdict",
              "from dataclasses import dataclass, field", "from dataclasses import dataclass"]:
    SEEN_IMPORTS.add(indiv)

# Add unique imports extracted from source files
for imp in sorted(stdlib_imports):
    # Skip internal/numpy imports already handled
    if "numpy" in imp: continue
    if imp in SEEN_IMPORTS: continue
    if imp.startswith("from spm_rating"): continue
    if imp.startswith("from ."): continue
    SEEN_IMPORTS.add(imp)
    IMPORT_LINES.append(imp)

# Add scipy optional
IMPORT_LINES.append("")
IMPORT_LINES.append("# scipy only needed for Nelder-Mead (optional)")
IMPORT_LINES.append("try:")
IMPORT_LINES.append("    from scipy.optimize import minimize")
IMPORT_LINES.append("except ImportError:")
IMPORT_LINES.append("    minimize = None")

# Add pandas optional (used by aggregate.py)
IMPORT_LINES.append("")
IMPORT_LINES.append("# pandas only needed for playtest loading (optional)")
IMPORT_LINES.append("try:")
IMPORT_LINES.append("    import pandas as pd")
IMPORT_LINES.append("except ImportError:")
IMPORT_LINES.append("    pd = None")

imports_section = "\n".join(IMPORT_LINES)


# ============================================================
# Write standalone file
# ============================================================
HEADER = '''#!/usr/bin/env python
"""
SPM Rating — 独立 SR 计算器 (Sigmoid 玩家准度聚合 + 特征修正层)

单文件, 零外部依赖 (除 numpy)。放入任意目录即可使用。

用法:
  python spm_calc_standalone.py                     # 扫描当前目录的 .osu
  python spm_calc_standalone.py "D:/maps/"          # 扫描指定目录
  python spm_calc_standalone.py chart.osu           # 计算单张谱面

模型:
  - Enhanced 模式 (Cross/Release/Shield/Inverse 全量分量)
  - Sigmoid 准确度聚合 (k=2.09, C=3.97, γ=0.196)
  - 特征修正层 (7 个谱面特征, L2 正则化, λ=0.01)
  - 311 张 playtest 谱面调优, MAE=0.2180, CV Test Loss=0.862

构建时间: """ + __import__("datetime").datetime.now().strftime("%Y-%m-%d") + """
"""

'''

TUNED_PARAMS_SECTION = f"""
# ============================================================================
# Tuned Parameters (from tuned_params_sigmoid.json, MAE=0.2180)
# ============================================================================

TUNED_PARAMS = {tuned_params_block}
TUNED_PARAMS["use_sigmoid_aggregation"] = 1


# ============================================================================
# Correction Layer (7 features, L2 regularized, λ=0.01)
# ============================================================================

CORRECTION_WEIGHTS = {correction_weights_block}

CORRECTION_POSTPROCESS = {postprocess_block}

FEATURE_NAMES = ["speed", "burst", "chord", "pj", "hs", "lb", "fj"]

FEATURE_PARAMS = {{
    "spd_dt": 150, "spd_dc": 3,
    "bst_dt": 100, "ch_order": 4,
    "hs_dt": 200, "lb_dt": 150, "fj_dt": 100,
}}

"""

CLI_SECTION = '''
# ============================================================================
# Feature Computation (Correction Layer)
# ============================================================================

def compute_features(cache, params=None):
    """Compute map-level features for the correction layer.

    These features capture pattern densities that the base D(t) formula
    systematically under- or over-estimates.
    """
    if params is None:
        params = FEATURE_PARAMS

    ns = cache["note_seq"]
    times = np.array([n[1] for n in ns], dtype=np.float64)
    cols = np.array([n[0] for n in ns], dtype=np.int32)
    duration_s = max((times[-1] - times[0]) / 1000.0, 1.0)
    features = {}
    if len(ns) < 2:
        return features

    dt_ns = np.diff(times).astype(np.float64)
    dc_ns = np.abs(np.diff(cols)).astype(np.float64)

    # Speed: fast cross-hand notes per second
    features["speed"] = float(
        np.sum((dt_ns < params["spd_dt"]) & (dc_ns >= int(round(params["spd_dc"]))))
    ) / duration_s

    # Burst: triplet density per second
    features["burst"] = float(
        sum(1 for j in range(2, len(times)) if times[j] - times[j-2] < params["bst_dt"])
    ) / duration_s

    # Chord: fraction of notes that are part of chords (>=ch_order simultaneous)
    ch_order = int(round(params["ch_order"]))
    chord_count = 0
    for j in range(len(ns)):
        t = times[j]
        cnt = 1
        k = j - 1
        while k >= 0 and abs(times[k] - t) < 2:
            cnt += 1
            k -= 1
        if cnt >= ch_order:
            chord_count += 1
    features["chord"] = chord_count / max(len(ns), 1)

    # PJ ratio: stream/jack balance
    Jbar = cache["Jbar_base"]
    Pbar = cache["Pbar_base"]
    features["pj"] = (
        float(np.mean(Pbar) / (np.mean(Jbar) + 1))
        if len(Jbar) > 0 and len(Pbar) > 0
        else 1.5
    )

    # Hand-switch: cross-hand transitions per second
    hand_mask = (
        ((cols[:-1] < 3) & (cols[1:] >= 4)) |
        ((cols[:-1] >= 4) & (cols[1:] < 3))
    )
    features["hs"] = float(
        np.sum(hand_mask & (dt_ns < params["hs_dt"]))
    ) / duration_s

    # Light burst: 4-note burst density
    features["lb"] = float(
        sum(1 for j in range(3, len(times)) if times[j] - times[j-3] < params["lb_dt"])
    ) / duration_s

    # Fast jack: same-column rapid notes per second
    same_col = dc_ns == 0
    features["fj"] = (
        float(np.sum(dt_ns[same_col] < params["fj_dt"])) / duration_s
        if np.any(same_col)
        else 0.0
    )

    return features


# ============================================================================
# CLI Entry Point
# ============================================================================

def _p(params, key, default):
    """Get param from dict with default."""
    if params is None:
        return default
    return params.get(key, default)


def _load_params():
    """Return tuned params as the base parameter dict."""
    return dict(TUNED_PARAMS)


def compute_sr_map(osu_path, params=None):
    """Compute Star Rating for a single .osu chart with correction layer.

    Args:
        osu_path: path to .osu file
        params: optional param overrides (dict), uses TUNED_PARAMS if None

    Returns:
        sr: float Star Rating value
        details: dict with diagnostic info (D_all, D_solved, correction, features, etc.)
    """
    if params is None:
        params = dict(TUNED_PARAMS)
    else:
        # Merge with tuned params
        p = dict(TUNED_PARAMS)
        p.update(params)
        params = p
    params["use_sigmoid_aggregation"] = 1

    # Step 1-2: Standard SPM precompute + combine
    cache = precompute(osu_path, use_enhanced=True, params=params)
    sr_base, details = combine(cache, params=params)
    D_full = details["D_all"]
    C_arr = details["C_arr"]

    # Step 3: D calibration
    cal_a = params.get("calib_a", 0.893)
    cal_b = params.get("calib_b", 0.031)
    D_calib = cal_a * D_full + cal_b

    # Step 4: Compute features and correction
    features = compute_features(cache)
    correction = sum(
        CORRECTION_WEIGHTS.get(fn, 0.0) * features.get(fn, 0.0)
        for fn in FEATURE_NAMES
    )

    # Step 5: Apply correction
    D_new = np.maximum(D_calib + correction, 0.01)

    # Step 6: Aggregation with corrected postprocess params
    total_notes = compute_total_notes(cache["note_seq"], cache["LN_seq"])
    SR, agg_details = compute_SR_sigmoid(
        cache["all_corners"], C_arr, D_new, total_notes, cache["LN_seq"],
        sigmoid_k=params.get("agg_sigmoid_k", 2.09),
        sigmoid_C=params.get("agg_sigmoid_C", 3.969),
        sigmoid_gamma=params.get("agg_sigmoid_ref_gamma", 0.196),
        note_norm_N0=CORRECTION_POSTPROCESS["N0"],
        rescale_threshold=CORRECTION_POSTPROCESS["threshold"],
        rescale_divisor=CORRECTION_POSTPROCESS["divisor"],
        global_scale=CORRECTION_POSTPROCESS["scale"],
    )

    details.update({
        "sr_base": sr_base,
        "sr_corrected": SR,
        "correction": correction,
        "features": features,
    })

    return SR, details


def main():
    print("=" * 55)
    print("  SPM Rating — Sigmoid 聚合 + 特征修正 SR 计算器")
    print("=" * 55)
    print(f"  k={TUNED_PARAMS['agg_sigmoid_k']:.2f}, C={TUNED_PARAMS['agg_sigmoid_C']:.2f}, "
          f"gamma={TUNED_PARAMS['agg_sigmoid_ref_gamma']:.3f}")
    print(f"  训练 MAE=0.2180, CV Test Loss=0.862")
    print(f"  修正层: {len(FEATURE_NAMES)} 个特征")
    print()

    args = sys.argv[1:]
    target = args[0] if args else os.path.dirname(os.path.abspath(__file__))

    # Collect .osu files
    if os.path.isfile(target) and target.endswith(".osu"):
        osu_files = [target]
    elif os.path.isdir(target):
        osu_files = []
        for root, dirs, files in os.walk(target):
            for f in files:
                if f.endswith('.osu'):
                    osu_files.append(os.path.join(root, f))
        osu_files.sort()
    else:
        print(f"  无效目标: {target}")
        sys.exit(1)

    if not osu_files:
        print(f"  未找到 .osu 文件")
        sys.exit(1)

    if len(osu_files) == 1:
        fpath = osu_files[0]
        print(f"  计算: {os.path.basename(fpath)}")
        sr, d = compute_sr_map(fpath)
        print(f"  SR (base):     {d.get('sr_base', 0):.4f}")
        print(f"  SR (corrected): {sr:.4f}")
        print(f"  Correction:     {d.get('correction', 0):+.4f}")
        print(f"  D_all 范围: [{d.get('D_min',0):.2f}, {d.get('D_max',0):.2f}]")
        print(f"  D_solved: {d.get('D_solved',0):.2f}")
        if 'n_raw' in d:
            print(f"  notes: {d.get('n_raw',0)}, LN: {d.get('n_LN',0)}")
        return

    print(f"  计算 {len(osu_files)} 张谱面...")
    print()
    results = []
    errors = 0
    t0 = time.time()
    for i, fpath in enumerate(osu_files):
        fname = os.path.basename(fpath)
        try:
            sr, _ = compute_sr_map(fpath)
            results.append((fname, sr))
            print(f"  [{i+1}/{len(osu_files)}] {fname}  SR={sr:.4f}")
        except Exception as e:
            errors += 1
            print(f"  [{i+1}/{len(osu_files)}] {fname}  [失败: {e}]")

    elapsed = time.time() - t0
    print()
    print("-" * 55)
    print(f"  完成 {len(results)} OK, {errors} 失败 ({elapsed:.1f}s)")
    if results:
        srs = [r[1] for r in results]
        print(f"  SR 范围: {min(srs):.4f} ~ {max(srs):.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
'''

# ============================================================
# Assemble final file
# ============================================================
output_path = os.path.join(PROJECT_ROOT, "spm_calc_standalone.py")

with open(output_path, "w", encoding="utf-8") as out:
    out.write(HEADER)
    out.write("\n")
    out.write(imports_section)
    out.write("\n\n")
    out.write(TUNED_PARAMS_SECTION)
    out.write("\n")

    for label, code in code_blocks:
        out.write(f"\n\n{label}\n")
        out.write(code)
        out.write("\n")

    out.write(CLI_SECTION)

# ============================================================
# Post-process: remove module prefixes from merged code
# ============================================================
with open(output_path, encoding="utf-8") as f:
    final_code = f.read()

# Replace module-qualified calls with direct calls
# These are the namespace aliases used in rating.py after stripping imports
replacements = [
    (r'\bparser\.', ''),
    (r'\bpreprocessor\.', ''),
    (r'\b_anchor\.', ''),
    (r'\b_jack\.', ''),
    (r'\b_cross_enh\.', ''),
    (r'\b_cross\.', ''),
    (r'\b_stream\.', ''),
    (r'\b_release_enh\.', ''),
    (r'\b_release\.', ''),
    (r'\b_shield\.', ''),
    (r'\b_inverse\.', ''),
    (r'\b_stamina\.', ''),
    (r'\b_CROSS_CFG\b', 'CROSS'),
    (r'\b_RELEASE_CFG\b', 'RELEASE'),
    (r'\b_SHIELD_CFG\b', 'SHIELD'),
]

for pattern, replacement in replacements:
    final_code = re.sub(pattern, replacement, final_code)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(final_code)

# Quick syntax check
try:
    compile(final_code, output_path, "exec")
    print("  Post-process syntax check: OK")
except SyntaxError as e:
    print(f"  Post-process syntax check: FAILED — {e}")

print(f"Written: {output_path}")
print(f"  {len(code_blocks)} code blocks merged")
print(f"  Total lines: ~{sum(1 for l in open(output_path, encoding='utf-8'))}")

# Quick syntax check
try:
    compile(open(output_path, encoding="utf-8").read(), output_path, "exec")
    print("  Syntax check: OK")
except SyntaxError as e:
    print(f"  Syntax check: FAILED — {e}")
