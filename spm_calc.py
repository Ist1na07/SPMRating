#!/usr/bin/env python
"""
SPM Rating — 单文件谱面 SR 计算器（Sigmoid 聚合 + 特征修正层）

用法:
  python spm_calc.py                     # 扫描脚本所在目录的 .osu
  python spm_calc.py "D:/maps/"          # 扫描指定目录
  python spm_calc.py chart.osu           # 计算单张谱面

特性:
  - Enhanced 模式 (Cross/Release/Shield/Inverse 全开启)
  - Sigmoid 玩家准度聚合 (k=2.09, C=3.97, γ=0.196)
  - 特征修正层 (7 个谱面特征，L2 正则化)
  - 自动加载 tuned_params_sigmoid.json 和 tuned_correction.json 最优参数
  - 无需预计算缓存 (precompute 即时完成)

依赖: numpy, scipy (仅用于可选优化), pandas (仅用于 playtest 评估)
"""

import os, sys, json, time, glob
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from spm_rating import rating
from spm_rating.combine_rc_ln import compute_total_notes
from spm_rating.aggregate_sigmoid import compute_SR_sigmoid


# 特征计算参数（固定默认值，不参与优化）
FEATURE_PARAMS = {
    "spd_dt": 150, "spd_dc": 3,
    "bst_dt": 100, "ch_order": 4,
    "hs_dt": 200, "lb_dt": 150, "fj_dt": 100,
}

FEATURE_NAMES = ["speed", "burst", "chord", "pj", "hs", "lb", "fj"]


def load_params():
    """加载最优 sigmoid 参数。"""
    param_file = os.path.join(SCRIPT_DIR, "tuned_params_sigmoid.json")
    if not os.path.exists(param_file):
        print("[警告] 未找到 tuned_params_sigmoid.json，使用默认参数")
        from spm_rating.config import get_default_params
        d = get_default_params()
        return {k: v[0] for k, v in d.items()}, None, None

    with open(param_file, encoding="utf-8") as f:
        tuned = json.load(f)
    params = dict(tuned["params"])
    print(f"  加载 Sigmoid 参数: {tuned['method']}")
    print(f"  训练 MAE={tuned['mae']:.4f}, Loss={tuned['loss']:.4f}")

    # 加载修正层参数
    correction_file = os.path.join(SCRIPT_DIR, "tuned_correction.json")
    if not os.path.exists(correction_file):
        print("[警告] 未找到 tuned_correction.json，跳过特征修正层")
        return params, None, None

    with open(correction_file, encoding="utf-8") as f:
        corr = json.load(f)
    weights = corr["correction_weights"]
    postprocess = corr["postprocess"]
    print(f"  加载修正层: {len(weights)} 个特征权重, λ={corr.get('regularization_lambda', '?')}")
    print(f"  CV Test Loss={corr['cv_test_loss']:.4f}")

    return params, weights, postprocess


def compute_features(cache, params=None):
    """计算谱面级特征用于修正层。"""
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

    # Speed: 快速跨手音符密度
    features["speed"] = float(
        np.sum((dt_ns < params["spd_dt"]) & (dc_ns >= int(round(params["spd_dc"]))))
    ) / duration_s

    # Burst: 三音组爆发密度
    features["burst"] = float(
        sum(1 for j in range(2, len(times)) if times[j] - times[j-2] < params["bst_dt"])
    ) / duration_s

    # Chord: 和弦比例
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

    # PJ ratio: 流/jack 平衡
    Jbar = cache["Jbar_base"]
    Pbar = cache["Pbar_base"]
    features["pj"] = (
        float(np.mean(Pbar) / (np.mean(Jbar) + 1))
        if len(Jbar) > 0 and len(Pbar) > 0
        else 1.5
    )

    # Hand-switch: 手切密度
    hand_mask = (
        ((cols[:-1] < 3) & (cols[1:] >= 4)) |
        ((cols[:-1] >= 4) & (cols[1:] < 3))
    )
    features["hs"] = float(
        np.sum(hand_mask & (dt_ns < params["hs_dt"]))
    ) / duration_s

    # Light burst: 四音组轻爆发密度
    features["lb"] = float(
        sum(1 for j in range(3, len(times)) if times[j] - times[j-3] < params["lb_dt"])
    ) / duration_s

    # Fast jack: 同列快打密度
    same_col = dc_ns == 0
    features["fj"] = (
        float(np.sum(dt_ns[same_col] < params["fj_dt"])) / duration_s
        if np.any(same_col)
        else 0.0
    )

    return features


def compute_sr(osu_path, params, weights=None, postprocess=None):
    """计算单张谱面的 Star Rating（含修正层）。"""
    try:
        # Step 1-2: 标准 SPM precompute + combine
        cache = rating.precompute(osu_path, use_enhanced=True, params=params)
        sr_base, details = rating.combine(cache, params=params)

        if weights is None or postprocess is None:
            # 无修正层，直接返回基础结果
            return sr_base, details

        # Step 3: D 校准
        D_full = details["D_all"]
        C_arr = details["C_arr"]
        cal_a = params.get("calib_a", 0.893)
        cal_b = params.get("calib_b", 0.031)
        D_calib = cal_a * D_full + cal_b

        # Step 4: 计算特征和修正量
        features = compute_features(cache)
        correction = sum(
            weights.get(fn, 0.0) * features.get(fn, 0.0)
            for fn in FEATURE_NAMES
        )

        # Step 5: 应用修正
        D_new = np.maximum(D_calib + correction, 0.01)

        # Step 6: 使用修正后的后处理参数聚合
        total_notes = compute_total_notes(cache["note_seq"], cache["LN_seq"])
        SR, _ = compute_SR_sigmoid(
            cache["all_corners"], C_arr, D_new, total_notes, cache["LN_seq"],
            sigmoid_k=params.get("agg_sigmoid_k", 2.09),
            sigmoid_C=params.get("agg_sigmoid_C", 3.969),
            sigmoid_gamma=params.get("agg_sigmoid_ref_gamma", 0.196),
            note_norm_N0=postprocess["N0"],
            rescale_threshold=postprocess["threshold"],
            rescale_divisor=postprocess["divisor"],
            global_scale=postprocess["scale"],
        )

        details.update({
            "sr_base": sr_base,
            "sr_corrected": SR,
            "correction": correction,
            "features": features,
        })
        return SR, details

    except Exception as e:
        return None, str(e)


def main():
    print("=" * 55)
    print("  SPM Rating — Sigmoid 聚合 + 特征修正 SR 计算器")
    print("=" * 55)

    # 加载参数
    params, weights, postprocess = load_params()
    params["use_sigmoid_aggregation"] = 1
    print()

    # 解析命令行
    args = sys.argv[1:]
    if args:
        target = args[0]
    else:
        target = SCRIPT_DIR

    # 收集 .osu 文件
    if os.path.isfile(target) and target.endswith(".osu"):
        osu_files = [target]
    elif os.path.isdir(target):
        osu_files = sorted(glob.glob(os.path.join(target, "*.osu")))
        if not osu_files:
            # 递归搜索
            osu_files = []
            for root, dirs, files in os.walk(target):
                for f in files:
                    if f.endswith(".osu"):
                        osu_files.append(os.path.join(root, f))
            osu_files.sort()
    else:
        print(f"  无效目标: {target}")
        return

    if not osu_files:
        print(f"  未找到 .osu 文件")
        return

    print(f"  计算 {len(osu_files)} 张谱面...")
    print()

    results = []
    errors = 0
    t0 = time.time()

    for i, fpath in enumerate(osu_files):
        fname = os.path.basename(fpath)
        sr, info = compute_sr(fpath, params, weights, postprocess)
        if sr is not None:
            results.append((fname, sr))
            if len(osu_files) == 1:
                # 单文件模式：打印详细信息
                d = info
                print(f"  谱面: {fname}")
                if weights is not None:
                    print(f"  SR (base):     {d.get('sr_base', sr):.4f}")
                    print(f"  SR (corrected): {sr:.4f}")
                    print(f"  Correction:     {d.get('correction', 0):+.4f}")
                else:
                    print(f"  SR:   {sr:.4f}")
                print(f"  D_all 范围: [{d.get('D_min',0):.2f}, {d.get('D_max',0):.2f}]")
                print(f"  D_weighted_mean: {d.get('D_weighted_mean',0):.2f}")
                print(f"  n_note: {d.get('n_raw',0)}, n_LN: {d.get('n_LN',0)}")
            else:
                print(f"  [{i+1}/{len(osu_files)}] {fname}  SR={sr:.4f}")
        else:
            errors += 1
            print(f"  [{i+1}/{len(osu_files)}] {fname}  [失败: {info}]")

    elapsed = time.time() - t0
    print()
    print("-" * 55)
    print(f"  完成 {len(results)} OK, {errors} 失败 ({elapsed:.1f}s)")
    if results and len(results) > 1:
        srs = [r[1] for r in results]
        print(f"  SR 范围: {min(srs):.4f} ~ {max(srs):.4f}")
        print(f"  SR 平均: {sum(srs)/len(srs):.4f}")
        print()
        print(f"  SR 排序:")
        results.sort(key=lambda x: -x[1])
        for fname, sr in results:
            print(f"    {sr:.4f}  {fname}")
    print("=" * 55)


if __name__ == "__main__":
    main()
