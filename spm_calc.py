#!/usr/bin/env python
"""
SPM Rating — 单文件谱面 SR 计算器（Sigmoid 聚合）

用法:
  python spm_calc.py                     # 扫描脚本所在目录的 .osu
  python spm_calc.py "D:/maps/"          # 扫描指定目录
  python spm_calc.py chart.osu           # 计算单张谱面

特性:
  - Enhanced 模式 (Cross/Release/Shield/Inverse 全开启)
  - Sigmoid 玩家准度聚合 (k=2.09, C=3.97, γ=0.196)
  - 自动加载 tuned_params_sigmoid.json 最优参数
  - 无需预计算缓存 (precompute 即时完成)

依赖: numpy, scipy (仅用于可选优化), pandas (仅用于 playtest 评估)
"""

import os, sys, json, time, glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from spm_rating import rating


def load_params():
    """加载最优 sigmoid 参数。"""
    param_file = os.path.join(SCRIPT_DIR, "tuned_params_sigmoid.json")
    if not os.path.exists(param_file):
        print("[警告] 未找到 tuned_params_sigmoid.json，使用默认参数")
        from spm_rating.config import get_default_params
        d = get_default_params()
        return {k: v[0] for k, v in d.items()}

    with open(param_file, encoding="utf-8") as f:
        tuned = json.load(f)
    params = dict(tuned["params"])
    print(f"  加载参数: {tuned['method']}")
    print(f"  训练 MAE={tuned['mae']:.4f}, Loss={tuned['loss']:.4f}")
    return params


def compute_sr(osu_path, params):
    """计算单张谱面的 Star Rating。"""
    try:
        cache = rating.precompute(osu_path, use_enhanced=True, params=params)
        sr, details = rating.combine(cache, params=params)
        return sr, details
    except Exception as e:
        return None, str(e)


def main():
    print("=" * 55)
    print("  SPM Rating — Sigmoid 聚合 SR 计算器")
    print("=" * 55)

    # 加载参数
    params = load_params()
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
        sr, info = compute_sr(fpath, params)
        if sr is not None:
            results.append((fname, sr))
            if len(osu_files) == 1:
                # 单文件模式：打印详细信息
                d = info
                print(f"  谱面: {fname}")
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
