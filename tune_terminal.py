#!/usr/bin/env python
"""
SPM Rating — 调参终端

集成了所有调参相关操作：
  status      查看数据概况和缓存状态
  evaluate    快速评估（~5秒）
  random      随机搜索调参
  tune        CMA-ES完整调参
  compare     clone vs enhanced 对比
  params      查看 saved 参数
  analyze     按标签/难度带分析
  recache     强制重新预计算缓存
  help        帮助
  exit        退出

首次 evaluate 会预计算171张谱面的中间数据（~3分钟），
之后所有操作都在~5秒内完成。
"""

import os, sys, time, json, pickle
import numpy as np

_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_root)

CACHE_DIR = os.path.join(_project_root, "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "precomputed.pkl")
TUNED_FILE = os.path.join(_project_root, "tuned_params.json")

# ===================== 数据/参数加载 =====================

_entries = None
_cached_entries = []  # [(entry, cache_dict), ...]
_cache_use_enhanced = None


def load_entries():
    global _entries
    if _entries is None:
        from tuning.data_loader import load_playtest_data
        _entries = load_playtest_data(maps_root=_project_root)
    return _entries


def get_default_params():
    from spm_rating.config import get_default_params
    d = get_default_params()
    return {k: v[0] for k, v in d.items()}


def load_tuned_params():
    """加载 tuned_params.json，存在则覆盖默认值，不存在返回默认。"""
    d = get_default_params()
    if os.path.exists(TUNED_FILE):
        with open(TUNED_FILE) as f:
            tuned = json.load(f)
        for k, v in tuned.get("params", {}).items():
            d[k] = v
    return d


# ===================== 缓存管理 =====================

def ensure_cache(use_enhanced=False, force=False):
    """确保预计算缓存存在，返回 [(entry, cache)] 列表。"""
    global _cached_entries, _cache_use_enhanced

    if not force and _cached_entries and _cache_use_enhanced == use_enhanced:
        return _cached_entries

    entries = load_entries()
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 检查磁盘缓存
    if not force and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                stored = pickle.load(f)
            if stored.get("version") == 7 and stored.get("use_enhanced") == use_enhanced:
                cache_map = stored.get("caches", {})
                result = []
                for e in entries:
                    c = cache_map.get(e["mapfile"])
                    if c is not None:
                        result.append((e, c))
                if len(result) == len(entries):
                    _cached_entries = result
                    _cache_use_enhanced = use_enhanced
                    return result
        except Exception:
            pass

    # 预计算
    print(f"  预计算 {len(entries)} 张谱面 (首次运行, 约3分钟)...")
    t0 = time.time()

    from spm_rating import rating
    result = []
    errors = 0
    for i, e in enumerate(entries):
        try:
            cache = rating.precompute(e["osu_path"], use_enhanced=use_enhanced,
                                          params={"stream_booster_scale": 8.5e-7})
            result.append((e, cache))
        except Exception as ex:
            errors += 1
            print(f"    [{i+1}/{len(entries)}] {e['mapfile'][:50]}... 失败: {ex}")
        if (i + 1) % 30 == 0:
            print(f"    [{i+1}/{len(entries)}] ...")

    elapsed = time.time() - t0
    print(f"  预计算完成: {len(result)} 成功, {errors} 失败, 用时 {elapsed:.0f}s")

    # 保存缓存
    cache_map = {entry["mapfile"]: c for entry, c in result}
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({
            "version": 7,
            "use_enhanced": use_enhanced,
            "caches": cache_map,
        }, f)
    print(f"  缓存保存至: {CACHE_FILE}")

    _cached_entries = result
    _cache_use_enhanced = use_enhanced
    return result


# ===================== 核心评估（使用缓存，~5秒） =====================

def evaluate_fast(params, use_enhanced, entries_limit=0):
    """
    快速评估：只用 combine()，不重新解析和计算分量。

    Args:
        params: 参数字典
        use_enhanced: 增强模式
        entries_limit: 限制谱面数量（0=全部）

    Returns:
        (loss, analysis, results)
    """
    cached = ensure_cache(use_enhanced)
    from tuning.scorer import score_batch
    from tuning.analyze import analyze_results
    from spm_rating import rating

    if entries_limit > 0:
        cached = cached[:entries_limit]

    results = []
    for entry, cache in cached:
        try:
            sr, _ = rating.combine(cache, params=params)
            results.append({
                "mapfile": entry["mapfile"],
                "sr_pred": sr,
                "sr_ref": entry["sr_ref"],
                "sr_error": entry["sr_error"],
                "tags": entry.get("tags", ""),
                "error": None,
            })
        except Exception as ex:
            results.append({
                "mapfile": entry["mapfile"],
                "sr_pred": None,
                "sr_ref": entry["sr_ref"],
                "sr_error": entry["sr_error"],
                "tags": entry.get("tags", ""),
                "error": str(ex),
            })

    valid = [r for r in results if r.get("error") is None and r.get("sr_pred") is not None]
    if not valid:
        return None, None, results

    preds = [r["sr_pred"] for r in valid]
    refs = [r["sr_ref"] for r in valid]
    errs = [r["sr_error"] for r in valid]
    loss, _, details = score_batch(preds, refs, errs)
    analysis = analyze_results(results)
    return loss, analysis, results


# ===================== 命令处理 =====================

def cmd_status():
    entries = load_entries()
    d = get_default_params()

    dan = sum(1 for e in entries if e["source"] == "dan")
    tour = sum(1 for e in entries if e["source"] == "tournament")
    rc = sum(1 for e in entries if e["d_rc"] is not None)
    ln_map = sum(1 for e in entries if e["d_ln"] is not None)
    sr_refs = [e["sr_ref"] for e in entries]

    print(f"  数据概况")
    print(f"  ─────────")
    print(f"  总有效条目:      {len(entries)}")
    print(f"    Dan段位:       {dan}")
    print(f"    Tournament:    {tour}")
    print(f"  有RC评分:        {rc}")
    print(f"  有LN评分:        {ln_map}")
    print(f"  难度范围:        {min(sr_refs):.2f} ~ {max(sr_refs):.2f} SR")
    print(f"  平均难度:        {np.mean(sr_refs):.2f} SR")
    print()
    print(f"  系统状态")
    print(f"  ─────────")
    print(f"  缓存:            {'存在' if os.path.exists(CACHE_FILE) else '无 (首次需预计算)'}")
    print(f"  调参结果:        {'存在' if os.path.exists(TUNED_FILE) else '无'}")
    if os.path.exists(CACHE_FILE):
        cache_size = os.path.getsize(CACHE_FILE) / 1024 / 1024
        print(f"  缓存大小:        {cache_size:.1f} MB")
    print()


def cmd_evaluate(args):
    use_enhanced = "enhanced" in args
    limit = 0
    for a in args:
        try: limit = int(a)
        except ValueError: pass

    params = load_tuned_params()
    mode = "enhanced" if use_enhanced else "clone"
    total = limit if limit > 0 else len(load_entries())

    print(f"  评估 ({mode} mode, {total} 张)")
    t0 = time.time()
    loss, analysis, results = evaluate_fast(params, use_enhanced, limit)
    elapsed = time.time() - t0

    if analysis is None:
        print("  [错误] 评估失败")
        return

    print(f"  {analysis['n_valid']}/{total} 有效, 用时 {elapsed:.1f}s")
    print(f"  MAE:         {analysis['mae']:.4f}")
    print(f"  相关性:      {analysis['correlation']:.4f}")
    print(f"  在误差内:    {analysis['inside_error_ratio']:.1%}")
    print(f"  RMSE:        {analysis['rmse']:.4f}")
    print(f"  平均残差:    {analysis['mean_residual']:+.4f}")
    print()


def cmd_random(args):
    use_enhanced = "enhanced" in args
    trials = 20
    for a in args:
        try: trials = int(a)
        except ValueError: pass

    entries = load_entries()
    d = get_default_params()

    # 基线
    _, analysis0, _ = evaluate_fast(d, use_enhanced)
    print(f"  默认: MAE={analysis0['mae']:.4f}")
    print(f"  随机搜索 {trials} 次...")
    print()

    best_loss = analysis0['mae'] if analysis0 else float('inf')
    best_d = d.copy()

    for i in range(trials):
        trial = dict(d)
        trial["S_w1"]  = np.clip(d["S_w1"] + np.random.normal(0, 0.1), 0.1, 0.9)
        trial["S_p"]   = np.clip(d["S_p"] + np.random.normal(0, 0.3), 0.5, 4.0)
        trial["alpha_P"] = np.clip(d["alpha_P"] + np.random.normal(0, 0.2), 0.1, 3.0)
        trial["alpha_R"] = np.clip(d["alpha_R"] + np.random.normal(0, 5), 5, 100)
        trial["D_beta1"] = np.clip(d["D_beta1"] + np.random.normal(0, 0.5), 0.5, 10.0)
        trial["D_beta2"] = np.clip(d["D_beta2"] + np.random.normal(0, 0.1), 0.05, 1.0)
        trial["w_93"]  = np.clip(d["w_93"] + np.random.normal(0, 0.05), 0.05, 0.5)
        trial["w_83"]  = np.clip(d["w_83"] + np.random.normal(0, 0.05), 0.05, 0.5)
        trial["w_mean"]= np.clip(d["w_mean"] + np.random.normal(0, 0.05), 0.2, 0.8)
        trial["global_scale"] = np.clip(d["global_scale"] + np.random.normal(0, 0.02), 0.9, 1.1)
        trial["coeff_93"] = np.clip(d["coeff_93"] + np.random.normal(0, 0.05), 0.5, 1.5)

        loss, analysis, _ = evaluate_fast(trial, use_enhanced)
        # 用 MAE 比较（直观）
        mae = analysis['mae'] if analysis else float('inf')
        if mae < best_loss:
            best_loss = mae
            best_d = trial.copy()
            print(f"  [{i+1}/{trials}] MAE={mae:.4f} (NEW BEST, 改善 {(1-mae/max(analysis0['mae'],0.001))*100:.1f}%)")
        elif mae < float('inf'):
            print(f"  [{i+1}/{trials}] MAE={mae:.4f}")

    # 保存
    output = {"loss": best_loss, "loss_default": analysis0['mae'], "mode": "clone" if not use_enhanced else "enhanced",
              "method": "random_search", "trials": trials, "params": best_d}
    with open(TUNED_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  已保存: {TUNED_FILE}")
    print(f"  MAE: {best_loss:.4f} (默认: {analysis0['mae']:.4f}, 改善 {(1-best_loss/max(analysis0['mae'],0.001))*100:.1f}%)")
    print()


def cmd_tune(args):
    # 策略: cma / de / refine
    strategy = "de"  # default: differential_evolution
    use_enhanced = False
    gens = 60

    for a in args:
        al = a.lower()
        if al in ("de", "cma", "refine"): strategy = al
        if al == "enhanced": use_enhanced = True
        try: gens = int(a)
        except ValueError: pass

    mode = "enhanced" if use_enhanced else "clone"
    entries = load_entries()
    ensure_cache(use_enhanced)

    d = get_default_params()
    _, analysis0, _ = evaluate_fast(d, use_enhanced)
    loss0 = analysis0['mae'] if analysis0 else float('inf')

    # 如果有tuned_params.json, 已保存的参数也作为候选
    best_seed = None
    if os.path.exists(TUNED_FILE):
        with open(TUNED_FILE) as f:
            prev = json.load(f)
        if prev.get("params"):
            test = dict(d)
            test.update(prev["params"])
            _, a_test, _ = evaluate_fast(test, use_enhanced)
            if a_test and a_test['mae'] < loss0:
                loss0 = a_test['mae']
                best_seed = prev["params"]

    from scripts.run_focused_tune import get_param_info, build_params_dict
    names, lowers, uppers, defaults = get_param_info(enhanced=use_enhanced)
    n_params = len(names)

    print(f"  策略: {strategy.upper()}, {mode}, {gens}次迭代, {len(entries)}张谱面")
    print(f"  默认 MAE: {loss0:.4f}")
    if best_seed:
        print(f"  从上轮最佳参数开始")
    print()

    best_mae = loss0
    best_params_list = None

    def _eval_focused(focused_params):
        """将聚焦参数合并到完整默认值中再评估。"""
        full = dict(d)
        full.update(focused_params)
        # Enhanced 模式下强制启用所有增强组件
        if use_enhanced:
            full.update({
                "use_enhanced_release": 1,
                "use_column_distance": 1,
                "use_shield": 1,
                "use_inverse": 1,
                "use_stamina": 1,
            })
        _, a, _ = evaluate_fast(full, use_enhanced)
        return a['mae'] if a else 999.0

    # ====== 方案1: Local Evolution Strategy (围绕当前最佳高斯扰动) ======
    # NOTE: 标准DE不适合此问题，因为随机全范围参数会导致MAE从0.27暴涨到4.0。
    # 改用(1+λ)-ES: 从最佳种子出发，加高斯噪声生成试验点，自适应步长。
    if strategy == "de":
        popsize = 15        # 每代试验点数量
        sigma_init = 0.04   # 初始扰动幅度 (归一化空间)
        sigma = sigma_init
        rng = np.random.RandomState(42)

        # 起始点: 归一化的默认值
        x0 = np.clip((defaults - lowers) / (uppers - lowers + 1e-12), 0.01, 0.99)
        x_best = x0.copy()
        if best_seed:
            seed_vec = np.array([(best_seed.get(n, float(dv))) for n, dv in zip(names, defaults)])
            seed_vec = np.clip((seed_vec - lowers) / (uppers - lowers + 1e-12), 0.01, 0.99)
            x_best = seed_vec

        best_fit = _eval_focused(build_params_dict(x_best, names, lowers, uppers))
        print(f"  (1+λ)-ES 搜索中 (popsize={popsize}, {gens}代 ≈ {popsize*gens}次评估, σ={sigma:.3f})...")
        t_start = time.time()
        print(f"  Gen 0: best={best_fit:.4f}")

        success_streak = 0
        for gen in range(1, gens + 1):
            improved = False
            for i in range(popsize):
                # 自适应扰动: 不同参数用不同大小的噪声
                trial = x_best + rng.normal(0, sigma, n_params)
                trial = np.clip(trial, 0.01, 0.99)

                pdict = build_params_dict(trial, names, lowers, uppers)
                tf = _eval_focused(pdict)
                if tf < best_fit:
                    best_fit = tf
                    x_best = trial.copy()
                    improved = True

            # 自适应步长: 连续成功→扩大探索; 连续失败→收敛
            if improved:
                success_streak += 1
                if success_streak >= 3:
                    sigma = min(sigma * 1.2, 0.15)
                    success_streak = 0
            else:
                success_streak = 0
                sigma *= 0.95  # 逐渐缩小

            if gen == 1 or gen % 5 == 0 or improved:
                impr = (1 - best_fit / max(loss0, 0.001)) * 100
                print(f"  Gen {gen}: best={best_fit:.4f} (改善{impr:.1f}%, σ={sigma:.3f}, {(time.time()-t_start):.0f}s)")

        elapsed = time.time() - t_start
        best_params_list = build_params_dict(x_best, names, lowers, uppers)
        print(f"  ES 完成: {elapsed:.0f}s, 每代平均 {elapsed/max(gens,1):.1f}s")

    # ====== 方案2: CMA-ES (无diagonal限制) ======
    elif strategy == "cma":
        import cma

        x0 = (defaults - lowers) / (uppers - lowers + 1e-12)
        if best_seed:
            seed = np.array([(best_seed.get(n, float(dv))) for n, dv in zip(names, defaults)])
            x0 = (seed - lowers) / (uppers - lowers + 1e-12)
            x0 = np.clip(x0, 0.05, 0.95)

        es = cma.CMAEvolutionStrategy(x0, 0.3, {
            'bounds': [0.0, 1.0], 'popsize': 20, 'maxfevals': gens * 20,
            'verbose': -1, 'seed': 42,
            # 关键修改: 去掉 CMA_diagonal, 让完整协方差矩阵发挥作用
            'tolfun': 1e-9, 'tolx': 1e-8,
        })

        best_mae = loss0
        best_x = x0.copy()
        t_start = time.time()

        for gen_idx in range(gens):
            if es.stop():
                print(f"  CMA-ES 提前停止: {es.stop()}")
                break
            sols = es.ask()
            losses = []
            for sol in sols:
                pdict = build_params_dict(sol, names, lowers, uppers)
                losses.append(_eval_focused(pdict))
            es.tell(sols, losses)
            gen_best = min(losses)
            if gen_best < best_mae:
                best_mae = gen_best
                best_x = sols[np.argmin(losses)].copy()
                elapsed = time.time() - t_start
                print(f"  Gen {gen_idx+1}: {gen_best:.4f} (NEW BEST, {(1-gen_best/loss0)*100:.1f}%, {elapsed:.0f}s)")
            if (gen_idx + 1) % 10 == 0:
                print(f"  Gen {gen_idx+1}: best={best_mae:.4f}, {(time.time()-t_start):.0f}s")

        best_params_list = build_params_dict(best_x, names, lowers, uppers)

    # ====== 方案3: 局部精调 (从 random 结果出发) ======
    elif strategy == "refine":
        try:
            with open(TUNED_FILE) as f:
                prev = json.load(f)
        except:
            prev = {"params": {}}

        from scipy.optimize import minimize

        def eval_local(x):
            pdict = build_params_dict(x, names, lowers, uppers)
            return _eval_focused(pdict)

        x0 = (defaults - lowers) / (uppers - lowers + 1e-12)
        for k, v in prev.get("params", {}).items():
            if k in names:
                idx = names.index(k)
                x0[idx] = np.clip((v - lowers[idx]) / (uppers[idx] - lowers[idx] + 1e-12), 0.01, 0.99)

        print(f"  Nelder-Mead 局部精调...")
        t_start = time.time()
        result = minimize(eval_local, x0, method='Nelder-Mead',
                          options={'maxiter': gens * 10, 'xatol': 1e-6, 'fatol': 1e-6})
        elapsed = time.time() - t_start

        best_mae = result.fun
        best_params_list = build_params_dict(result.x, names, lowers, uppers)
        print(f"  精调完成: {elapsed:.0f}s")

    # ====== 保存 ======
    if best_params_list:
        output = {"loss": best_mae, "loss_default": loss0, "mode": mode,
                  "method": f"{strategy}-es", "generations": gens, "params": best_params_list}
        with open(TUNED_FILE, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n  已保存: {TUNED_FILE}")
        print(f"  MAE: {best_mae:.4f} (默认: {loss0:.4f}, 改善 {(1-best_mae/max(loss0,0.001))*100:.1f}%)")
    print()


def cmd_compare():
    d = get_default_params()
    print("  Clone vs Enhanced 对比 (默认参数, 全量)")
    print()

    for mode in ["clone", "enhanced"]:
        _, analysis, _ = evaluate_fast(d, mode == "enhanced")
        if analysis:
            print(f"  {mode:10s}  MAE={analysis['mae']:.4f}  相关性={analysis['correlation']:.4f}  在误差内={analysis['inside_error_ratio']:.1%}  平均残差={analysis['mean_residual']:+.4f}")


def cmd_params():
    if not os.path.exists(TUNED_FILE):
        print("  tuned_params.json 不存在")
        return
    with open(TUNED_FILE) as f:
        d = json.load(f)
    print(f"  来源: {TUNED_FILE}")
    print(f"  Loss:        {d.get('loss', '?')}")
    print(f"  Loss默认:    {d.get('loss_default', '?')}")
    print(f"  改善:        {(1-d['loss']/max(d['loss_default'],0.001))*100:.1f}%" if d.get('loss') and d.get('loss_default') else "")
    print(f"  方法:        {d.get('method', d.get('mode', '?'))}")
    print()
    print(f"  关键参数 (自默认值有变动的):")
    for k, v in sorted(d.get("params", {}).items()):
        from spm_rating.config import get_default_params
        default_val = get_default_params().get(k, (None,))[0]
        if default_val is not None and abs(v - default_val) > 0.001:
            print(f"    {k}: {default_val} → {v}")
    print()


def cmd_analyze(args):
    json_file = TUNED_FILE
    for a in args:
        if a.endswith(".json"):
            json_file = a
    if not os.path.exists(json_file):
        print(f"  文件不存在: {json_file}")
        return

    with open(json_file) as f:
        d = json.load(f)
    params = d.get("params", d)
    use_enhanced = d.get("mode", "clone") == "enhanced"

    _, analysis, _ = evaluate_fast(params, use_enhanced)
    if analysis is None:
        print("  评估失败")
        return

    print(f"  全量评估: {analysis['n_valid']} 张")
    print(f"  MAE: {analysis['mae']:.4f}, 相关性: {analysis['correlation']:.4f}")
    print(f"  RMSE: {analysis['rmse']:.4f}, 在误差内: {analysis['inside_error_ratio']:.1%}")
    print()
    print(f"  按难度带:")
    for band, info in sorted(analysis.get("per_band", {}).items()):
        print(f"    SR {band:8s}: n={info['count']:3d}  err={info['mean_residual']:+.3f}  mae={info['mae']:.3f}")
    print()
    print(f"  按标签:")
    for tag, info in sorted(analysis.get("per_tag", {}).items(), key=lambda x: -x[1]['count'])[:10]:
        print(f"    {tag:25s}: n={info['count']:3d}  err={info['mean_residual']:+.3f}  mae={info['mae']:.3f}")
    print()


def cmd_recache():
    print("  强制重新预计算...")
    ensure_cache(use_enhanced=False, force=True)
    print("  完成")
    print()


# ===================== 主循环 =====================

def print_help():
    print("  status      查看数据、缓存、调参状态")
    print("  evaluate [mode] [N]  评估 (clone/enhanced, 可选前N张)")
    print("  random [mode] [N]    随机搜索调参 (默认20次, 适合快速摸底)")
    print("  tune de [gens]       (1+λ)-ES局部进化策略 (推荐, 默认60代)")
    print("  tune cma [gens]      CMA-ES搜索 (默认60代)")
    print("  tune refine [gens]   从上次结果局部精调 (默认60步)")
    print("  compare     clone vs enhanced 默认参数对比")
    print("  params      查看 tuned_params.json 内容")
    print("  analyze [f] 分析评估结果（按标签/难度带）")
    print("  recache     强制重新预计算缓存")
    print("  help        显示此帮助")
    print("  exit        退出")
    print()


def main():
    print("  ╔═══════════════════════════════════════╗")
    print("  ║  SPM Rating — 调参终端             ║")
    print("  ╚═══════════════════════════════════════╝")
    print()
    print_help()
    cmd_status()

    while True:
        try:
            raw = input("  (tune) > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见!")
            break

        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0].lower()
        args = parts[1:]

        dispatch = {
            "exit": lambda: sys.exit(0),
            "quit": lambda: sys.exit(0),
            "q":    lambda: sys.exit(0),
            "help": print_help,
            "status": cmd_status,
            "evaluate": lambda: cmd_evaluate(args),
            "eval": lambda: cmd_evaluate(args),
            "random": lambda: cmd_random(args),
            "tune": lambda: cmd_tune(args),
            "compare": cmd_compare,
            "params": cmd_params,
            "analyze": lambda: cmd_analyze(args),
            "recache": cmd_recache,
        }

        if cmd in dispatch:
            try:
                dispatch[cmd]()
            except Exception as e:
                import traceback
                print(f"  [错误] {e}")
                traceback.print_exc()
                print()
        else:
            print(f"  未知: {cmd}, 输入 help 查看可用命令")
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
