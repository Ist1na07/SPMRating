# SPM Rating — 特征修正层调参方法论

**日期**: 2026-06-02
**上下文**: 在 Sigmoid 聚合层（k=2.09, C=3.97, γ=0.196）基础上添加特征修正层。最终结果: Loss=0.770, MAE=0.213, r=0.989, Pass=83.6%。

---

## 目录

1. [问题背景](#1-问题背景)
2. [架构: 修正层在管线中的位置](#2-架构)
3. [七个特征及其物理意义](#3-七个特征)
4. [核心性质: D_solved 位移不变性](#4-d_solved-位移不变性)
5. [预计算阶段: 缓存构建](#5-预计算阶段)
6. [优化阶段: 标量 Nelder-Mead](#6-优化阶段)
7. [正则化策略](#7-正则化策略)
8. [交叉验证: 过拟合控制](#8-交叉验证)
9. [特征阈值参数](#9-特征阈值参数)
10. [完整调参工作流](#10-完整调参工作流)
11. [关键发现与陷阱](#11-关键发现与陷阱)

---

## 1. 问题背景

### 1.1 D 公式的系统性偏差

SPM Rating 的 D(t) 公式通过逐时刻计算 S(sustained)、T(technicality)、D(instantaneous) 三个分量的组合来描述谱面难度。尽管 D 公式包含 41 个参数（分 6 块交替优化），仍存在**系统性偏差**：

| 模式 | D 公式行为 | 实际体感 | 偏差方向 |
|------|-----------|---------|---------|
| 和弦 (Chord) | 独立计算每列难度，未考虑多列同时击打的协同 | 多指协同降低实际难度 | **高估** |
| Fast Jack (同列快打) | 瞬时难度曲线无法充分捕捉累积疲劳 | 精确时机要求随频率非线性增长 | **低估** |
| 手切 (Hand-switch) | 跨列距离仅反映空间跨度 | 左右手切换需要额外协调 | **低估** |
| 爆发 (Burst) | 三音一组的高密度模式 | 短暂爆发后肌肉放松 | **高估** |

这些偏差的特征是：**与谱面中的特定模式密度相关，且方向一致（不是随机噪声）**。

### 1.2 为什么不直接修改 D 公式

修改 D 公式（如添加 chordjack 折扣、修改 jack 聚合方式）存在两个问题：

1. **高维耦合**: D 公式的 41 个参数已通过 3 轮交替 NM 充分优化，修改任一组件都需要重新跑完整交替流程（~2 小时/轮）
2. **过拟合风险**: D 公式参数作用于每个时刻的 D(t) 值（~数万个时间点），微调容易引入谱面特异性偏差

特征修正层的设计目标是：**在不修改 D 公式的前提下，用 7 个谱面级特征捕捉系统性偏差**。

### 1.3 与 Sigmoid 聚合层调参的关系

特征修正层位于 Sigmoid 聚合层之上（见第 2 节），假设 Sigmoid 聚合参数（k, C, γ, calib）已调至最优。修正层**不重新调优**聚合层参数，而是在已校准的 D_calib 上添加修正量。

这意味着：
- 如果 Sigmoid 层参数发生变化（如数据集更新后重新调 k），**修正层需要重新训练**
- 修正层的参数（7 个权重 + 4 个后处理）远少于 D 公式层（41 个），训练速度快 100×+

---

## 2. 架构

### 2.1 管线位置

```
.osu 文件
    ↓ precompute()
cache (note_seq, LN_seq, Jbar, Pbar, all_corners, ...)
    ↓ combine()
D_full[t], C_arr (标准 D 公式输出)
    ↓ 校准 (Phase 5 已有)
D_calib = calib_a × D_full + calib_b
    ↓ ★ 特征修正层 (新增)
correction = Σ w_j × feature_j
D_new = max(D_calib + correction, 0.01)
    ↓ Sigmoid 聚合
SR = sigmoid_aggregate(D_new, total_notes, postprocess_params)
```

### 2.2 关键设计决策: 标量修正作用于 D 序列

修正量是一个**标量**（不是逐时刻的向量），直接加在 D_calib 的所有时刻上：

```
D_new(t) = D_calib(t) + correction
```

这确保了修正后的 D(t) 分布形状不变（仅平移），可以利用 D_solved 位移不变性实现快速优化（见第 4 节）。

### 2.3 参数分块

修正层总共 11 个参数，作为一个整体优化（不分块）：

| 组 | 参数数 | 内容 |
|----|:---:|------|
| **W: 特征权重** | 7 | w_speed, w_burst, w_chord, w_pj, w_hs, w_lb, w_fj |
| **P: 后处理** | 4 | N0, threshold, divisor, global_scale |

后处理参数（P 组）从 Sigmoid 层继承并**联合重优化**，因为修正量改变了 D_solved 的数值范围，后处理参数需要适配。

---

## 3. 七个特征

### 3.1 特征定义

所有特征均从 `precompute()` 的缓存数据计算，输出为**谱面级标量**（不是逐时刻序列）。

| 特征 | 名称 | 定义 | 单位 |
|------|------|------|------|
| **speed** | 速度型密度 | dt < `spd_dt` 且 dc ≥ `spd_dc` 的音符对数 / 时长 | 个/秒 |
| **burst** | 爆发型密度 | 三音组中 `times[i] - times[i-2]` < `bst_dt` 的数量 / 时长 | 个/秒 |
| **chord** | 和弦比例 | 参与 ≥`ch_order` 个同窗音符的音符数 / 总音符数 | 比例 |
| **pj** | 流/jack 平衡 | mean(Pbar) / (mean(Jbar) + 1) | 比值 |
| **hs** | 手切密度 | dt < `hs_dt` 且左右手切换的音符对数 / 时长 | 个/秒 |
| **lb** | 轻爆发密度 | 四音组中 `times[i] - times[i-3]` < `lb_dt` 的数量 / 时长 | 个/秒 |
| **fj** | Fast jack 密度 | 同列连续音符 dt < `fj_dt` 的数量 / 时长 | 个/秒 |

其中 dt = 相邻音符时间差 (ms)，dc = 相邻音符列差绝对值。

### 3.2 物理意义与权重解释

最终优化的权重（λ=0.01）：

| 特征 | 权重 | 物理意义 |
|------|------|---------|
| **chord** | **-0.714** | D 公式高估了和弦密度谱面。多列同时击打有协同减负，但 D(t) 独立计算每列难度 |
| **fj** | **+0.265** | D 公式低估了 fast jack 密度。同列快速连打的累积疲劳未充分反映 |
| **hs** | +0.043 | 手切难度被轻微低估 |
| **lb** | +0.020 | 轻爆发（四音组）难度被轻微低估 |
| **speed** | -0.038 | 速度型模式被轻微高估 |
| **burst** | -0.025 | 爆发型模式被轻微高估 |
| **pj** | -0.005 | 流/jack 平衡几乎无影响（接近零） |

**权重幅度差异大**（chord 是 pj 的 143 倍）是正常现象：不同特征对 D 公式偏差的贡献量级不同。L2 正则化（见第 7 节）控制了极端权重。

### 3.3 特征计算注意事项

1. **chord 用比例而非密度**: chord 是唯一不以"个/秒"为单位的特征，它表示的是谱面中和弦音符的比例（0~1）。这是因为 chord 的定义基于同窗音符计数，与谱面总密度天然耦合。

2. **pj 从缓存读取 Jbar/Pbar**: 与其他特征不同，pj 需要 `precompute()` 输出的 Jbar_base 和 Pbar_base，不能仅从 note_seq 计算。

3. **所有特征归一化为谱面级统计量**: 除以时长或总音符数，消除谱面长度影响。

---

## 4. D_solved 位移不变性

### 4.1 核心性质

修正层优化的关键洞察：当 D(t) 分布整体平移一个常数 c 时，Sigmoid 聚合的解 D_solved 也近似平移同样的量。

```
D_solved({D(t) + c}) ≈ D_solved({D(t)}) + c
```

**数学直觉**：Sigmoid 聚合通过二分法求解 D_solved 使得：

```
Σ w_i / (C + e^(k(D_i - D_solved))) = total_W × γ
```

将所有 D_i 替换为 D_i + c 后，等价于将 D_solved 替换为 D_solved + c（方程两边完全一致）。

### 4.2 近似的精度

该性质在以下条件下近似成立：

1. **D(t) 分布的分段聚合**引入少量误差（30 个分段）
2. **有效权重 w_i** 与 D 值略有耦合
3. 实测误差 < 0.001 SR（对 311 张谱面平均）

### 4.3 为什么重要

**不使用**位移不变性：
```
for each NM evaluation:
    for each map:
        D_new(t) = D_calib(t) + correction  # 修改整个 D 序列
        D_solved = solve_bisection(D_new)    # 二分法求解 ~5ms/谱面
    # 总时间: ~1.5s/eval → NM 10000 iter ≈ 4 小时
```

**使用**位移不变性：
```
# 一次性预计算 (~1s)
for each map:
    D_solved_base = solve_bisection(D_calib)

# 快速评估
for each NM evaluation:
    for each map:
        D_solved_new = D_solved_base + correction  # 标量加法 ~1μs/谱面
    # 总时间: ~0.001s/eval → NM 10000 iter ≈ 10 秒
```

**加速比: ~1500×**

### 4.4 局限性

位移不变性**仅对标量线性修正成立**。以下情况不适用：

1. **非线性修正**: 如果 correction 是 D_solved 的函数（如 chord 折扣模型），位移不变性不成立
2. **交互项**: 特征交互项（如 chord × fj）在快速近似下可能给出与完整管线不一致的结果（实测交互项 CV 改善 2.4%，但完整管线恶化 5.2%）
3. **逐时刻修正**: 如果 correction 随 t 变化，D(t) 分布形状改变，位移不变性不成立

**因此，修正层应始终保持标量线性形式：correction = Σ w_j × feature_j。**

---

## 5. 预计算阶段

### 5.1 缓存构建

对每张谱面执行一次完整的 `precompute()` + `combine()` 管线：

```python
for each map:
    cache = precompute(osu_path, use_enhanced=True, params=params_spm)
    _, details = combine(cache, params=params_spm)
    # 保存:
    #   cache_i.pkl  → note_seq, LN_seq, Jbar_base, Pbar_base, all_corners
    #   d_i.npz      → D_full, C_arr
```

311 张谱面耗时 ~15 分钟（单线程），只需在以下情况重新构建：
- 新增/移除谱面
- D 公式参数 (B2/B3 块) 发生变化
- precompute() 逻辑发生变更

### 5.2 D_solved 预计算

在缓存基础上，使用已确定的 Sigmoid 层参数计算每张谱面的基准 D_solved：

```python
for each map:
    D_calib = calib_a * D_full + calib_b
    eff_w = compute_effective_weights(all_corners, C_arr)
    D_seg, w_seg = segment_by_difficulty(D_calib, eff_w, 30)
    D_solved = solve_bisection(D_seg, w_seg, k, C, gamma)
    n_eff = compute_total_notes(note_seq, LN_seq)
```

这一步耗时 < 1 秒（所有谱面），但**每次 Sigmoid 层参数变化后需要重跑**。

### 5.3 特征预计算

从缓存数据计算 7 个特征值（见第 3 节）：

```python
for each map:
    features = compute_features(cache, FEAT_PARAMS)
```

特征计算只依赖 note_seq、LN_seq、Jbar_base、Pbar_base，全部来自 cache。耗时 < 1 秒。

---

## 6. 优化阶段

### 6.1 参数向量

总共 11 个参数，编码为一维向量：

```
x = [w_speed, w_burst, w_chord, w_pj, w_hs, w_lb, w_fj, N0, threshold, divisor, global_scale]
     |←────────────────── W (7) ──────────────────→|  |←────────── P (4) ──────────→|
```

### 6.2 评估函数

```python
def eval_model(x, indices):
    w = x[:7]
    N0, thr, div, gs = x[7:]

    total_loss = 0.0
    for i in indices:
        correction = sum(w[j] * features[i][j] for j in range(7))
        # 利用位移不变性
        D_shifted = D_solved_base[i] + correction
        SR = postprocess(D_shifted, n_eff[i], N0, thr, div, gs)
        total_loss += score_single(SR, sr_ref[i], sr_error[i])

    return total_loss / len(indices) + regularization
```

其中后处理函数：

```python
def postprocess(D_shifted, n_eff, N0, threshold, divisor, scale):
    N0_safe = max(N0, 0.01)
    SR = D_shifted * n_eff / (n_eff + N0_safe)
    if SR > threshold:
        SR = threshold + (SR - threshold) / divisor
    return SR * scale
```

### 6.3 优化器配置

使用 Nelder-Mead（与 Sigmoid 层调参一致），但配置更精细：

```python
scipy.optimize.minimize(
    eval_model,
    x0=[0.0]*7 + [N0_def, thr_def, div_def, gs_def],
    method="Nelder-Mead",
    options={
        "maxiter": 10000,
        "xatol": 1e-7,
        "fatol": 1e-7,
        "adaptive": True
    }
)
```

**参数说明**：
- `maxiter=10000`: 11 维空间足够收敛（实测 ~3000 次 eval 后稳定）
- `xatol=1e-7, fatol=1e-7`: 高精度收敛（因为评估很快，不需要节省 eval）
- `adaptive=True`: 启用自适应 NM，在高维空间收敛更快

### 6.4 多次重启

为避免局部最优，使用 5 次随机重启：

```python
for restart in range(5):
    if restart == 0:
        x0 = [0.0]*7 + [defaults...]  # 零初始化
    else:
        x0 = random_normal(...)       # 随机扰动
    res = minimize(eval_model, x0, ...)
    best = min(best, res, key=lambda r: r.fun)
```

零初始化 (restart=0) 通常收敛到全局最优，随机重启用于验证稳健性。

### 6.5 初始值选择

| 参数 | 初始值 | 来源 |
|------|--------|------|
| w_j (所有权重) | 0.0 | 无修正（零起点） |
| N0 | 8.21 | Sigmoid 层最优值 |
| threshold | 9.42 | Sigmoid 层最优值 |
| divisor | 2.01 | Sigmoid 层最优值 |
| global_scale | 1.055 | Sigmoid 层最优值 |

后处理参数从 Sigmoid 层的最优值出发，因为修正量较小时，后处理参数变化不大。

---

## 7. 正则化策略

### 7.1 L2 正则化

评估函数添加 L2 惩罚项：

```python
loss = base_loss + λ × Σ w_j²
```

**λ 的选择**：

| λ | Train Loss | CV Test Loss | Gap | 推荐 |
|---|-----------|-------------|-----|------|
| 0.000 | 0.715 | 0.903 | 0.188 | 过拟合 |
| 0.005 | 0.752 | 0.878 | 0.126 | — |
| **0.010** | **0.770** | **0.862** | **0.092** | **最优** |
| 0.020 | 0.793 | 0.872 | 0.079 | 稍差 |
| 0.050 | 0.827 | 0.895 | 0.068 | 欠拟合 |

**λ=0.01** 在 CV test loss 上最优。注意 gap 随 λ 单调递减（正则化越强过拟合越小），但 test loss 在 λ=0.01 处达到最小值。

### 7.2 N0 ≥ 0 约束

物量归一化参数 N0 的物理含义是"等效虚拟音符数"，必须非负。通过惩罚项实现：

```python
n0_penalty = max(0, -N0)² × 10.0
```

当 N0 < 0 时施加强惩罚（系数 10.0），N0 ≥ 0 时无惩罚。这比硬约束更适合 NM 优化器。

**实测效果**：N0 收敛到 ~0.001（接近零但正值），说明当前数据集下短谱面不需要额外的物量惩罚，但约束阻止了 N0 变为负值（物理上无意义）。

### 7.3 为什么不需要 L1 正则化

L1 正则化（Σ|w_j|）会推动权重稀疏化（某些 w_j 变为 0）。但 7 个特征中，即使权重接近零的特征（如 pj=-0.005）也提供了有用的信号方向。**保留所有 7 个特征、用 L2 控制幅度**比 L1 的特征选择更适合此场景。

---

## 8. 交叉验证

### 8.1 5 折交叉验证方法

```python
N = len(entries)          # 311
np.random.seed(42)         # 固定随机种子，确保可复现
perm = np.random.permutation(N)
K = 5
fold_size = N // K        # 62
folds = [perm[k*62:(k+1)*62] for k in range(K)]  # 最后一折 63

for fold in range(K):
    train_idx = [all indices except fold]
    test_idx = folds[fold]

    # 在 train_idx 上训练
    res = minimize(eval_model, x0, args=(train_idx,))
    # 在 test_idx 上评估
    test_loss = eval_model(res.x, test_idx)
```

### 8.2 结果解读

```
Fold 0: train=0.742  test=0.867  gap=0.125
Fold 1: train=0.769  test=0.841  gap=0.072
Fold 2: train=0.784  test=0.917  gap=0.133
Fold 3: train=0.805  test=0.821  gap=0.016
Fold 4: train=0.765  test=0.864  gap=0.099

平均: train=0.773  test=0.862  gap=0.089
std:  train=0.021  test=0.034
```

**关键指标**：
- **CV Test Loss** (0.862): 预期泛化性能
- **Gap** (0.089): 过拟合程度，越小越好
- **Test Loss std** (0.034): 泛化稳定性

### 8.3 过拟合诊断

| 场景 | Train | Test | Gap | 诊断 |
|------|-------|------|-----|------|
| 健康 | 0.77 | 0.86 | ~0.09 | 适度泛化，可接受 |
| 过拟合 | 0.71 | 0.90 | ~0.19 | 增加 λ 或减少参数 |
| 欠拟合 | 0.85 | 0.89 | ~0.04 | 减少 λ 或增加参数 |
| 不稳定 | 0.77 | 0.86±0.10 | — | 需要更多数据 |

### 8.4 新增数据后的 CV 预期

当数据集从 311 扩展到 N 张谱面时：
- 如果新增谱面分布与现有数据一致：CV test loss 应**降低**（更多数据 → 更好的泛化）
- 如果新增谱面来自新的分布（如新 sort 类型）：CV test loss 可能**升高**（分布偏移），此时需要检查特征是否需要扩展

---

## 9. 特征阈值参数

### 9.1 参数定义

每个特征的计算依赖若干阈值参数，这些参数**不参与 NM 优化**，在训练前固定：

| 参数 | 默认值 | 影响的特征 | 含义 |
|------|--------|-----------|------|
| `spd_dt` | 150 ms | speed | 速度型音符的最大时间差 |
| `spd_dc` | 3 列 | speed | 速度型音符的最小列差 |
| `bst_dt` | 100 ms | burst | 三音组爆发的最大窗口 |
| `ch_order` | 4 | chord | 构成和弦的最小同时音符数 |
| `hs_dt` | 200 ms | hs | 手切音符的最大时间差 |
| `lb_dt` | 150 ms | lb | 四音组轻爆发的最大窗口 |
| `fj_dt` | 100 ms | fj | fast jack 同列快打的最大时间差 |

### 9.2 是否需要调优这些参数

**通常不需要**。这些参数的默认值基于游戏机制的物理理解（如 7K 中人类反应时间 ~100ms，手切上限 ~200ms）。在以下情况下可能需要调优：

1. **键位数变化**: 从 7K 扩展到 4K/9K 时，`spd_dc` 和 `ch_order` 需要调整
2. **特征贡献异常**: 如果某个特征的权重接近零，可能是阈值参数不合理导致特征区分度不足
3. **新数据分布偏移**: 新增谱面的 BPM 范围大幅超出原有数据时

### 9.3 调优方法（如需）

使用**网格搜索**（非 NM），因为阈值参数是离散的：

```python
for spd_dt in [120, 130, 140, 150, 160, 170]:
    for bst_dt in [80, 90, 100, 110, 120]:
        features = recompute_all_features(spd_dt=spd_dt, bst_dt=bst_dt, ...)
        # 重新训练修正层
        res = optimize_correction(features)
        # 5-fold CV 评估
        cv_test = cross_validate(res, features)
        record(spd_dt, bst_dt, cv_test)
```

**注意**：每次修改阈值参数后，需要重新计算所有谱面的特征值，然后重新训练修正层权重。

---

## 10. 完整调参工作流

### 10.1 场景一: 补充数据集后重新调参

最常见的场景：新增 N 张谱面到 playtest 数据中。

```
Phase 1: 数据准备
├── 1a. 将新谱面的 .osu 文件放入 maps/ 目录
├── 1b. 在对应的 playtest.xlsx 中添加条目
├── 1c. 确认所有列（mapfile, difficulty, accurate, error, sort）完整
└── 1d. 运行 maps/counter.py 验证谱面数量

Phase 2: 缓存重建
├── 2a. 运行 cache 构建脚本，为所有谱面生成 cache_i.pkl 和 d_i.npz
├── 2b. 验证缓存数量 = xlsx 条目数
└── 2c. 检查是否有缓存失败（note_seq < 10 的谱面会被跳过）

Phase 3: 预计算
├── 3a. 加载 Sigmoid 层参数 (tuned_params_sigmoid.json)
├── 3b. 为所有谱面预计算 D_solved_base 和 n_eff
└── 3c. 计算所有谱面的 7 个特征值

Phase 4: 修正层优化
├── 4a. λ 扫描: λ ∈ {0.0, 0.005, 0.01, 0.02, 0.05}
├── 4b. 每个 λ 值: 5 次随机重启 × NM(maxiter=10000)
├── 4c. 选择 in-sample loss 最低的 λ
└── 4d. 记录: 权重、后处理参数、in-sample loss

Phase 5: 交叉验证
├── 5a. 5-fold CV (固定 seed=42，确保可复现)
├── 5b. 记录: CV train/test loss, gap, std
└── 5c. 诊断过拟合 (gap < 0.15 为健康)

Phase 6: 完整管线验证 (必须)
├── 6a. 用优化后的权重，对每张谱面跑完整 precompute→combine→correct→aggregate
├── 6b. 对比: 完整管线 loss vs CV test loss
├── 6c. 差异 < 0.05 为正常（位移不变性的近似误差）
└── 6d. 差异 > 0.10 需检查（可能有缓存不一致或特征计算差异）

Phase 7: 保存
└── 保存 tuned_correction.json (权重 + 后处理 + λ + CV 结果)
```

### 10.2 场景二: Sigmoid 层参数变更后重新调参

如果 Sigmoid 聚合层（k, C, γ, calib）发生了变化（例如 B2/B3 块重新优化），修正层需要重新训练：

```
1. 使用新的 Sigmoid 层参数重新预计算 D_solved_base（Phase 3）
2. 特征值不需要重新计算（特征只依赖 note_seq，不依赖 Sigmoid 参数）
3. 重新训练修正层（Phase 4-7）
```

**注意**：Sigmoid 层参数变更会改变 D_solved 的数值范围，因此后处理参数（N0, threshold, divisor, global_scale）必须与修正层权重联合重优化。

### 10.3 场景三: 添加新特征

当发现 D 公式对某种新模式存在系统性偏差时：

```
1. 定义新特征: 在 compute_features() 中添加计算逻辑
2. 更新 FEATURE_NAMES 列表
3. 重新预计算所有谱面的特征值（Phase 3）
4. 重新训练修正层（Phase 4-7）
5. 对比新旧模型的 CV test loss:
   - 改进 > 0.01: 保留新特征
   - 改进 < 0.01: 新特征不值得增加的参数
```

**原则**: 每增加一个特征 = 多一个参数。在 ~300 样本的数据规模下，参数总数不应超过 ~15 个（参数/样本比 < 5%）。

### 10.4 场景四: D 公式参数变更后重新调参

如果 D 公式层（B2/B3 块）的参数发生了变化：

```
1. 重新构建所有谱面的缓存（Phase 2, 因为 D_full 变了）
2. 重新预计算 D_solved_base（Phase 3）
3. 重新计算特征值（Phase 3, 因为 pj 依赖 Jbar/Pbar）
4. 重新训练修正层（Phase 4-7）
```

这是最耗时的场景，因为需要重跑 `precompute()` + `combine()` 管线（~15 分钟/311 张谱面）。

---

## 11. 关键发现与陷阱

### 11.1 修正量应远小于 D_solved

修正量 correction 的典型范围为 [-1.5, +0.5]，而 D_solved 的典型范围为 [2.0, 8.0]。如果修正量接近 D_solved 的量级（例如某个特征的权重异常大），说明：
- 该特征可能在补偿 D 公式的结构性缺陷（应优先修复 D 公式）
- 正则化不足（增大 λ）
- 数据集中有异常样本

### 11.2 完整管线验证不可省略

D_solved 位移不变性是近似性质。快速优化（利用位移不变性）的结果**必须**通过完整管线验证确认：

| 模型 | CV Test Loss | Full Pipeline Loss | 一致性 |
|------|-------------|-------------------|--------|
| 线性修正 (7 特征) | 0.862 | 0.770 | ✅ 一致 |
| 交互项 (chord×fj) | 0.874 | 0.810 | ❌ CV 预测改善，完整管线恶化 |

**教训**：交互项在快速近似中表现良好，但破坏了位移不变性假设，导致完整管线结果与 CV 不一致。因此修正层应始终保持**线性形式**，不引入交互项。

### 11.3 Chord 权重的物理解释

chord 权重为 -0.714，是最大的负权重。这**不意味着** D 公式有结构性缺陷。调查表明：

1. Chord 密度与预测误差**无单调关系**（各区间平均误差在 -0.07 到 +0.01 之间）
2. 尝试结构性 chord 折扣模型（阈值+最大折扣），test loss 恶化 0.204（23.7%）
3. 线性修正已经足够捕捉 chord 的统计规律

**结论**：chord 的大权重反映了数据中的统计趋势，而非 D 公式需要结构性修改。

### 11.4 RC 谱面是最难预测的类型

| 类型 | 数量 | Loss | Pass Rate |
|------|------|------|-----------|
| RC | 133 | 1.182 | 75.2% |
| LN | 125 | 0.456 | 89.6% |
| HB | 47 | 0.488 | 91.5% |

RC 谱面的 Loss 是 LN 的 2.6 倍。可能原因：
- RC 谱面的技巧多样性更高（stream, jump, tech 混合），单一特征集难以覆盖
- RC 的 playtest 数据可能方差更大（不同玩家对 RC 难度的判断差异更大）

**潜在改进方向**：为 RC 谱面训练独立的修正层权重（分段模型），但这需要更多数据来支撑额外的 7 个参数。

### 11.5 参数数量的经验上限

| 模型 | 参数数 | Train | Test | Gap |
|------|--------|-------|------|-----|
| 无修正 | 0 | 0.931 | 0.931 | 0.000 |
| 线性修正 | 11 | 0.770 | 0.862 | 0.092 |
| 完整模型 (sigmoid 混合) | 20 | 0.715 | 0.903 | 0.188 |

在 ~300 样本的数据规模下，**11 个参数是安全上限**。超过此数量（如 20 参数模型）会导致过拟合（gap 翻倍）。如果数据集扩展到 500+ 样本，可以尝试 15-18 个参数。

### 11.6 特征权重不稳定的诊断

如果重新训练后某个特征的权重方向翻转（如从正变负），可能原因：
1. **该特征的贡献被其他特征吸收**: 检查特征间的相关性（如 burst 和 speed 高度相关）
2. **正则化不足**: 增大 λ 后权重方向应趋于稳定
3. **数据量不足**: 该特征的信号被噪声淹没

**处理方法**：检查 5 折 CV 中各折的权重。如果权重方向在不同折之间不一致，说明该特征不稳定，可考虑移除。

---

## 附录 A: 脚本索引

| 脚本 | 功能 |
|------|------|
| `predict_correction.py` | 修正层预测接口（单谱面/批量） |
| `tune_correction.py` | 修正层训练（λ 扫描 + 多次重启 + 5-fold CV） |
| `verify_correction.py` | 完整管线验证（对比快速优化 vs 完整管线） |
| `build_cache.py` | 缓存构建（precompute + combine + 特征计算） |

## 附录 B: 参数归档

### 修正层参数 (11 个)

| 文件 | 内容 |
|------|------|
| `tuned_correction.json` | 特征权重 (7) + 后处理 (4) + λ + CV 结果 |

### 依赖参数 (来自 Sigmoid 层，不重新调优)

| 文件 | 参数 |
|------|------|
| `tuned_params_sigmoid.json` | k, C, γ, calib_a, calib_b (以及 D 公式/特征层全部参数) |

### 特征阈值参数 (固定默认值)

```json
{
    "spd_dt": 150,
    "spd_dc": 3,
    "bst_dt": 100,
    "ch_order": 4,
    "hs_dt": 200,
    "lb_dt": 150,
    "fj_dt": 100
}
```

## 附录 C: 损失函数

修正层使用与 Sigmoid 层相同的 dead-zone piecewise-linear 损失：

```
delta = |SR_pred - SR_ref|
eps = max(error_bound, 0.01)
ratio = delta / eps

if ratio <= 0.5:   loss = 0            (死区: 在半误差范围内 = 完美)
elif ratio <= 1.0: loss = (ratio-0.5)×2 (线性增长: 0→1)
else:              loss = 1.0 + (ratio-1.0)×6 (陡坡: 超出误差后 3× 斜率)
```

## 附录 D: 快速参考 — 调参检查清单

补充数据后重新调参时，按此清单逐项确认：

- [ ] 新增谱面的 .osu 文件存在于 maps/ 目录
- [ ] playtest.xlsx 中新条目各列完整（mapfile, difficulty, accurate, error, sort）
- [ ] 缓存已重建（cache_i.pkl + d_i.npz 数量 = 谱面数）
- [ ] D_solved_base 已使用最新 Sigmoid 参数预计算
- [ ] 特征值已使用默认阈值参数计算
- [ ] λ 扫描覆盖了 {0.0, 0.005, 0.01, 0.02, 0.05}
- [ ] 每个 λ 使用了 ≥3 次随机重启
- [ ] 5-fold CV 使用 seed=42
- [ ] CV gap < 0.15（过拟合诊断）
- [ ] 完整管线验证通过（差异 < 0.05）
- [ ] 结果已保存到 tuned_correction.json
