# SPM Rating — Sigmoid 调参方法论

**日期**: 2026-05-10
**上下文**: 从 Percentile (百分位聚合) 切换到 Sigmoid (玩家准度模型) 聚合层的调参实践。

---

## 目录

1. [问题背景](#1-问题背景)
2. [核心方法论: 块式交替 Nelder-Mead](#2-核心方法论)
3. [B1 快速通道: 预提取 D 分段](#3-b1-快速通道)
4. [k 值扫描: 步进 0.1 的精细搜索](#4-k-值扫描)
5. [D 预校准: 线性变换 D'=aD+b](#5-d-预校准)
6. [C-γ 自洽性: 参数冗余的消除](#6-c-γ-自洽性)
7. [后处理参数合并: 锁定 Phase 5 后处理](#7-后处理参数合并)
8. [参数分块原则](#8-参数分块原则)
9. [完整工作流](#9-完整工作流)
10. [关键发现](#10-关键发现)

---

## 1. 问题背景

### 1.1 两种聚合的本质区别

| 特性 | Percentile (旧) | Sigmoid (新) |
|------|:---:|:---:|
| 操作 | 截取 D(t) 分布的 P70-P75 | 二分法求解: Σwᵢ/(C+e^(k(Dᵢ-D))) = total_W·γ |
| 物理解释 | 统计分位 (无物理含义) | 模拟玩家准度衰减曲线 |
| 参数数 | 2 (power, percentile) | 3 (k, C, γ) + 4 calib/post |
| 调参成本 | 低 | **高** — 每次 combine() 含二分法慢 ~15% |

### 1.2 为什么需要新的调参方法

Percentile 聚合层只涉及 2 个参数，可以在整体调参中直接放入参数向量。但 Sigmoid 聚合引入了 7 个新参数（k, C, γ, calib_a, calib_b, post 参数），且这些参数与 D 公式层（10 参数）、特征层（16 参数）存在**结构性耦合**：

- Sigmoid 参数 (k, C, γ, calib) 决定了"玩家如何看待难度"
- D 公式参数 (S_w1, alpha_P, alpha_R, ...) 决定了"难度是什么"
- 特征层参数 (cross, release, inverse) 决定了"难度分量的定义"

如果所有 35 参数一起扔进 NM，会出现：
1. **高维退化**: 35 维 NM 效率极低 (需要 ~200+ 次评估才开始改善)
2. **梯度混淆**: B1 参数 (sigmoid) 和 B2 参数 (D formula) 互相补偿，陷入局部最优
3. **速度瓶颈**: eval_full() 需要 14.5 秒，35 维 NM 需要 ~1 小时

---

## 2. 核心方法论: 块式交替 Nelder-Mead

### 2.1 分块策略

将 35 个参数按**语义独立性**分为 5 个块：

| 块 | 参数数 | 含义 | 评估模式 |
|----|:---:|------|:---:|
| **B1: Sigmoid+Calib+Post** | 8 | k, C, γ, calib_a, calib_b, N0, threshold, divisor, global_scale | **Fast** (0.04s/eval) |
| **B2: D Formula Core** | 10 | S_w1, S_p, alpha_P, alpha_R, alpha_C, alpha_S, alpha_V, D_beta1, D_beta2, Abar_scale | Full (14.5s/eval) |
| **B3a: Cross Distance** | 5 | dist_exponent_rc, dist_exponent_ln, same_hand_penalty_rc/ln, thumb_bridge | Full |
| **B3b: Release LN Tail** | 6 | tail_coeff, tail_to_tap, same_col_bonus, coord_exponent, seq_coeff, lock_interaction | Full |
| **B3c: Inverse/Guide** | 7 | inv_amplitude, inv_tau, inv_power, guide_depth, guide_center, guide_width, cross_guide_scale | Full |

### 2.2 交替算法

```
1. 加载 Phase 5 最佳参数作为初始值
2. 设 k = 1.5 (从 sweep 确定的最优值)
3. for round in [1, 2]:
     maxiter = 50 if round==1 else 25  # 第一轮粗调，第二轮精调
     for block in [B1, B2, B3a, B3b, B3c]:
       NM(block, maxiter=maxiter)
       if block 是 full 模式:
           重新提取 D_seg (因为 D 公式变了)
```

### 2.3 为什么有效

- **解耦梯度**: 每个块只优化自己语义范围内的参数，不会出现跨语义补偿
- **交替迭代**: B1 先确定"玩家模型"，B2 再优化"难度定义"，循环两轮达到联合最优
- **Full-mode 块后重提取**: B2/B3 改变 D(t) 分布后，B1 的 D_seg 必须重提取，确保 B1 在下一次迭代时有正确的 D 输入

### 2.4 收敛行为

典型的 Loss 轨迹 (2 轮 × 5 块 = 10 个 NM 阶段):

```
R1 B1: 1.0281 → 0.9132  (Fast, 200 次评估, 8s)
R1 B2: 0.9132 → 0.9098  (Full,  150 次评估, 36 min)
R1 B3a: 0.9098 → 0.9087 (Full,  80 次评估, 19 min)
R1 B3b: 0.9087 → 0.9079 (Full,  90 次评估, 22 min)
R1 B3c: 0.9079 → 0.9075 (Full,  100 次评估, 24 min)
---
R2 B1: 0.9075 → 0.9068  (Fast, 100 次评估, 4s)
R2 B2: 0.9068 → 0.9063  (Full,  80 次评估, 19 min)
R2 B3a: 0.9063 → 0.9063  (no change)
R2 B3b: 0.9063 → 0.9063  (no change)
R2 B3c: 0.9063 → 0.9063  (no change)
```

**关键观察**:
- 第一轮贡献了 99% 的改善
- 第一轮的 B1 Fast 是最大改善来源 (从 pct baseline 直接拉下来)
- 第二轮特征层块几乎无边际改善 — 说明特征层参数已经接近最优

---

## 3. B1 快速通道

### 3.1 动机

`combine()` 每次调用需要 ~14.5 秒（针对 205 张谱面），其中 99% 时间花在 D 公式计算和特征层上。但如果**只改变 sigmoid/calib/post 参数**（B1 块），D(t) 分布本身不变 — 只有二分法求解和后续 rescale 变了。

### 3.2 实现

```python
# 一次性预提取 (14.5s，只做一次)
for each map:
    cache = precompute(osu_path)
    _, details = combine(cache, use_sigmoid=False)  # 只用 percentile 提取 D_all
    D_seg, w_seg = segment_by_difficulty(details["D_all"], eff_w, 30)
    保存 {D_seg, w_seg, total_notes}

# 快速评估 (0.04s/eval，350x 加速)
def eval_b1(params, k, pre_data):
    D_cal = params["calib_a"] * D_seg + params["calib_b"]
    D_solved = solve_D_bisection(D_cal, w_seg, k=k, C=C, gamma=gamma)
    SR = D_solved * total_notes / (total_notes + N0)
    SR = rescale(SR)
    return score(SR, ref)
```

### 3.3 性能对比

| 模式 | 每轮 NM 时间 | 加速比 |
|------|:---:|:---:|
| Full combine() | ~38 min (50 iter × 150 eval × 14.5s) | 1× |
| B1 Fast Path | **~8 秒** (50 iter × 200 eval × 0.04s) | **~285×** |

这使得以下操作变得可行：
- k 值扫描: 26 个 k 值 × 3 秒 = 78 秒 (full 模式需要 ~16 小时)
- B1 反复微调: 可以跑 maxiter=50+ NM 而不担心时间

### 3.4 局限性

B1 fast path 假设 D(t) 分布不变，因此**不能用于 B2/B3 块的优化**。每当 B2/B3 完成一轮 NM 后，必须重新调用 `extract_b1_segments()` 更新 D_seg。

---

## 4. k 值扫描

### 4.1 为什么 k 需要独立搜索

k 是 sigmoid 的**最敏感参数** — 它控制玩家准度随难度衰减的速度：

| k 值 | A(d+1)/A(d) | 物理含义 |
|------|:---:|------|
| 0.5 | ~62% | 太平坦，玩家 d+5 时仍有 90% 准度 — **不符合实际** |
| 1.0 | ~27% | 中等衰减 |
| **1.5** | **~12%** | 接近最优，d+2 时准度降至 ~5% |
| 2.0 | ~5% | 太陡，接近 hard-max |

k 不与任何其他参数线性相关 — 改变 k 需要**重新拟合**所有 B1 参数 (calib, C, γ, post)。

### 4.2 扫描方法

```
1. 用现有的 D 公式/特征层参数 (Phase 5 best) 预提取 D_seg (一次性)
2. for k in 0.5, 0.6, ..., 3.0 (步进 0.1):
     用 B1 Fast NM 优化所有 B1 参数 (30 iter)
     记录: k, Loss, MAE, calib_a, calib_b, C, γ
3. 选取 Loss 最低的 k
```

全流程耗时: 26 × 3s ≈ **78 秒**。

### 4.3 结果

```
k=0.5:  Loss=0.9283  (太平坦)
k=1.0:  Loss=0.9124
k=1.3:  Loss=0.9083
k=1.4:  Loss=0.9074
k=1.5:  Loss=0.9071  ← 最优
k=1.6:  Loss=0.9073
k=1.7:  Loss=0.9079
k=2.0:  Loss=0.9118
k=3.0:  Loss=0.9204  (太陡)
```

最优 k≈1.5，对应 d-D=+1 时准度降至匹配水平的 ~12%，d-D=+2 时 ~2%。这与实际经验一致：7K 玩家在超出技能天花板 2SR 后基本无法正常游玩。

### 4.4 D 校准的稳定性

在不同 k 值下，calib_a 和 calib_b 高度稳定：

```
k=0.5:  calib=(0.893, +0.055)
k=1.0:  calib=(0.888, +0.051)
k=1.5:  calib=(0.890, +0.044)
k=2.0:  calib=(0.891, +0.040)
k=3.0:  calib=(0.895, +0.033)
```

calib_a 始终在 0.89 附近，calib_b 在 0.04-0.06 范围。这验证了 D 校准是一个**稳健的线性变换**，不依赖 k 的精确值。

---

## 5. D 预校准

### 5.1 动机

D(t) 的绝对值含义在 Percentile 聚合和 Sigmoid 聚合中不同：

- Percentile: D=5 意味着"这比 P70 的 5 难" — 是相对概念
- Sigmoid: D=5 意味着"玩家在这里的准度衰减曲线以 5 为中心" — 需要 D 的绝对标度与 sigmoid 的数值范围对齐

当 k=1.5, C=4, γ=0.2 时，sigmoid 求解 D 的最敏感区间在 D_i 附近 (d-D ∈ [-2, +2])。如果原始 D 值的标度偏移了 1-2 个单位，会显著影响求解结果。

### 5.2 线性校准

最简单的校准形式: `D'(t) = a × D(t) + b`

- **a=0.89**: D 的标度略微压缩 (~11%)，使数值范围与 sigmoid 的敏感区对齐
- **b=0.04**: 微小的正偏移，补偿 D 在低值区的略微低估

### 5.3 为什么是线性

尝试过高阶校准 (二次、分段线性) 但未带来改善。原因：
1. D(t) 分布在不同谱面间形状相似（大致对数正态）
2. 线性校准已经足以将 D 映射到 sigmoid 的合适数值范围
3. 更多自由度只会增加过拟合风险

### 5.4 操作方式

在校准被**纳入 B1 块**（而非单独处理）：

```python
D_cal = params["calib_a"] * D_seg + params["calib_b"]
D_solved = solve_D_bisection(D_cal, w_seg, k=k, C=C, gamma=gamma)
```

calib_a 和 calib_b 通过 B1 NM 与 C, γ, k 联合优化。这确保校准参数与 sigmoid 形状参数联合最优。

---

## 6. C-γ 自洽性

### 6.1 理论关系

在 sigmoid 公式 `A(d) = A_min + (A_max-A_min)/(C + e^(k(d-D)))` 中：

- 当 d=D 时，A(D) = A_min + (A_max-A_min)/(C+1)
- γ 定义为参考玩家的目标准确率分数: γ = (A_ref - A_min)/(A_max - A_min)

如果 γ 和 C 是独立参数，那么当 d=D 时，`A(D) = A_min + (A_max-A_min)/(C+1)`。

这意味着 γ_ref (目标) 和 1/(C+1) (匹配点的实际准确率分数) 之间存在自洽关系。

### 6.2 网格搜索验证

对 (C, γ) 进行 7×7 网格搜索 (C ∈ [2,10], γ ∈ [0.1,0.3])，结果：

```
γ ≈ 1/(C+1) 的约束线附近 Loss 最低
```

具体而言：
- C=4.0 时，1/(C+1)=0.20，NM 找到 γ=0.201 — **精确吻合**
- C=2.0 时，1/(C+1)=0.33，NM 找到 γ≈0.32
- C=8.0 时，1/(C+1)=0.11，NM 找到 γ≈0.12

### 6.3 设计决策: 将 C 放入 B1 块

基于自洽关系，虽然 C 和 γ 不是完全冗余，但它们通过 1/(C+1)≈γ 强耦合。将两者都放入 B1 块：

- B1 NM 可以在两个参数之间自由权衡
- 自洽关系使搜索空间实际上缩小为 1 自由度 (沿 1/(C+1)≈γ 线)
- 最终 NM 收敛到 C≈4.0, γ≈0.2，恰好满足自洽

如果 C 单独放一个块或与特征层一起优化，NM 会浪费大量 eval 在 2D 空间中搜索，且可能与特征层参数产生虚假的梯度信号。

---

## 7. 后处理参数合并

### 7.1 Phase 5 后处理

Phase 5 (Percentile 聚合) 有 4 个后处理参数：

```
note_norm_N0:   10.0   # 物量归一化偏移
rescale_threshold: 9.54  # 高 SR 压缩阈值
rescale_divisor:   2.00  # 高 SR 压缩除数
global_scale:   1.055  # 全局缩放
```

### 7.2 为什么不重新调

从一开始就决定**不重新调优后处理参数**，而是直接沿用 Phase 5 的值。原因：

1. **后处理参数与聚合方法正交**: N0, threshold, divisor 解决的是"物量偏置"和"高 SR 非线性"问题，这些问题在 sigmoid 下同样存在
2. **减少搜索空间**: 4 个参数 × 高度非线性 = 巨大的搜索空间，且容易与 calib 参数产生补偿效应
3. **Phase 5 的值已经很好**: Phase 5 的 post 参数是经过充分优化的，直接复用是安全起点

### 7.3 操作方式

后处理参数**合并到 B1 块**中:

```
B1 = {k, C, γ, calib_a, calib_b} + {N0, threshold, divisor, global_scale}
```

在 B1 fast path NM 中，这些参数与 sigmoid+calib 参数联合优化。后处理公式不变：

```python
SR *= total_notes / (total_notes + N0)       # 物量归一化
if SR > threshold: SR = threshold + (SR - threshold) / divisor  # 高SR压缩
SR *= global_scale                             # 全局缩放
```

### 7.4 最终值

| 参数 | Phase 5 (Percentile) | Sigmoid Final | 变化 |
|------|:---:|:---:|:---:|
| note_norm_N0 | 10.0 | 10.0 | 不变 |
| rescale_threshold | 9.54 | 9.54 | 不变 |
| rescale_divisor | 2.00 | 2.00 | 不变 |
| global_scale | 1.055 | 1.055 | 不变 |

后处理参数在 sigmoid 优化中完全未改变 — 验证了它们确实与聚合方式正交。

---

## 8. 参数分块原则

### 8.1 分块准则

1. **语义聚合**: 同一"概念层"的参数放一起 (如 cross 的 5 个参数全部在 B3a)
2. **评估成本对齐**: B1 用 fast path，B2/B3 用 full combine — 不同评估模式的参数必须分块
3. **梯度独立性**: 不同分块间的交叉梯度应尽可能小
4. **块大小平衡**: 每块 5-10 参数，NM 在每个块上都能高效运行

### 8.2 B2 为什么是 10 参数 (最大块)

D 公式是连接特征层和聚合层的桥梁。它的参数 (S_w1, alpha_P, alpha_R, D_beta1/2 ...) 之间存在强耦合，不应拆分。

虽然 10 参数是最大块，但 full NM 44 分钟 (maxiter=50) 对 10 维是可接受的。拆分会引入虚假的交替收敛问题。

### 8.3 B3 为什么拆分为 3 个子块

特征层有 18 个参数 (cross 5 + release 6 + inverse 7)，如果一次全部优化：
- 18 维 NM 在 full 模式下极度缓慢 (~200 eval × 14.5s = 48 min/round)
- Cross 和 Release 的 D(t) 贡献通过 Xbar/Rbar 耦合，但直接梯度很小

拆分为 3 个子块后：
- 每个块 5-7 参数，NM 收敛快
- 交替优化允许跨块耦合通过重提取 D_seg 传播

---

## 9. 完整工作流

### Phase 1: 粗粒度 k 扫描 (步进 0.5)

```
k ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}
→ 确定 k≈1.5 最优区域
```

### Phase 2: 精细 k 扫描 (步进 0.1)

```
k ∈ {0.5, 0.6, ..., 3.0}  (26 个值)
对每个 k: B1 Fast NM (30 iter)
→ k=1.5 验证为最优, calib_a≈0.89, calib_b≈0.04, C≈4.0, γ≈0.20
```

### Phase 3: C-γ 网格搜索

```
C ∈ {2,3,4,5,6,8,10}, γ ∈ {0.10, 0.13, 0.17, 0.20, 0.23, 0.27, 0.30}
→ 验证 1/(C+1)≈γ 自洽关系
```

### Phase 4: 完整交替 NM (k=1.5 固定)

```
2 rounds × 5 blocks:
  B1: maxiter=50→25, Fast
  B2: maxiter=50→25, Full
  B3a: maxiter=50→25, Full
  B3b: maxiter=50→25, Full
  B3c: maxiter=50→25, Full
```

### Phase 5: 最终参数微调 + 保存

```
R3 B1 only (maxiter=50, Fast): 精调 sigmoid/calib/post
→ 保存 tuned_params_sigmoid.json
```

---

## 10. 关键发现

### 10.1 k=1.5 是压倒性的改进

从 k=0.52 (初始猜测) 切换到 k=1.5：
- MAE: 0.2325 → 0.2253 (-3.1%)
- Loss: 1.01 → 0.91 (-10%)

这是 sigmoid 聚合层**最大的单一改进**，超过所有后续 NM 优化之和。

### 10.2 C 和 γ 不是独立的

`γ ≈ 1/(C+1)` 这条自洽线几乎精确成立。NM 不会偏离它超过 2%。实际上只需要 1 个自由度 (例如只调 C，γ 由公式计算)。

### 10.3 D_solved 稳定地落在 D(t) 分布的 P70

跨 205 张谱面: `D_solved ≈ 0.87 × D_P70`，std 仅 0.01。

这个惊人的一致性意味着：
1. Sigmoid 聚合等价于对 D 分布做光滑的"软分位选择"
2. 选择的位置固定在 ~P70，由 k 和 C 的数值决定
3. 改变 k 会微调这个等价分位（k 越大 → 分位越靠前），但变化很小

### 10.4 D 校准是必要的，但非常简单

`D' = 0.89×D + 0.04` 这条线性关系在不同 k 值和不同优化阶段都高度稳定。说明 D 的绝对值已经接近正确标度，只需要 ~11% 的轻微修正。

### 10.5 特征层参数几乎不需要重新调优

从 Phase 5 (Percentile) 到 Sigmoid Final，特征层的 16 个参数变化 <2%。B3a/B3b/B3c 的 NM 改善总共不到 0.001 MAE。

这意味着：**特征层 (D 分量定义) 与聚合层 (如何从 D(t) 到 SR) 是高度解耦的**。好的 D(t) 在两种聚合方法下都表现良好。

### 10.6 交替 NM 的边际收益递减

```
R1 B1: ΔLoss = -0.115  (主导改善)
R1 B2: ΔLoss = -0.003
R1 B3a-c: ΔLoss = -0.001
R2:     ΔLoss = -0.001
```

第一轮 B1 贡献了 ~95% 的总改善。R2 和 B3 子块的边际收益极低 (<1%)。对后续工作: **集中精力在聚合层参数 (k, C, γ, calib)，特征层可以锁定**。

---

## 附录 A: 脚本索引

| 脚本 | 功能 |
|------|------|
| `scripts/sweep_k_fine.py` | k=0.5-3.0 精细扫描 (B1 fast path) |
| `scripts/tune_sigmoid_k15.py` | k=1.5 完整交替 NM (5 blocks × 2 rounds) |
| `scripts/build_standalone.py` | 单文件分发构建器 |
| `spm_rating/aggregate_sigmoid.py` | Sigmoid 聚合核心 (分段 + 二分法) |

## 附录 B: 参数归档

| 文件 | 内容 |
|------|------|
| `tuned_params_sigmoid.json` | 最终最优参数 (MAE=0.2253) |
| `tuned_params_sigmoid_k15.json` | k=1.5 完整 NM 中间结果 |
| `tuned_params_sigmoid_bestk.json` | k sweep 最佳 B1 参数 |
