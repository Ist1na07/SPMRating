# SPM Rating — Sigmoid 调参方法论

**日期**: 2026-05-19
**上下文**: 从 Percentile (百分位聚合) 切换到 Sigmoid (玩家准度模型) 聚合层的调参实践。R3 最终结果: k=2.09, C=3.97, γ=0.196, MAE=0.2180。

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

将 41 个参数按**语义独立性**分为 6 个块，另有 3 个参数有证据排除：

| 块 | 参数数 | 含义 | 评估模式 |
|----|:---:|------|:---:|
| **B1: Sigmoid+Calib+Post** | 8 | k, C, γ, calib_a, calib_b, N0, threshold, divisor, global_scale | **Fast** (0.04s/eval) |
| **B2: D Formula Core** | 10 | S_w1, S_p, alpha_P, alpha_R, alpha_C, alpha_S, alpha_V, D_beta1, D_beta2, Abar_scale | Full (14.5s/eval) |
| **B3a: Cross Distance** | 5 | dist_exponent_rc, dist_exponent_ln, same_hand_penalty_rc/ln, thumb_bridge | Full |
| **B3b: Release LN Tail** | 8 | tail_coeff, tail_to_tap, same_col_bonus, coord_exponent, seq_coeff, lock_interaction, short_ln_threshold, short_ln_reduction | Full |
| **B3c: Inverse/Guide** | 8 | inv_amplitude, inv_tau, inv_power, guide_depth, guide_center, guide_width, cross_guide_scale, inverse_same_col_bonus | Full |
| **B3d: Jack** | 2 | jack_aggregation_power, multi_jack_boost | Full |
| *排除* | 3 | shield_tau_ms, shield_anchor_mod, shield_coord_factor | — |

**Shield 参数排除理由**：Shield (Sbar) 组件对 MAE 的总贡献仅 ~0.003（禁用后 MAE 变化 <1.5%）。3 个 shield 参数的优化成本（full NM ~15min）远超其可能带来的改善（<0.0005 MAE）。在 R3 中 shield 参数有轻微偏离默认值（见 8.4 节），但这是 B2 NM 的噪声级偏移，非有意义的优化。不纳入独立调参块。

### 2.2 交替算法

```
1. 加载 Phase 5 最佳参数作为初始值
2. 设 k = 2.09 (从 sweep 确定的最优值)
3. for round in [1, 2, 3]:
     maxiter = 50 if round <= 2 else varies  # R1-2 粗调+精调，R3 修正上限后微调
     for block in [B1, B2, B3a, B3b, B3c, B3d]:
       NM(block, maxiter=maxiter)
       if block 是 full 模式:
           重新提取 D_seg (因为 D 公式变了)
4. R3 完成后: B1 Fast NM (maxiter=50) 做最终精调
```

**R3 特殊处理**：R1-R2 将 `release_tail_to_tap` 锁定在 ≤4.0，R3 放宽上限至 ≤6.0 后重跑全流程。这使得 Release 参数找到了更优的配置（tail_to_tap 从 ~2.8 升至 4.22）。

### 2.3 为什么有效

- **解耦梯度**: 每个块只优化自己语义范围内的参数，不会出现跨语义补偿
- **交替迭代**: B1 先确定"玩家模型"，B2 再优化"难度定义"，循环两轮达到联合最优
- **Full-mode 块后重提取**: B2/B3 改变 D(t) 分布后，B1 的 D_seg 必须重提取，确保 B1 在下一次迭代时有正确的 D 输入

### 2.4 收敛行为

典型的 Loss 轨迹 (3 轮 × 6 块 + B1 终调 = 19 个 NM 阶段，R3 实际数据):

```
R3 B1: 0.9346  (Fast)
R3 B2: 0.9346 → 0.9346  (Full, 无变化 — D Formula 已收敛)
R3 B3a: 0.9346 → 0.9338 (Full, Cross Distance 微调)
R3 B3b: 0.9338 → 0.9335 (Full, Release LN Tail, tail_to_tap≤6.0)
R3 B3c: 0.9335 → 0.9329 (Full, Inverse/Guide)
R3 B3d: 0.9329 → 0.9329 (Full, Jack, 几乎无变化)
R3 B1 Final: 0.9329 → 0.9321 (Fast, 精调 sigmoid/calib/post)
```

**关键观察**:
- B1 Fast 和 B3c (Inverse) 是 R3 的主要改善来源
- B2 (D Formula) 已经完全收敛，NM 无法再找到下降方向
- B3d (Jack) 边际收益接近于零 — jack_aggregation_power 和 multi_jack_boost 接近最优
- R3 是在放宽 `release_tail_to_tap` 上限 (4.0→6.0) 后重跑的，Release 参数有显著移动

---

## 3. B1 快速通道

### 3.1 动机

`combine()` 每次调用需要 ~14.5 秒（针对 311 张谱面），其中 99% 时间花在 D 公式计算和特征层上。但如果**只改变 sigmoid/calib/post 参数**（B1 块），D(t) 分布本身不变 — 只有二分法求解和后续 rescale 变了。

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
| 1.5 | ~12% | d+2 时准度降至 ~5% |
| **2.09** | **~5%** | 最优，d+1 时准度降至 ~5%，d+0.5 时 ~18% |
| 3.0 | ~1% | 太陡，接近 hard-max |

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
k=0.5:  Loss=0.9412  (太平坦)
k=1.0:  Loss=0.9351
k=1.5:  Loss=0.9334
k=1.8:  Loss=0.9328
k=2.0:  Loss=0.9324
k=2.09: Loss=0.9321  ← 最优
k=2.2:  Loss=0.9325
k=2.5:  Loss=0.9338
k=3.0:  Loss=0.9362  (太陡)
```

最优 k≈2.09，比早期的 k=1.5 高约 40%。较高的 k 值意味着 sigmoid 更陡峭：玩家准度在超出技能边界后衰减更快。k=2.09 时 d-D=+1 对应准度降至匹配水平的 ~5%，d-D=+0.5 对应 ~18%。这与 7K 高难度谱面的实际体验一致。

### 4.4 D 校准的稳定性

在不同 k 值下，calib_a 和 calib_b 保持稳定，但 calib_b 随 k 增大而减小：

```
k=1.0:  calib=(0.891, +0.048)
k=1.5:  calib=(0.892, +0.042)
k=2.0:  calib=(0.893, +0.033)
k=2.09: calib=(0.893, +0.031)
k=2.5:  calib=(0.895, +0.025)
k=3.0:  calib=(0.896, +0.018)
```

calib_a 始终在 0.89 附近，calib_b 随 k 增大而递减（更高的 k 需要更小的正偏移）。这验证了 D 校准是一个**稳健的线性变换**，calib_a 几乎不依赖 k 的精确值。

---

## 5. D 预校准

### 5.1 动机

D(t) 的绝对值含义在 Percentile 聚合和 Sigmoid 聚合中不同：

- Percentile: D=5 意味着"这比 P70 的 5 难" — 是相对概念
- Sigmoid: D=5 意味着"玩家在这里的准度衰减曲线以 5 为中心" — 需要 D 的绝对标度与 sigmoid 的数值范围对齐

当 k=2.09, C=3.97, γ=0.196 时，sigmoid 求解 D 的最敏感区间在 D_i 附近 (d-D ∈ [-1.5, +1.5])。如果原始 D 值的标度偏移了 1-2 个单位，会显著影响求解结果。更高的 k 值使敏感区间更窄，对 D 校准的精度要求更高。

### 5.2 线性校准

最简单的校准形式: `D'(t) = a × D(t) + b`

- **a=0.893**: D 的标度略微压缩 (~11%)，使数值范围与 sigmoid 的敏感区对齐
- **b=0.031**: 微小的正偏移，补偿 D 在低值区的略微低估

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
- C=3.97 时，1/(C+1)=0.201，NM 找到 γ=0.196 — **吻合 (偏差 2.5%)**
- C=2.0 时，1/(C+1)=0.33，NM 找到 γ≈0.32
- C=8.0 时，1/(C+1)=0.11，NM 找到 γ≈0.12

### 6.3 设计决策: 将 C 放入 B1 块

基于自洽关系，虽然 C 和 γ 不是完全冗余，但它们通过 1/(C+1)≈γ 强耦合。将两者都放入 B1 块：

- B1 NM 可以在两个参数之间自由权衡
- 自洽关系使搜索空间实际上缩小为 1 自由度 (沿 1/(C+1)≈γ 线)
- 最终 NM 收敛到 C≈3.97, γ≈0.196，满足自洽关系 (1/(3.97+1)=0.201≈0.196)

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

| 参数 | Phase 5 (Percentile) | Sigmoid R3 Final | 变化 |
|------|:---:|:---:|:---:|
| note_norm_N0 | 10.0 | 8.21 | -18% |
| rescale_threshold | 9.54 | 9.42 | -1.3% |
| rescale_divisor | 2.00 | 2.01 | +0.5% |
| global_scale | 1.055 | 1.055 | 不变 |

后处理参数在 sigmoid 优化中有小幅调整：N0 从 10.0 降至 8.21，减轻了对短谱面的物量惩罚；threshold 和 divisor 基本稳定。global_scale 完全不变 — 验证了全局缩放与其他参数正交。

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

### 8.3 B3 为什么拆分为 4 个子块

特征层有 23 个参数 (cross 5 + release 8 + inverse 8 + jack 2)，如果一次全部优化：
- 23 维 NM 在 full 模式下极度缓慢 (~300 eval × 14.5s = 72 min/round)
- Cross 和 Release 的 D(t) 贡献通过 Xbar/Rbar 耦合，但直接梯度很小

拆分为 4 个子块后：
- 每个块 2-8 参数，NM 收敛快
- 交替优化允许跨块耦合通过重提取 D_seg 传播
- Jack 块独立出来因为 `jack_aggregation_power` 控制 Jbar 的跨列聚合方式，语义上独立于 cross/release/inverse

### 8.4 Shield 参数处理

Shield (Sbar) 是防范 LN 头前同列音符误触的概率模型。其公式为 `Sbar = sum(exp(-dt/tau)) * (1 + anchor_mod * coord_factor * lock_bonus)`。

**实证**：禁用 Shield 后 MAE 仅升高 ~0.003（变化 <1.5%）。跨 311 张谱面的 MAE 贡献是所有分量中最低的。

Shield 的 3 个参数 (`shield_tau_ms`, `shield_anchor_mod`, `shield_coord_factor`) 未被纳入任何独立的 NM 块，但通过 B2 (D Formula) 中的 `alpha_S` 间接影响。在 R3 最终参数中，shield 参数有轻微偏离默认值：
- `shield_tau_ms`: 100 → 56.2 (-44%)
- `shield_anchor_mod`: 1.0 → 0.806 (-19%)
- `shield_coord_factor`: 1.0 → 1.003 (+0.3%)

这些偏移可能是 B2 NM 的随机游走而非有意义的优化 — 因为改变 shield 参数对 Loss 的影响在数值噪声级别 (<0.0001)。**不建议为 shield 参数创建独立的调参块。**

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
→ k=2.09 验证为最优, calib_a≈0.893, calib_b≈0.031, C≈3.97, γ≈0.196
```

### Phase 3: C-γ 网格搜索

```
C ∈ {2,3,4,5,6,8,10}, γ ∈ {0.10, 0.13, 0.17, 0.20, 0.23, 0.27, 0.30}
→ 验证 1/(C+1)≈γ 自洽关系
```

### Phase 4: 完整交替 NM (k=2.09 固定)

```
R1-R2: 2 rounds × 6 blocks (release_tail_to_tap ≤ 4.0):
  B1:  maxiter=50→25, Fast
  B2:  maxiter=50→25, Full
  B3a: maxiter=50→25, Full
  B3b: maxiter=50→25, Full
  B3c: maxiter=50→25, Full
  B3d: maxiter=50→25, Full

R3: 放宽 release_tail_to_tap 上限至 6.0，重跑 6 blocks
  原因: R1-R2 的 tail_to_tap 收敛到上限 (4.0)，
        表明真正最优点在边界之外
```

### Phase 5: 最终参数微调 + 保存

```
R3 B1 Final (maxiter=50, Fast): 精调 sigmoid/calib/post
→ 保存 tuned_params_sigmoid.json (MAE=0.2180)
```

---

## 10. 关键发现

### 10.1 k=2.09 是压倒性的改进

从 k=0.52 (初始猜测) 切换到 k=2.09：
- MAE: 0.2299 → 0.2180 (-5.2%)
- Loss: 1.15 → 0.932 (-19%)

这是 sigmoid 聚合层**最大的单一改进**，超过所有后续 NM 优化之和。k 值从早期的 1.5 进一步升至 2.09，主要受益于放宽 `release_tail_to_tap` 上限后的重调。

### 10.2 C 和 γ 不是独立的

`γ ≈ 1/(C+1)` 这条自洽线几乎精确成立。NM 不会偏离它超过 2%。实际上只需要 1 个自由度 (例如只调 C，γ 由公式计算)。

### 10.3 D_solved 稳定地落在 D(t) 分布的 P70

跨 311 张谱面: `D_solved ≈ 0.87 × D_P70`，std 仅 0.01。

这个惊人的一致性意味着：
1. Sigmoid 聚合等价于对 D 分布做光滑的"软分位选择"
2. 选择的位置固定在 ~P70，由 k 和 C 的数值决定
3. 改变 k 会微调这个等价分位（k 越大 → 分位越靠前），但变化很小

### 10.4 D 校准是必要的，但非常简单

`D' = 0.893×D + 0.031` 这条线性关系在不同 k 值和不同优化阶段都高度稳定。说明 D 的绝对值已经接近正确标度，只需要 ~11% 的轻微压缩和微小的正偏移。

### 10.5 特征层参数几乎不需要重新调优

从 Phase 5 (Percentile) 到 Sigmoid R3 Final，特征层的 23 个参数变化 <5%。B3a/B3b/B3c/B3d 的 NM 改善总共约 0.0017 Loss (0.9346→0.9329)。

这意味着：**特征层 (D 分量定义) 与聚合层 (如何从 D(t) 到 SR) 是高度解耦的**。好的 D(t) 在两种聚合方法下都表现良好。

### 10.6 交替 NM 的边际收益递减

```
R3 B1:     ΔLoss = -0.0025  (主导改善)
R3 B3a-c:  ΔLoss = -0.0017
R3 B3d:    ΔLoss = -0.0000  (无边际改善)
R3 B1 Final: ΔLoss = -0.0008
```

B1 (聚合层) 贡献了 ~50% 的总改善，B3c (Inverse) 是特征层中唯一有显著改善的块。B3d (Jack) 边际收益为零。对后续工作: **集中精力在聚合层参数 (k, C, γ, calib) 和 Inverse，其余特征层可以锁定**。

### 10.7 RC 子模型调参

RC 子模型禁用 LN 专属分量 (Rbar, Sbar, Vbar)，使用 RC-only D 公式。使用与主模型相同的分块 NM 方法调参：

| 指标 | Total SR 用于 RC 标签 | RC 子模型 (调参后) | 改善 |
|------|:---:|:---:|:---:|
| MAE | 0.3250 | 0.2366 | -27% |
| Loss | 2.197 | 1.359 | -38% |
| r | — | 0.9870 | — |

**关键发现**：
- RC sigmoid k=2.31，比主模型 k=2.09 更高 — RC 谱面的难度感知更陡峭
- `note_norm_N0_rc` 收敛到 0 — RC 谱面不需要物量归一化（注量分布更均匀）
- RC 子模型显著优于直接用 Total SR 预测 RC 标签，验证了分量裁剪的有效性

### 10.8 LN 子模型的架构限制

LN 子模型使用简化的加法 D 公式 (`D_ln = alpha_R*Rbar + alpha_S*Sbar + alpha_V*Vbar + alpha_P_ln*Pbar`)，不含 Jbar、Xbar 和复杂的 S-T 混合项。

**调参结果不理想**：

| 指标 | Total SR 用于 LN 标签 | LN 子模型 (调参后) |
|------|:---:|:---:|
| MAE | **0.2049** | 0.8162 |
| Loss | 0.440 | 7.568 |
| r | — | 0.827 |

Total SR 作为 LN 预测器 (MAE=0.205) 远优于专用 LN 子模型 (MAE=0.816)。这说明当前 LN D 公式过于简化，缺少 Jbar (jack 分量)、Xbar (tech 分量) 等关键信息。**LN 子模型需要架构重设计，而非继续调参。** 可能的改进方向：
- 引入 Jbar/Xbar 到 LN D 公式中
- 使用与主模型一致的 S-T 混合公式结构
- 或在 LN masking 下直接复用主模型 D 公式

---

## 附录 A: 脚本索引

| 脚本 | 功能 |
|------|------|
| `scripts/tune_sigmoid_k15.py` | k 固定完整交替 NM (6 blocks × 3 rounds) |
| `scripts/tune_rc.py` | RC 子模型 NM 调参 (3 blocks × 2 rounds) |
| `scripts/tune_ln.py` | LN 子模型 NM 调参 (3 blocks × 2 rounds) |
| `scripts/train_sort_classifier.py` | 谱面分类器训练 (RC/LN/HB/Mix) |
| `scripts/train_tag_classifier.py` | 模式标签分类器训练 (14 标签) |
| `scripts/build_standalone.py` | 单文件分发构建器 |
| `spm_rating/aggregate_sigmoid.py` | Sigmoid 聚合核心 (分段 + 二分法) |

## 附录 B: 参数归档

| 文件 | 内容 |
|------|------|
| `tuned_params_sigmoid.json` | R3 最终最优参数 (MAE=0.2180, k=2.09) |
| `tuned_params_rc.json` | RC 子模型最优参数 |
| `tuned_params_ln.json` | LN 子模型最优参数 |

## 附录 C: 完整参数清单

### 调参参数 (41 个，分 6 块)

**B1 — Sigmoid+Calib+Post (9p, k 在 Phase 1-2 扫描中确定)**
`agg_sigmoid_k`, `calib_a`, `calib_b`, `agg_sigmoid_C`, `agg_sigmoid_ref_gamma`, `note_norm_N0`, `rescale_threshold`, `rescale_divisor`, `global_scale`

**B2 — D Formula Core (10p)**
`S_w1`, `S_p`, `alpha_P`, `alpha_R`, `alpha_C`, `alpha_S`, `alpha_V`, `D_beta1`, `D_beta2`, `Abar_scale`

**B3a — Cross Distance (5p)**
`cross_dist_exponent_rc`, `cross_dist_exponent_ln`, `cross_same_hand_penalty_rc`, `cross_same_hand_penalty_ln`, `cross_thumb_bridge_factor`

**B3b — Release LN Tail (8p)**
`release_tail_coeff`, `release_tail_to_tap`, `release_same_col_bonus`, `release_coord_exponent`, `release_seq_coeff`, `lock_interaction_coeff`, `short_ln_threshold`, `short_ln_reduction`

**B3c — Inverse/Guide (8p)**
`inv_amplitude`, `inv_tau`, `inv_power`, `guide_depth`, `guide_center`, `guide_width`, `cross_guide_scale`, `inverse_same_col_bonus`

**B3d — Jack (2p)**
`jack_aggregation_power`, `multi_jack_boost`

### 排除参数 (3 个，锁定默认值)

| 参数 | 默认值 | 排除原因 |
|------|--------|---------|
| `shield_tau_ms` | 100 | Shield 总贡献 ~0.003 MAE |
| `shield_anchor_mod` | 1.0 | 同上，且与 coord_factor 乘性耦合 |
| `shield_coord_factor` | 1.0 | 同上 |

### 非调参参数 (固定或废弃)

| 参数 | 状态 |
|------|------|
| `inverse_peak_width` | 固定 (2.0) |
| `inverse_window_ms` | 固定 (200) |
| `shield_smooth_window` | 固定 (500) |
| `shield_scale` | 固定 (0.001) |
| `stream_booster_scale` | 固定 (需 recache) |
| `stream_short_window` | 固定 |
| `D_gamma_e` (Stamina) | 禁用 (正值破坏 MAE) |
| `V_alpha` | 未使用 (存在于 params 中但代码未读取) |
