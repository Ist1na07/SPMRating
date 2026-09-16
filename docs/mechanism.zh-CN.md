# SPM Rating v1.1.1 算法机制规范（完全复刻级标准）

> 本文档为 SPM Rating v1.1.1 的**完备规范文档**。任何开发者或研究者仅需阅读本文档中的数学推导与计算流程，即可在任何编程语言中完全复刻出数值对齐（ulp 级一致）的 SPM Rating 算法。
>
> 发布参数权威文件：[params/spm-rating-1.1.1.json](../params/spm-rating-1.1.1.json)（共 276 项独立参数）。
> 配套技术文档：[《研制历程与结论》](history.zh-CN.md) 记录从 v0.5.0/v1.0.0 到 v1.1.0/v1.1.1 的演化与实证决策，[《维护指南》](maintenance.zh-CN.md) 记录日常运行与门禁规范。

---

## 目录
- [0. 核心设计原则与符号定义](#0-核心设计原则与符号定义)
- [1. 输入解析与几何映射 (Parser)](#1-输入解析与几何映射-parser)
- [2. 行网格与特征张量构建 (Grid & Struct)](#2-行网格与特征张量构建-grid--struct)
- [3. 击打宽容度与 OD 映射 (Leniency)](#3-击打宽容度与-od-映射-leniency)
- [4. 连打通道群：Jm, Jc, Ja (Jacks)](#4-连打通道群jm-jc-ja-jacks)
- [5. 流与和弦手势定价通道：Pbar, Xbar, Abar (Stream)](#5-流与和弦手势定价通道pbar-xbar-abar-stream)
- [6. 长键释放动力学：Rbar (Release)](#6-长键释放动力学rbar-release)
- [7. 长键协调几何矩阵：Cbar v2 (LNCoord)](#7-长键协调几何矩阵cbar-v2-lncoord)
- [8. 生理约束与辅助通道：HCd, SHd, EyeR, Recov](#8-生理约束与辅助通道hcd-shd-eyer-recov)
- [9. 局部瞬时难度合成：D(t) (Combine)](#9-局部瞬时难度合成dt-combine)
- [10. 两级物量加权分段聚合：agg_mode=4 (Aggregation)](#10-两级物量加权分段聚合agg_mode4-aggregation)
- [11. 谱面级后处理与全局标定 (Postprocess)](#11-谱面级后处理与全局标定-postprocess)
- [附录 A：手部拓扑与几何常数表](#附录-a手部拓扑与几何常数表)
- [附录 B：发布参数全集表 (276 项)](#附录-b发布参数全集表-272-项)

---

## 0. 核心设计原则与符号定义

1. **时序因果性与物理局域性**：难度只能来源于谱面局部的时间结构，绝对禁止使用谱面全局标量统计修正层。
2. **段落级鲁棒性与可加性**：算法对谱面切片、空白休息段、重复段落具有数学不变性，杜绝因外部填充产生难度稀释或膨胀。
3. **连续性与微扰免疫**：彻底剔除浮点极小值（如 $10^{-9}\mathrm{s}$ ）门控与硬折线截断，所有物理通道必须保证关于时间戳微扰（ $\pm 1\mathrm{ms}$ ）的一阶平滑连续。
4. **统一时间量纲**：
   - 内部时间戳 $t$ 、时间间隔 $\Delta t$ 、窗口半宽等一律以**毫秒 (ms)** 为原生存储与计算单位；
   - 在公式中换算为物理速度（如每秒击键数）或使用标准衰减常数时，明确换算为**秒 (s)**（记为 $\delta = \Delta t / 1000$ ）。

---

## 1. 输入解析与几何映射 (Parser)

### 1.1 `.osu` 文件解析
算法读取 `.osu` 文件的 `[General]`, `[Difficulty]`, `[HitObjects]` 三个区块：
- `Mode`: 必须为 `3`（osu!mania 模式）；
- `CircleSize`: 原生键数 $K_{\mathrm{orig}}$ （支持 4K, 6K, 7K）；
- `OverallDifficulty`: 判定难度 $\mathrm{OD} \in [0.0, 10.0]$ ；
- 速率系数 $\mathrm{rate}$ （标准速度为 $1.0$ ，加速模组如 DT 为 $1.5$ ）。

每个击打物件（HitObject）解析为四元组 `(t_raw, c_raw, t_end_raw, is_ln)` ：
- **原生列计算**：限制在 $[0, K_{\mathrm{orig}}-1]$ ：

$$
c_{\mathrm{raw}} = \left\lfloor \frac{x \cdot K_{\mathrm{orig}}}{512} \right\rfloor
$$

- **击键起始时间**： $t = t_{\mathrm{raw}} / \mathrm{rate}$ （ms）；
- **物件判定**：若类型二进制第 7 位为 1（ `type & 128` ）且终止时间满足 `t_end_raw > t_raw`，则判定为长键（LN），实际结束时间计算为 `t_end = t_end_raw / rate`，标记 `is_ln = True`；否则为普通单键（Normal Note），结束时间取 `t_end = t`，标记 `is_ln = False`。

### 1.2 统一 7 轨道物理对齐 (Geometry Embedding)
所有下游算法均运行在固定的 **7 轨物理坐标系**（列索引 $k \in \{0, 1, 2, 3, 4, 5, 6\}$ ）上：
- **7K 映射**：直接对应 $c = c_{\mathrm{raw}}$ ；
- **4K 对称嵌入**：

$$
\text{4K 原生列 } [0, 1, 2, 3] \longrightarrow \text{7 轨坐标 } [1, 2, 4, 5]
$$

  物理意义：列 0, 3, 6 置空，左手落于列 1、2，右手落于列 4、5，中缝拇指列 3 置空。保证了 4K 玩家双手中指与食指的间距和生理跨手距离与 7K 完全一致。
- **6K 映射**：默认采用 $(0, 1, 2, 4, 5, 6)$ ，拇指列 3 置空。

所有音符按照 $t$ 升序排序；若 $t$ 相同，按 $c$ 升序排序。

---

## 2. 行网格与特征张量构建 (Grid & Struct)

算法不采用事件驱动，而是将谱面投影到严格按时间单调递增的**行网格 (Row Grid)** 上。

### 2.1 行去重与比特掩码

对于 $N$ 个音符，找出所有严格不同的击打时间戳序列：

$$
t_0 < t_1 < \dots < t_{R-1}, \quad r \in [0, R-1]
$$

- **行时间差分**：

$$
\Delta t_r = t_{r+1} - t_r \quad (\mathrm{ms}), \quad \Delta t_{R-1} = 0, \quad \delta_r = \frac{\Delta t_r}{1000} \quad (\mathrm{s})
$$

- **7 位按键掩码**：

$$
\mathrm{mask}_r = \sum_{k \in \mathrm{notes}(r)} 2^k \in [1, 127]
$$

- **行和弦尺寸**：

$$
\mathrm{size}_r = \mathrm{popcount}(\mathrm{mask}_r) \in [1, 7]
$$

- **瞬时长键覆盖状态**：在时刻 $t_r$，若存在任意长键满足 $t_{\mathrm{head}} < t_r < t_{\mathrm{tail}}$，则该长键在该列处于按住（Held）状态。记 $\mathrm{heldmask}_r$ 为被长键占用的 7 位掩码：

$$
\mathrm{held}_r = \mathrm{popcount}(\mathrm{heldmask}_r)
$$

### 2.2 局域物量计数 $C(t)$ 与有效列数 $K_s$
- **基础角点物量 $C_r$**：以时刻 $t_r$ 为中心的 $\pm 500\mathrm{ms}$ 对称时间窗口内的总音符数：

$$
C_r = \sum_{j: |t_j - t_r| \le 500\mathrm{ms}} \mathrm{size}_j
$$

- **有效活跃列数 $K_{s, r}$**：

$$
K_{s, r} = \begin{cases} 4.0, & \text{若为 4K 谱面} \\\\ 7.0, & \text{若为 7K 谱面} \end{cases}
$$

- **长条身体占用时间** `ln_body_ms[r]`：在区间 $[t_r, t_{r+1})$ 内处于被按住状态的长条数量与时长的乘积。

---

## 3. 击打宽容度与 OD 映射 (Leniency)

判定难度 $\mathrm{OD}$ 决定了玩家的物理时间窗口，算法将其映射为击打宽容度基准标量 $x$ （秒）：

1. 若速率 $\mathrm{rate} \ne 1.0$ ，将有效 $\mathrm{OD}$ 依最高判定窗等比例换算为有效判定 $\mathrm{OD}_{\mathrm{eff}}$ （对应变量 `eff_od` ）；
2. 宽容度核心公式：

$$
\begin{aligned}
q &= \frac{64.5 - \lceil 3 \cdot \mathrm{OD}_{\mathrm{eff}} \rceil}{500.0} \\\\\\\\
x_0 &= 0.3 \cdot \sqrt{\max(q, 10^{-6})} \\\\\\\\
x &= \min\big(x_0,\; 0.6 \cdot (x_0 - 0.09) + 0.09\big)
\end{aligned}
$$

   对于 $\mathrm{OD} = 8.0$ ，得到 $x \approx 0.0834\mathrm{s}$ 。
3. **各通道击打宽容度幂次响应**：
   - 连打通道： $x_{\mathrm{jm}} = x^{0.750426}$ （对应参数 `od_mult_jm`）；
   - 流通道： $x_p = x^{1.126485}$ （对应参数 `od_mult_p`）；
   - 跨列通道： $x_x = x^{1.165131}$ （对应参数 `od_mult_x`）；
   - 释放通道： $x_r = x^{1.064324}$ （对应参数 `od_mult_r`）。

---

## 4. 连打通道群：Jm, Jc, Ja (Jacks)

连打按解剖学与打击模式拆分为三条并行的物理通道，各自独立计算并施加特征饱和。

### 4.1 逐列连打统计量
对每一列 $k \in [0, 6]$ ，提取该列连续两个音符的时间戳对 $(t_{k, j}, t_{k, j+1})$ ，间隔 $\Delta t_j = t_{k, j+1} - t_{k, j}$ （ms），秒单位 $g_j = \Delta t_j / 1000$ ：
- 同列两连打核函数：

$$
\mathrm{kern}_j = \max(g_j, x_{\mathrm{jm}})^{-2.0}
$$

- **连打稀释因子**（修复 A1 超集单调性）：
  令 $\mathrm{other}_j$ 为该两连打期间其它列发生的音符总数， $\mathrm{nrows}_j$ 为期间总行数。

$$
\mathrm{dil}_j = \frac{1.0}{1.0 + 0.080 \cdot \left(\frac{\mathrm{other}_j}{\mathrm{nrows}_j}\right)^{0.9615}}
$$

  在 v1.1.0 中，参数取值为 `mj_dilute = 0.080`，`mj_dilute_exp = 0.9615`。若当前行处于多押和弦，启用参数 `chord_no_dilute = 1.0`，强制令 $\mathrm{dil}_j = 1.0$，杜绝和弦对叠键的错误扣分。
- **三连打加成 (Triple Gain)**：
  若存在同列三连击且总跨度 $t_{k, j+1} - t_{k, j-1} \le 135.97\mathrm{ms}$ （参数 `j_triple_tau`），则：

$$
\mathrm{gain}_j = 1.0 + \gamma_{\mathrm{trip}} \cdot \mathrm{trip}_j, \quad \gamma_{\mathrm{trip}} = 0.0193 \quad (\text{4K 为 } 0.5356)
$$

### 4.2 三通道解耦
1. **孤立小叠 $J_m$ (Minijack)**：

$$
v_{\mathrm{jm}, j} = \mathrm{kern}_j \cdot \mathrm{dil}_j \cdot \mathrm{gain}_j
$$

   并入同列极快连打补偿通道： $J_m \leftarrow J_m + c_{\mathrm{fj}} \cdot \mathrm{FJ}$ （参数 `c_fj` 为 $1.1224$，4K 为 $1.6385$）。
2. **多押大叠 $J_c$ (Chordjack)**：

$$
v_{\mathrm{jc}, j} = \mathrm{kern}_j \cdot \left(\frac{\max(\mathrm{load}_j, 10^{-6})}{1.9758}\right)^{1.0152}
$$

   其中 $\mathrm{load}_j$ 为该连打点局部的和弦按键负荷，参数取值 `cj_norm = 1.9758`，`cj_size_exp = 1.0152`。
3. **同轨长跑 $J_a$ (Anchor/Long-jack)**：

$$
\begin{aligned}
e_j &= \max(\mathrm{runlen}_j + 2.0 - 4.0742,\; 0.0) \\\\
\mathrm{runf}_j &= \left(\frac{e_j}{e_j + 5.9814}\right)^{1.0048} \\\\
v_{\mathrm{ja}, j} &= \mathrm{kern}_j \cdot \mathrm{runf}_j \cdot \left(1.0 + \frac{\mathrm{other}_j}{\mathrm{nrows}_j}\right)^{0.6808}
\end{aligned}
$$

   参数取值：`ja_norm = 5.9814`，`ja_len_exp = 1.0048`，`ja_len_floor = 4.0742`，`ja_coord_exp = 0.6808`。

### 4.3 跨列加权广义幂平均
将各列的阶梯值散射回行网格后，使用高斯滑动窗口平滑（参数 `w_jm = 842.9ms`），最终以权重 $w_j = 1 / g_j$ 做幂次 $p_{\mathrm{agg}} = 4.2205$（参数 `j_agg_pow`）的加权幂均值：

$$
J = \left( \frac{\sum_k w_k \cdot \mathrm{smooth}(v_k)^{p_{\mathrm{agg}}}}{\sum_k w_k} \right)^{1 / p_{\mathrm{agg}}}
$$

---

## 5. 流与和弦手势定价通道：Pbar, Xbar, Abar (Stream)

### 5.1 全局流密度通道 $P_{\mathrm{bar}}$
令全局行间隙为 $\delta_r = \Delta t_r / 1000$ （秒）。
1. **连续和弦时间窗与多押检测**：
   在时间常数 $\tau_c = 28.0\mathrm{ms}$ （参数 `chord_tau_c`）内，使用三角加权窗统计当前行左右手的局部有效按键数 $u_L, u_R$，得到连续有效和弦尺寸 $u = u_L + u_R$。
2. **多押定价权重**：
   - 2 押到 7 押的离散定价参数表：

$$
\{p_{\mathrm{chw2}}: 0.700,\; p_{\mathrm{chw3}}: 1.471,\; p_{\mathrm{chw4}}: 0.821,\; p_{\mathrm{chw5}}: 0.868,\; p_{\mathrm{chw6}}: 0.747,\; p_{\mathrm{chw7}}: 1.033\}
$$

   - 依据连续有效尺寸 $u$ ，在线性插值下获得手势定价因子 $\mathrm{chw}_u$ 。
3. **脉冲增益与流密度**：

$$
\begin{aligned}
q &= \min\left(\delta - \frac{x_p}{2.0},\; \frac{x_p}{6.0}\right) \\\\
f_p &= \left( \max\left( \frac{p_{\mathrm{scale}}}{x_p} \cdot \left(1.0 - \frac{p_{\mathrm{lam3}}}{x_p} \cdot q^2\right),\; 0.0 \right) \right)^{0.25}
\end{aligned}
$$

   其中参数取值 `p_scale = 0.05862`，`p_lam3 = 27.838`。
4. **短爆发与行增量**：
   加入 100ms 与 150ms 的短爆发（Burst）计数加成：

$$
\mathrm{inc} = \frac{1}{\delta} \cdot f_p \cdot \mathrm{chw}_u \cdot \mathrm{mix} + 0.0575 \cdot \mathrm{cnt}_{100} \cdot 10.0 + 0.0416 \cdot \mathrm{cnt}_{150} \cdot 6.6667
$$

   通过滑动平滑窗口（参数 `w_p = 806.2ms`）获得平滑曲线 $P_{\mathrm{bar}}$，其中短爆发加成参数取值 `p_burst3 = 0.0575`，`p_burst4 = 0.0416`。

### 5.2 跨列交替与几何跳跃 $X_{\mathrm{bar}}$
分析相邻列边界上的敲击转移：
1. **边界事件链与内外切方向拆分**：
   针对 8 条相邻列边界，提取相邻事件间隔 $g$。若转移方向朝向中心拇指列（内切），加权参数取值 `x_din = -0.5959`；若远离中心列（外切），加权参数取值 `x_dout = 0.9859`：

$$
X_b = x_{\mathrm{amp}} \cdot \max(g, x_x)^{-2.0} \cdot \big(1.0 + x_{\mathrm{din}} \cdot \max(dd, 0) + x_{\mathrm{dout}} \cdot \max(-dd, 0)\big)
$$

   其中振幅参数取值 `x_amp = 0.1478`。
2. **双向快速交替几何平均**：
   相邻两条边界的快速交替项取几何平均： $\sqrt{\mathrm{fc}_b \cdot \mathrm{fc}_{b+1}}$ ，加权入总流。
3. **几何大跳通道 (Jump Channel)**：
   以权重参数 `x_jump_w = -0.8523` 引入跨越多轨的几何大跳。
4. 通过滑动平滑窗口（参数 `w_x = 939.3ms`）生成平滑曲线 $X_{\mathrm{bar}}$。

### 5.3 节奏不均匀度 $A_{\mathrm{bar}}$
度量相邻活跃列敲击的时间节奏离散度：
1. 对相邻两活跃列 $k_0, k_1$ ，计算同列间隔绝对差 $\Delta d = |d_{k0} - d_{k1}|$ 与最大间隔 $mx = \max(d_{k0}, d_{k1})$ ；
2. **10ms 节奏焊接与软边界**（参数 `abar_weld = 1.0`）：

$$
dd = \Delta d + w_{\mathrm{mx}} \cdot \max(0, mx - 0.0886)
$$

$$
A_{\mathrm{val}} = \begin{cases}
\min(c_0 + w_{\mathrm{mx}} \cdot mx,\; 1.0), & dd < 0.0168 \\\\
\min(c_0 + k \cdot (dd - 0.0168) + w_{\mathrm{mx}} \cdot mx,\; 1.0), & 0.0168 \le dd < 0.0945 \\\\
1.0, & dd \ge 0.0945
\end{cases}
$$

   参数取值：$c_0 = 0.7299$，$k = 5.8693$，`thr_lo = 0.0168s`（约 17ms），`thr_hi = 0.0945s`（约 95ms），`abar_mx_w = 0.0716`，`abar_mx_thr = 0.0886`。
3. 经平滑窗口（参数 `w_a = 60.79ms`）乘全局标度参数 `abar_scale = 1.00986`，得到不均匀度因子 $A \in [0.65, 1.0]$。

---

## 6. 长键释放动力学：Rbar (Release)

长键释放要求肌肉在特定时机精准松开，算法将其分解为三大动力学过程：

### 6.1 逐尾释放 (Per-tail Release)
每个长键尾部 $t_i$ ，查找后续最近击键事件（头部或尾部），间隔为 $dtr = \max((t_{\mathrm{next}} - t_i)/1000.0,\; dtr_{\min})$ ，其中安全截断下限 $dtr_{\min} = 0.015\mathrm{s}$ （15ms）：
- **释放难度基础项**：

$$
r_v = \frac{r_{\mathrm{tail}}}{\sqrt{dtr}} \cdot x_r^{-1.0} \cdot (1.0 + w_I \cdot I)
$$

  参数取值：$r_{\mathrm{tail}} = 0.028985$，偏离权重 $w_I = 1.0611$（参数 `r_I_w`）。
- **释放偏离指数 $I$**（陡度参数 $s_I = 3.8156$ 对应 `r_I_steep`，偏置参数 $o_I = 1.6798$ 对应 `r_I_off`）：

$$
I = \frac{2.0}{2.0 + \exp\big(-s_I \cdot (|dur - 80|/1000/x - o_I)\big)}
$$

- **手位协调权重幂次**：

$$
r_v \leftarrow r_v \cdot \mathrm{cw}^{e_{\mathrm{coord}} \cdot e_{\mathrm{tt}}}
$$

  其中 $\mathrm{cw} = w_{\mathrm{coord}}(c_i, c_{\mathrm{next}})$ （同列 $0.982$, 同手 $0.425$, 拇指 $0.366$, 跨手 $0.690$ ）。以严格正幂次保证难度单调性，杜绝负难度。
- **短长条折减**：持续时间 $< 440\mathrm{ms}$ 的长条按比例线性折减，最低折减至 $0.109$ 。
- **锁手加成与跨指独立性矩阵**：
  结合当前处于按住状态的其它列，施加锁手加成参数 `r_lock = 0.1775`，并经四组独立手指解剖学系数（`r_ain0` ~ `r_ain3`）加成。

### 6.2 尾到尾时序 (Tail-to-Tail Sequence)
相邻两个释放事件间隔 $dtr$，以序列参数 `r_seq = 0.06596` 计算连续抬指成本：

$$
s_v = \frac{r_{\mathrm{seq}}}{\sqrt{dtr}} \cdot x_r^{-1.0} \cdot \big(1.0 + w_I(I_1 + I_2)\big) \cdot \mathrm{cw}^{e_{\mathrm{coord}}}
$$

### 6.3 释放顺序三元组 (Release-Order Triple)
编码“向内滚奏抬指比向外翻转抬指简单”的经验物理：
若连续三次同手释放轨道为 $c_1, c_2, c_3$，时间跨度在 $404.3\mathrm{ms}$ 内（参数 `r_order_tau`），计算跳跃超出跨距的超额量：

$$
\mathrm{excess} = \max\Big(|c_1 - c_2| + |c_2 - c_3| - \big(\max(c_1,c_2,c_3) - \min(c_1,c_2,c_3)\big),\; 0.0\Big)
$$

以惩罚权重参数 `r_order_pen = 0.04983` 加成入释放通道。

全释放分量经滑动平滑窗口（参数 `w_r = 1349.4ms`）输出 $R_{\mathrm{bar}}$。

---

## 7. 长键协调几何矩阵：Cbar v2 (LNCoord)

当某根手指按住长键时，其余手指运动会遭受严重的生理肌腱牵制。`lncoord_v2` 在行网格上计算 7 个独立的几何子项，通过解剖学跨列代价矩阵 $M$ 综合：

### 7.1 解剖学跨列代价矩阵 $M$
对于轨道对 $(c_1, c_2)$ ，依据所属手位类别查找对应代价：
- 同手相邻列 (`sh_adj`): $0.9898$ ；
- 同手相隔一列 (`sh_split`): $0.6943$ ；
- 拇指跨接 (`1h_split`): $0.6670$ ；
- 跨手近邻 (`2h_split`): $0.6285$ ；
- 跨手远端 (`cross`): $0.6384$ 。

### 7.2 七大几何分量定义
1. **重叠按住 (Overlap, `cbv_overlap` = 0.9864)**：多个长键并行按住时的持续时间与列对约束重叠；
2. **局部护盾 (Shield, `cbv_shield` = 1.1201)**：长键头部前极短时间内同轨或相邻轨单键的敲击干扰；
3. **锁手叠 (Lockstack, `cbv_lockstack` = 0.5382)**：一只手指按住长键时，同手其余手指遭遇同轨连打；
4. **锁手长跑 (Lockanchor, `cbv_lockanchor` = 1.3024)**：在长键按住背景下，其余手指承受高速单轨长跑；
5. **按住时长累积 (Holdage, `cbv_holdage` = 3.4071)**：长键超长按住造成的局部肌肉僵直负担；
6. **长键切换频次 (Churn, `cbv_churn` = 2.7804)**：高频交替点按/松开长键的动作转换阻力；
7. **按住同时敲击 (Holdtap, `cbv_holdtap` = -1.0227)**：按住长键期间伴随的高密度点按。

各分量经加权和并经滑动平滑窗口（参数 `w_cb = 1716.9ms`）输出 $C_{\mathrm{bar}}$。

---

## 8. 生理约束与辅助通道：HCd, SHd, EyeR, Recov

### 8.1 A4 双手并行公理修复：同手超加性牵制费 (HCd)
为消除“单手 3 押比双手 2+1 押计算更便宜”的 A4 公理倒挂，在流分支注入同手超加性牵制费：
1. 依据 $\tau_c = 28\mathrm{ms}$ 的三角窗计算左手按键负荷 $u_L$ 与右手按键负荷 $u_R$ ；
2. 计算单手超额超加性惩罚（指数参数 `hc_e = 1.500`）：

$$
\mathrm{HCd} = \max(0, u_L - 1.0)^{1.5} + \max(0, u_R - 1.0)^{1.5}
$$

3. 以权重 $d_{\mathrm{hc}} = 0.400$ 注入总流通道。单手同时按键数 $\ge 2$ 时必然产生陡增的牵制费，彻底消除倒挂。

### 8.2 视觉规整度 (EyeR)
评估击打时间与局部节奏拍点的对齐熵。视觉极为规整的谱面会被纯生理手部模型系统性低估，因此以正向密度门控交互形式进入：

$$
E_{\mathrm{term}} = d_{\mathrm{eye}} \cdot P_{\mathrm{bar}} \cdot \mathrm{EyeR}, \quad d_{\mathrm{eye}} = 0.08187 \ (\text{4K 为 } 0.01401)
$$

### 8.3 恢复折扣 (Recov)
连续高难密集段落后若紧随稀疏段，肌肉进入恢复状态，以 $d_{\mathrm{rec}} = 0.78125$ 实施动态难度折减。

---

## 9. 局部瞬时难度合成：D(t) (Combine)

在每一行 $r$ 上，将上述各通道融合成标量总难度 $D_r$ ：

### 9.1 分支非线性饱和函数

定义饱和函数 $\mathrm{branch}(v, \mathrm{mult})$：

$$
\mathrm{branch}(v, \mathrm{mult}) = A^{\mathrm{mult} / K_s} \cdot \min\big(\max(v, 0), \; \mathrm{cap}_a + \mathrm{cap}_b \cdot \max(v, 0)\big)
$$

其中饱和截断参数取值 $\mathrm{cap}_a = 10.8467, \; \mathrm{cap}_b = 0.7670$。各连打分支饱和计算如下（分支指数参数 $a_j = 3.4083$）：

$$
J_{m, \mathrm{sat}} = \mathrm{branch}(J_m, a_j), \quad J_{c, \mathrm{sat}} = \mathrm{branch}(J_c, a_j), \quad J_{a, \mathrm{sat}} = \mathrm{branch}(J_a, a_j)
$$

### 9.2 流分支聚合 (Stream Branch)

$$
\mathrm{stream} = A^{a_p} \cdot \left( a_p \cdot P_{\mathrm{bar}} + \frac{a_r \cdot R_{\mathrm{bar}}}{C + a_c} + \frac{a_{cb} \cdot C_{\mathrm{bar}}}{C + a_{cbc}} + d_{\mathrm{sh}} \cdot \mathrm{SHd} + d_{\mathrm{hc}} \cdot \mathrm{HCd} + E_{\mathrm{term}} \right)
$$

参数（7K / 4K 独立覆盖）：
- 7K: $a_p = 0.4571, a_r = 59.955, a_{cb} = 4.000, a_c = 10.717, a_{cbc} = 10.311$ ；
- 4K 覆盖: $a_{p, 4\mathrm{K}} = 0.9672, a_{r, 4\mathrm{K}} = 0.0, a_{cb, 4\mathrm{K}} = 11.357$。

### 9.3 连打与流的广义幂和 $S$
采用广义幂和指数 $s_p = 1.1397$（参数 `s_p`）：

$$
S = \left( c_{\mathrm{jm}} \cdot J_{m, \mathrm{sat}}^{s_p} + c_{\mathrm{jc}} \cdot J_{c, \mathrm{sat}}^{s_p} + c_{\mathrm{ja}} \cdot J_{a, \mathrm{sat}}^{s_p} + c_s \cdot \mathrm{stream}^{s_p} \right)^{1 / s_p}
$$

权重： $c_{\mathrm{jm}} = 0.5453, c_{\mathrm{jc}} = 0.01839, c_{\mathrm{ja}} = -0.00125, c_s = 0.3931$ 。

### 9.4 扭曲因子 $T$ 与总难度 $D$

$$
\begin{aligned}
X_t &= \max(X_{\mathrm{bar}}, 0) + t_{\mathrm{jm}} \cdot J_{m\_b}, \quad t_{\mathrm{jm}} = -0.3555 \\\\
T &= A^{a_t / K_s} \cdot \frac{X_t}{X_t + S + 0.4820}, \quad a_t = 5.7043 \\\\
D(t) &= d_{b1} \cdot S^{d_{\mathrm{ds}}} \cdot \max(T, 10^{-9})^{d_{\mathrm{dt}}} + d_{b2} \cdot S
\end{aligned}
$$

参数取值：$d_{b1} = 1.4529, d_{b2} = 0.4343, d_{\mathrm{ds}} = 0.5437, d_{\mathrm{dt}} = 1.3838$，偏置参数 `t_s_off = 0.4820`。

---

## 10. 两级物量加权分段聚合：agg_mode=4 (Aggregation)

针对 v1.0.0 旧聚合的 T1-T4 压力测试缺陷，v1.1.0 建立两级物量加权聚合体系：

### 10.1 参与物量质量 Engagement Mass $m_r$
每一行 $r$ 的有效物理物量定义为：

$$
m_r = \big(\max(C_r, 0)\big)^{1.155} \cdot \min(\Delta t_r,\; 175.549)^{0.966}
$$

- 密度物量指数：参数 `aggC_wc = 1.15498`；
- 间隙上限截断：参数 `aggC_gc = 175.549ms`（攻克 T1 散键抢权）；
- 间隙平滑指数：参数 `aggC_gw = 0.96625`。

### 10.2 物理内容切段与定长窗口网格
1. **静音断句**：当 $\Delta t_r \ge 1854.5\mathrm{ms}$ （参数 `aggC_gap`）时，判定为物理分段重置，空白休息段物量严格为 0（攻克 T3）；
2. **窗口化**：在每个内容段内，以段首为锚点切分定长窗口，窗长 $W_{\mathrm{ms}} = 6902.95\mathrm{ms}$ （约 6.9s，参数 `aggC_ms`），步长 $500\mathrm{ms}$。

### 10.3 Level 1: 窗内物量加权幂平均
在每个窗口 $w$ 内，计算物量加权幂平均作为该窗的局域代表难度 $d_1[w]$ （幂次参数 `aggC_pw1 = 2.13656`）：

$$
\begin{aligned}
M_w &= \sum_{r \in w} m_r \\\\
d_1[w] &= \left( \frac{\sum_{r \in w} m_r \cdot D_r^{p_1}}{\max(M_w, 10^{-12})} \right)^{1 / p_1}, \quad p_1 = 2.13656
\end{aligned}
$$

### 10.4 Level 2: 跨窗峰包络积分方程与 26 次二分求根
跨窗口聚合建立在带休息衰减的连续峰包络积分方程上：
1. 跨窗休息衰减系数（衰减常数参数 `aggC_tau = 4000.57ms`）：

$$
\mathrm{DEC}_w = \exp\left(-\frac{\Delta t_{\mathrm{gap}, w}}{\tau}\right), \quad \tau = 4000.57\mathrm{ms}
$$
2. 目标函数二分求解：二分搜索区间 $[0, \max(d_1) + 1.0]$ ，进行严格 **26 次二分迭代**。在每轮中测试 $\mathrm{mid}$ ：

$$
\begin{aligned}
F_w &= \max\big(0.0,\; F_{w-1} \cdot \mathrm{DEC}_w + M_w \cdot (d_1[w] - \mathrm{mid})\big) \\\\
\mathrm{peak} &= \max_w F_w
\end{aligned}
$$

   若 $\mathrm{peak} \le \kappa$ （收敛阈值参数 `aggC_kap = 0.81587`），说明 $\mathrm{mid}$ 是可行解，收缩上界；否则收缩下界。
3. 26 次迭代后返回区间中点，即为最终全谱聚合难度 $\mathrm{SR}_{\mathrm{agg}}$ 。

---

## 11. 谱面级后处理与全局标定 (Postprocess)

得到 $\mathrm{SR}_{\mathrm{agg}}$ 后，通过单段全局线性映射输出最终官方星数：

1. **短谱物量渐进补偿**（补偿参数 `pp_n0 = 2.8826`）：

$$
\mathrm{SR}_1 = \mathrm{SR}_{\mathrm{agg}} \cdot \frac{N_{\mathrm{notes}}}{N_{\mathrm{notes}} + 2.8826} 
$$

2. **极高星级软压缩**：

$$
\mathrm{SR}_2 = \begin{cases}
\mathrm{SR}_1, & \mathrm{SR}_1 \le 8.3100 \\\\
8.3100 + \frac{\mathrm{SR}_1 - 8.3100}{1.5281}, & \mathrm{SR}_1 > 8.3100
\end{cases}
$$

   参数取值：`pp_thr = 8.3100`，`pp_div = 1.5281`。
3. **全局线性仿射标定**：

$$
\mathrm{SR} = \max\big(0.0,\; \mathrm{SR}_2 \cdot 0.98662 \cdot 0.98771 + 0.68240\big)
$$

   参数取值：`pp_scale = 0.97524`，`calib_a = 0.98747`，`calib_b = 0.77542`。

输出标量 $\mathrm{SR}$ 即为 SPM Rating v1.1.1 官方 Star Rating。

---

## 附录 A：手部拓扑与几何常数表

| 7 轨坐标 | 对应手指 (7K) | 对应手指 (4K 嵌入) | 解剖学 AIN 组 | 所属手位 HAND |
|---|---|---|---|---|
| 列 0 | 左手无名指 | （置空） | 0 (无名指) | 0 (左手) |
| 列 1 | 左手中指 | 左手中指 | 1 (中指) | 0 (左手) |
| 列 2 | 左手食指 | 左手食指 | 2 (食指) | 0 (左手) |
| 列 3 | 拇指 | （置空） | 3 (拇指) | 1 (拇指) |
| 列 4 | 右手食指 | 右手食指 | 2 (食指) | 2 (右手) |
| 列 5 | 右手中指 | 右手中指 | 1 (中指) | 2 (右手) |
| 列 6 | 右手无名指 | （置空） | 0 (无名指) | 2 (右手) |

---

## 附录 B：发布参数全集表 (276 项)

以下参数为 `params/spm-rating-1.1.1.json` 中保存的单一真值，单文件引擎 `interface/python/spm_engine.py` 严格内嵌相同数值并通过自动化测试守护：

```json
{
{
  "a_c": 13.380682264603966,
  "a_cb": 3.290737504274206,
  "a_cb_4k": 10.329032565184669,
  "a_cbc": 4.546298626916172,
  "a_p": 0.457050886903945,
  "a_p_4k": 1.0271932840768654,
  "a_r": 71.93325028558291,
  "a_r_4k": 0.0,
  "abar_active": 0.02531712913365081,
  "abar_c0": 0.729854838334696,
  "abar_c1": 0.6730069593477146,
  "abar_k": 5.869324658227848,
  "abar_mx": 0.48830826958952744,
  "abar_mx_thr": 0.10636359112398937,
  "abar_mx_w": 0.45817570312814426,
  "abar_scale": 1.009862473386292,
  "abar_thr_hi": 0.09447563280275387,
  "abar_thr_lo": 0.01682395583948343,
  "abar_weld": 1.0,
  "aggC_gap": 1919.100392694976,
  "aggC_gap_4k": 0.0,
  "aggC_gc": 206.24663914625867,
  "aggC_gw": 1.0018741768102066,
  "aggC_hw": 1.0,
  "aggC_kap": 0.8250936661252044,
  "aggC_ms": 6903.358465661703,
  "aggC_ms_4k": 0.0,
  "aggC_pw1": 2.1462221289544194,
  "aggC_pw2": 5.549120840435838,
  "aggC_tau": 4027.709861084976,
  "aggC_wc": 1.1528146438243094,
  "agg_C": 5.7765188650680015,
  "agg_gamma": 0.3142734186203886,
  "agg_gap_w": 1.0873927743990874,
  "agg_k": 3.923290274033005,
  "agg_mode": 4.0,
  "agg_nseg": 30.0,
  "agg_wc": 1.145022164188944,
  "aggw_gap_cap": 0.0,
  "aggw_src": 0.0,
  "aj": 3.4082562260373974,
  "an_a0": 0.17043768459177114,
  "an_a1": 0.34850214310770977,
  "an_cubic": 6.806495193662656,
  "an_on_p": -0.3268168035336469,
  "ap": 1.039645612096299,
  "at": 5.704296617075204,
  "c_chord2": 0.0,
  "c_fj": 1.1223698076832407,
  "c_fj_4k": 1.6384853006440867,
  "c_ja": -0.001250731346377573,
  "c_jc": 0.018393884652087668,
  "c_jm": 0.5452603821867452,
  "c_s": 0.3931375724076219,
  "c_stam": 0.0,
  "calib_a": 0.9877134644832144,
  "calib_b": 0.6823998306913243,
  "cap_a": 10.84672199638674,
  "cap_b": 0.7669642744433789,
  "cb_holdtap": 0.0,
  "cb_holdtap_pow": 1.0,
  "cb_lock_pow": 1.0,
  "cb_lockanchor": 0.0,
  "cb_lockjack": 0.0,
  "cb_par": 0.0,
  "cb_shield": 0.0,
  "cb_shield_lock": 0.8062,
  "cb_shield_tau": 56.24,
  "cbv_churn": 2.7804415879691047,
  "cbv_churn_4k": 0.0018149727082640878,
  "cbv_holdage": 3.4070743387277163,
  "cbv_holdage_4k": 0.0017655285135421454,
  "cbv_holdtap": -1.0227403453455164,
  "cbv_lock_pow": 0.9662307730974806,
  "cbv_lockanchor": 1.3023578337914123,
  "cbv_lockstack": 0.5382064568725327,
  "cbv_ls_1h_split": 0.6669564049783417,
  "cbv_ls_2h_split": 0.628481439284875,
  "cbv_ls_cross": 0.6384201937709314,
  "cbv_ls_sh_adj": 0.9897529278156161,
  "cbv_ls_sh_split": 0.694320421215397,
  "cbv_ov_1h_split": 0.8912801595852077,
  "cbv_ov_2h_split": 0.628660224600841,
  "cbv_ov_cap": 2451.2427577736403,
  "cbv_ov_cross": 0.43216200109373926,
  "cbv_ov_dt_max": 1511.5378347371243,
  "cbv_ov_sh_adj": 1.2272220519138874,
  "cbv_ov_sh_split": 1.2153000673016874,
  "cbv_overlap": 0.9863986916717904,
  "cbv_sh_dt_max": 256.0406131304308,
  "cbv_sh_ln_tau": 145.443782024068,
  "cbv_sh_ln_w": 1.0790072249120857,
  "cbv_sh_lock": 1.0870780764260457,
  "cbv_sh_tau": 75.33463318028997,
  "cbv_shield": 1.1200999548581216,
  "cbv_str_1h_split": 0.5323308514011785,
  "cbv_str_2h_split": 0.17479688357773077,
  "cbv_str_bal": 1.141728630652899,
  "cbv_str_cross": 0.09922357503493356,
  "cbv_str_h0": 0.8793037998512705,
  "cbv_str_h1": 0.9113686383081036,
  "cbv_str_h2": 0.8557205717895041,
  "cbv_str_h3": 0.5028456653288493,
  "cbv_str_maxd": 2.048176307602251,
  "cbv_str_sh_adj": 0.19036255960279166,
  "cbv_str_sh_split": 0.9209759560022021,
  "cbv_straddle": 0.45854924271224995,
  "cbv_w": 1140.2267181326677,
  "chord2_hard": 1.0,
  "chord2_w": 998.3068456460403,
  "chord_no_dilute": 1.0,
  "chord_soft_att": 1.0,
  "chord_tau_c": 28.00006103515625,
  "churn_event_rate": 0.0,
  "churn_sat": 14.712127862753169,
  "churn_w": 1022.5551356347157,
  "cj_norm": 1.9757525590733565,
  "cj_size_exp": 1.0152415724691892,
  "d_b1": 1.4529112152056358,
  "d_b2": 0.4343400978218659,
  "d_ds": 0.5436886594030598,
  "d_dt": 1.3838263378115971,
  "d_eye": 0.0818717181241305,
  "d_eye_4k": 0.014013206098578592,
  "d_hc": 0.4,
  "d_read": 0.0,
  "d_read_4k": 0.0,
  "d_rec": 0.78125,
  "d_sh": 0.0,
  "env_ms": 395.28032205483703,
  "env_w": 0.0033333333333333335,
  "eye_tau": 0.2,
  "eye_w_ref": 4000.0,
  "eye_w_s": 500.0,
  "fj_bounded": 0.0,
  "ga_alpha": 0.0,
  "ga_gam": 1.0,
  "ga_pw": 1.0,
  "ga_q": 1.0,
  "hc_e": 1.5,
  "holdage_cap": 0.9712508526439506,
  "holdage_th": 0.25073103183213685,
  "hw_a": -4.833333333333334,
  "hw_half": 51.932695284873475,
  "j_agg_pow": 4.220517945172583,
  "j_c1": 0.11994305455701491,
  "j_nerf_a": -2.266651862989442e-05,
  "j_nerf_c": 0.07443739821757173,
  "j_nerf_off": 0.15369105717534054,
  "j_triple_gain": 0.01930565188193239,
  "j_triple_gain_4k": 0.5356436393950172,
  "j_triple_pow": 1.005976267076746,
  "j_triple_tau": 135.97491132645644,
  "ja_coord_exp": -9.268498548764114e-05,
  "ja_gap_thr": 253.00356478769385,
  "ja_len_exp": 1.0047596122947928,
  "ja_len_floor": 4.074244535477681,
  "ja_norm": 5.981391018054386,
  "ln_eff_tail_ms": 0.0,
  "ln_note_level": 0.0,
  "mean_pow": 6.465267031930503,
  "mj_dilute": 0.08,
  "mj_dilute_exp": 0.9615171298692908,
  "note_chord": 1.0,
  "od_mult_ja": -1.0,
  "od_mult_jc": -0.37109374999999944,
  "od_mult_jm": 0.7504257038347726,
  "od_mult_p": 1.1264849156945338,
  "od_mult_r": 1.064323992348447,
  "od_mult_x": 1.1651306736110387,
  "out_a": 0.8686939253627315,
  "out_b": -0.9591165705858606,
  "p83_c": 2.667686959362696,
  "p93_c": 2.505172683584436,
  "p_boost": -1e-05,
  "p_boost_hi": 401.3793072521453,
  "p_boost_lo": 186.3466709871835,
  "p_burst3": 0.05749725160072005,
  "p_burst4": 0.041571581046874806,
  "p_bv_max": 1.023169107328028,
  "p_chw2": 0.6997350782823039,
  "p_chw3": 1.4710731528472405,
  "p_chw4": 0.82149995851254,
  "p_chw5": 0.8676070127257947,
  "p_chw6": 0.7474127532548505,
  "p_chw7": 1.03284160083206,
  "p_lam2": 0.006084380903675146,
  "p_lam3": 27.838293996532915,
  "p_sat_a": 10.690152688495372,
  "p_sat_b": 2.0349325289971496,
  "p_scale": 0.05862260059258437,
  "p_v_norm": -0.005007331887365899,
  "pp_div": 1.559992046815954,
  "pp_n0": 0.0,
  "pp_scale": 0.9866220368284715,
  "pp_thr": 8.416405509249257,
  "r_I_off": 1.679847001293981,
  "r_I_steep": 3.8155719025104933,
  "r_I_w": 1.0611113361746765,
  "r_ain0": 1.402277599248007,
  "r_ain1": 0.8487725298502042,
  "r_ain2": 0.9889530992577216,
  "r_ain3": 1.4808396303580251,
  "r_coord_e": 0.6807568539174652,
  "r_cw_cross": 0.6899948287076584,
  "r_cw_hand": 0.42489506847315023,
  "r_cw_same": 0.9824539263359018,
  "r_cw_thumb": 0.3660514384223894,
  "r_dt_max": 5985.197267885601,
  "r_dtr_min": 0.015,
  "r_lock": 0.1775454477373913,
  "r_order_pen": 0.04982726415201301,
  "r_order_tau": 404.3308885287425,
  "r_same_col": 0.39414011153048933,
  "r_seq": 0.06596340070980779,
  "r_short_red": 0.10936100877002214,
  "r_short_thr": 440.09681230961536,
  "r_sim_tau": 10.0,
  "r_simul": 0.08534094291675064,
  "r_soft_edges": 1.0,
  "r_straddle": -0.11540999412449468,
  "r_tail": 0.02898542543998874,
  "r_tt": 3.5417544104085192,
  "read_clip_hi": 1.5,
  "read_clip_lo": 0.5,
  "read_n_win": 16.0,
  "read_tol": 0.18,
  "rec_half": 0.4886690846205094,
  "recov_w": 989.9492044110127,
  "recov_wl": 21003.231873446508,
  "s_p": 1.1396559261442123,
  "shd_w": 1071.1117107781824,
  "stam_grp": 30.0,
  "stam_tau": 8000.0,
  "t_jack_mix": -0.35552500051452174,
  "t_s_off": 0.4819735114526039,
  "use_cbar_v2": 1.0,
  "use_cbv_holdtap": 1.0,
  "use_cbv_lockanchor": 1.0,
  "use_cbv_lockstack": 1.0,
  "use_cbv_overlap": 1.0,
  "use_cbv_shield": 1.0,
  "use_cbv_straddle": 1.0,
  "use_eye": 1.0,
  "use_hb": 1.0,
  "use_read": 0.0,
  "w83": 0.4815139944602175,
  "w93": 0.3605157897072774,
  "w_a": 60.794584215917084,
  "w_cb": 1716.856156378797,
  "w_ja": 0.0,
  "w_jc": -0.4781249999999999,
  "w_jm": 842.9060364136417,
  "w_p": 806.2241563795392,
  "w_r": 1349.4219390385326,
  "w_x": 939.2935347185421,
  "wmean": 2.1613377064577026,
  "x_amp": 0.14776064699735425,
  "x_cw0": 0.1460113252705887,
  "x_cw1": 0.3535388218642357,
  "x_cw2": 0.3389326726836199,
  "x_cw3": 0.04918990446028124,
  "x_din": -0.595854461464495,
  "x_dir_in": 0.2845860534263308,
  "x_dir_out": -0.016749922744166222,
  "x_dout": 0.9858637139438629,
  "x_fc_a": 0.541953977488367,
  "x_fc_floor": 0.04577642898224647,
  "x_fc_off": 60.317605380705,
  "x_fc_w": 1.6631718126735746,
  "x_jd_e": 1.0765985987293114,
  "x_jd_p": 0.0456784859552089,
  "x_jh_e": 1.6812589290379765,
  "x_jh_p": 1.0261744953668672,
  "x_jt": 0.5592172136082718,
  "x_jump_w": -0.852278272911138
}
```
