# SPM Rating v0.4.0 — 7K osu!mania 难度评级算法

[English](README_EN.md) | 中文

一个开源的 osu!mania **7K** 难度评级算法。

## 起源

本算法以 **[Star-Rating-Rebirth](https://github.com/SunnyO8/Star-Rating-Rebirth)**（sunny rework）为基础，经以下方向的修改演化而来：

- **难度分量扩展**：在原始 jack / stream 等分量之外，新增 inverse（密度反转惩罚）、shield（护盾保护）、release（LN 释放交互）等分量，并重写 cross（列间距离加权）。
- **D(t) 公式重构**：将分量合并方式从线性叠加改为非线性组合（`β1·√S·T^1.5 + β2·S + α·分量` 形式），更贴近实际疲劳叠加规律。
- **玩家准度聚合模型**：以准度方程 `A(d) = A_max / (C + e^{k(d-D)})` 替代原始分位点截取，对每张谱面求解使加权平均准度等于目标值 `γ` 的难度 `D_solved`，作为原始 SR。该聚合方式提供物理可解释的难度-准度映射。
- **D 预校准**：在聚合前对 D(t) 做线性变换 `D' = 0.893·D + 0.031`，补偿分量公式带来的系统性量级偏差。
- **特征修正层**：在主公式之上叠加 9 个谱面级特征的线性修正，捕捉 D 公式难以表达的结构性偏差（详见下文）。
- **后处理重优化**：物量归一化、高 SR 压缩、全局缩放三段式后处理参数与修正层联合训练。

## 快速开始

### 方式一：独立单文件（推荐）

`spm_calc_standalone.py` 是**完全独立**的单文件，除 numpy 外零依赖：

```bash
pip install numpy
python spm_calc_standalone.py chart.osu         # 单张谱面
python spm_calc_standalone.py "D:/osu/Songs/"  # 批量扫描
```

内联了全部算法代码与最优参数，无需目录结构。

### 方式二：模块导入

```bash
python spm_calc.py chart.osu                    # 依赖 spm_rating/ 包
```

### 编程接口

```python
from spm_calc_standalone import compute_sr_map

sr, details = compute_sr_map("chart.osu")
print(f"SR = {sr:.4f}")
print(f"D_solved = {details['D_solved']:.2f}")
```

## 测试表现

| 指标 | 值 |
|------|-----|
| **Loss** | 0.6935 |
| **MAE** | 0.2068 |
| **相关性 r** | 0.9889 |
| **Inside%**（误差在容忍范围内） | 83.3% |
| **配对 t-test p** | 0.0007 |

相对前版（v0.3.0）改善：Loss −9.9%，统计显著。改善主要来自 RC（rice）与 LN 谱面。

## 算法架构

### 1. 难度分量层

在 ~500Hz 时间格点上计算 7 个分量：

| 分量 | 物理含义 |
|------|---------|
| **Jbar** | 叠键密度 |
| **Xbar** | 列间距离加权难度 |
| **Pbar** | 连续打击密度 |
| **Rbar** | LN 释放交互难度 |
| **Abar** | 锚点 / 卡手配置 |
| **Sbar** | 护盾保护（贡献小） |
| **Vbar** | 密度反转惩罚（贡献显著） |

各分量合并为瞬时难度 D(t)：

```
S(t) = w1 · Jbar + (1-w1) · p_norm(Xbar, Pbar, exponent=p)
T(t) = p_norm(Rbar, Abar, exponent=p)

D(t) = β1 · √S · T^1.5 + β2 · S
        + α_P · Pbar + α_R · Rbar/(C_step + α_C)
        + α_S · Sbar(t) + α_V · Vbar(t)
```

### 2. 玩家准度聚合模型

不使用分位点截取，而是通过准度模型求解：

$$A(d) = \frac{A_{max}}{C + e^{k(d-D)}}$$

其中：
- **k = 2.09**：衰减速度（d-D=+1 → 准度跌至 9%，+2 → 2%）
- **C = 3.97**：曲线形状（匹配点准度 = 1/(C+1) ≈ 20%）
- **γ = 0.196**：目标平均准度分数

对每张谱面求解 D 使得加权平均准度等于目标：

$$\sum \frac{w_i}{C + e^{k(D_i-D)}} = total\_W \cdot \gamma$$

通过二分法求解 → D_solved 即为原始 SR。

### 3. 特征修正层

D 公式存在**系统性偏差**（如高估和弦密度、低估 fast jack 疲劳），修正层通过 9 个谱面级特征捕捉这些偏差：

```
correction = Σ w_j × feature_j    (j ∈ {speed, burst, chord, pj, hs, lb, fj, nps_std, chord2})
D_new(t) = D_calib(t) + correction
```

| 特征 | 权重 | 物理意义 |
|------|------|---------|
| **chord2** | -0.656 | 双押密度（恰好 2 列同时击打的事件占比；v0.4.0 新增） |
| **chord** | -0.769 | 和弦密度（≥4 列同时击打协同减负） |
| **fj** | +0.031 | Fast jack（同列快打累积疲劳） |
| hs | +0.073 | 手切（左右手切换协调） |
| lb | +0.016 | 轻爆发（四音组） |
| speed | -0.047 | 速度型模式 |
| burst | -0.029 | 爆发型模式 |
| pj | +0.002 | 流 / jack 平衡 |
| **nps_std** | -0.014 | 密度时变波动（500ms 窗口 NPS 标准差；v0.4.0 新增） |

**v0.4.0 新增特征说明**：

- **chord2**（双押密度）：原 `chord` 特征阈值是 ≥4 列，只覆盖大和弦；chord2 覆盖最常见的双押（jumpstream / chordstream），与 chord 互补。双押密度高表示谱面偏向双指同时发力，每指负担比纯 stream 重但 Pbar（全局 NPS）会高估，故负权重补偿。经拆分实验确认：3 音和弦（chord3）无效、4+ 音和弦（chord4p）与原 chord 冗余（相关性 0.955），仅双押有效。
- **nps_std**（密度时变波动）：将谱面按 500ms 分窗，计算每窗 NPS 标准差。高 nps_std = 爆发段 + 休息段交替（有恢复）；低 nps_std = 全程均匀密度（持续疲劳）。捕捉现有 7 特征缺失的"时变"维度，与 chord2 正交。

修正层是**标量线性模型**（不随时间变化），利用 D_solved 位移不变性实现 ~1500× 加速。L2 正则化（λ=0.01）控制权重幅度，防止过拟合。

完整方法论见 `docs/TUNING_CORRECTION_LAYER.md`。

### 4. 后处理

修正层联合重优化了后处理参数：

```
SR = D_solved × n_eff/(n_eff + N0)       # 物量归一化 (N0=1.029)
if SR > 9.11: SR = 9.11 + (SR - 9.11)/1.97  # 高 SR 压缩
SR *= 1.094                                  # 全局缩放
```

### 5. D 预校准

D(t) 在送入准度聚合前经过线性预校准：

$$D'(t) = 0.893 \cdot D(t) + 0.031$$

补偿分量聚合带来的量级偏差。

## 参数文件

| 文件 | 内容 |
|------|------|
| `tuned_params_sigmoid.json` | 主公式参数（分量权重、聚合参数、预校准） |
| `tuned_correction.json` | 特征修正层权重（v0.4.0：9 特征 + 4 后处理） |
| `tuned_params_rc.json` | RC 子模型参数 |
| `tuned_params_ln.json` | LN 子模型参数 |

## 目录结构

```
SPMRating-Z-Release/
├── README.md                        # 本文档
├── README_EN.md                     # English version
├── LICENSE                          # MIT
├── spm_calc_standalone.py           # ★ 独立单文件 SR 计算器（推荐，含修正层）
├── spm_calc.py                      # 模块版计算器（含修正层）
├── tune_terminal.py                 # 交互式调参终端
├── tuned_params_sigmoid.json        # 最优主公式参数
├── tuned_correction.json            # ★ 特征修正层权重 (v0.4.0：9 特征 + 4 后处理)
├── tuned_params_rc.json             # RC 子模型参数
├── tuned_params_ln.json             # LN 子模型参数
├── docs/
│   ├── TUNING_CORRECTION_LAYER.md   # ★ 修正层调参方法论
│   ├── TUNING_CORRECTION_LAYER_EN.md
├── spm_rating/                      # 核心算法
│   ├── rating.py                    # precompute() + combine() 入口
│   ├── aggregate_sigmoid.py         # 准度聚合（二分求解）
│   ├── aggregate.py                 # 百分位聚合（对照用）
│   ├── combine.py                   # D(t) 公式
│   ├── combine_rc_ln.py             # RC/LN 子模型
│   ├── config.py                    # 参数定义
│   ├── parser.py                    # osu! 谱面解析
│   ├── preprocessor.py              # 预处理
│   ├── utils.py                     # 数学工具
│   └── components/                  # 难度分量
│       ├── jack.py                  # 叠键
│       ├── cross.py / cross_enhanced.py
│       ├── stream.py                # 连击密度
│       ├── release.py / release_enhanced.py
│       ├── anchor.py                # 锚点
│       ├── shield.py                # 护盾
│       ├── inverse.py               # 密度反转
│       └── stamina.py               # 耐力（禁用）
├── tuning/                          # 调参工具
│   ├── data_loader.py               # Playtest 数据加载
│   └── scorer.py                    # 评分函数
└── scripts/                         # 调参 / 训练脚本
    ├── tune_sigmoid_k15.py          # 核心：主公式 NM 调参
    ├── tune_sigmoid_alternating.py  # 交替块式 NM
    ├── tune_rc.py / tune_ln.py      # 子模型调参
    ├── fit_ln_masked.py             # LN 掩码聚合
    ├── fit_dan_regression.py        # 段位映射
    ├── sweep_k_fine.py              # k 值精细扫描
    ├── rebuild_enhanced_cache.py    # 缓存重建
    ├── build_standalone.py          # 构建独立单文件
    ├── retrain_correction_zver.py   # v0.4.0 修正层重训脚本
    ├── residual_diagnosis.py        # 残差诊断
    └── verify_release.py            # 发布验证
```

## 调参方法

### 主公式层：交替块式 Nelder-Mead

参数通过**交替块式 Nelder-Mead** 优化（6 块 × 2 轮，50→25 次迭代）：

| 块 | 参数 | 模式 | 贡献 |
|----|------|------|------|
| B1 | k, C, γ, calib, N0, threshold, divisor, scale | Fast (~0.04s) | ~95% 改善 |
| B2 | D 公式权重与指数 | Full (~14.5s) | 微调 |
| B3a | Cross 特征层参数 | Full | 改善极小 |
| B3b | Release 特征层参数（含 short_ln） | Full | 改善极小 |
| B3c | Inverse 特征层参数（含 same_col_bonus） | Full | 改善极小 |
| B3d | Jack 聚合参数 | Full | 改善极小 |

核心发现：**k=2.09 最优**（最大单一改善）。完整方法见 `docs/TUNING_METHODOLOGY.md`。

### 修正层：L2 正则化线性回归

修正层在主公式之上独立训练（9 个特征权重 + 4 个后处理参数）：

- **优化器**：Nelder-Mead（maxiter=10000, xatol=1e-7, fatol=1e-7, adaptive=True）
- **正则化**：L2（λ=0.01），控制权重幅度
- **交叉验证**：5-fold CV（seed=42）
- **关键技巧**：利用 D_solved 位移不变性加速 ~1500×

完整方法见 `docs/TUNING_CORRECTION_LAYER.md`。

## 技术细节

### k 值的物理含义

玩家在难度超过自身水平时准度急剧下降：

| d - D | A/A_max | 含义 |
|-------|---------|------|
| 0 | 20% | 匹配玩家天花板 |
| +1 | 9% | 略超水平 |
| +2 | 2% | 远超水平 |
| +3 | 0.3% | 完全无法应对 |

符合实际经验：7K 玩家在超出技能天花板 2SR 后基本无法正常游玩。

### C-γ 自洽性

**γ ≈ 1/(C+1)** 精确成立：C=3.97 时 γ≈0.196 ≈ 1/5.1。目标准度等于匹配点准度，验证了模型的内部自洽性。

## 插件

本算法驱动 **SPM Map Analyser** tosu 游戏内悬浮窗插件：

- 仓库：[Ist1na07/spm_rating_map_analyser](https://github.com/Ist1na07/spm_rating_map_analyser)
- 提供实时难度显示，含段位映射、ML 键型分类

## 依赖

- **numpy**（必需）
- **scipy**（仅 Nelder-Mead 调参需要）
- **pandas**（仅 playtest 评估需要）

## 更新日志

### v0.4.0
- **特征修正层扩展**：新增 2 个谱面级特征 **nps_std**（密度时变波动）与 **chord2**（双押密度），特征数 7 → 9
- 经批量特征筛选 + forward selection 确认最优组合；chord2 拆分实验确认仅双押有效（3 音无效、4+ 音与原 chord 冗余）
- In-sample Loss: 0.770 → **0.694**（−9.9%），配对 t-test p=0.0007
- MAE: 0.213 → **0.207**
- 修正层权重与后处理参数联合重训练（5 restarts + 5-fold CV）

### v0.3.0
- **新增特征修正层**：7 个谱面级特征（speed, burst, chord, pj, hs, lb, fj），L2 正则化线性模型
- In-sample Loss: 0.932 → **0.770**（-17.4%）
- CV Test Loss: **0.862**（5-fold, gap=0.092）
- MAE: 0.218 → **0.213**，相关性: 0.988 → **0.989**
- 后处理参数与修正层联合重优化
- 完整调参方法论：`docs/TUNING_CORRECTION_LAYER.md`

### v0.2.0
- 数据集从 213 扩展到 **311 张**（新增 86 Ranked + 12 Tournament）
- k 值重新扫描优化：1.5 → **2.09**（粗扫描 + 精细扫描验证）
- C-γ 自洽性验证：γ ≈ 1/(C+1)，C=3.97, γ=0.196
- 调参块从 5 块扩展为 **6 块**：新增 B3d (Jack)、B3b/B3c 各新增 2/1 参数
- RC 子模型重新训练：MAE=0.2366（改善 27%）
- LN 子模型重新训练：MAE=0.8162（需架构重设计）

### v0.1.0
- 初始发布：Sigmoid 聚合 (k=1.5, C=4.0, γ=0.20) + 7 分量 D 公式
- 基于 213 张谱面训练，MAE=0.2253

## License

MIT — 详见 [LICENSE](LICENSE)
