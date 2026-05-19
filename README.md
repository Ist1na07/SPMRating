# SPM Rating — 7K osu!mania 难度评级算法

[English](README_EN.md) | 中文

一个开源的 osu!mania **7K** 难度评级算法。

**核心设计**：7 个独立的难度分量从多维度提取谱面结构特征——叠键密度 (Jack)、列间协调 (Cross)、连击密度 (Stream)、LN 释放交互 (Release)、锚点/卡手 (Anchor)、护盾保护 (Shield)、密度反转 (Inverse)。所有分量在统一的 Precompute/Combine 架构下按 ~500Hz 时间分辨率合成瞬时难度 D(t)，最后通过 Sigmoid 玩家准度模型聚合为与社区共识高度一致的星数评级。

基于 **311 张谱面**（148 Dan + 57 Tournament + 20 Graveyard + 86 Ranked），通过交替块式 Nelder-Mead 优化训练各分量权重与聚合参数。

## 快速开始

### 方式一：独立单文件（推荐）

`spm_calc_standalone.py` 是**完全独立**的单文件，除 numpy 外零依赖：

```bash
pip install numpy
python spm_calc_standalone.py chart.osu         # 单张谱面
python spm_calc_standalone.py "D:/osu/Songs/"  # 批量扫描
```

内联了全部算法代码 + 最优参数，无需目录结构。

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

## 结果

| 指标 | 值 |
|------|-----|
| **MAE** | 0.2180 |
| **Loss** | 0.9321 |
| **相关性** | 0.9881 |
| **Pass@0.5** | 89.4% |

按来源 (311 张):
- Dan 段位 (148 张): MAE ~0.20
- Tournament (57 张): MAE ~0.22
- Graveyard (20 张): MAE ~0.38
- Ranked (86 张): MAE ~0.25

## 算法架构

### 1. 特征层（36 参数）

在 ~500Hz 时间格点上计算 7 个分量：

| 分量 | 物理含义 |
|------|---------|
| **Jbar** | 叠键密度 |
| **Xbar** | 列间距离加权难度 |
| **Pbar** | 连续打击密度 |
| **Rbar** | LN 释放交互难度 |
| **Abar** | 锚点/卡手配置 |
| **Sbar** | 护盾保护（贡献小） |
| **Vbar** | 密度反转惩罚（贡献显著） |

各分量合并为瞬时难度 D(t):

```
S(t) = w1 · Jbar + (1-w1) · p_norm(Xbar, Pbar, exponent=p)
T(t) = p_norm(Rbar, Abar, exponent=p)

D(t) = β1 · √S · T^1.5 + β2 · S
        + α_P · Pbar + α_R · Rbar/(C_step + α_C)
        + α_S · Sbar(t) + α_V · Vbar(t)
```

### 2. Sigmoid 玩家准度聚合

不使用分位点截取，而是通过准度模型求解：

$$A(d) = \frac{A_{max}}{C + e^{k(d-D)}}$$

其中:
- **k = 2.09**: 衰减速度（d-D=+1 → 准度跌至 9%，+2 → 2%）
- **C = 3.97**: 曲线形状（匹配点准度 = 1/(C+1) ≈ 20%）
- **γ = 0.196**: 目标平均准度分数

对每张谱面求解 D 使得加权平均准度等于目标：

$$\sum \frac{w_i}{C + e^{k(D_i-D)}} = total\_W \cdot \gamma$$

通过二分法求解 → D_solved 即为原始 SR。

### 3. 后处理

```
SR = D_solved × n_eff/(n_eff + N0)       # 物量归一化 (N0=8.21)
if SR > 9.42: SR = 9.42 + (SR - 9.42)/2.01  # 高 SR 压缩
SR *= 1.055                                # 全局缩放
```

### 4. D 预校准

D(t) 在送入 Sigmoid 前经过线性预校准：

$$D'(t) = 0.893 \cdot D(t) + 0.031$$

补偿百分位聚合校准带来的偏差。

### 5. 子模型

- **RC 模型**: 禁用 Rbar/Sbar/Vbar（纯 Rice，LN 头当作单点）
- **LN 模型**: 仅用 LN 段落掩码聚合（排除 RC 主导段干扰）

## 参数文件

| 文件 | 内容 |
|------|------|
| `tuned_params_sigmoid.json` | Total SR 参数（Sigmoid 聚合, MAE=0.2180） |
| `tuned_params_rc.json` | RC 子模型参数 |
| `tuned_params_ln.json` | LN 子模型参数 |

## 目录结构

```
spm_rating/
├── README.md                        # 本文档
├── README_EN.md                     # English version
├── LICENSE                          # MIT
├── spm_calc_standalone.py           # ★ 独立单文件 SR 计算器 (推荐)
├── spm_calc.py                      # 模块版计算器
├── tune_terminal.py                 # 交互式调参终端
├── tuned_params_sigmoid.json        # 最优 Sigmoid 参数
├── tuned_params_rc.json             # RC 子模型参数
├── tuned_params_ln.json             # LN 子模型参数
├── docs/
│   ├── TUNING_METHODOLOGY.md       # 调参方法论
│   └── TUNING_METHODOLOGY_EN.md    # English version
├── spm_rating/                      # 核心算法
│   ├── rating.py                    # precompute() + combine() 入口
│   ├── aggregate_sigmoid.py         # Sigmoid 聚合（二分求解）
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
│       └── stamina.py               # 耐力 (禁用)
├── tuning/                          # 调参工具
│   ├── data_loader.py               # Playtest 数据加载
│   └── scorer.py                    # 评分函数
└── scripts/                         # 调参/训练脚本
    ├── tune_sigmoid_k15.py          # 核心：Sigmoid NM 调参
    ├── tune_sigmoid_alternating.py  # 交替块式 NM
    ├── tune_rc.py / tune_ln.py      # 子模型调参
    ├── fit_ln_masked.py             # LN 掩码聚合
    ├── fit_dan_regression.py        # 段位映射
    ├── sweep_k_fine.py              # k 值精细扫描
    ├── rebuild_enhanced_cache.py    # 缓存重建
    ├── build_standalone.py          # 构建独立单文件
    ├── train_sort_classifier.py     # 谱面类型分类器
    └── train_tag_classifier.py      # 键型标签分类器
```

## 技术细节

### D_solved 与 D(t) 分布的关系

Sigmoid 求解的 D_solved 极其稳定地落在 D(t) 加权分布的 **P70** 附近：

$$D_{solved} \approx 0.87 \cdot D_{P70}$$

std 仅 0.01（跨 311 张谱面高度一致）。这验证了 Sigmoid 模型的物理一致性——不是"随机"聚合，而是通过准度方程等价于对 D 分布做光滑的软分位选择。

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

**γ ≈ 1/(C+1)** 精确成立: C=3.97 时 γ≈0.196 ≈ 1/5.1。目标准度等于匹配点准度，验证了模型的内部自洽性。

## 评估

在 playtest 数据集上评估算法:

```python
from tuning.data_loader import load_playtest_data
from tuning.scorer import score_single
from spm_rating.rating import precompute, combine
import json

entries = load_playtest_data()  # 需 maps/ 目录含 Excel + .osu 文件
params = json.load(open("tuned_params_sigmoid.json"))["params"]

for entry in entries:
    cache = precompute(entry["osu_path"], use_enhanced=True, params=params)
    sr, _ = combine(cache, params=params)
    print(f"{entry['mapfile']}: SR={sr:.2f}, Ref={entry['sr_ref']:.2f}")
```

## 调参方法

参数通过**交替块式 Nelder-Mead** 优化（6 块 × 2 轮，50→25 次迭代）:

| 块 | 参数 | 模式 | 贡献 |
|----|------|------|------|
| B1 | k, C, γ, calib, N0, threshold, divisor, scale | Fast (~0.04s) | ~95% 改善 |
| B2 | D 公式权重与指数 | Full (~14.5s) | 微调 |
| B3a | Cross 特征层参数 | Full | 改善极小 |
| B3b | Release 特征层参数 (含 short_ln) | Full | 改善极小 |
| B3c | Inverse 特征层参数 (含 same_col_bonus) | Full | 改善极小 |
| B3d | Jack 聚合参数 | Full | 改善极小 |

核心发现: **k=2.09 最优**（最大单一改善）。

完整方法见 `docs/TUNING_METHODOLOGY.md`。

## 版本历史

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

## 插件

本算法驱动 **SPM Map Analyser** tosu 游戏内悬浮窗插件:
- 仓库: [Ist1na07/spm_rating_map_analyser](https://github.com/Ist1na07/spm_rating_map_analyser)
- 提供实时难度显示，含 RC/LN 子模型、段位映射、ML 键型分类

## 依赖

- **numpy** (必需)
- **scipy** (仅 Nelder-Mead 调参需要)
- **pandas** (仅 playtest 评估需要)

## License

MIT — 详见 [LICENSE](LICENSE)
