# 4K 参考标定说明

[English](calibration-4k.md) · [使用说明](../README.zh-CN.md) · [评估与取舍](evaluation.zh-CN.md)

`detail-4k` 底层运行与 v1.3.0 `detail`（`V13-D1`）完全一致的原生评分核，并通过在宽跨度 4K 段位参考集（约 741 谱）上拟合的严格单调递增分段线性映射，将 **4K raw 分**映射到 4K Dan 参考尺度。该映射保序且不改变原始分、谱面排位与局部需求场；其参考值仅在 4K 体系内对齐社区 Dan，既不是 osu! 官方星数，也不代表跨键数等效刻度。

## 留出样本检查

在完全独立留出的外部 4K 面板（139 条记录、124 个谱系，原速 `1.0×`）上，按谱系加权的 Dan 平均绝对误差（MAE）较 RC3 `detail` 下降 **58%**：从 **0.574 降至 0.241 Dan**（同谱配对改善 **−0.3324 Dan**，95% 谱系 bootstrap 区间 **[−0.3828, −0.2771]**），中位系统偏差从 **−0.540 Dan** 收敛至 **−0.021 Dan**。（作为对比，历史上基于幂函数的旧 RC3 `detail-4k` 在该面板 MAE 为 `0.385 Dan`。）

| 4K 键型子集 | 记录数 | RC3 `detail` MAE | v1.3.0 `detail-4k` MAE |
|---|---:|---:|---:|
| `jack` | 46 | 0.535 | **0.180** |
| `speed` | 22 | 0.637 | **0.298** |
| `stamina` | 38 | 0.367 | **0.197** |
| `tech` | 33 | 0.824 | **0.340** |

上述 139 条留出集主要集中在高段单点/高速/技术/叠键曲目；在前述宽跨度 4K 参考集的 102 张长条（LN）谱上，Dan MAE 亦从 RC3 的 **0.736** 改善至 **0.704**（`detail`）与 **0.664**（`compact`）。关于 `stamina` 子集的原生排序权衡，详见[评估与模型取舍](evaluation.zh-CN.md)。

## 使用方式与旧版复现

- **CLI**：`python -m spm_rating "4k-chart.osu" --model detail-4k`
- **Python**：`rate_file(path, "detail-4k", scale="reference")`（省略 `scale` 时 `score` 返回 `raw_score`，参考 Dan 值另见 `reference_score`）。
- **桌面界面**：选择 **4K Ref / 4K 标定**（与 **Detailed / 完整** `detail` 共享同套 v1.3.0 4K 标定）。
- 如需复现旧版 RC3（或旧幂函数 `detail-4k`）分数，可通过 `load_model("detail-4k", model_dir="path/to/rc3_models")` 显式加载 Git 历史中的归档权重目录。
