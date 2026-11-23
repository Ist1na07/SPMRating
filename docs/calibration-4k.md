# 4K reference calibration

[简体中文](calibration-4k.zh-CN.md) · [Usage](../README.md) · [Evaluation](evaluation.md) · [Full Mechanism](mechanism.md)

`detail-4k` runs the same native v1.3.0 `detail` (`V13-D1`) scoring engine and maps **4K raw scores** to community Dan grades through a strictly increasing piecewise-linear calibration fitted on a wide-span 4K Dan reference panel (~741 charts). The monotone map preserves raw scores, within-4K rank order, and local demand curves. Its reference output targets 4K Dan alignment only and is neither an osu! Star Rating nor a cross-key equivalence scale.

## Held-out check

On a disjoint external 4K holdout panel (139 records across 124 chart families; 1.0× playback rate), family-weighted Dan MAE fell by **58%** relative to RC3 `detail` (**0.574 → 0.241 Dan**; paired difference **−0.3324 Dan**, 95% family bootstrap interval **[−0.3828, −0.2771]**), with median bias moving from **−0.540 Dan** to **−0.021 Dan**. (For comparison, the archived RC3 `detail-4k` power-curve overlay had `0.385` Dan MAE on the same panel.)

| 4K pattern | Records | RC3 `detail` MAE | v1.3.0 `detail-4k` MAE |
|---|---:|---:|---:|
| `jack` | 46 | 0.535 | **0.180** |
| `speed` | 22 | 0.637 | **0.298** |
| `stamina` | 38 | 0.367 | **0.197** |
| `tech` | 33 | 0.824 | **0.340** |

The 139-record holdout panel concentrates in high-Dan tap/speed/tech/jack charts; on the separate 102 long-note charts inside the wide-span 4K panel, Dan MAE also improved from **0.736** (RC3) to **0.704** (`detail`) and **0.664** (`compact`). See [Evaluation and tradeoffs](evaluation.md) for rank-correlation tradeoffs on the `stamina` subset.

## Usage and historical compatibility

- **CLI**: `python -m spm_rating "4k-chart.osu" --model detail-4k`
- **Python**: `rate_file(path, "detail-4k", scale="reference")` (omitting `scale` returns `raw_score` as `score` and `reference_score` alongside it).
- **Desktop GUI**: select **4K Ref / 4K 标定** (or **Detailed / 完整** (`detail`), which shares the same v1.3.0 4K calibration).
- To reproduce legacy RC3 scores (including the archived RC3 power-curve `detail-4k`), explicitly load an archived RC3 model directory via `load_model("detail-4k", model_dir="path/to/rc3_models")`.
