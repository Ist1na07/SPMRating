# SPM Rating v1.1.0

[English](README.md)

SPM Rating 是面向 osu!mania 谱面的高精度难度评分算法（以 7K 优先，兼顾 4K 与 6K）。它对谱面时间轴计算多维手部动作难度曲线，合成瞬时总难度 $D(t)$ ，再经由两级物量聚合与全局标定输出最终难度星数（Star Rating, SR）。

在全量公开基准数据集上，v1.1.0 的成绩（斯皮尔曼等级相关系数 Spearman sp，衡量算法排序与人工标注排序的一致性）达到：
- **7 键 (7K)**：**0.98490**（较 v1.0.0 进一步提升，配对 bootstrap 90% CI $[-0.0016, +0.0023]$ ）；
- **4 键 (4K)**：**0.98531**（保持在 1$\sigma$ 噪声底内高位持平）；
- **6 键 (6K)**：**0.95595**（零样本检验域，从未参与调参，处于粗标签不可分辨边界）。

相较于 v1.0.0，**v1.1.0 彻底解决了多项长久存在的结构性缺陷**：
- **聚合机制重做（两级物量聚合 mode 4）**：以音符物量分布驱动聚合，彻底攻克远处散键稀释难度（T1 从 0.309 降至 **0.0035**）、复制粘贴重复段膨胀（T2 从 0.343 降至 **0.0043**）以及长空白休息段占权稀释（T3 从 0.141 降至 **0.0123**）；
- **多押连续手势检测与双手结构化**：多押检测由离散时间切片升级为连续手势级定价，彻底消除 $\pm 1$ms 时间抖动引起的断崖跳变，并统一 2-7 押定价架构；
- **公理倒挂修复**：严格修复超集倒挂公理 A1（同时按更多键反而更简单）与双手并行公理 A4（单手一把抓比双手分摊更便宜）；
- **段落级鲁棒性验证**：经由标准合成排列与真实谱面切片双重检验，段落孤立与平铺的一致性失真相比 v1.0.0 缩小近 10 倍。

算法继承自 [sunny-rework](https://github.com/sunnyxxy/Star-Rating-Rebirth) 架构并经过深层重构，详细演进与消融实验参见《研制历程与结论》。

## 文档

| 文档 | 核心内容 |
|---|---|
| [docs/mechanism.zh-CN.md](docs/mechanism.zh-CN.md) | 从谱面到星数的完整计算管线、各通道公式与 272 个发布参数的物理含义 |
| [docs/history.zh-CN.md](docs/history.zh-CN.md) | 从 sunny 到 v1.0.0 再到 v1.1.0 的演化历程、消融对比、证伪探索与方法论 |
| [docs/maintenance.zh-CN.md](docs/maintenance.zh-CN.md) | 环境搭建、日常调用、合规回归检查（5/5）、发版打包流程与开发避坑指南 |

每份文档均提供对应的英文原版（[docs/mechanism.md](docs/mechanism.md)、[docs/history.md](docs/history.md)、[docs/maintenance.md](docs/maintenance.md)）。

## 快速开始

运行环境推荐 Python 3.10 及以上版本，依赖仅需 `numpy` 与 `scipy`：

```bash
pip install -r requirements.txt
```

单张谱面评分通过标准对外接口完成：

```python
import sys
sys.path.insert(0, "interface/python")
import spm_interface as spm

# 支持传入谱面文件绝对/相对路径
result = spm.analyze("path/to/chart.osu", input_type="path")
print("Star Rating:", result["star"])    # 失败时返回 -255，原因记录在 error 字段

# 也支持直接传入 .osu 文件文本内容
chart_content = open("path/to/chart.osu", encoding="utf-8").read()
result = spm.analyze(chart_content, input_type="content")
print("Star Rating:", result["star"])
```

接口提供 `minimal` 与 `full` 两种输出模式。详细约定请参阅 [interface/manifest.json](interface/manifest.json) 与 [《维护与升级》第二节](docs/maintenance.zh-CN.md#2-日常评分)。

## 回归与合规检查

仓库内置了标准测试集与接口守卫，确保发版质量与跨环境可靠性：

```bash
# 执行 5 项全套接口合规与预期值对比检查（全部通过输出 5/5）
python tests/compliance_check.py

# 执行单元测试与参数引擎同步守卫（必须 13/13 全部通过）
python -m pytest tests/ -q
```

## 仓库布局

```
core/          算法核心：解析器、行网格、手部动作曲线、通道合并、两级聚合与标定
  components/  长键协调 v2 等关键通道实现
interface/     对外接口：版本与能力清单、统一调用入口及自包含单文件引擎 (spm_engine.py)
params/        参数文件，发布版为 spm-rating-1.1.0.json (272 参数)
tests/         回归测试：内置谱面 (cases/)、预期星数 (expected.json)、合规脚本与单元测试
docs/          算法机制说明、研制历程报告与维护手册（中英双语）
```

## 许可与致谢

代码基于 MIT 许可协议开源，详见 [LICENSE](LICENSE)。第三方素材与前置工作致谢见 [NOTICE](NOTICE)：
- [tests/cases/](tests/cases/) 下的谱面取自公开基准评测集，仅用于回归测试，版权归原谱师所有；
- 基础架构思想借鉴并拓展了 [sunny-rework](https://github.com/sunnyxxy/Star-Rating-Rebirth) 的优秀探索。

osu! 是 ppy Pty Ltd 的注册商标；本项目为学术与算法技术研究，与 osu! 官方团队无关。
