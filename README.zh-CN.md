# SPM Rating

[English](README.md)

SPM Rating 用于估计 osu!mania 谱面的难度。算法先沿时间线建立局部难度曲线，再将各处的动作需求聚合为整谱评分。模型沿着 [sunny-rework](https://github.com/sunnyxxy/Star-Rating-Rebirth) 的思路发展，重点处理重复按下、和弦和长条协调。

## 版本选择

| 版本 | 位置 | 用途 |
|---|---|---|
| **v1.2.0-rc3** | [候选包](candidates/v1.2.0-rc3/) | 完整与精简模型、整谱与局部评分、速度倍率及桌面评分器 |
| **v1.1.1** | `core/`、`params/`、`interface/` | 现有稳定接口与回归用例 |

候选版支持4键、6键和7键。完整与精简模型在开发评估中的加权排序相关系数分别为 **0.988334** 和 **0.988271**；同一评估口径下，v1.1.1 为 **0.981346**。[评估说明](candidates/v1.2.0-rc3/docs/evaluation.zh-CN.md) 提供样本量、权重和模型取舍。

## 试用候选版

使用 Python 3.13，在仓库根目录运行：

```bash
cd candidates/v1.2.0-rc3
python -m pip install -r requirements.txt
python -m spm_rating examples/Control_P_120.osu
python -m spm_rating examples/Control_Q_120.osu --model compact
```

Windows 用户可双击 `open_scorer.cmd` 打开桌面界面。[候选版指南](candidates/v1.2.0-rc3/README.zh-CN.md) 介绍文件与目录评分、CSV导出、Python调用、局部段落和输出尺度。[模型说明](candidates/v1.2.0-rc3/docs/model.zh-CN.md) 介绍算法组件及其源码位置。

## 稳定的 v1.1.1 接口

安装仓库根目录的 `requirements.txt` 后调用：

```python
import sys
sys.path.insert(0, "interface/python")
import spm_interface as spm
result = spm.analyze("path/to/chart.osu", input_type="path")
print(result["star"])
```

接口约定见 [interface/manifest.json](interface/manifest.json)。现有[模型](docs/mechanism.zh-CN.md)、[历史](docs/history.zh-CN.md)和[维护](docs/maintenance.zh-CN.md)文档描述 v1.1.1。

## 许可与致谢

代码采用 [MIT 许可](LICENSE)。[NOTICE](NOTICE) 记录基础架构来源及稳定版回归谱面的权属。候选包中的少量P/Q示例是生成的游玩控制谱。osu! 商标属于 ppy；本项目独立开发。
