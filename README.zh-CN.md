# SPM Rating v1.0.0

[English](README.md)

SPM Rating 是面向 osu!mania 谱面的难度评分算法。它对一张谱面计算一条随
时间变化的难度曲线，再聚合成一个星数。

v1.0.0 在基准数据集上的成绩（斯皮尔曼等级相关，即算法排序与人工标注排序
的相关程度）：7 键 0.98449，4 键 0.98535，分别高出此前最高基线 0.0053 与
0.0062。另有一个从未参与调参的 6 键数据集，v1.0.0 在其上的零样本成绩为
0.95796，高于 sunny-rework 参考值 0.95515。

算法在 [sunny-rework](https://github.com/sunnyxxy/Star-Rating-Rebirth) 架构
上发展而来，机制差异见《研制历程与结论》。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/mechanism.zh-CN.md](docs/mechanism.zh-CN.md) | 从谱面到星数的完整计算过程，各步骤的公式与参数值 |
| [docs/history.zh-CN.md](docs/history.zh-CN.md) | 与前代算法的差异、研制过程、全部结论、方法论与后续优化方向 |
| [docs/maintenance.zh-CN.md](docs/maintenance.zh-CN.md) | 环境搭建、日常评分、回归检查、发版流程、重调参原则与常见陷阱 |

每份文档都有对应的英文版（docs/mechanism.md、docs/history.md、docs/maintenance.md）。

只需直接使用算法时，读完本页的快速开始即可；需要理解机制细节时，阅读
[《机制说明》](docs/mechanism.zh-CN.md)；接手维护工作则三份均需阅读。

## 快速开始

运行环境为 Python 3.10 或更高版本，依赖 numpy 与 scipy。

```
pip install -r requirements.txt
```

单张谱面评分通过标准接口完成：

```python
import sys
sys.path.insert(0, "interface/python")
import spm_interface as spm

result = spm.analyze("path/to/chart.osu", input_type="path")
print(result["star"])    # 星数；失败时为 -255，原因写在 error 字段
```

`input_type` 也接受 `"content"`，直接传入谱面文本即可。`analyze` 提供
`minimal` 与 `full` 两种输出模式，`full` 模式返回更完整的字段，未实现的
字段以哨兵值表示（例如 `star_rc = -255`）。接口的详细约定见
[interface/manifest.json](interface/manifest.json) 与
[《维护与升级》第二节](docs/maintenance.zh-CN.md#2-日常评分)。

## 回归检查

[tests/cases/](tests/cases/) 下有十张内置谱面，
[tests/expected.json](tests/expected.json) 记录每张谱面的预期星数。运行

```
python tests/compliance_check.py
```

会用当前代码重算全部谱面并与预期值比对（容差 0.001），同时检查接口在独立
目录下能否运行、输出字段是否完整、错误输入是否按约定处理，共五项检查，
全部通过时输出 5/5。

## 仓库布局

```
core/          模型主体：解析、行结构、难度曲线、合并、聚合与标定
  components/  各通道的实现
interface/     对外接口：版本与能力声明、自包含单文件引擎、接口入口
params/        参数文件，当前版本为 spm-rating-1.0.0.json
tests/         回归测试：内置谱面、预期分值、合规检查脚本、单元测试
docs/          上述三份文档，英文为主，中文带 zh-CN 后缀
```

## 许可

代码采用 MIT 许可，见 [LICENSE](LICENSE)。第三方素材见 [NOTICE](NOTICE)：
[tests/cases/](tests/cases/) 下的谱面取自公开基准集，仅用于回归测试，版权归原
谱师所有；基础架构的来源与许可状态也在该文件中说明。

osu! 是 ppy 公司的注册商标；本项目为独立研究，与 osu! 官方无关。
