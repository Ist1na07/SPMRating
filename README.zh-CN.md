# SPM Rating v1.2.0-rc3

[English](README.md)

候选包提供完整模型 `detail` 和精简模型 `compact`，用于计算 osu!mania 难度。两支均支持4键、6键、7键、速度倍率，以及整谱和局部评分。

## 安装与运行

使用 Python 3.13，在本目录运行：

```bash
python -m pip install -r requirements.txt
python -m spm_rating examples/Control_P_120.osu
```

Windows 用户可运行 `install_dependencies.cmd` 安装依赖，双击 `open_scorer.cmd` 打开桌面评分器。其他系统的 Python 若包含 Tk，可使用 `python -m spm_rating.gui`。首次评分时，Numba 编译数值函数可能花费额外时间。

## 模型选择

| | 完整模型：`detail` | 精简模型：`compact` |
|---|---|---|
| 建议用途 | 日常评分，尤其是包含长条的谱面 | 使用稍简单的模型进行比较与研究 |
| 开发评估加权排序相关系数 | 0.988334 | 0.988271 |
| 局部12段排序相关系数 | 0.866219 | 0.866219 |
| 长条集合排序相关系数 | 0.904663 | 0.899101 |

两支模型都衡量手指近期持续持键后继续动作的费用。完整模型还保留条件持续时间组件。精简模型关闭这一组件，使两个幅度以及持续时间阈值、上限退出有效作用；其余协调组件继续共用。

每份参数文件保留291个配置字段，包括固定值和关闭分支，另有一个手指角色幅度。精简是在一个特征较多的模型内进行的适度调整。[评估说明](docs/evaluation.zh-CN.md) 介绍指标定义与样本。

## 命令行

```bash
python -m spm_rating "path/to/chart.osu" --model detail
python -m spm_rating "path/to/chart.osu" --model compact --rate 1.5
python -m spm_rating "path/to/Songs" --csv scores.csv
python -m spm_rating "path/to/chart.osu" --start-ms 30000 --end-ms 45000
python -m spm_rating "path/to/chart.osu" --start-ms 30000 --end-ms 45000 --scope isolated
```

命令接受文件或目录，每张谱面输出一个JSON对象。计算失败的谱面带有 `error` 字段，其他谱面继续处理。如果有谱面失败，命令以状态码1退出。

## Python 接口

```python
from spm_rating import load_model, rate_file, rate_content

model = load_model("detail")
whole = rate_file("path/to/chart.osu", model, scale="reference")
segment = rate_file(
    "path/to/chart.osu", model,
    start_ms=30000, end_ms=45000,
    scope="context", include_curve=True,
)
print(whole["score"], segment["score"])
print(segment["query"], segment["field"])

with open("path/to/chart.osu", encoding="utf-8-sig") as file:
    same_chart = rate_content(file.read(), model)
```

`rate_notes` 接受已经解析的谱面。也可直接传入模型名称，例如 `rate_file(path, "compact")`。模型对象提供只读的参数映射。

### 分数

`raw_score` 是模型原始输出；`reference_score` 通过正幂函数对齐到各键数的标注尺度。`score` 跟随所选 `scale`：命令行默认使用 `reference`，Python 默认使用 `raw`。4键、6键和7键分别校准，跨键数比较仍需额外验证。osu! 官方星数使用自身的尺度。

### 局部段落

边界的单位是**应用速度倍率之后的毫秒**。区间包含起点、排除终点。例如速度倍率为1.5时，原谱30000至45000毫秒对应参数20000至30000毫秒。

`context` 从整谱读取局部需求，再聚合选定的按下。`isolated` 重新建立选定谱面，丢弃来自区间前的长条，并将去向长条截至区间末尾。局部至少需要两个物理按下；没有按下的纯静持区间暂不提供评分。

`include_curve=True` 返回 NumPy 数组 `query` 和 `field`。对于原始时间含小数的谱面，曲线是各相位曲线的平均，标量则平均各相位分别聚合后的评分。[时间处理](docs/time.zh-CN.md) 进一步说明两者的区别。

## 输入与性能

同键上完全重复的对象会合并。同刻同键但尾部冲突的对象会报错。解析器将4键与6键嵌入对称的七列坐标。

普通整数毫秒谱及其变速版本进行一次原生评分。原始时间含小数时，精确相位平均可能需要多次评分。Python 调用者可传入 `cancel=callback`，程序在相位之间检查停止信号；桌面评分器也提供停止按钮。

普遍楼梯叠与切键仍是有待发展的建模问题。当前手指角色费用针对持续占用后继续动作的负担。更多谱面集合和玩家对照有助于了解它在其他键型中的表现。

## 源码与检查

| 路径 | 内容 |
|---|---|
| `spm_rating/api.py` | 整谱和局部接口 |
| `spm_rating/engine.py` | 两个镜像视角的对称评分流程 |
| `spm_rating/_core/` | 谱面结构、局部组件、组合与聚合 |
| `spm_rating/hand.py`、`role.py` | 共同接触预算与手指占用费用 |
| `spm_rating/geometry.py`、`coordinates.py` | 跨列转移 |
| `spm_rating/recovery.py`、`clock.py` | 恢复上下文与时间延拓 |
| `spm_rating/models/` | 完成的模型参数和输出尺度 |

运行 `python -m unittest discover -s tests -v` 可执行附带检查。[模型说明](docs/model.zh-CN.md)、[评估结果](docs/evaluation.zh-CN.md)和[时间处理](docs/time.zh-CN.md)均有英文版本。

## 历史版本

[v1.1.1源码快照](https://github.com/Ist1na07/SPMRating/tree/11064c7d73370e9d8b7d1fff1d9ccf740829c6af)保留该版本的原始实现与 `spm_interface` 入口。更早版本可通过 Git 历史记录获取。

代码采用 [MIT许可](LICENSE)，架构来源与示例说明见[项目说明与致谢](NOTICE.zh-CN.md)。
