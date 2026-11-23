# SPM Rating

[English](README.md)

SPM Rating 为 osu!mania 的 4 键（4K）、6 键（6K）及 7 键（7K）谱面提供整谱与局部区间难度评估。系统基于 sunny-rework 坐标系演进而来，v1.3.0 提供两支冻结生产模型——日常高精度模型 **`detail`**（`V13-D1`）与高速批量模型 **`compact`**（`V13-C`）——并保留 **`detail-4k`** 兼容现有 4K 段位调用接口。

## 安装与运行

请使用 Python 3.13，并在仓库根目录运行：

```bash
python -m pip install -r requirements.txt
python -m spm_rating examples/Control_P_120.osu
```

Windows 用户可双击或运行 `install_dependencies.cmd` 安装依赖，运行 `open_scorer.cmd` 启动桌面评分界面。其他系统的 Python 若包含 Tk，可执行 `python -m spm_rating.gui`。首次调用会编译 Numba 数值内核，耗时略长属正常现象。

## 模型选择

| 配置名 | 参数与计算模式 | 适用场景 |
|---|---|---|
| **完整** · `detail` | 78 个连续拟合参数；双镜像对称前向计算 | 默认常规评分（64 张 7K 开发谱已解析输入下中位耗时约 73.9 ms/谱） |
| **精简** · `compact` | 70 个连续拟合参数；规范化单方向前向计算（省略 `lockanchor` 与 `shield`） | 大批量高速评分（热态中位耗时约 36.4 ms/谱，约为 RC3 `detail` 的 2.3 倍速度） |
| **4K 段位** · `detail-4k` | 底层复用 `detail` 原生评分器 | 保留既有 4K 调用名称，并绑定 v1.3.0 分段线性 4K 段位参考映射 |

两支模型均采用非负原生行网格长条协调、三次 Hill 连打-技术门控、因果有界耐力以及按有效接触物量加权的 Log-Logistic 行级 Sigmoid 求根聚合。评估结果、合成控谱检查与尚存取舍见[评估与模型取舍](docs/evaluation.zh-CN.md)。

如需在 Python 中显式加载已归档的历史模型目录，可传入 `model_dir`：

```python
load_model("detail", model_dir="path/to/archived_models")
```

早期版本参数可通过 Git 历史获取。当前 `detail-4k` 采用 v1.3.0 单调分段线性 Dan 映射，不再沿用早期幂函数覆盖曲线；详见[4K 参考标定说明](docs/calibration-4k.zh-CN.md)。

## 命令行使用

```bash
python -m spm_rating "path/to/chart.osu" --model detail
python -m spm_rating "path/to/4k-chart.osu" --model detail-4k
python -m spm_rating "path/to/chart.osu" --model compact --rate 1.5
python -m spm_rating "path/to/Songs" --csv scores.csv
python -m spm_rating "path/to/chart.osu" --start-ms 30000 --end-ms 45000
python -m spm_rating "path/to/chart.osu" --start-ms 30000 --end-ms 45000 --scope isolated
```

命令接受单个 `.osu` 文件或目录，为每张谱面输出一条 JSON 记录（也可配合 `--csv` 导出表格）。单张谱面解析或校验失败时会在该条记录写入 `error` 字段，其余谱面正常继续；若存在任何失败项，进程以退出码 `1` 结束。

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

`rate_notes` 支持直接传入已解析谱面对象；也可直接传模型名称字符串（例如 `rate_file(path, "compact")`）。模型对象的 `parameters` 属性暴露只读参数字典。

### 分数尺度

`raw_score` 为模型原生输出，具备严格正尺度齐次性；`reference_score` 则经由各键数独立拟合的冻结严格单调递增分段线性映射转换：4K 为 `dan`，6K 为 `lv`，7K 为 `dan_sr`。`score` 字段随 `scale` 切换（CLI 默认输出 `reference`，Python 默认输出 `raw`）。不同键数各自独立标定，跨键数直接横向对比仍需额外校准；osu! 官方星数采用独立体系。

### 局部段落

时间边界使用**应用倍速 `--rate` / `rate` 之后的毫秒数**，按左闭右开 `[start_ms, end_ms)` 切片。例如倍率为 `1.5` 时，原谱 `30000–45000 ms` 对应 `--start-ms 20000 --end-ms 30000`。

- `context` 模式先在整谱上计算上下文需求场，再对目标区间内行聚合评分；
- `isolated` 模式独立重建切片区间，丢弃区间前延续进来的长条，并将穿出 `end_ms` 的长条在终点截断。

每个受评区间至少需包含两个物理音头；无音头的纯长条持键区间不在接口支持范围内。设置 `include_curve=True` 时会返回 NumPy 数组 `query` 与 `field`。当原谱带有小数时间戳时，`field` 为各相位的平均局部曲线，标量分则为各相位分别完成非线性聚合后的积分平均；详见[时间处理](docs/time.zh-CN.md)。

## 输入处理与性能特征

同列完全重复对象会在评分前自动合并；同一时刻同一物理列出现尾部冲突的双重对象会报错退出。解析器将 4 键与 6 键统一嵌入对称七列坐标系。

常规整数毫秒 `.osu` 谱面及常用倍速仅运行一次原生前向计算，且聚合原点相位极差严格为 `0.0`。原始时刻含小数的谱面则会触发原谱时钟相位精确积分（[时间处理](docs/time.zh-CN.md)）。Python 接口可传 `cancel=callback`，程序在相位步间检查取消信号；桌面界面提供 **Stop / 停止** 按钮。

v1.3.0 保持时间与区间契约兼容，但原始分及参考分数值不与 RC3 或早期 v1.x 版本直接通用。

## 源码与测试

| 路径 | 内容 |
|---|---|
| `spm_rating/api.py` | 整谱与局部评分接口、分段线性与历史幂函数参考映射 |
| `spm_rating/v130_rev2.py` | v1.3.0 原生需求核、Hill 门控与物量加权行 Sigmoid 求解器 |
| `spm_rating/v130_engine.py`、`phase_average.py` | 结构状态封装与双镜像后平均支撑 |
| `spm_rating/_core/` | 谱面结构容器、分项曲线、长条协调与基础原语 |
| `spm_rating/hand.py`、`role.py` | 手部共同接触分配与持续占键条件负担 |
| `spm_rating/geometry.py`、`coordinates.py` | 跨列转移几何与对称七列映射 |
| `spm_rating/clock.py` | 原谱小数时刻连续相位延拓 |
| `spm_rating/models/` | 打包模型权重与域参考映射（`detail`、`compact`、`detail-4k`） |

运行 `pytest tests -q -p no:cacheprovider`（或 `python -m unittest discover -s tests -v`）执行测试套件。更多技术说明见[模型原理](docs/model.zh-CN.md)、[评估与模型取舍](docs/evaluation.zh-CN.md)、[4K 参考标定说明](docs/calibration-4k.zh-CN.md)及[时间处理](docs/time.zh-CN.md)（均备有英文版）。

## 历史版本与许可

[v1.1.1 源码快照](https://github.com/Ist1na07/SPMRating/tree/11064c7d73370e9d8b7d1fff1d9ccf740829c6af)保留了初代实现与 `spm_interface` 入口，其余更早版本见 Git 历史。

代码基于 [MIT 许可](LICENSE)开源，架构来源与示例素材说明见[项目说明与致谢](NOTICE.zh-CN.md)。
