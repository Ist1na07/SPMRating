# SPM Rating v1.0.0

[简体中文文档](README.zh-CN.md)

SPM Rating is a difficulty rating algorithm for osu!mania charts. For each chart it computes a difficulty curve that varies over time, then aggregates that curve into a single star rating.

On the benchmark datasets, v1.0.0 scores (Spearman rank correlation, i.e. how well the algorithm's ordering agrees with the human labels' ordering): 0.98449 on 7-key and 0.98535 on 4-key, exceeding the previous best baselines by 0.0053 and 0.0062 respectively. On a separate 6-key dataset that never took part in tuning, v1.0.0 scores 0.95796 zero-shot, above the sunny-rework reference value of 0.95515.

The algorithm is built on the [sunny-rework](https://github.com/sunnyxxy/Star-Rating-Rebirth) architecture; the mechanism differences are described in the history document.

## Documents

| Document | Contents |
|---|---|
| [docs/mechanism.md](docs/mechanism.md) | The complete computation from chart to star rating, with the formulas and parameter values of each stage |
| [docs/history.md](docs/history.md) | Differences from predecessor algorithms, the development process, all findings, methodology, and future directions |
| [docs/maintenance.md](docs/maintenance.md) | Environment setup, everyday rating, regression checks, the release process, retuning principles, and common mistakes |

Each document has a Chinese counterpart (`docs/mechanism.zh-CN.md`, `docs/history.zh-CN.md`, `docs/maintenance.zh-CN.md`).

If you only need to use the algorithm, the quick start below is enough. For the details of how it works, read the [mechanism document](docs/mechanism.md). If you are taking over maintenance, read all three.

## Quick Start

The runtime is Python 3.10 or newer, with numpy and scipy as dependencies.

```
pip install -r requirements.txt
```

Single-chart rating goes through the standard interface:

```python
import sys
sys.path.insert(0, "interface/python")
import spm_interface as spm

result = spm.analyze("path/to/chart.osu", input_type="path")
print(result["star"])    # Star rating; -255 on failure, with the reason in the error field
```

`input_type` also accepts `"content"`, which takes the chart text directly. `analyze` offers two output modes, `minimal` and `full`; `full` mode returns a more complete set of fields, and unimplemented fields are reported with sentinel values (e.g. `star_rc = -255`). For the full interface contract, see [interface/manifest.json](interface/manifest.json) and [Section 2 of the maintenance guide](docs/maintenance.md#2-day-to-day-rating).

## Regression Check

[tests/cases/](tests/cases/) contains ten built-in charts, and [tests/expected.json](tests/expected.json) records the expected star rating for each one. Run

```
python tests/compliance_check.py
```

It recomputes all charts with the current code and compares them against the expected values (tolerance 0.001), and additionally checks that the interface runs from an independent directory, that all output fields are present, and that invalid inputs are handled per the contract — five checks in total, reported as 5/5 when all pass.

## Repository Layout

```
core/          Model core: parsing, row structure, difficulty curves, merging, aggregation, and calibration
  components/  Implementations of the individual channels
interface/     Public interface: version and capability declarations, self-contained single-file engine, interface entry point
params/        Parameter files; the current release is spm-rating-1.0.0.json
tests/         Regression tests: built-in charts, expected scores, compliance check script, unit tests
docs/          The three documents listed above, English primary, Chinese with a zh-CN suffix
```

## License

The code is under the MIT license, see [LICENSE](LICENSE). Third-party material is described in [NOTICE](NOTICE): the charts under [tests/cases/](tests/cases/) come from public benchmark sets and are used for regression testing only, with copyright belonging to the original mappers, and the base architecture is acknowledged there with its licensing status.

osu! is a registered trademark of ppy; this project is independent research and is not affiliated with osu! official.
