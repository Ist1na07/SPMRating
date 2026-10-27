# SPM Rating v1.2.0-rc3

[简体中文](README.zh-CN.md)

This candidate provides detailed (`detail`) and compact (`compact`) difficulty models for osu!mania. Both support 4-key, 6-key, and 7-key charts, speed changes, and whole-chart or segment scoring.

## Install and run

Use Python 3.13 and run these commands from this directory:

```bash
python -m pip install -r requirements.txt
python -m spm_rating examples/Control_P_120.osu
```

On Windows, `install_dependencies.cmd` installs dependencies and `open_scorer.cmd` opens the desktop scorer. On other systems, use `python -m spm_rating.gui` if Python includes Tk. The first score may take longer while Numba compiles numerical functions.

## Choose a model

| | Detailed: `detail` | Compact: `compact` |
|---|---|---|
| Suggested use | General scoring, especially charts with long notes | A slightly simpler model for comparison and research |
| Weighted development ranking correlation | 0.988334 | 0.988271 |
| Local 12-segment ranking correlation | 0.866219 | 0.866219 |
| Long-note set ranking correlation | 0.904663 | 0.899101 |

Both models price actions performed by fingers recently occupied by long notes. The detailed model also retains a conditional hold-duration component. The compact model disables that component, retiring two amplitudes and their duration threshold and cap from active use. The remaining coordination components are shared.

Each parameter file keeps 291 configuration fields, including fixed values and disabled branches, plus one finger-role price. This is a modest simplification within a feature-rich model. See the [evaluation notes](docs/evaluation.md) for the ranking metric and sample definitions.

## Command line

```bash
python -m spm_rating "path/to/chart.osu" --model detail
python -m spm_rating "path/to/chart.osu" --model compact --rate 1.5
python -m spm_rating "path/to/Songs" --csv scores.csv
python -m spm_rating "path/to/chart.osu" --start-ms 30000 --end-ms 45000
python -m spm_rating "path/to/chart.osu" --start-ms 30000 --end-ms 45000 --scope isolated
```

The command accepts files or directories and prints one JSON object per chart. A failed chart receives an `error` field; other charts continue. The command exits with status 1 if any chart failed.

## Python interface

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

`rate_notes` accepts an already parsed chart. A model name also works directly: `rate_file(path, "compact")`. The model object exposes a read-only parameter mapping.

### Scores

`raw_score` is the model's native output. `reference_score` applies a positive power-law alignment to each key-count's supplied labels. `score` follows the selected `scale`; the default is `reference` on the command line and `raw` in Python. The 4-key, 6-key, and 7-key reference scales are calibrated separately. Cross-key comparisons need additional validation. osu!'s official star-rating system uses its own scale.

### Segments

Boundaries use milliseconds **after applying the speed rate**. The interval includes its start and excludes its end. At rate 1.5, original times 30000–45000 ms correspond to segment arguments 20000–30000 ms.

`context` reads demand from the full chart and aggregates the selected heads. `isolated` rebuilds the selected chart, dropping incoming long notes and clipping outgoing tails at its end. A rated segment needs at least two physical heads. Headless holding segments are outside the current scoring interface.

`include_curve=True` returns NumPy arrays `query` and `field`. For fractional original times, the curve averages phase-specific curves; the scalar averages their individually aggregated scores. [Time handling](docs/time.md) explains the distinction.

## Inputs and performance

Exact duplicate objects on one key are merged. Simultaneous objects on one physical key with conflicting tails raise an error. The parser embeds 4-key and 6-key charts in a symmetric seven-column frame.

Integer-millisecond charts and their speed-modified versions use one native scoring pass. Fractional original timestamps use exact phase averaging and may require more passes. Python callers can supply `cancel=callback`; cancellation is checked between phase passes. The desktop scorer offers a Stop button.

General staircase-jack and switching patterns remain an open modeling area. The present finger-role cost addresses sustained occupation followed by further actions. Additional chart collections and player comparisons can help assess how this transfers to other patterns.

## Source and checks

| Path | Contents |
|---|---|
| `spm_rating/api.py` | Whole-chart and segment interface |
| `spm_rating/engine.py` | Symmetric two-view scoring pipeline |
| `spm_rating/_core/` | Chart structures, component curves, combination, aggregation |
| `spm_rating/hand.py`, `role.py` | Shared contact accounting and finger occupation |
| `spm_rating/geometry.py`, `coordinates.py` | Cross-column transitions |
| `spm_rating/recovery.py`, `clock.py` | Recovery context and time extension |
| `spm_rating/models/` | Completed model parameters and output alignments |

Run `python -m unittest discover -s tests -v` for the included checks. Read the [model guide](docs/model.md), [evaluation](docs/evaluation.md), and [time handling](docs/time.md) for further details. Each document has a Chinese counterpart.

## Earlier versions

The [v1.1.1 source snapshot](https://github.com/Ist1na07/SPMRating/tree/11064c7d73370e9d8b7d1fff1d9ccf740829c6af) contains its original implementation and `spm_interface` entry point. Earlier versions remain available through the Git history.

Code uses the [MIT license](LICENSE). Architectural attribution and example provenance are recorded in [NOTICE](NOTICE).
