# SPM Rating

[简体中文](README.zh-CN.md)

SPM Rating estimates pattern and physical difficulty for 4-key (4K), 6-key (6K), and 7-key (7K) osu!mania charts and local segments. Built on the sunny-rework coordinate representation, v1.3.0 ships two frozen scoring profiles—**`detail`** (`V13-D1`) for general use and **`compact`** (`V13-C`) for faster batch evaluation—together with **`detail-4k`** for 4K Dan reference workflows.

## Install and run

Requires Python 3.13. Run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m spm_rating examples/Control_P_120.osu
```

On Windows, `install_dependencies.cmd` installs dependencies and `open_scorer.cmd` launches the desktop scorer. On other systems whose Python build includes Tk, run `python -m spm_rating.gui`. The first scoring call may take slightly longer while Numba compiles numerical kernels.

## Model profiles

| Profile | Parameters & pass mode | Intended use |
|---|---|---|
| **Detail** · `detail` | 78 fitted continuous parameters; symmetric dual mirrored pass | Default general-purpose scoring (~73.9 ms/chart warm median on 64 parsed 7K charts) |
| **Compact** · `compact` | 70 fitted continuous parameters; single canonical pass (omits `lockanchor` and `shield`) | High-throughput batch scoring (~36.4 ms/chart warm median; ~2.3× faster than RC3 `detail`) |
| **4K Dan** · `detail-4k` | Identical native scorer to `detail` | Dedicated 4K CLI/API/GUI workflows using the v1.3.0 piecewise-linear 4K Dan calibration |

All profiles use nonnegative native row-grid long-note coordination, a cubic jack-versus-technique Hill gate, bounded causal stamina, and exact engagement-mass-weighted Log-Logistic Row Sigmoid aggregation. Model tradeoffs, synthetic control checks, and benchmark results are documented in [Evaluation and tradeoffs](docs/evaluation.md).

To load an archived legacy model directory explicitly in Python:

```python
load_model("detail", model_dir="path/to/archived_models")
```

Historical model weights remain accessible through Git history. Current `detail-4k` applies the v1.3.0 monotone piecewise-linear Dan map rather than historical power curves; see [4K reference calibration](docs/calibration-4k.md).

## Command line

```bash
python -m spm_rating "path/to/chart.osu" --model detail
python -m spm_rating "path/to/4k-chart.osu" --model detail-4k
python -m spm_rating "path/to/chart.osu" --model compact --rate 1.5
python -m spm_rating "path/to/Songs" --csv scores.csv
python -m spm_rating "path/to/chart.osu" --start-ms 30000 --end-ms 45000
python -m spm_rating "path/to/chart.osu" --start-ms 30000 --end-ms 45000 --scope isolated
```

The CLI accepts single `.osu` files or directories and emits one JSON object per chart (or writes `--csv`). Charts that fail parsing or validation record an `error` field while remaining charts continue; the process exits with code `1` if any chart failed.

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

`rate_notes` scores an already parsed note structure, and string profile names are accepted directly (`rate_file(path, "compact")`). `model.parameters` exposes a read-only parameter mapping.

### Scores

`raw_score` is the model's native scale-homogeneous output. `reference_score` applies a frozen, strictly increasing piecewise-linear map aligned separately per key count: `dan` for 4K, `lv` for 6K, and `dan_sr` for 7K. `score` follows the selected `scale` (`reference` by default on the CLI, `raw` by default in Python). Key-count scales are calibrated independently and should not be compared across key counts without separate cross-key validation. osu!'s official Star Rating uses a different scale.

### Segments

Boundaries use milliseconds **after applying `--rate` / `rate`**, half-open as `[start_ms, end_ms)`. For example, at rate `1.5`, original timestamps `30000–45000 ms` correspond to arguments `start_ms=20000, end_ms=30000`.

`context` computes demand over the entire chart and aggregates the selected rows; `isolated` rebuilds the trimmed chart segment, dropping incoming long notes and clipping outgoing tails at `end_ms`. Every segment requires at least two physical note heads; headless holding-only spans fall outside the interface.

`include_curve=True` returns NumPy arrays `query` and `field`. For charts authored with fractional original timestamps, `field` averages phase-specific demand curves while the scalar averages individually aggregated scores; see [Time handling](docs/time.md).

## Inputs and performance

Exact duplicate objects on the same column are merged before scoring; simultaneous objects on the same physical key with conflicting long-note tails raise an error. The parser embeds 4K and 6K layouts in a symmetric seven-column coordinate frame.

Integer-millisecond `.osu` charts and their speed-modified versions run in one native forward pass with exact `0.0` aggregation-origin phase spread. Charts with fractional original-clock timestamps use exact shared-phase integration over the rounding interval ([Time handling](docs/time.md)). Python callers can pass `cancel=callback`, checked between clock phase passes; the desktop UI provides a **Stop** control.

Input timing and segment semantics are unchanged across v1.x, though numeric scores from v1.3.0 are not directly interchangeable with earlier releases.

## Source and tests

| Path | Contents |
|---|---|
| `spm_rating/api.py` | Whole-chart and segment API, piecewise-linear and legacy power calibration |
| `spm_rating/v130_rev2.py` | v1.3.0 native demand kernels, Hill gate, and mass-weighted Row Sigmoid root solver |
| `spm_rating/v130_engine.py`, `phase_average.py` | Structural state representation and mirror-averaging support |
| `spm_rating/_core/` | Chart containers, component demand curves, coordination, and primitives |
| `spm_rating/hand.py`, `role.py` | Shared hand-contact accounting and sustained finger-occupation costs |
| `spm_rating/geometry.py`, `coordinates.py` | Cross-column transition geometry and symmetric 7-column embedding |
| `spm_rating/clock.py` | Fractional original-clock continuous phase extension |
| `spm_rating/models/` | Packaged model weights and reference calibrations (`detail`, `compact`, `detail-4k`) |

Run `pytest tests -q -p no:cacheprovider` (or `python -m unittest discover -s tests -v`) to verify the test suite. Further technical documentation is available in [Model architecture](docs/model.md), [Evaluation and tradeoffs](docs/evaluation.md), [4K reference calibration](docs/calibration-4k.md), and [Time handling](docs/time.md) (each with a Simplified Chinese counterpart).

## Earlier versions and license

The [v1.1.1 source snapshot](https://github.com/Ist1na07/SPMRating/tree/11064c7d73370e9d8b7d1fff1d9ccf740829c6af) preserves the original implementation and `spm_interface` entry point; earlier snapshots remain available in Git history.

Licensed under the [MIT License](LICENSE). Architectural provenance and example asset credits are documented in [NOTICE](NOTICE).
