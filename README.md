# SPM Rating v1.1.0

[简体中文文档](README.zh-CN.md)

SPM Rating is a high-precision difficulty rating algorithm for osu!mania beatmaps (prioritizing 7K, while comprehensively supporting 4K and 6K). It calculates multi-dimensional physical difficulty curves along the chart's timeline, synthesizes an instantaneous total difficulty curve $D(t)$, and aggregates it into a single Star Rating (SR) via two-level sectioned mass aggregation and global calibration.

On the full benchmark dataset, v1.1.0 achieves (measured by Spearman's rank correlation coefficient $sp$, which quantifies the agreement between algorithmic ranking and human ground-truth labels):
- **7-Key (7K)**: **0.98490** (further improved over v1.0.0, paired bootstrap 90% CI $[-0.0016, +0.0023]$);
- **4-Key (4K)**: **0.98531** (retains ultra-high correlation, within $1\sigma$ noise floor);
- **6-Key (6K)**: **0.95595** (zero-shot evaluation domain, never tuned, sitting at the coarse-label indistinguishability boundary).

Compared to v1.0.0, **v1.1.0 resolves several long-standing structural defects**:
- **Two-Level Sectioned Mass Aggregation (Mode 4)**: Driven by note mass rather than wall-clock duration, completely eliminating distant sparse note dilution (T1 reduced from 0.309 to **0.0035**), repeated section inflation (T2 reduced from 0.343 to **0.0043**), and long silence gap dilution (T3 reduced from 0.141 to **0.0123**);
- **Continuous Gesture Chord Detection & Structured Bilateral Pricing**: Upgraded chord detection from discrete millisecond slicing to continuous gesture-level detection, eradicating $\pm 1$ms jitter cliffs and establishing unified 2-7 note chord pricing;
- **Axiomatic Inversion Fixes**: Fixed superset monotonicity axiom A1 (pressing more keys simultaneously must not be easier) and dual-hand parallelism axiom A4 (one hand taking multiple keys must not be cheaper than sharing across hands);
- **Segment-Level Difficulty Robustness**: Extensively verified via synthetic patterns and real sliced charts, reducing segment isolation vs. tiling distortion by nearly 10x.

The algorithm is built upon the [sunny-rework](https://github.com/sunnyxxy/Star-Rating-Rebirth) architecture and substantially redesigned. For detailed ablations and evolution, see the History document.

## Documentation

| Document | Key Contents |
|---|---|
| [docs/mechanism.md](docs/mechanism.md) | Complete pipeline from chart to star rating, formulas for each channel, and physical meanings of all 272 release parameters |
| [docs/history.md](docs/history.md) | Evolution from sunny to v1.0.0 and v1.1.0, ablation benchmarks, falsified avenues, and empirical methodology |
| [docs/maintenance.md](docs/maintenance.md) | Environment setup, API consumption, regression checks (5/5), bundling process, and development pitfalls |

Each document has a Chinese counterpart ([docs/mechanism.zh-CN.md](docs/mechanism.zh-CN.md), [docs/history.zh-CN.md](docs/history.zh-CN.md), [docs/maintenance.zh-CN.md](docs/maintenance.zh-CN.md)).

## Quick Start

Python 3.10 or newer is recommended. Dependencies are minimal, requiring only `numpy` and `scipy`:

```bash
pip install -r requirements.txt
```

Rating an osu!mania chart is straightforward via the public interface:

```python
import sys
sys.path.insert(0, "interface/python")
import spm_interface as spm

# Analyze from file path
result = spm.analyze("path/to/chart.osu", input_type="path")
print("Star Rating:", result["star"])    # Returns -255 on failure with details in 'error'

# Or analyze directly from raw .osu string content
with open("path/to/chart.osu", encoding="utf-8") as f:
    result = spm.analyze(f.read(), input_type="content")
print("Star Rating:", result["star"])
```

Both `minimal` and `full` output modes are supported. For full API contracts, see [interface/manifest.json](interface/manifest.json) and [Section 2 of the Maintenance Guide](docs/maintenance.md#2-day-to-day-rating).

## Regression & Compliance Checks

The repository includes a built-in test suite to safeguard reproducibility and cross-environment reliability:

```bash
# Run interface compliance and expected value checks (5/5 must pass)
python tests/compliance_check.py

# Run unit tests and parameter-engine sync guard (13/13 must pass)
python -m pytest tests/ -q
```

## Repository Layout

```
core/          Model core: parser, row grid, difficulty curves, merging, two-level aggregation, calibration
  components/  Key channel components including lncoord_v2
interface/     Public interface: manifests, unified wrapper, self-contained single-file engine (spm_engine.py)
params/        Parameter files; release parameter is spm-rating-1.1.0.json (272 parameters)
tests/         Regression suite: test cases (cases/), expected ratings (expected.json), compliance runner
docs/          Detailed mechanism specification, historical development report, and maintenance guide
```

## License & Acknowledgements

The code is licensed under the MIT License, see [LICENSE](LICENSE). Third-party assets and acknowledgements are listed in [NOTICE](NOTICE):
- Charts under [tests/cases/](tests/cases/) are drawn from public benchmark sets for regression verification only; copyright belongs to the original mappers;
- Foundational architectural concepts build upon and extend the pioneering work of [sunny-rework](https://github.com/sunnyxxy/Star-Rating-Rebirth).

osu! is a registered trademark of ppy Pty Ltd. This project is an independent research initiative and is not affiliated with the osu! team.
