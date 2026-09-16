# SPM Rating v1.1.1

[简体中文文档](README.zh-CN.md)

SPM Rating is a high-precision difficulty rating algorithm for osu!mania beatmaps (prioritizing 7K, while comprehensively supporting 4K and 6K). It calculates multi-dimensional physical difficulty curves along the chart's timeline, synthesizes an instantaneous total difficulty curve $D(t)$, and aggregates it into a single Star Rating (SR) via two-level sectioned mass aggregation and global calibration.

On updated and expanded benchmark datasets, v1.1.1 achieves (measured by Spearman's rank correlation coefficient $sp$, which quantifies the agreement between algorithmic ranking and human ground-truth labels):
- **7-Key (7K)**: **0.97899** on the expanded benchmark set (historical high with deadzone loss $dz$ reduced by 5.6%; retaining **0.98437** on the classic benchmark subset within statistical noise floor);
- **4-Key (4K)**: **0.98573** (statistically significant positive improvement, paired bootstrap 90% CI $[+0.00003, +0.00083]$);
- **6-Key (6K)**: **0.95417** (zero-shot evaluation domain, strictly preserved within tolerance without over-tuning).

### What's New in v1.1.1
v1.1.1 retunes and calibrates algorithm hyperparameters using **updated and expanded benchmark data**:
- **Long-Note Coordination & Release Dynamics**: Re-calibrated long-note release scaling ($a_r$) and damping ($a_c$) alongside multi-finger chord suppression ($a_{cbc}$), eliminating over-penalization in dense long-note chords while capturing true release resistance;
- **Refined Stream & Jack Balance**: Fine-tuned combination weights ($c_{jm}, c_{jc}, c_s$) for more accurate separation of high-speed streams and bursts;
- **Decoupled Multi-Key Dynamics**: Independent key-count parameters ($a_{cb\_4k}, a_{p\_4k}$) prevent inter-mode interference;
- **Stress-Test Invariance Verified**: With $pp\_n0 = 0.0$ strictly preserved, repeated section inflation (T2) maximum deviation is pushed down to **0.00856** (passing the strict $<0.010$ ceiling), while T1 (0.0029), T3 (0.0135), and T4 (0.0567) remain fully robust.

Compared to v1.0.0, **v1.1.x resolves several long-standing structural defects**:
- **Two-Level Sectioned Mass Aggregation (Mode 4)**: Driven by note mass rather than wall-clock duration, completely eliminating distant sparse note dilution (T1), repeated section inflation (T2), and long silence gap dilution (T3);
- **Continuous Gesture Chord Detection & Structured Bilateral Pricing**: Upgraded chord detection from discrete millisecond slicing to continuous gesture-level detection, eradicating $\pm 1$ms jitter cliffs and establishing unified 2-7 note chord pricing;
- **Axiomatic Inversion Fixes**: Fixed superset monotonicity axiom A1 (pressing more keys simultaneously must not be easier) and dual-hand parallelism axiom A4 (one hand taking multiple keys must not be cheaper than sharing across hands);
- **Segment-Level Difficulty Robustness**: Extensively verified via synthetic patterns and real sliced charts, reducing segment isolation vs. tiling distortion by nearly 10x.

The algorithm is built upon the [sunny-rework](https://github.com/sunnyxxy/Star-Rating-Rebirth) architecture and substantially redesigned. For detailed ablations and evolution, see the History document.

## Documentation

| Document | Key Contents |
|---|---|
| [docs/mechanism.md](docs/mechanism.md) | Complete pipeline from chart to star rating, formulas for each channel, and physical meanings of all 276 release parameters |
| [docs/history.md](docs/history.md) | Evolution from sunny to v1.0.0, v1.1.0, and v1.1.1, ablation benchmarks, falsified avenues, and empirical methodology |
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
params/        Parameter files; release parameter is spm-rating-1.1.1.json (276 parameters)
tests/         Regression suite: test cases (cases/), expected ratings (expected.json), compliance runner
docs/          Detailed mechanism specification, historical development report, and maintenance guide
```

## License & Acknowledgements

The code is licensed under the MIT License, see [LICENSE](LICENSE). Third-party assets and acknowledgements are listed in [NOTICE](NOTICE):
- Charts under [tests/cases/](tests/cases/) are drawn from public benchmark sets for regression verification only; copyright belongs to the original mappers;
- Foundational architectural concepts build upon and extend the pioneering work of [sunny-rework](https://github.com/sunnyxxy/Star-Rating-Rebirth).

osu! is a registered trademark of ppy Pty Ltd. This project is an independent research initiative and is not affiliated with the osu! team.
