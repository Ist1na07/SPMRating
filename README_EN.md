# SPM Rating v4.0 — 7K osu!mania Difficulty Rating

English | [中文](README.md)

An open-source difficulty rating algorithm for osu!mania **7K** beatmaps.

## Origin

This algorithm is based on **[Star-Rating-Rebirth](https://github.com/sunnyxxy/Star-Rating-Rebirth)** (sunny rework) and evolved through the following modifications:

- **Difficulty component expansion**: Beyond the original jack / stream components, new components were added — inverse (density-reversal penalty), shield (shield protection), release (LN release interaction) — and cross (column-distance weighting) was rewritten.
- **D(t) formula restructuring**: Component composition changed from linear superposition to a nonlinear form (`β1·√S·T^1.5 + β2·S + α·component`), better matching realistic fatigue accumulation.
- **Player accuracy aggregation model**: Replaced the original percentile-truncation with an accuracy equation `A(d) = A_max / (C + e^{k(d-D)})`. For each chart, solve for the difficulty `D_solved` such that the weighted average accuracy equals the target value `γ`; this serves as the raw SR. The aggregation provides a physically interpretable difficulty-accuracy mapping.
- **D pre-calibration**: Apply a linear transform `D' = 0.893·D + 0.031` before aggregation, compensating for systematic magnitude bias from the component formula.
- **Feature correction layer**: A linear correction from 9 chart-level features is layered on top of the main formula, capturing structural biases that D(t) cannot express (see below).
- **Post-processing re-optimization**: Note-count normalization, high-SR compression, and global scale — trained jointly with the correction layer.

## Quick Start

### Option 1: Standalone single file (Recommended)

`spm_calc_standalone.py` is a **fully self-contained** single file with zero dependencies beyond numpy:

```bash
pip install numpy
python spm_calc_standalone.py chart.osu         # single chart
python spm_calc_standalone.py "D:/osu/Songs/"  # batch scan
```

All algorithm code and tuned parameters are inlined; no directory structure required.

### Option 2: Module import

```bash
python spm_calc.py chart.osu                    # depends on spm_rating/ package
```

### Programming interface

```python
from spm_calc_standalone import compute_sr_map

sr, details = compute_sr_map("chart.osu")
print(f"SR = {sr:.4f}")
print(f"D_solved = {details['D_solved']:.2f}")
```

## Algorithm Architecture

### 1. Difficulty component layer

Seven components are computed on a ~500Hz time grid:

| Component | Meaning |
|-----------|---------|
| **Jbar** | Jack density |
| **Xbar** | Column-distance-weighted difficulty |
| **Pbar** | Stream density |
| **Rbar** | LN release interaction difficulty |
| **Abar** | Anchor / hand-lock configuration |
| **Sbar** | Shield protection (small contribution) |
| **Vbar** | Inverse penalty (significant contribution) |

Components are combined into instantaneous difficulty D(t):

```
S(t) = w1 · Jbar + (1-w1) · p_norm(Xbar, Pbar, exponent=p)
T(t) = p_norm(Rbar, Abar, exponent=p)

D(t) = β1 · √S · T^1.5 + β2 · S
        + α_P · Pbar + α_R · Rbar/(C_step + α_C)
        + α_S · Sbar(t) + α_V · Vbar(t)
```

### 2. Player accuracy aggregation model

Instead of percentile truncation, an accuracy model is solved:

$$A(d) = \frac{A_{max}}{C + e^{k(d-D)}}$$

where:
- **k = 2.09**: decay rate (d-D=+1 → accuracy drops to 9%, +2 → 2%)
- **C = 3.97**: curve shape (matching-point accuracy = 1/(C+1) ≈ 20%)
- **γ = 0.196**: target average accuracy score

For each chart, solve for D such that the weighted average accuracy equals the target:

$$\sum \frac{w_i}{C + e^{k(D_i-D)}} = total\_W \cdot \gamma$$

Solved via bisection → D_solved is the raw SR.

### 3. Feature correction layer

The D formula has **systematic biases** (e.g., overestimating chord density, underestimating fast-jack fatigue). The correction layer captures these biases via 9 chart-level features:

```
correction = Σ w_j × feature_j    (j ∈ {speed, burst, chord, pj, hs, lb, fj, nps_std, chord2})
D_new(t) = D_calib(t) + correction
```

| Feature | Weight | Meaning |
|---------|--------|---------|
| **chord2** | -0.656 | Two-note chord density (exactly-2-column simultaneous events; v4.0 new) |
| **chord** | -0.769 | Chord density (≥4-column simultaneous events reduce per-finger load) |
| **fj** | +0.031 | Fast jack (same-column rapid-fire fatigue accumulation) |
| hs | +0.073 | Hand switch (left-right hand coordination) |
| lb | +0.016 | Light burst (4-note groups) |
| speed | -0.047 | Speed-type patterns |
| burst | -0.029 | Burst-type patterns |
| pj | +0.002 | Stream / jack balance |
| **nps_std** | -0.014 | Density temporal variation (500ms-window NPS std; v4.0 new) |

**v4.0 new features**:

- **chord2** (two-note chord density): The original `chord` feature threshold is ≥4 columns, covering only large chords. chord2 covers the most common two-note chords (jumpstream / chordstream) and is complementary to chord. A high chord2 indicates the chart leans toward two-finger simultaneous effort; per-finger load is heavier than pure stream but Pbar (global NPS) overestimates it, hence the negative-weight compensation. Split experiments confirmed: 3-note chords (chord3) are ineffective; 4+ note chords (chord4p) are redundant with the original chord (correlation 0.955); only two-note chords are effective.
- **nps_std** (density temporal variation): Split the chart into 500ms windows and compute the per-window NPS standard deviation. High nps_std = burst + rest alternation (recovery available); low nps_std = uniform density throughout (sustained fatigue). Captures the "temporal" dimension missing from the existing 7 features, orthogonal to chord2.

The correction layer is a **scalar linear model** (time-invariant), achieving ~1500× speedup via the D_solved translation-invariance property. L2 regularization (λ=0.01) controls weight magnitude to prevent overfitting.

Full methodology in `docs/TUNING_CORRECTION_LAYER_EN.md`.

### 4. Post-processing

The correction layer is jointly re-optimized with post-processing parameters:

```
SR = D_solved × n_eff/(n_eff + N0)       # note-count normalization (N0=1.029)
if SR > 9.11: SR = 9.11 + (SR - 9.11)/1.97  # high-SR compression
SR *= 1.094                                  # global scale
```

### 5. D pre-calibration

D(t) passes through a linear pre-calibration before entering the accuracy aggregation:

$$D'(t) = 0.893 \cdot D(t) + 0.031$$

Compensates for magnitude bias from component aggregation.

## Parameter Files

| File | Contents |
|------|----------|
| `tuned_params_sigmoid.json` | Main-formula parameters (component weights, aggregation params, pre-calibration) |
| `tuned_correction.json` | Feature correction layer weights (v4.0: 9 features + 4 postprocess) |
| `tuned_params_rc.json` | RC sub-model parameters |
| `tuned_params_ln.json` | LN sub-model parameters |

## Directory Structure

```
SPMRating-Z-Release/
├── README.md                        # This document
├── README_EN.md                     # English version
├── LICENSE                          # MIT
├── spm_calc_standalone.py           # ★ Standalone single-file SR calculator (recommended, with correction layer)
├── spm_calc.py                      # Module-version calculator (with correction layer)
├── tune_terminal.py                 # Interactive tuning terminal
├── tuned_params_sigmoid.json        # Optimal main-formula parameters
├── tuned_correction.json            # ★ Feature correction layer weights (v4.0: 9 features + 4 postprocess)
├── tuned_params_rc.json             # RC sub-model parameters
├── tuned_params_ln.json             # LN sub-model parameters
├── docs/
│   ├── TUNING_CORRECTION_LAYER.md   # ★ Correction-layer tuning methodology
│   ├── TUNING_CORRECTION_LAYER_EN.md
├── spm_rating/                      # Core algorithm
│   ├── rating.py                    # precompute() + combine() entry
│   ├── aggregate_sigmoid.py         # Accuracy aggregation (bisection)
│   ├── aggregate.py                 # Percentile aggregation (for comparison)
│   ├── combine.py                   # D(t) formula
│   ├── combine_rc_ln.py             # RC/LN sub-models
│   ├── config.py                    # Parameter definitions
│   ├── parser.py                    # osu! beatmap parser
│   ├── preprocessor.py              # Preprocessing
│   ├── utils.py                     # Math utilities
│   └── components/                  # Difficulty components
│       ├── jack.py                  # Jack
│       ├── cross.py / cross_enhanced.py
│       ├── stream.py                # Stream density
│       ├── release.py / release_enhanced.py
│       ├── anchor.py                # Anchor
│       ├── shield.py                # Shield
│       ├── inverse.py               # Inverse
│       └── stamina.py               # Stamina (disabled)
├── tuning/                          # Tuning tools
│   ├── data_loader.py               # Playtest data loader
│   └── scorer.py                    # Scoring function
└── scripts/                         # Tuning / training scripts
    ├── tune_sigmoid_k15.py          # Core: main-formula NM tuning
    ├── tune_sigmoid_alternating.py  # Alternating-block NM
    ├── tune_rc.py / tune_ln.py      # Sub-model tuning
    ├── fit_ln_masked.py             # LN-masked aggregation
    ├── fit_dan_regression.py        # Dan-tier mapping
    ├── sweep_k_fine.py              # Fine k sweep
    ├── rebuild_enhanced_cache.py    # Cache rebuild
    ├── build_standalone.py          # Build standalone single file
    ├── retrain_correction_zver.py   # v4.0 correction-layer retraining
    ├── residual_diagnosis.py        # Residual diagnosis
    └── verify_release.py            # Release verification
```

## Tuning Methodology

### Main-formula layer: alternating-block Nelder-Mead

Parameters are tuned via **alternating-block Nelder-Mead** (6 blocks × 2 rounds, 50→25 iterations):

| Block | Parameters | Mode | Contribution |
|-------|-----------|------|--------------|
| B1 | k, C, γ, calib, N0, threshold, divisor, scale | Fast (~0.04s) | ~95% of improvement |
| B2 | D-formula weights and exponents | Full (~14.5s) | Fine-tuning |
| B3a | Cross feature-layer params | Full | Minor |
| B3b | Release feature-layer params (incl. short_ln) | Full | Minor |
| B3c | Inverse feature-layer params (incl. same_col_bonus) | Full | Minor |
| B3d | Jack aggregation params | Full | Minor |

Key finding: **k=2.09 is optimal** (largest single improvement). Full methodology in `docs/TUNING_METHODOLOGY.md`.

### Correction layer: L2-regularized linear regression

The correction layer is trained independently on top of the main formula (9 feature weights + 4 postprocess params):

- **Optimizer**: Nelder-Mead (maxiter=10000, xatol=1e-7, fatol=1e-7, adaptive=True)
- **Regularization**: L2 (λ=0.01), controlling weight magnitude
- **Cross-validation**: 5-fold CV (seed=42)
- **Key trick**: ~1500× speedup via D_solved translation-invariance

Full methodology in `docs/TUNING_CORRECTION_LAYER_EN.md`.

## Technical Details

### Physical meaning of k

Player accuracy drops sharply when difficulty exceeds their level:

| d - D | A/A_max | Meaning |
|-------|---------|---------|
| 0 | 20% | Matched player ceiling |
| +1 | 9% | Slightly above level |
| +2 | 2% | Far above level |
| +3 | 0.3% | Cannot cope |

Consistent with experience: 7K players are essentially unable to play normally beyond 2SR above their skill ceiling.

### C-γ self-consistency

**γ ≈ 1/(C+1)** holds exactly: with C=3.97, γ≈0.196 ≈ 1/5.1. Target accuracy equals matching-point accuracy, verifying the model's internal self-consistency.

## Plugin

This algorithm powers the **SPM Map Analyser** tosu in-game overlay plugin:

- Repo: [Ist1na07/spm_rating_map_analyser](https://github.com/Ist1na07/spm_rating_map_analyser)
- Provides real-time difficulty display, with dan-tier mapping and ML key-pattern classification

## Dependencies

- **numpy** (required)
- **scipy** (only for Nelder-Mead tuning)
- **pandas** (only for playtest evaluation)

## License

MIT — see [LICENSE](LICENSE)
