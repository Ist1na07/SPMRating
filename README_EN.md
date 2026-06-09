# SPM Rating — 7K osu!mania Difficulty Rating

English | [中文](README.md)

An open-source difficulty rating algorithm for osu!mania **7K** beatmaps.

**Core design**: 7 independent difficulty components extract multi-dimensional structural features — jack density, column-distance-weighted cross coordination, stream density, LN release interaction, anchor unevenness, shield protection, and inverse penalty. All components are composited into instantaneous difficulty D(t) at ~500Hz resolution under a unified Precompute/Combine architecture, aggregated through a Sigmoid player accuracy model, then refined by a **feature correction layer** (7 chart-level features, L2-regularized linear model) to capture systematic biases in the D formula, producing star ratings highly consistent with community consensus.

Trained on **311 beatmaps** (148 Dan + 57 Tournament + 20 Graveyard + 86 Ranked) via alternating-block Nelder-Mead optimization, with the correction layer independently trained via L2-regularized linear regression.

## Quick Start

### Option 1: Standalone single file (Recommended)

`spm_calc_standalone.py` is a **fully self-contained** single file with zero dependencies except numpy:

```bash
pip install numpy
python spm_calc_standalone.py chart.osu              # Single chart
python spm_calc_standalone.py "D:/osu/Songs/"        # Batch scan
```

All 3800+ lines of algorithm code + optimal parameters are inlined. No directory structure needed.

### Option 2: Module import

```bash
python spm_calc.py chart.osu                 # Uses spm_rating/ package
```

### API

```python
from spm_calc_standalone import compute_sr_map

sr, details = compute_sr_map("chart.osu")
print(f"SR = {sr:.4f}")
print(f"D_solved = {details['D_solved']:.2f}")
```

## Results

| Metric | Value |
|--------|-------|
| **MAE** | 0.213 |
| **In-sample Loss** | 0.770 |
| **CV Test Loss** | 0.862 (5-fold) |
| **Correlation** | 0.989 |
| **Pass@0.5** | 83.6% |

By source (311 charts):
- Dan (148 charts): MAE ~0.20
- Tournament (57 charts): MAE ~0.22
- Graveyard (20 charts): MAE ~0.38
- Ranked (86 charts): MAE ~0.25

**Correction layer improvement**: On top of Sigmoid aggregation, the correction layer reduces in-sample loss from 0.932 to 0.770 (-17.4%) via 7 chart-level features capturing systematic D formula biases.

## Algorithm

### 1. Feature Layer (36 parameters)

7 per-point difficulty components on ~500Hz time grid:

| Component | Meaning |
|-----------|---------|
| **Jbar** | Jack intensity |
| **Xbar** | Cross-column distance weighting |
| **Pbar** | Stream/trill density |
| **Rbar** | LN release interaction |
| **Abar** | Anchor/strain configuration |
| **Sbar** | Shield protection (minor contribution) |
| **Vbar** | Density inverse spike penalty (significant contribution) |

Components are combined into per-point instantaneous difficulty D(t):

```
S(t) = w1 · Jbar + (1-w1) · p_norm(Xbar, Pbar, exponent=p)
T(t) = p_norm(Rbar, Abar, exponent=p)

D(t) = β1 · √S · T^1.5 + β2 · S
        + α_P · Pbar + α_R · Rbar/(C_step + α_C)
        + α_S · Sbar(t) + α_V · Vbar(t)
```

### 2. Sigmoid Player Accuracy Aggregation

Instead of percentile cutoff, a physical player accuracy model:

$$A(d) = \frac{A_{max}}{C + e^{k(d-D)}}$$

where:
- **k = 2.09**: Decay rate (d-D=+1 → accuracy drops to 9%, +2 → 2%)
- **C = 3.97**: Curve shape (matching accuracy = 1/(C+1) ≈ 20%)
- **γ = 0.196**: Target average accuracy score

For each chart, solve D such that weighted average accuracy equals target:

$$\sum \frac{w_i}{C + e^{k(D_i-D)}} = total\_W \cdot \gamma$$

Solved via bisection → D_solved is the raw SR.

### 3. Feature Correction Layer

The D formula has **systematic biases** (e.g., overestimating chord density, underestimating fast jack fatigue). The correction layer captures these via 7 chart-level features:

```
correction = Σ w_j × feature_j    (j ∈ {speed, burst, chord, pj, hs, lb, fj})
D_new(t) = D_calib(t) + correction
```

| Feature | Weight | Physical Meaning |
|---------|--------|------------------|
| **chord** | -0.714 | Chord density (multi-column simultaneous coordination benefit) |
| **fj** | +0.265 | Fast jack (same-column rapid hit cumulative fatigue) |
| hs | +0.043 | Hand-switch (left-right hand coordination) |
| lb | +0.020 | Light burst (4-note groups) |
| speed | -0.038 | Speed patterns |
| burst | -0.025 | Burst patterns |
| pj | -0.005 | Stream/jack balance |

The correction layer is a **scalar linear model** (time-invariant), exploiting D_solved translation invariance for ~1500× speedup. L2 regularization (λ=0.01) controls weight magnitudes to prevent overfitting.

Full methodology: see `docs/TUNING_CORRECTION_LAYER_EN.md`.

### 4. Post-processing

The correction layer jointly re-optimized post-processing parameters:

```
SR = D_solved × n_eff/(n_eff + N0)        # Note count normalization (N0=0.0005)
if SR > 9.40: SR = 9.40 + (SR - 9.40)/1.98  # High SR compression
SR *= 1.061                                 # Global scale
```

### 5. D Pre-calibration

D(t) is linearly pre-calibrated before sigmoid aggregation:

$$D'(t) = 0.893 \cdot D(t) + 0.031$$

This compensates for calibration bias from percentile-based tuning.

### 6. Sub-models

- **RC Model**: Disables Rbar/Sbar/Vbar (Rice-only, LN heads treated as taps)
- **LN Model**: Uses LN-only masked aggregation (excludes RC-dominant sections)

## Parameter Files

| File | Contents |
|------|----------|
| `tuned_params_sigmoid.json` | Total SR parameters (Sigmoid aggregation, MAE=0.2180) |
| `tuned_correction.json` | Feature correction layer weights (7 features + 4 postprocess, CV Test Loss=0.862) |
| `tuned_params_rc.json` | RC sub-model parameters |
| `tuned_params_ln.json` | LN sub-model parameters |

## Package Structure

```
spm_rating/
├── README.md                     # This document
├── LICENSE                       # MIT
├── spm_calc_standalone.py                # ★ Standalone SR calculator (Recommended, with correction)
├── spm_calc.py                   # Module-based calculator (with correction)
├── tune_terminal.py              # Interactive tuning terminal
├── tuned_params_sigmoid.json     # Optimal sigmoid params
├── tuned_correction.json         # ★ Feature correction layer (7 features + 4 postprocess)
├── tuned_params_rc.json          # RC sub-model params
├── tuned_params_ln.json          # LN sub-model params
├── docs/
│   ├── TUNING_CORRECTION_LAYER.md   # ★ Correction layer methodology
│   ├── TUNING_CORRECTION_LAYER_EN.md# English version
│   ├── TUNING_METHODOLOGY.md        # Sigmoid layer methodology
│   └── TUNING_METHODOLOGY_EN.md     # English version
├── spm_rating/                   # Core algorithm
│   ├── rating.py                 # precompute() + combine() entry
│   ├── aggregate_sigmoid.py      # Sigmoid aggregation (bisection)
│   ├── aggregate.py              # Percentile aggregation (reference)
│   ├── combine.py                # D(t) formula
│   ├── combine_rc_ln.py          # RC/LN sub-models
│   ├── config.py                 # Parameter definitions
│   ├── parser.py                 # osu! chart parser
│   ├── preprocessor.py           # Preprocessing
│   ├── utils.py                  # Math utilities
│   └── components/               # Difficulty components
│       ├── anchor.py             # Anchor configuration
│       ├── cross.py / cross_enhanced.py
│       ├── inverse.py            # Density inverse
│       ├── jack.py               # Jack intensity
│       ├── release.py / release_enhanced.py
│       ├── shield.py             # Shield protection
│       ├── stamina.py            # Stamina (disabled)
│       └── stream.py             # Stream density
├── tuning/                       # Tuning toolkit
│   ├── data_loader.py            # Playtest data loader
│   └── scorer.py                 # Scoring functions
└── scripts/                      # Tuning & training scripts
    ├── tune_sigmoid_k15.py       # Primary: sigmoid NM tuning
    ├── tune_sigmoid_alternating.py  # Alternating-block NM
    ├── tune_rc.py / tune_ln.py   # Sub-model tuning
    ├── fit_ln_masked.py          # LN masked aggregation
    ├── fit_dan_regression.py     # Dan mapping
    ├── sweep_k_fine.py           # k-parameter fine scan
    ├── rebuild_enhanced_cache.py # Cache rebuild
    ├── build_standalone.py       # Build spm_calc_standalone.py
    ├── train_sort_classifier.py  # Map type classifier
    └── train_tag_classifier.py   # Pattern tag classifier
```

## Technical Details

### D_solved vs D(t) Distribution

The bisection-solved D_solved consistently falls near **P70** of the weighted D(t) distribution:

$$D_{solved} \approx 0.87 \cdot D_{P70}$$

std = 0.01 (extremely stable across 311 charts). This validates the physical consistency of the sigmoid model — it's not "random" aggregation but a smooth soft-percentile selection via the accuracy equation.

### Physical Meaning of k

Player accuracy drops sharply when difficulty exceeds their level:

| d - D | A/A_max | Interpretation |
|-------|---------|----------------|
| 0 | 20% | At player's ceiling |
| +1 | 9% | Slightly beyond |
| +2 | 2% | Far beyond |
| +3 | 0.3% | Completely unable |

Matches real-world experience: 7K players cannot meaningfully play charts 2 SR above their ceiling.

### C-γ Self-Consistency

The relation **γ ≈ 1/(C+1)** holds precisely: with C=3.97, γ≈0.196 ≈ 1/5.1. This means the target accuracy equals the matching-point accuracy, confirming the model's internal consistency.

## Evaluation

To evaluate the algorithm on playtest data:

```python
from tuning.data_loader import load_playtest_data
from tuning.scorer import score_single
from spm_rating.rating import precompute, combine
import json

entries = load_playtest_data()  # Requires maps/ directory with Excel + .osu files
params = json.load(open("tuned_params_sigmoid.json"))["params"]

for entry in entries:
    cache = precompute(entry["osu_path"], use_enhanced=True, params=params)
    sr, _ = combine(cache, params=params)
    print(f"{entry['mapfile']}: SR={sr:.2f}, Ref={entry['sr_ref']:.2f}")
```

## Tuning

### Sigmoid Layer: Alternating-Block Nelder-Mead

Parameters were optimized via **alternating-block Nelder-Mead** (6 blocks × 2 rounds, 50→25 iterations):

| Block | Parameters | Mode | Contribution |
|-------|-----------|------|-------------|
| B1 | k, C, γ, calib, N0, threshold, divisor, scale | Fast (~0.04s) | ~95% of improvement |
| B2 | D formula weights & exponents | Full (~14.5s) | Fine-tuning |
| B3a | Cross feature params | Full | Minimal improvement |
| B3b | Release feature params (incl. short_ln) | Full | Minimal improvement |
| B3c | Inverse feature params (incl. same_col_bonus) | Full | Minimal improvement |
| B3d | Jack aggregation params | Full | Minimal improvement |

Key finding: **k=2.09 is optimal** (largest single improvement). Full methodology: see `docs/TUNING_METHODOLOGY_EN.md`.

### Correction Layer: L2-Regularized Linear Regression

The correction layer is independently trained on top of the Sigmoid layer (11 parameters, not blocked):

- **Optimizer**: Nelder-Mead (maxiter=10000, xatol=1e-7, fatol=1e-7, adaptive=True)
- **Regularization**: L2 (λ=0.01), controls weight magnitudes
- **Cross-validation**: 5-fold CV (seed=42), CV Test Loss=0.862
- **Key technique**: Exploits D_solved translation invariance for ~1500× speedup

Full methodology: see `docs/TUNING_CORRECTION_LAYER_EN.md`.

## Version History

### v0.3.0 (Current)
- **Added feature correction layer**: 7 chart-level features (speed, burst, chord, pj, hs, lb, fj), L2-regularized linear model
- In-sample Loss: 0.932 → **0.770** (-17.4%)
- CV Test Loss: **0.862** (5-fold, gap=0.092)
- MAE: 0.218 → **0.213**, Correlation: 0.988 → **0.989**
- Post-processing parameters jointly re-optimized with correction layer
- Full tuning methodology: `docs/TUNING_CORRECTION_LAYER_EN.md`

### v0.2.0
- Dataset expanded from 213 to **311 charts** (added 86 Ranked + 12 Tournament)
- k-value re-optimized: 1.5 → **2.09** (coarse + fine scan verified)
- C-γ self-consistency verified: γ ≈ 1/(C+1), C=3.97, γ=0.196
- Tuning blocks expanded from 5 to **6 blocks**: added B3d (Jack), B3b/B3c each gained 2/1 new params
- RC sub-model retrained: MAE=0.2366 (27% improvement)
- LN sub-model retrained: MAE=0.8162 (needs architectural redesign)

### v0.1.0
- Initial release: Sigmoid aggregation (k=1.5, C=4.0, γ=0.20) + 7-component D formula
- Trained on 213 charts, MAE=0.2253

## Plugin

This algorithm powers the **SPM Map Analyser** tosu overlay plugin:
- Repository: [Ist1na07/spm_rating_map_analyser](https://github.com/Ist1na07/spm_rating_map_analyser)
- Provides real-time in-game difficulty display, including RC/LN sub-models, Dan mapping, and ML pattern classification

## Dependencies

- **numpy** (required for all modes)
- **scipy** (required for Nelder-Mead tuning only)

## License

MIT — see [LICENSE](LICENSE)
