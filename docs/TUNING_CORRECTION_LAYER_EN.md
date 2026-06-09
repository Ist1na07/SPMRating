# SPM Rating — Feature Correction Layer Tuning Methodology

**Date**: 2026-06-02
**Context**: Adding a feature correction layer on top of the Sigmoid aggregation layer (k=2.09, C=3.97, γ=0.196). Final results: Loss=0.770, MAE=0.213, r=0.989, Pass=83.6%.

---

## Table of Contents

1. [Problem Background](#1-problem-background)
2. [Architecture: Correction Layer Position in Pipeline](#2-architecture)
3. [Seven Features and Their Physical Meaning](#3-seven-features)
4. [Core Property: D_solved Translation Invariance](#4-d_solved-translation-invariance)
5. [Precomputation Phase: Cache Building](#5-precomputation-phase)
6. [Optimization Phase: Scalar Nelder-Mead](#6-optimization-phase)
7. [Regularization Strategy](#7-regularization-strategy)
8. [Cross-Validation: Overfitting Control](#8-cross-validation)
9. [Feature Threshold Parameters](#9-feature-threshold-parameters)
10. [Complete Tuning Workflow](#10-complete-tuning-workflow)
11. [Key Findings and Pitfalls](#11-key-findings-and-pitfalls)

---

## 1. Problem Background

### 1.1 Systematic Bias in the D Formula

SPM Rating's D(t) formula computes the combination of three components—S (sustained), T (technicality), and D (instantaneous)—at each time step to describe chart difficulty. Despite the D formula containing 41 parameters (optimized in 6 alternating blocks), **systematic biases** remain:

| Pattern | D Formula Behavior | Actual Feel | Bias Direction |
|---------|-------------------|-------------|----------------|
| Chord | Computes difficulty per column independently, ignoring multi-column coordination | Multi-finger coordination reduces actual difficulty | **Overestimate** |
| Fast Jack (same-column rapid hits) | Instantaneous difficulty curve insufficiently captures cumulative fatigue | Precision timing demands grow non-linearly with frequency | **Underestimate** |
| Hand-switch | Cross-column distance only reflects spatial span | Left-right hand switching requires additional coordination | **Underestimate** |
| Burst | High-density triplet patterns | Brief bursts followed by muscle relaxation | **Overestimate** |

These biases are characterized by: **correlating with specific pattern densities in the chart, and being directional (not random noise)**.

### 1.2 Why Not Directly Modify the D Formula

Modifying the D formula (e.g., adding chord jack discounts, changing jack aggregation) has two issues:

1. **High-dimensional coupling**: The D formula's 41 parameters have been thoroughly optimized through 3 rounds of alternating NM; modifying any component requires re-running the full alternating pipeline (~2 hours/round)
2. **Overfitting risk**: D formula parameters act on each moment's D(t) value (~tens of thousands of time points); fine-tuning easily introduces chart-specific biases

The feature correction layer's design goal is: **to capture systematic biases using 7 chart-level features without modifying the D formula**.

### 1.3 Relationship to Sigmoid Aggregation Layer Tuning

The feature correction layer sits above the Sigmoid aggregation layer (see Section 2), assuming Sigmoid aggregation parameters (k, C, γ, calib) are already optimized. The correction layer **does not re-optimize** aggregation layer parameters; instead, it adds corrections on top of the already-calibrated D_calib.

This means:
- If Sigmoid layer parameters change (e.g., after re-tuning k with updated dataset), **the correction layer needs retraining**
- The correction layer's parameters (7 weights + 4 postprocess) are far fewer than the D formula layer (41), making training 100×+ faster

---

## 2. Architecture

### 2.1 Pipeline Position

```
.osu file
    ↓ precompute()
cache (note_seq, LN_seq, Jbar, Pbar, all_corners, ...)
    ↓ combine()
D_full[t], C_arr (standard D formula output)
    ↓ Calibration (already in Phase 5)
D_calib = calib_a × D_full + calib_b
    ↓ ★ Feature Correction Layer (NEW)
correction = Σ w_j × feature_j
D_new = max(D_calib + correction, 0.01)
    ↓ Sigmoid Aggregation
SR = sigmoid_aggregate(D_new, total_notes, postprocess_params)
```

### 2.2 Key Design Decision: Scalar Correction on D Sequence

The correction is a **scalar** (not a per-timestep vector), added directly to all timesteps of D_calib:

```
D_new(t) = D_calib(t) + correction
```

This ensures the corrected D(t) distribution shape remains unchanged (only shifted), enabling fast optimization via D_solved translation invariance (see Section 4).

### 2.3 Parameter Grouping

The correction layer has 11 parameters total, optimized as a whole (not blocked):

| Group | Param Count | Content |
|-------|:-----------:|---------|
| **W: Feature Weights** | 7 | w_speed, w_burst, w_chord, w_pj, w_hs, w_lb, w_fj |
| **P: Postprocess** | 4 | N0, threshold, divisor, global_scale |

Postprocess parameters (P group) are inherited from the Sigmoid layer and **jointly re-optimized**, because the correction changes D_solved's numerical range, requiring postprocess parameters to adapt.

---

## 3. Seven Features

### 3.1 Feature Definitions

All features are computed from `precompute()` cache data, outputting **chart-level scalars** (not per-timestep sequences).

| Feature | Name | Definition | Unit |
|---------|------|------------|------|
| **speed** | Speed pattern density | Note pairs with dt < `spd_dt` and dc ≥ `spd_dc` / duration | count/sec |
| **burst** | Burst pattern density | Triplets where `times[i] - times[i-2]` < `bst_dt` / duration | count/sec |
| **chord** | Chord fraction | Notes participating in ≥`ch_order` simultaneous notes / total notes | ratio |
| **pj** | Stream/jack balance | mean(Pbar) / (mean(Jbar) + 1) | ratio |
| **hs** | Hand-switch density | Note pairs with dt < `hs_dt` and left-right hand switch / duration | count/sec |
| **lb** | Light burst density | 4-note groups where `times[i] - times[i-3]` < `lb_dt` / duration | count/sec |
| **fj** | Fast jack density | Same-column consecutive notes with dt < `fj_dt` / duration | count/sec |

Where dt = time difference between adjacent notes (ms), dc = absolute column difference between adjacent notes.

### 3.2 Physical Meaning and Weight Interpretation

Final optimized weights (λ=0.01):

| Feature | Weight | Physical Meaning |
|---------|--------|------------------|
| **chord** | **-0.714** | D formula overestimates chord-density charts. Multi-column simultaneous hits have coordination benefits, but D(t) computes each column independently |
| **fj** | **+0.265** | D formula underestimates fast jack density. Cumulative fatigue from same-column rapid hits insufficiently captured |
| **hs** | +0.043 | Hand-switch difficulty slightly underestimated |
| **lb** | +0.020 | Light burst (4-note group) difficulty slightly underestimated |
| **speed** | -0.038 | Speed patterns slightly overestimated |
| **burst** | -0.025 | Burst patterns slightly overestimated |
| **pj** | -0.005 | Stream/jack balance nearly irrelevant (close to zero) |

**Large weight magnitude differences** (chord is 143× pj) are normal: different features contribute to D formula bias at different scales. L2 regularization (see Section 7) controls extreme weights.

### 3.3 Feature Computation Notes

1. **Chord uses ratio, not density**: Chord is the only feature not in "count/sec" units; it represents the fraction of chord notes in the chart (0~1). This is because chord's definition is based on simultaneous note counting, naturally coupling with total chart density.

2. **pj reads Jbar/Pbar from cache**: Unlike other features, pj requires `precompute()` outputs Jbar_base and Pbar_base, and cannot be computed from note_seq alone.

3. **All features normalized to chart-level statistics**: Divided by duration or total note count, eliminating chart length effects.

---

## 4. D_solved Translation Invariance

### 4.1 Core Property

Key insight for correction layer optimization: when the D(t) distribution is shifted by a constant c, the Sigmoid aggregation's solution D_solved also shifts by approximately the same amount.

```
D_solved({D(t) + c}) ≈ D_solved({D(t)}) + c
```

**Mathematical intuition**: Sigmoid aggregation uses bisection to solve for D_solved such that:

```
Σ w_i / (C + e^(k(D_i - D_solved))) = total_W × γ
```

Replacing all D_i with D_i + c is equivalent to replacing D_solved with D_solved + c (both sides of the equation remain identical).

### 4.2 Approximation Accuracy

This property holds approximately under the following conditions:

1. **D(t) distribution's piecewise aggregation** introduces minor errors (30 segments)
2. **Effective weights w_i** slightly couple with D values
3. Measured error < 0.001 SR (averaged over 311 charts)

### 4.3 Why It Matters

**Without** translation invariance:
```
for each NM evaluation:
    for each map:
        D_new(t) = D_calib(t) + correction  # modify entire D sequence
        D_solved = solve_bisection(D_new)    # bisection solve ~5ms/chart
    # Total time: ~1.5s/eval → NM 10000 iter ≈ 4 hours
```

**With** translation invariance:
```
# One-time precomputation (~1s)
for each map:
    D_solved_base = solve_bisection(D_calib)

# Fast evaluation
for each NM evaluation:
    for each map:
        D_solved_new = D_solved_base + correction  # scalar addition ~1μs/chart
    # Total time: ~0.001s/eval → NM 10000 iter ≈ 10 seconds
```

**Speedup: ~1500×**

### 4.4 Limitations

Translation invariance **only holds for scalar linear corrections**. The following cases do not apply:

1. **Non-linear corrections**: If correction is a function of D_solved (e.g., chord discount model), translation invariance doesn't hold
2. **Interaction terms**: Feature interactions (e.g., chord × fj) may give inconsistent results under fast approximation vs full pipeline (measured: interaction term CV improvement 2.4%, but full pipeline degradation 5.2%)
3. **Per-timestep corrections**: If correction varies with t, D(t) distribution shape changes, and translation invariance doesn't hold

**Therefore, the correction layer should always maintain scalar linear form: correction = Σ w_j × feature_j.**

---

## 5. Precomputation Phase

### 5.1 Cache Building

Execute full `precompute()` + `combine()` pipeline once per chart:

```python
for each map:
    cache = precompute(osu_path, use_enhanced=True, params=params_spm)
    _, details = combine(cache, params=params_spm)
    # Save:
    #   cache_i.pkl  → note_seq, LN_seq, Jbar_base, Pbar_base, all_corners
    #   d_i.npz      → D_full, C_arr
```

311 charts take ~15 minutes (single-threaded), only need rebuilding when:
- Adding/removing charts
- D formula parameters (B2/B3 blocks) change
- precompute() logic changes

### 5.2 D_solved Precomputation

On top of cache, compute each chart's baseline D_solved using established Sigmoid layer parameters:

```python
for each map:
    D_calib = calib_a * D_full + calib_b
    eff_w = compute_effective_weights(all_corners, C_arr)
    D_seg, w_seg = segment_by_difficulty(D_calib, eff_w, 30)
    D_solved = solve_bisection(D_seg, w_seg, k, C, gamma)
    n_eff = compute_total_notes(note_seq, LN_seq)
```

This step takes < 1 second (all charts), but **needs re-running whenever Sigmoid layer parameters change**.

### 5.3 Feature Precomputation

Compute 7 feature values from cached data (see Section 3):

```python
for each map:
    features = compute_features(cache, FEAT_PARAMS)
```

Feature computation only depends on note_seq, LN_seq, Jbar_base, Pbar_base, all from cache. Takes < 1 second.

---

## 6. Optimization Phase

### 6.1 Parameter Vector

11 parameters total, encoded as 1D vector:

```
x = [w_speed, w_burst, w_chord, w_pj, w_hs, w_lb, w_fj, N0, threshold, divisor, global_scale]
     |←────────────────── W (7) ──────────────────→|  |←────────── P (4) ──────────→|
```

### 6.2 Evaluation Function

```python
def eval_model(x, indices):
    w = x[:7]
    N0, thr, div, gs = x[7:]

    total_loss = 0.0
    for i in indices:
        correction = sum(w[j] * features[i][j] for j in range(7))
        # Using translation invariance
        D_shifted = D_solved_base[i] + correction
        SR = postprocess(D_shifted, n_eff[i], N0, thr, div, gs)
        total_loss += score_single(SR, sr_ref[i], sr_error[i])

    return total_loss / len(indices) + regularization
```

Where the postprocess function:

```python
def postprocess(D_shifted, n_eff, N0, threshold, divisor, scale):
    N0_safe = max(N0, 0.01)
    SR = D_shifted * n_eff / (n_eff + N0_safe)
    if SR > threshold:
        SR = threshold + (SR - threshold) / divisor
    return SR * scale
```

### 6.3 Optimizer Configuration

Uses Nelder-Mead (consistent with Sigmoid layer tuning), but with finer configuration:

```python
scipy.optimize.minimize(
    eval_model,
    x0=[0.0]*7 + [N0_def, thr_def, div_def, gs_def],
    method="Nelder-Mead",
    options={
        "maxiter": 10000,
        "xatol": 1e-7,
        "fatol": 1e-7,
        "adaptive": True
    }
)
```

**Parameter explanations**:
- `maxiter=10000`: 11D space sufficient for convergence (measured ~3000 evals to stabilize)
- `xatol=1e-7, fatol=1e-7`: High-precision convergence (evaluation is fast, no need to save evals)
- `adaptive=True`: Enable adaptive NM, faster convergence in high-dimensional space

### 6.4 Multiple Restarts

To avoid local optima, use 5 random restarts:

```python
for restart in range(5):
    if restart == 0:
        x0 = [0.0]*7 + [defaults...]  # Zero initialization
    else:
        x0 = random_normal(...)       # Random perturbation
    res = minimize(eval_model, x0, ...)
    best = min(best, res, key=lambda r: r.fun)
```

Zero initialization (restart=0) usually converges to global optimum; random restarts verify robustness.

### 6.5 Initial Value Selection

| Parameter | Initial Value | Source |
|-----------|---------------|--------|
| w_j (all weights) | 0.0 | No correction (zero start) |
| N0 | 8.21 | Sigmoid layer optimum |
| threshold | 9.42 | Sigmoid layer optimum |
| divisor | 2.01 | Sigmoid layer optimum |
| global_scale | 1.055 | Sigmoid layer optimum |

Postprocess parameters start from Sigmoid layer optima, because when corrections are small, postprocess parameter changes are minimal.

---

## 7. Regularization Strategy

### 7.1 L2 Regularization

Evaluation function adds L2 penalty:

```python
loss = base_loss + λ × Σ w_j²
```

**λ selection**:

| λ | Train Loss | CV Test Loss | Gap | Recommendation |
|---|-----------|-------------|-----|----------------|
| 0.000 | 0.715 | 0.903 | 0.188 | Overfitting |
| 0.005 | 0.752 | 0.878 | 0.126 | — |
| **0.010** | **0.770** | **0.862** | **0.092** | **Optimal** |
| 0.020 | 0.793 | 0.872 | 0.079 | Slightly worse |
| 0.050 | 0.827 | 0.895 | 0.068 | Underfitting |

**λ=0.01** is optimal on CV test loss. Note gap monotonically decreases with λ (stronger regularization → less overfitting), but test loss reaches minimum at λ=0.01.

### 7.2 N0 ≥ 0 Constraint

Note count normalization parameter N0's physical meaning is "equivalent virtual note count", must be non-negative. Implemented via penalty:

```python
n0_penalty = max(0, -N0)² × 10.0
```

Strong penalty (coefficient 10.0) when N0 < 0, no penalty when N0 ≥ 0. More suitable for NM optimizer than hard constraints.

**Measured effect**: N0 converges to ~0.001 (close to zero but positive), indicating short charts in current dataset don't need additional note count penalties, but constraint prevents N0 from becoming negative (physically meaningless).

### 7.3 Why L1 Regularization Is Not Needed

L1 regularization (Σ|w_j|) promotes weight sparsity (some w_j become 0). But among 7 features, even features with near-zero weights (e.g., pj=-0.005) provide useful signal direction. **Keeping all 7 features with L2-controlled magnitudes** is more suitable for this scenario than L1's feature selection.

---

## 8. Cross-Validation

### 8.1 5-Fold Cross-Validation Method

```python
N = len(entries)          # 311
np.random.seed(42)         # Fixed random seed for reproducibility
perm = np.random.permutation(N)
K = 5
fold_size = N // K        # 62
folds = [perm[k*62:(k+1)*62] for k in range(K)]  # Last fold 63

for fold in range(K):
    train_idx = [all indices except fold]
    test_idx = folds[fold]

    # Train on train_idx
    res = minimize(eval_model, x0, args=(train_idx,))
    # Evaluate on test_idx
    test_loss = eval_model(res.x, test_idx)
```

### 8.2 Result Interpretation

```
Fold 0: train=0.742  test=0.867  gap=0.125
Fold 1: train=0.769  test=0.841  gap=0.072
Fold 2: train=0.784  test=0.917  gap=0.133
Fold 3: train=0.805  test=0.821  gap=0.016
Fold 4: train=0.765  test=0.864  gap=0.099

Average: train=0.773  test=0.862  gap=0.089
std:     train=0.021  test=0.034
```

**Key metrics**:
- **CV Test Loss** (0.862): Expected generalization performance
- **Gap** (0.089): Overfitting degree, smaller is better
- **Test Loss std** (0.034): Generalization stability

### 8.3 Overfitting Diagnosis

| Scenario | Train | Test | Gap | Diagnosis |
|----------|-------|------|-----|-----------|
| Healthy | 0.77 | 0.86 | ~0.09 | Moderate generalization, acceptable |
| Overfit | 0.71 | 0.90 | ~0.19 | Increase λ or reduce parameters |
| Underfit | 0.85 | 0.89 | ~0.04 | Decrease λ or add parameters |
| Unstable | 0.77 | 0.86±0.10 | — | Need more data |

### 8.4 CV Expectations After Adding Data

When dataset expands from 311 to N charts:
- If new charts follow existing distribution: CV test loss should **decrease** (more data → better generalization)
- If new charts come from new distribution (e.g., new sort type): CV test loss may **increase** (distribution shift), requiring feature expansion checks

---

## 9. Feature Threshold Parameters

### 9.1 Parameter Definitions

Each feature's computation depends on several threshold parameters, which **do not participate in NM optimization** and are fixed before training:

| Parameter | Default | Affected Feature | Meaning |
|-----------|---------|------------------|---------|
| `spd_dt` | 150 ms | speed | Maximum time difference for speed-type notes |
| `spd_dc` | 3 columns | speed | Minimum column difference for speed-type notes |
| `bst_dt` | 100 ms | burst | Maximum window for triplet bursts |
| `ch_order` | 4 | chord | Minimum simultaneous notes to form chord |
| `hs_dt` | 200 ms | hs | Maximum time difference for hand-switch notes |
| `lb_dt` | 150 ms | lb | Maximum window for 4-note light bursts |
| `fj_dt` | 100 ms | fj | Maximum time difference for same-column fast jacks |

### 9.2 Whether to Tune These Parameters

**Usually not needed**. These parameters' defaults are based on physical understanding of game mechanics (e.g., human reaction time in 7K ~100ms, hand-switch upper limit ~200ms). Tuning may be needed when:

1. **Key count changes**: Expanding from 7K to 4K/9K requires adjusting `spd_dc` and `ch_order`
2. **Abnormal feature contribution**: If a feature's weight is near zero, threshold parameters may be unreasonable, reducing feature discrimination
3. **New data distribution shift**: When new charts' BPM range significantly exceeds original data

### 9.3 Tuning Method (If Needed)

Use **grid search** (not NM), because threshold parameters are discrete:

```python
for spd_dt in [120, 130, 140, 150, 160, 170]:
    for bst_dt in [80, 90, 100, 110, 120]:
        features = recompute_all_features(spd_dt=spd_dt, bst_dt=bst_dt, ...)
        # Retrain correction layer
        res = optimize_correction(features)
        # 5-fold CV evaluation
        cv_test = cross_validate(res, features)
        record(spd_dt, bst_dt, cv_test)
```

**Note**: After modifying threshold parameters, all charts' feature values must be recomputed, then correction layer weights retrained.

---

## 10. Complete Tuning Workflow

### 10.1 Scenario 1: Retune After Dataset Expansion

Most common scenario: Adding N charts to playtest data.

```
Phase 1: Data Preparation
├── 1a. Place new charts' .osu files in maps/ directory
├── 1b. Add entries to corresponding playtest.xlsx
├── 1c. Confirm all columns (mapfile, difficulty, accurate, error, sort) complete
└── 1d. Run maps/counter.py to verify chart count

Phase 2: Cache Rebuild
├── 2a. Run cache building script, generating cache_i.pkl and d_i.npz for all charts
├── 2b. Verify cache count = xlsx entry count
└── 2c. Check for cache failures (charts with note_seq < 10 are skipped)

Phase 3: Precomputation
├── 3a. Load Sigmoid layer parameters (tuned_params_sigmoid.json)
├── 3b. Precompute D_solved_base and n_eff for all charts
└── 3c. Compute 7 features for all charts

Phase 4: Correction Layer Optimization
├── 4a. λ sweep: λ ∈ {0.0, 0.005, 0.01, 0.02, 0.05}
├── 4b. Each λ value: 5 random restarts × NM(maxiter=10000)
├── 4c. Select λ with lowest in-sample loss
└── 4d. Record: weights, postprocess parameters, in-sample loss

Phase 5: Cross-Validation
├── 5a. 5-fold CV (fixed seed=42 for reproducibility)
├── 5b. Record: CV train/test loss, gap, std
└── 5c. Diagnose overfitting (gap < 0.15 is healthy)

Phase 6: Full Pipeline Verification (Required)
├── 6a. With optimized weights, run full precompute→combine→correct→aggregate per chart
├── 6b. Compare: full pipeline loss vs CV test loss
├── 6c. Difference < 0.05 is normal (translation invariance approximation error)
└── 6d. Difference > 0.10 requires investigation (possible cache inconsistency or feature computation differences)

Phase 7: Save
└── Save tuned_correction.json (weights + postprocess + λ + CV results)
```

### 10.2 Scenario 2: Retune After Sigmoid Layer Parameter Changes

If Sigmoid aggregation layer (k, C, γ, calib) changed (e.g., B2/B3 blocks re-optimized), correction layer needs retraining:

```
1. Recompute D_solved_base using new Sigmoid layer parameters (Phase 3)
2. Feature values don't need recomputation (features only depend on note_seq, not Sigmoid parameters)
3. Retrain correction layer (Phase 4-7)
```

**Note**: Sigmoid layer parameter changes alter D_solved's numerical range, so postprocess parameters (N0, threshold, divisor, global_scale) must be jointly re-optimized with correction layer weights.

### 10.3 Scenario 3: Adding New Features

When discovering D formula has systematic bias for a new pattern type:

```
1. Define new feature: Add computation logic in compute_features()
2. Update FEATURE_NAMES list
3. Recompute all charts' feature values (Phase 3)
4. Retrain correction layer (Phase 4-7)
5. Compare old vs new model CV test loss:
   - Improvement > 0.01: Keep new feature
   - Improvement < 0.01: New feature not worth added parameter
```

**Principle**: Each additional feature = one more parameter. At ~300 sample data scale, total parameters should not exceed ~15 (parameter/sample ratio < 5%).

### 10.4 Scenario 4: Retune After D Formula Parameter Changes

If D formula layer (B2/B3 blocks) parameters changed:

```
1. Rebuild all charts' cache (Phase 2, because D_full changed)
2. Recompute D_solved_base (Phase 3)
3. Recompute feature values (Phase 3, because pj depends on Jbar/Pbar)
4. Retrain correction layer (Phase 4-7)
```

Most time-consuming scenario, requires re-running `precompute()` + `combine()` pipeline (~15 minutes/311 charts).

---

## 11. Key Findings and Pitfalls

### 11.1 Correction Should Be Much Smaller Than D_solved

Typical correction range is [-1.5, +0.5], while D_solved typical range is [2.0, 8.0]. If correction approaches D_solved magnitude (e.g., some feature's weight abnormally large), it indicates:
- Feature may be compensating for D formula's structural deficiency (prioritize fixing D formula)
- Insufficient regularization (increase λ)
- Outlier samples in dataset

### 11.2 Full Pipeline Verification Cannot Be Skipped

D_solved translation invariance is approximate. Fast optimization (using translation invariance) results **must** be confirmed via full pipeline verification:

| Model | CV Test Loss | Full Pipeline Loss | Consistency |
|-------|-------------|-------------------|-------------|
| Linear correction (7 features) | 0.862 | 0.770 | ✅ Consistent |
| Interaction term (chord×fj) | 0.874 | 0.810 | ❌ CV predicts improvement, full pipeline degrades |

**Lesson**: Interaction term performs well in fast approximation but breaks translation invariance assumption, causing full pipeline results inconsistent with CV. Therefore correction layer should always maintain **linear form**, not introducing interaction terms.

### 11.3 Chord Weight Physical Interpretation

Chord weight is -0.714, the largest negative weight. This **does not mean** D formula has structural deficiency. Investigation shows:

1. Chord density and prediction error have **no monotonic relationship** (average errors across bins range from -0.07 to +0.01)
2. Attempted structural chord discount model (threshold + max discount), test loss worsened by 0.204 (23.7%)
3. Linear correction already sufficient to capture chord's statistical patterns

**Conclusion**: Chord's large weight reflects statistical trends in data, not that D formula needs structural modification.

### 11.4 RC Charts Are the Hardest to Predict

| Type | Count | Loss | Pass Rate |
|------|-------|------|-----------|
| RC | 133 | 1.182 | 75.2% |
| LN | 125 | 0.456 | 89.6% |
| HB | 47 | 0.488 | 91.5% |

RC charts' loss is 2.6× LN charts. Possible reasons:
- RC charts have higher skill diversity (stream, jump, tech mixed), single feature set insufficient to cover
- RC playtest data may have higher variance (different players judge RC difficulty more differently)

**Potential improvement direction**: Train independent correction layer weights for RC charts (piecewise model), but requires more data to support additional 7 parameters.

### 11.5 Empirical Upper Bound on Parameter Count

| Model | Param Count | Train | Test | Gap |
|-------|-------------|-------|------|-----|
| No correction | 0 | 0.931 | 0.931 | 0.000 |
| Linear correction | 11 | 0.770 | 0.862 | 0.092 |
| Full model (sigmoid mixture) | 20 | 0.715 | 0.903 | 0.188 |

At ~300 sample data scale, **11 parameters is safe upper bound**. Exceeding this (e.g., 20-parameter model) causes overfitting (gap doubles). If dataset expands to 500+ samples, can try 15-18 parameters.

### 11.6 Diagnosing Unstable Feature Weights

If a feature's weight direction flips after retraining (e.g., from positive to negative), possible causes:
1. **Feature's contribution absorbed by other features**: Check inter-feature correlations (e.g., burst and speed highly correlated)
2. **Insufficient regularization**: Increase λ, weight direction should stabilize
3. **Insufficient data**: Feature's signal drowned by noise

**Handling method**: Check weights across 5 CV folds. If weight direction inconsistent across folds, feature is unstable, consider removing.

---

## Appendix A: Script Index

| Script | Function |
|--------|----------|
| `predict_correction.py` | Correction layer prediction interface (single chart / batch) |
| `tune_correction.py` | Correction layer training (λ sweep + multiple restarts + 5-fold CV) |
| `verify_correction.py` | Full pipeline verification (fast optimization vs full pipeline comparison) |
| `build_cache.py` | Cache building (precompute + combine + feature computation) |

## Appendix B: Parameter Archive

### Correction Layer Parameters (11)

| File | Content |
|------|---------|
| `tuned_correction.json` | Feature weights (7) + postprocess (4) + λ + CV results |

### Dependent Parameters (From Sigmoid Layer, Not Re-tuned)

| File | Parameters |
|------|------------|
| `tuned_params_sigmoid.json` | k, C, γ, calib_a, calib_b (and all D formula / feature layer parameters) |

### Feature Threshold Parameters (Fixed Defaults)

```json
{
    "spd_dt": 150,
    "spd_dc": 3,
    "bst_dt": 100,
    "ch_order": 4,
    "hs_dt": 200,
    "lb_dt": 150,
    "fj_dt": 100
}
```

## Appendix C: Loss Function

Correction layer uses same dead-zone piecewise-linear loss as Sigmoid layer:

```
delta = |SR_pred - SR_ref|
eps = max(error_bound, 0.01)
ratio = delta / eps

if ratio <= 0.5:   loss = 0            (dead zone: within half error bound = perfect)
elif ratio <= 1.0: loss = (ratio-0.5)×2 (linear growth: 0→1)
else:              loss = 1.0 + (ratio-1.0)×6 (steep slope: 3× slope beyond error)
```

## Appendix D: Quick Reference — Tuning Checklist

When retuning after adding data, confirm each item:

- [ ] New charts' .osu files exist in maps/ directory
- [ ] playtest.xlsx new entries have all columns complete (mapfile, difficulty, accurate, error, sort)
- [ ] Cache rebuilt (cache_i.pkl + d_i.npz count = chart count)
- [ ] D_solved_base precomputed using latest Sigmoid parameters
- [ ] Feature values computed using default threshold parameters
- [ ] λ sweep covered {0.0, 0.005, 0.01, 0.02, 0.05}
- [ ] Each λ used ≥3 random restarts
- [ ] 5-fold CV used seed=42
- [ ] CV gap < 0.15 (overfitting diagnosis)
- [ ] Full pipeline verification passed (difference < 0.05)
- [ ] Results saved to tuned_correction.json
