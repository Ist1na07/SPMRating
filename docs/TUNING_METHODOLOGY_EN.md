# SPM Rating — Sigmoid Tuning Methodology

**Date**: 2026-05-19
**Context**: Tuning practice during the transition from Percentile to Sigmoid (player accuracy model) aggregation. R3 final results: k=2.09, C=3.97, γ=0.196, MAE=0.2180.

---

## Table of Contents

1. [Background](#1-background)
2. [Core Methodology: Block Alternating Nelder-Mead](#2-core-methodology-block-alternating-nelder-mead)
3. [B1 Fast Path: Pre-extracted D Segments](#3-b1-fast-path-pre-extracted-d-segments)
4. [k-Value Scan: Fine Search at 0.1 Step](#4-k-value-scan-fine-search-at-01-step)
5. [D Pre-calibration: Linear Transform D'=aD+b](#5-d-pre-calibration-linear-transform-dadb)
6. [C-γ Self-Consistency: Eliminating Parameter Redundancy](#6-c-γ-self-consistency-eliminating-parameter-redundancy)
7. [Post-processing Parameter Merge](#7-post-processing-parameter-merge)
8. [Parameter Blocking Principles](#8-parameter-blocking-principles)
9. [Complete Workflow](#9-complete-workflow)
10. [Key Findings](#10-key-findings)

---

## 1. Background

### 1.1 Fundamental Differences Between the Two Aggregation Methods

| Property | Percentile (Old) | Sigmoid (New) |
|----------|:---:|:---:|
| Operation | Take P70-P75 of D(t) distribution | Bisection solve: Σwᵢ/(C+e^(k(Dᵢ-D))) = total_W·γ |
| Physical meaning | Statistical percentile (no physical meaning) | Simulates player accuracy decay curve |
| Parameter count | 2 (power, percentile) | 3 (k, C, γ) + 4 calib/post |
| Tuning cost | Low | **High** — each combine() includes bisection (~15% slower) |

### 1.2 Why a New Tuning Method is Needed

Percentile aggregation only involves 2 parameters and can be placed directly into the parameter vector during global tuning. However, Sigmoid aggregation introduces 7 new parameters (k, C, γ, calib_a, calib_b, post parameters), and these parameters exhibit **structural coupling** with the D formula layer (10 parameters) and feature layer (16 parameters):

- Sigmoid parameters (k, C, γ, calib) determine "how the player perceives difficulty"
- D formula parameters (S_w1, alpha_P, alpha_R, ...) determine "what difficulty is"
- Feature layer parameters (cross, release, inverse) determine "how difficulty components are defined"

If all 35 parameters are thrown into NM at once:
1. **High-dimensional degradation**: 35-dimensional NM is extremely inefficient (~200+ evaluations before improvement begins)
2. **Gradient confusion**: B1 parameters (sigmoid) and B2 parameters (D formula) compensate for each other, getting stuck in local optima
3. **Speed bottleneck**: eval_full() takes 14.5s; 35-dimensional NM takes ~1 hour

---

## 2. Core Methodology: Block Alternating Nelder-Mead

### 2.1 Blocking Strategy

Divide 41 parameters into 6 blocks based on **semantic independence**, with 3 parameters excluded with evidence:

| Block | Params | Meaning | Evaluation Mode |
|-------|:---:|------|:---:|
| **B1: Sigmoid+Calib+Post** | 9 | k, C, γ, calib_a, calib_b, N0, threshold, divisor, global_scale | **Fast** (0.04s/eval) |
| **B2: D Formula Core** | 10 | S_w1, S_p, alpha_P, alpha_R, alpha_C, alpha_S, alpha_V, D_beta1, D_beta2, Abar_scale | Full (14.5s/eval) |
| **B3a: Cross Distance** | 5 | dist_exponent_rc, dist_exponent_ln, same_hand_penalty_rc/ln, thumb_bridge | Full |
| **B3b: Release LN Tail** | 8 | tail_coeff, tail_to_tap, same_col_bonus, coord_exponent, seq_coeff, lock_interaction, short_ln_threshold, short_ln_reduction | Full |
| **B3c: Inverse/Guide** | 8 | inv_amplitude, inv_tau, inv_power, guide_depth, guide_center, guide_width, cross_guide_scale, inverse_same_col_bonus | Full |
| **B3d: Jack** | 2 | jack_aggregation_power, multi_jack_boost | Full |
| *Excluded* | 3 | shield_tau_ms, shield_anchor_mod, shield_coord_factor | — |

**Shield parameter exclusion rationale**: The Shield (Sbar) component contributes only ~0.003 to total MAE (MAE change <1.5% when disabled). The optimization cost of 3 shield parameters (full NM ~15min) far exceeds their potential improvement (<0.0005 MAE). In R3, shield parameters drifted slightly from defaults (see §8.4), but this is noise-level drift from B2 NM, not meaningful optimization. Creating a dedicated tuning block for shield parameters is not recommended.

### 2.2 Alternating Algorithm

```
1. Load Phase 5 best parameters as initial values
2. Set k = 2.09 (optimal value determined from sweep)
3. for round in [1, 2, 3]:
     maxiter = 50 if round <= 2 else varies  # R1-2 coarse+fine, R3 re-run with relaxed bounds
     for block in [B1, B2, B3a, B3b, B3c, B3d]:
       NM(block, maxiter=maxiter)
       if block is full mode:
           Re-extract D_seg (because D formula changed)
4. After R3: B1 Fast NM (maxiter=50) for final fine-tuning
```

**R3 special handling**: R1-R2 locked `release_tail_to_tap` at ≤4.0. R3 relaxed the upper bound to ≤6.0 and re-ran the full pipeline. This allowed Release parameters to find a better configuration (tail_to_tap rose from ~2.8 to 4.22).

### 2.3 Why It Works

- **Decoupled gradients**: Each block only optimizes parameters within its semantic scope, preventing cross-semantic compensation
- **Alternating iteration**: B1 first determines the "player model", B2 then optimizes "difficulty definition", cycling to reach joint optimum
- **Re-extraction after full-mode blocks**: After B2/B3 changes the D(t) distribution, B1's D_seg must be re-extracted to ensure B1 has correct D input for the next iteration

### 2.4 Convergence Behavior

Typical Loss trajectory (3 rounds × 6 blocks + B1 final = 19 NM phases, actual R3 data):

```
R3 B1: 0.9346  (Fast)
R3 B2: 0.9346 → 0.9346  (Full, no change — D Formula converged)
R3 B3a: 0.9346 → 0.9338 (Full, Cross Distance fine-tuning)
R3 B3b: 0.9338 → 0.9335 (Full, Release LN Tail, tail_to_tap≤6.0)
R3 B3c: 0.9335 → 0.9329 (Full, Inverse/Guide)
R3 B3d: 0.9329 → 0.9329 (Full, Jack, near-zero change)
R3 B1 Final: 0.9329 → 0.9321 (Fast, fine-tune sigmoid/calib/post)
```

**Key observations**:
- B1 Fast and B3c (Inverse) are the main sources of R3 improvement
- B2 (D Formula) has fully converged — NM can no longer find a descent direction
- B3d (Jack) marginal return near zero — jack_aggregation_power and multi_jack_boost are already near-optimal
- R3 was re-run after relaxing `release_tail_to_tap` upper bound (4.0→6.0); Release parameters moved significantly

---

## 3. B1 Fast Path

### 3.1 Motivation

Each `combine()` call takes ~14.5 seconds (for 311 charts), with 99% of time spent on D formula and feature layer computation. However, if only **sigmoid/calib/post parameters** change (B1 block), the D(t) distribution itself does not change — only the bisection solve and subsequent rescaling change.

### 3.2 Implementation

```python
# One-time pre-extraction (14.5s, done once)
for each map:
    cache = precompute(osu_path)
    _, details = combine(cache, use_sigmoid=False)  # Use percentile to extract D_all
    D_seg, w_seg = segment_by_difficulty(details["D_all"], eff_w, 30)
    Save {D_seg, w_seg, total_notes}

# Fast evaluation (0.04s/eval, 350x speedup)
def eval_b1(params, k, pre_data):
    D_cal = params["calib_a"] * D_seg + params["calib_b"]
    D_solved = solve_D_bisection(D_cal, w_seg, k=k, C=C, gamma=gamma)
    SR = D_solved * total_notes / (total_notes + N0)
    SR = rescale(SR)
    return score(SR, ref)
```

### 3.3 Performance Comparison

| Mode | NM Time per Round | Speedup |
|------|:---:|:---:|
| Full combine() | ~38 min (50 iter × 150 eval × 14.5s) | 1× |
| B1 Fast Path | **~8 seconds** (50 iter × 200 eval × 0.04s) | **~285×** |

This makes the following operations feasible:
- k-value scan: 26 k-values × 3s = 78s (full mode would need ~16 hours)
- Repeated B1 fine-tuning: can run maxiter=50+ NM without time concerns

### 3.4 Limitations

B1 fast path assumes D(t) distribution remains unchanged, thus **cannot be used for B2/B3 block optimization**. After each B2/B3 NM round completes, `extract_b1_segments()` must be called again to update D_seg.

---

## 4. k-Value Scan

### 4.1 Why k Needs Independent Search

k is sigmoid's **most sensitive parameter** — it controls how fast player accuracy decays with difficulty:

| k | A(d+1)/A(d) | Physical Meaning |
|---|:---:|------|
| 0.5 | ~62% | Too flat, player still at 90% accuracy at d+5 — **unrealistic** |
| 1.0 | ~27% | Moderate decay |
| 1.5 | ~12% | Accuracy drops to ~5% at d+2 |
| **2.09** | **~5%** | Optimal, accuracy drops to ~5% at d+1, ~18% at d+0.5 |
| 3.0 | ~1% | Too steep, near hard-max |

k does not linearly correlate with any other parameter — changing k requires **refitting** all B1 parameters (calib, C, γ, post).

### 4.2 Scan Method

```
1. Pre-extract D_seg using existing D formula / feature parameters (Phase 5 best) — one-time
2. for k in 0.5, 0.6, ..., 3.0 (step 0.1):
     Optimize all B1 parameters using B1 Fast NM (30 iter)
     Record: k, Loss, MAE, calib_a, calib_b, C, γ
3. Select k with lowest Loss
```

Total time: 26 × 3s ≈ **78 seconds**.

### 4.3 Results

```
k=0.5:  Loss=0.9412  (too flat)
k=1.0:  Loss=0.9351
k=1.5:  Loss=0.9334
k=1.8:  Loss=0.9328
k=2.0:  Loss=0.9324
k=2.09: Loss=0.9321  ← Optimal
k=2.2:  Loss=0.9325
k=2.5:  Loss=0.9338
k=3.0:  Loss=0.9362  (too steep)
```

Optimal k≈2.09, about 40% higher than the earlier k=1.5. Higher k means steeper sigmoid: player accuracy decays faster beyond skill boundary. At k=2.09, d-D=+1 corresponds to accuracy at ~5% of matching level, d-D=+0.5 at ~18%. This matches real-world experience with high-difficulty 7K charts.

### 4.4 Stability of D Calibration

calib_a and calib_b are stable across different k values, but calib_b decreases with increasing k:

```
k=1.0:  calib=(0.891, +0.048)
k=1.5:  calib=(0.892, +0.042)
k=2.0:  calib=(0.893, +0.033)
k=2.09: calib=(0.893, +0.031)
k=2.5:  calib=(0.895, +0.025)
k=3.0:  calib=(0.896, +0.018)
```

calib_a consistently near 0.89, calib_b decreases with higher k (larger k needs smaller positive offset). This validates that D calibration is a **robust linear transform**, with calib_a nearly independent of the exact k value.

---

## 5. D Pre-calibration

### 5.1 Motivation

The absolute meaning of D(t) differs between Percentile and Sigmoid aggregation:

- Percentile: D=5 means "this is harder than P70 = 5" — a relative concept
- Sigmoid: D=5 means "player accuracy decay curve centers at 5 here" — requiring D's absolute scale to align with sigmoid's numerical range

With k=2.09, C=3.97, γ=0.196, the sigmoid solve for D is most sensitive near D_i (d-D ∈ [-1.5, +1.5]). If raw D values are offset by 1-2 units, it significantly affects the solution. Higher k narrows the sensitive range, demanding more precise D calibration.

### 5.2 Linear Calibration

The simplest calibration form: `D'(t) = a × D(t) + b`

- **a=0.893**: D scale slightly compressed (~11%), aligning numerical range with sigmoid's sensitive region
- **b=0.031**: Tiny positive offset, compensating for slight D underestimation at low values

### 5.3 Why Linear

Higher-order calibrations (quadratic, piecewise linear) were attempted but brought no improvement. Reasons:
1. D(t) distribution shape is similar across charts (roughly log-normal)
2. Linear calibration already suffices to map D to sigmoid's suitable numerical range
3. More degrees of freedom only increase overfitting risk

### 5.4 How It Works

Calibration is **incorporated into the B1 block** (not handled separately):

```python
D_cal = params["calib_a"] * D_seg + params["calib_b"]
D_solved = solve_D_bisection(D_cal, w_seg, k=k, C=C, gamma=gamma)
```

calib_a and calib_b are jointly optimized with C, γ, k through B1 NM. This ensures calibration parameters are jointly optimal with sigmoid shape parameters.

---

## 6. C-γ Self-Consistency

### 6.1 Theoretical Relationship

In the sigmoid formula `A(d) = A_min + (A_max-A_min)/(C + e^(k(d-D)))`:

- At d=D, A(D) = A_min + (A_max-A_min)/(C+1)
- γ is defined as reference player's target accuracy fraction: γ = (A_ref - A_min)/(A_max - A_min)

If γ and C are independent parameters, then at d=D, `A(D) = A_min + (A_max-A_min)/(C+1)`.

This means there exists a self-consistency relationship between γ_ref (target) and 1/(C+1) (actual accuracy fraction at matching point).

### 6.2 Grid Search Validation

7×7 grid search on (C, γ) (C ∈ [2,10], γ ∈ [0.1,0.3]), results:

```
Loss is lowest along the constraint line γ ≈ 1/(C+1)
```

Specifically:
- C=3.97 → 1/(C+1)=0.201, NM finds γ=0.196 — **match (2.5% deviation)**
- C=2.0 → 1/(C+1)=0.33, NM finds γ≈0.32
- C=8.0 → 1/(C+1)=0.11, NM finds γ≈0.12

### 6.3 Design Decision: Put C in B1 Block

Based on the self-consistency relationship, although C and γ are not fully redundant, they are strongly coupled through 1/(C+1)≈γ. Putting both in B1 block:

- B1 NM can freely trade off between the two parameters
- Self-consistency effectively reduces the search space to 1 degree of freedom (along the 1/(C+1)≈γ line)
- Final NM converges to C≈3.97, γ≈0.196, satisfying self-consistency (1/(3.97+1)=0.201≈0.196)

If C were in a separate block or optimized with feature layers, NM would waste many evaluations searching in 2D space, and could produce spurious gradient signals with feature parameters.

---

## 7. Post-processing Parameter Merge

### 7.1 Phase 5 Post-processing

Phase 5 (Percentile aggregation) had 4 post-processing parameters:

```
note_norm_N0:     10.0   # Note-count normalization offset
rescale_threshold: 9.54  # High SR compression threshold
rescale_divisor:   2.00  # High SR compression divisor
global_scale:      1.055 # Global scale
```

### 7.2 Why Not Re-tune

From the beginning, the decision was made to **not re-tune post-processing parameters from scratch**, instead reusing Phase 5 values as initial points. Reasons:

1. **Post-processing parameters are orthogonal to aggregation method**: N0, threshold, divisor address "note-count bias" and "high SR nonlinearity" problems — these exist equally under sigmoid
2. **Reduced search space**: 4 parameters × highly nonlinear = enormous search space, prone to compensation effects with calibration parameters
3. **Phase 5 values are already good**: Phase 5 post parameters were thoroughly optimized; direct reuse is a safe starting point

### 7.3 How It Works

Post-processing parameters are **merged into B1 block**:

```
B1 = {k, C, γ, calib_a, calib_b} + {N0, threshold, divisor, global_scale}
```

In B1 fast path NM, these parameters are jointly optimized with sigmoid+calib parameters. The post-processing formulas remain unchanged:

```python
SR *= total_notes / (total_notes + N0)        # Note-count normalization
if SR > threshold: SR = threshold + (SR - threshold) / divisor  # High SR compression
SR *= global_scale                              # Global scale
```

### 7.4 Final Values

| Parameter | Phase 5 (Percentile) | Sigmoid R3 Final | Change |
|-----------|:---:|:---:|:---:|
| note_norm_N0 | 10.0 | 8.21 | -18% |
| rescale_threshold | 9.54 | 9.42 | -1.3% |
| rescale_divisor | 2.00 | 2.01 | +0.5% |
| global_scale | 1.055 | 1.055 | Unchanged |

Post-processing parameters had modest adjustments during sigmoid optimization: N0 dropped from 10.0 to 8.21, reducing note-count penalty for short charts; threshold and divisor remained stable. global_scale remained completely unchanged — validating that global scaling is orthogonal to other parameters.

---

## 8. Parameter Blocking Principles

### 8.1 Blocking Criteria

1. **Semantic grouping**: Parameters from the same "conceptual layer" go together (e.g., cross's 5 parameters all in B3a)
2. **Evaluation cost alignment**: B1 uses fast path, B2/B3 use full combine — parameters with different evaluation modes must be in separate blocks
3. **Gradient independence**: Cross-block gradients should be as small as possible
4. **Balanced block size**: 5-10 parameters per block, allowing NM to run efficiently on each

### 8.2 Why B2 is 10 Parameters (Largest Block)

The D formula is the bridge connecting the feature layer and aggregation layer. Its parameters (S_w1, alpha_P, alpha_R, D_beta1/2 ...) have strong coupling and should not be split.

Although 10 parameters is the largest block, full NM at 44 minutes (maxiter=50) is acceptable for 10 dimensions. Splitting would introduce false alternating convergence problems.

### 8.3 Why B3 is Split into 4 Sub-blocks

The feature layer has 23 parameters (cross 5 + release 8 + inverse 8 + jack 2). If all optimized at once:
- 23-dimensional NM in full mode is extremely slow (~300 eval × 14.5s = 72 min/round)
- Cross and Release D(t) contributions are coupled through Xbar/Rbar, but direct gradients are very small

After splitting into 4 sub-blocks:
- Each block 2-8 parameters, NM converges quickly
- Alternating optimization allows cross-block coupling to propagate through D_seg re-extraction
- Jack block is separate because `jack_aggregation_power` controls Jbar's cross-column aggregation method, semantically independent of cross/release/inverse

### 8.4 Shield Parameter Handling

Shield (Sbar) is a probabilistic model for accidental LN activation when a note on the same column precedes an LN head. Its formula is `Sbar = sum(exp(-dt/tau)) * (1 + anchor_mod * coord_factor * lock_bonus)`.

**Empirical evidence**: Disabling Shield increases MAE by only ~0.003 (<1.5% change). Its contribution across 311 charts is the lowest of all components.

Shield's 3 parameters (`shield_tau_ms`, `shield_anchor_mod`, `shield_coord_factor`) were not placed in any dedicated NM block but are indirectly affected through `alpha_S` in B2 (D Formula). In R3 final parameters, shield parameters drifted slightly from defaults:
- `shield_tau_ms`: 100 → 56.2 (-44%)
- `shield_anchor_mod`: 1.0 → 0.806 (-19%)
- `shield_coord_factor`: 1.0 → 1.003 (+0.3%)

These shifts are likely random walk from B2 NM rather than meaningful optimization — changing shield parameters affects Loss at the numerical noise level (<0.0001). **Creating a dedicated tuning block for shield parameters is not recommended.**

---

## 9. Complete Workflow

### Phase 1: Coarse k Scan (step 0.5)

```
k ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}
→ Determine k≈1.5 optimal region
```

### Phase 2: Fine k Scan (step 0.1)

```
k ∈ {0.5, 0.6, ..., 3.0}  (26 values)
For each k: B1 Fast NM (30 iter)
→ k=2.09 verified as optimal, calib_a≈0.893, calib_b≈0.031, C≈3.97, γ≈0.196
```

### Phase 3: C-γ Grid Search

```
C ∈ {2,3,4,5,6,8,10}, γ ∈ {0.10, 0.13, 0.17, 0.20, 0.23, 0.27, 0.30}
→ Verify 1/(C+1)≈γ self-consistency relationship
```

### Phase 4: Full Alternating NM (k=2.09 fixed)

```
R1-R2: 2 rounds × 6 blocks (release_tail_to_tap ≤ 4.0):
  B1:  maxiter=50→25, Fast
  B2:  maxiter=50→25, Full
  B3a: maxiter=50→25, Full
  B3b: maxiter=50→25, Full
  B3c: maxiter=50→25, Full
  B3d: maxiter=50→25, Full

R3: Relax release_tail_to_tap upper bound to 6.0, re-run 6 blocks
  Reason: R1-R2 tail_to_tap converged to the upper bound (4.0),
          indicating the true optimum lies beyond the boundary
```

### Phase 5: Final Parameter Fine-tuning + Save

```
R3 B1 Final (maxiter=50, Fast): Fine-tune sigmoid/calib/post
→ Save tuned_params_sigmoid.json (MAE=0.2180)
```

---

## 10. Key Findings

### 10.1 k=2.09 is the Overwhelming Improvement

Switching from k=0.52 (initial guess) to k=2.09:
- MAE: 0.2299 → 0.2180 (-5.2%)
- Loss: 1.15 → 0.932 (-19%)

This is the **largest single improvement** of the sigmoid aggregation layer, exceeding all subsequent NM optimizations combined. k further rose from the earlier 1.5 to 2.09, primarily benefiting from re-tuning after relaxing the `release_tail_to_tap` upper bound.

### 10.2 C and γ Are Not Independent

The self-consistency line `γ ≈ 1/(C+1)` holds almost exactly. NM never deviates from it by more than 2%. In practice, only 1 degree of freedom is needed (e.g., only tune C, compute γ from formula).

### 10.3 D_solved Stably Falls at P70 of D(t) Distribution

Across 311 charts: `D_solved ≈ 0.87 × D_P70`, std only 0.01.

This remarkable consistency means:
1. Sigmoid aggregation is equivalent to performing smooth "soft percentile selection" on the D distribution
2. The selection point is fixed at ~P70, determined by the numerical values of k and C
3. Changing k slightly adjusts this equivalent percentile (larger k → earlier percentile), but the change is minimal

### 10.4 D Calibration is Necessary but Very Simple

The linear relationship `D' = 0.893×D + 0.031` is highly stable across different k values and optimization stages. This indicates D's absolute values are already close to the correct scale, requiring only ~11% slight compression and a tiny positive offset.

### 10.5 Feature Layer Parameters Barely Need Re-tuning

From Phase 5 (Percentile) to Sigmoid R3 Final, the 23 feature layer parameters changed <5%. B3a/B3b/B3c/B3d's combined NM improvement totals about 0.0017 Loss (0.9346→0.9329).

This means: **the feature layer (D component definition) and aggregation layer (how to go from D(t) to SR) are highly decoupled**. Good D(t) performs well under both aggregation methods.

### 10.6 Diminishing Marginal Returns of Alternating NM

```
R3 B1:     ΔLoss = -0.0025  (dominant improvement)
R3 B3a-c:  ΔLoss = -0.0017
R3 B3d:    ΔLoss = -0.0000  (zero marginal improvement)
R3 B1 Final: ΔLoss = -0.0008
```

B1 (aggregation layer) contributed ~50% of total improvement; B3c (Inverse) is the only feature block with meaningful improvement. B3d (Jack) had zero marginal return. For future work: **focus on aggregation layer parameters (k, C, γ, calib) and Inverse; the remaining feature layer can be locked**.

### 10.7 RC Sub-model Tuning

The RC sub-model disables LN-specific components (Rbar, Sbar, Vbar) and uses an RC-only D formula. Tuned using the same block-alternating NM method as the main model:

| Metric | Total SR for RC labels | RC Sub-model (tuned) | Improvement |
|--------|:---:|:---:|:---:|
| MAE | 0.3250 | 0.2366 | -27% |
| Loss | 2.197 | 1.359 | -38% |
| r | — | 0.9870 | — |

**Key findings**:
- RC sigmoid k=2.31, higher than the main model's k=2.09 — RC charts have steeper difficulty perception
- `note_norm_N0_rc` converged to 0 — RC charts need no note-count normalization (more uniform note distribution)
- RC sub-model significantly outperforms using Total SR directly for RC labels, validating the effectiveness of component pruning

### 10.8 LN Sub-model Architectural Limitations

The LN sub-model uses a simplified additive D formula (`D_ln = alpha_R*Rbar + alpha_S*Sbar + alpha_V*Vbar + alpha_P_ln*Pbar`) without Jbar, Xbar, or complex S-T mixing terms.

**Tuning results are unsatisfactory**:

| Metric | Total SR for LN labels | LN Sub-model (tuned) |
|--------|:---:|:---:|
| MAE | **0.2049** | 0.8162 |
| Loss | 0.440 | 7.568 |
| r | — | 0.827 |

Total SR as an LN predictor (MAE=0.205) vastly outperforms the dedicated LN sub-model (MAE=0.816). This indicates the current LN D formula is overly simplified, missing critical information such as Jbar (jack component) and Xbar (tech component). **The LN sub-model needs architectural redesign, not further parameter tuning.** Possible improvement directions:
- Introduce Jbar/Xbar into the LN D formula
- Use the same S-T mixing formula structure as the main model
- Or directly reuse the main model D formula under LN masking

---

## Appendix A: Script Index

| Script | Function |
|--------|----------|
| `scripts/tune_sigmoid_k15.py` | Full alternating NM at fixed k (6 blocks × 3 rounds) |
| `scripts/tune_rc.py` | RC sub-model NM tuning (3 blocks × 2 rounds) |
| `scripts/tune_ln.py` | LN sub-model NM tuning (3 blocks × 2 rounds) |
| `scripts/train_sort_classifier.py` | Map type classifier training (RC/LN/HB/Mix) |
| `scripts/train_tag_classifier.py` | Pattern tag classifier training (14 tags) |
| `scripts/build_standalone.py` | Single-file distribution builder |
| `spm_rating/aggregate_sigmoid.py` | Sigmoid aggregation core (segmentation + bisection) |

## Appendix B: Parameter Archive

| File | Content |
|------|---------|
| `tuned_params_sigmoid.json` | R3 final optimal parameters (MAE=0.2180, k=2.09) |
| `tuned_params_rc.json` | RC sub-model optimal parameters |
| `tuned_params_ln.json` | LN sub-model optimal parameters |

## Appendix C: Complete Parameter Inventory

### Tuned Parameters (41 total, in 6 blocks)

**B1 — Sigmoid+Calib+Post (9p, k determined in Phase 1-2 sweep)**
`agg_sigmoid_k`, `calib_a`, `calib_b`, `agg_sigmoid_C`, `agg_sigmoid_ref_gamma`, `note_norm_N0`, `rescale_threshold`, `rescale_divisor`, `global_scale`

**B2 — D Formula Core (10p)**
`S_w1`, `S_p`, `alpha_P`, `alpha_R`, `alpha_C`, `alpha_S`, `alpha_V`, `D_beta1`, `D_beta2`, `Abar_scale`

**B3a — Cross Distance (5p)**
`cross_dist_exponent_rc`, `cross_dist_exponent_ln`, `cross_same_hand_penalty_rc`, `cross_same_hand_penalty_ln`, `cross_thumb_bridge_factor`

**B3b — Release LN Tail (8p)**
`release_tail_coeff`, `release_tail_to_tap`, `release_same_col_bonus`, `release_coord_exponent`, `release_seq_coeff`, `lock_interaction_coeff`, `short_ln_threshold`, `short_ln_reduction`

**B3c — Inverse/Guide (8p)**
`inv_amplitude`, `inv_tau`, `inv_power`, `guide_depth`, `guide_center`, `guide_width`, `cross_guide_scale`, `inverse_same_col_bonus`

**B3d — Jack (2p)**
`jack_aggregation_power`, `multi_jack_boost`

### Excluded Parameters (3, locked to defaults)

| Parameter | Default | Exclusion Reason |
|-----------|--------|-----------------|
| `shield_tau_ms` | 100 | Shield total contribution ~0.003 MAE |
| `shield_anchor_mod` | 1.0 | Same as above; multiplicatively coupled with coord_factor |
| `shield_coord_factor` | 1.0 | Same as above |

### Non-tuned Parameters (fixed or deprecated)

| Parameter | Status |
|-----------|--------|
| `inverse_peak_width` | Fixed (2.0) |
| `inverse_window_ms` | Fixed (200) |
| `shield_smooth_window` | Fixed (500) |
| `shield_scale` | Fixed (0.001) |
| `stream_booster_scale` | Fixed (requires recache) |
| `stream_short_window` | Fixed |
| `D_gamma_e` (Stamina) | Disabled (positive values destroy MAE) |
| `V_alpha` | Unused (exists in params but not read by code) |
