# SPM Rating — Sigmoid Tuning Methodology

**Date**: 2026-05-10
**Context**: Tuning practice during the transition from Percentile to Sigmoid (player accuracy model) aggregation.

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

Divide 35 parameters into 5 blocks based on **semantic independence**:

| Block | Parameters | Meaning | Evaluation Mode |
|-------|:---:|------|:---:|
| **B1: Sigmoid+Calib+Post** | 8 | k, C, γ, calib_a, calib_b, N0, threshold, divisor, global_scale | **Fast** (0.04s/eval) |
| **B2: D Formula Core** | 10 | S_w1, S_p, alpha_P, alpha_R, alpha_C, alpha_S, alpha_V, D_beta1, D_beta2, Abar_scale | Full (14.5s/eval) |
| **B3a: Cross Distance** | 5 | dist_exponent_rc, dist_exponent_ln, same_hand_penalty_rc/ln, thumb_bridge | Full |
| **B3b: Release LN Tail** | 6 | tail_coeff, tail_to_tap, same_col_bonus, coord_exponent, seq_coeff, lock_interaction | Full |
| **B3c: Inverse/Guide** | 7 | inv_amplitude, inv_tau, inv_power, guide_depth, guide_center, guide_width, cross_guide_scale | Full |

### 2.2 Alternating Algorithm

```
1. Load Phase 5 best parameters as initial values
2. Set k = 1.5 (optimal value determined from sweep)
3. for round in [1, 2]:
     maxiter = 50 if round==1 else 25  # Coarse first round, fine second round
     for block in [B1, B2, B3a, B3b, B3c]:
       NM(block, maxiter=maxiter)
       if block is full mode:
           Re-extract D_seg (because D formula changed)
```

### 2.3 Why It Works

- **Decoupled gradients**: Each block only optimizes parameters within its semantic scope, preventing cross-semantic compensation
- **Alternating iteration**: B1 first determines the "player model", B2 then optimizes "difficulty definition", cycling twice to reach joint optimum
- **Re-extraction after full-mode blocks**: After B2/B3 changes the D(t) distribution, B1's D_seg must be re-extracted to ensure B1 has correct D input for the next iteration

### 2.4 Convergence Behavior

Typical Loss trajectory (2 rounds × 5 blocks = 10 NM phases):

```
R1 B1: 1.0281 → 0.9132  (Fast, 200 evaluations, 8s)
R1 B2: 0.9132 → 0.9098  (Full, 150 evaluations, 36 min)
R1 B3a: 0.9098 → 0.9087 (Full,  80 evaluations, 19 min)
R1 B3b: 0.9087 → 0.9079 (Full,  90 evaluations, 22 min)
R1 B3c: 0.9079 → 0.9075 (Full, 100 evaluations, 24 min)
---
R2 B1: 0.9075 → 0.9068  (Fast, 100 evaluations, 4s)
R2 B2: 0.9068 → 0.9063  (Full,  80 evaluations, 19 min)
R2 B3a: 0.9063 → 0.9063  (no change)
R2 B3b: 0.9063 → 0.9063  (no change)
R2 B3c: 0.9063 → 0.9063  (no change)
```

**Key observations**:
- Round 1 contributes 99% of total improvement
- Round 1 B1 Fast is the largest source of improvement (pulling down from pct baseline directly)
- Round 2 feature-layer blocks show almost no marginal improvement — indicating feature parameters are already near-optimal

---

## 3. B1 Fast Path

### 3.1 Motivation

Each `combine()` call takes ~14.5 seconds (for 205 charts), with 99% of time spent on D formula and feature layer computation. However, if only **sigmoid/calib/post parameters** change (B1 block), the D(t) distribution itself does not change — only the bisection solve and subsequent rescaling change.

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
| **1.5** | **~12%** | Near optimal, accuracy drops to ~5% at d+2 |
| 2.0 | ~5% | Too steep, near hard-max |

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
k=0.5:  Loss=0.9283  (too flat)
k=1.0:  Loss=0.9124
k=1.3:  Loss=0.9083
k=1.4:  Loss=0.9074
k=1.5:  Loss=0.9071  ← Optimal
k=1.6:  Loss=0.9073
k=1.7:  Loss=0.9079
k=2.0:  Loss=0.9118
k=3.0:  Loss=0.9204  (too steep)
```

Optimal k≈1.5, corresponding to accuracy dropping to ~12% of matching level at d-D=+1, and ~2% at d-D=+2. This matches real-world experience: 7K players cannot meaningfully play charts 2 SR above their ceiling.

### 4.4 Stability of D Calibration

calib_a and calib_b are highly stable across different k values:

```
k=0.5:  calib=(0.893, +0.055)
k=1.0:  calib=(0.888, +0.051)
k=1.5:  calib=(0.890, +0.044)
k=2.0:  calib=(0.891, +0.040)
k=3.0:  calib=(0.895, +0.033)
```

calib_a consistently near 0.89, calib_b in 0.04-0.06 range. This validates that D calibration is a **robust linear transform** independent of the exact k value.

---

## 5. D Pre-calibration

### 5.1 Motivation

The absolute meaning of D(t) differs between Percentile and Sigmoid aggregation:

- Percentile: D=5 means "this is harder than P70 = 5" — a relative concept
- Sigmoid: D=5 means "player accuracy decay curve centers at 5 here" — requiring D's absolute scale to align with sigmoid's numerical range

When k=1.5, C=4, γ=0.2, the sigmoid solve for D is most sensitive near D_i (d-D ∈ [-2, +2]). If raw D values are offset by 1-2 units, it significantly affects the solution.

### 5.2 Linear Calibration

The simplest calibration form: `D'(t) = a × D(t) + b`

- **a=0.89**: D scale slightly compressed (~11%), aligning numerical range with sigmoid's sensitive region
- **b=0.04**: Tiny positive offset, compensating for slight D underestimation at low values

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
- C=4.0 → 1/(C+1)=0.20, NM finds γ=0.201 — **exact match**
- C=2.0 → 1/(C+1)=0.33, NM finds γ≈0.32
- C=8.0 → 1/(C+1)=0.11, NM finds γ≈0.12

### 6.3 Design Decision: Put C in B1 Block

Based on the self-consistency relationship, although C and γ are not fully redundant, they are strongly coupled through 1/(C+1)≈γ. Putting both in B1 block:

- B1 NM can freely trade off between the two parameters
- Self-consistency effectively reduces the search space to 1 degree of freedom (along the 1/(C+1)≈γ line)
- Final NM converges to C≈4.0, γ≈0.2, exactly satisfying self-consistency

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

From the beginning, the decision was made to **not re-tune post-processing parameters**, instead directly reusing Phase 5 values. Reasons:

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

| Parameter | Phase 5 (Percentile) | Sigmoid Final | Change |
|-----------|:---:|:---:|:---:|
| note_norm_N0 | 10.0 | 10.0 | Unchanged |
| rescale_threshold | 9.54 | 9.54 | Unchanged |
| rescale_divisor | 2.00 | 2.00 | Unchanged |
| global_scale | 1.055 | 1.055 | Unchanged |

Post-processing parameters were completely unchanged during sigmoid optimization — validating that they are indeed orthogonal to the aggregation method.

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

### 8.3 Why B3 is Split into 3 Sub-blocks

The feature layer has 18 parameters (cross 5 + release 6 + inverse 7). If all optimized at once:
- 18-dimensional NM in full mode is extremely slow (~200 eval × 14.5s = 48 min/round)
- Cross and Release D(t) contributions are coupled through Xbar/Rbar, but direct gradients are very small

After splitting into 3 sub-blocks:
- Each block 5-7 parameters, NM converges quickly
- Alternating optimization allows cross-block coupling to propagate through D_seg re-extraction

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
→ k=1.5 verified as optimal, calib_a≈0.89, calib_b≈0.04, C≈4.0, γ≈0.20
```

### Phase 3: C-γ Grid Search

```
C ∈ {2,3,4,5,6,8,10}, γ ∈ {0.10, 0.13, 0.17, 0.20, 0.23, 0.27, 0.30}
→ Verify 1/(C+1)≈γ self-consistency relationship
```

### Phase 4: Full Alternating NM (k=1.5 fixed)

```
2 rounds × 5 blocks:
  B1: maxiter=50→25, Fast
  B2: maxiter=50→25, Full
  B3a: maxiter=50→25, Full
  B3b: maxiter=50→25, Full
  B3c: maxiter=50→25, Full
```

### Phase 5: Final Parameter Fine-tuning + Save

```
R3 B1 only (maxiter=50, Fast): Fine-tune sigmoid/calib/post
→ Save tuned_params_sigmoid.json
```

---

## 10. Key Findings

### 10.1 k=1.5 is the Overwhelming Improvement

Switching from k=0.52 (initial guess) to k=1.5:
- MAE: 0.2325 → 0.2253 (-3.1%)
- Loss: 1.01 → 0.91 (-10%)

This is the **largest single improvement** of the sigmoid aggregation layer, exceeding all subsequent NM optimizations combined.

### 10.2 C and γ Are Not Independent

The self-consistency line `γ ≈ 1/(C+1)` holds almost exactly. NM never deviates from it by more than 2%. In practice, only 1 degree of freedom is needed (e.g., only tune C, compute γ from formula).

### 10.3 D_solved Stably Falls at P70 of D(t) Distribution

Across 205 charts: `D_solved ≈ 0.87 × D_P70`, std only 0.01.

This remarkable consistency means:
1. Sigmoid aggregation is equivalent to performing smooth "soft percentile selection" on the D distribution
2. The selection point is fixed at ~P70, determined by the numerical values of k and C
3. Changing k slightly adjusts this equivalent percentile (larger k → earlier percentile), but the change is minimal

### 10.4 D Calibration is Necessary but Very Simple

The linear relationship `D' = 0.89×D + 0.04` is highly stable across different k values and optimization stages. This indicates D's absolute values are already close to the correct scale, requiring only ~11% slight adjustment.

### 10.5 Feature Layer Parameters Barely Need Re-tuning

From Phase 5 (Percentile) to Sigmoid Final, the 16 feature layer parameters changed <2%. B3a/B3b/B3c's combined NM improvement totals less than 0.001 MAE.

This means: **the feature layer (D component definition) and aggregation layer (how to go from D(t) to SR) are highly decoupled**. Good D(t) performs well under both aggregation methods.

### 10.6 Diminishing Marginal Returns of Alternating NM

```
R1 B1: ΔLoss = -0.115  (dominant improvement)
R1 B2: ΔLoss = -0.003
R1 B3a-c: ΔLoss = -0.001
R2:     ΔLoss = -0.001
```

Round 1 B1 contributes ~95% of total improvement. Marginal returns from R2 and B3 sub-blocks are extremely low (<1%). For future work: **focus on aggregation layer parameters (k, C, γ, calib); the feature layer can be locked**.

---

## Appendix A: Script Index

| Script | Function |
|--------|----------|
| `scripts/sweep_k_fine.py` | Fine k scan from 0.5-3.0 (B1 fast path) |
| `scripts/tune_sigmoid_k15.py` | Full alternating NM at k=1.5 (5 blocks × 2 rounds) |
| `scripts/build_standalone.py` | Single-file distribution builder |
| `spm_rating/aggregate_sigmoid.py` | Sigmoid aggregation core (segmentation + bisection) |

## Appendix B: Parameter Archive

| File | Content |
|------|---------|
| `tuned_params_sigmoid.json` | Final optimal parameters (MAE=0.2253) |
| `tuned_params_sigmoid_k15.json` | k=1.5 full NM intermediate results |
| `tuned_params_sigmoid_bestk.json` | k sweep best B1 parameters |
