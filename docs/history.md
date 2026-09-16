# SPM Rating Algorithm History, Findings, and Methodology (Full Archive v1.0.0 to v1.1.0)

> This document contains the **complete historical record** of the SPM Rating algorithm, detailing its inception against early baselines, the **four phases of v1.0.0**, and the **four phases of v1.1.0** focusing on physiological axiom repairs and two-level mass aggregation. Every benchmark, falsified hypothesis, ablation verdict, and methodological standard is documented from code commits and experimental logs.
>
> Companion documentation: [Mechanism Specification](mechanism.md) (exhaustive reproduction standard and formulas), [Maintenance Guide](maintenance.md) (engineering standards and CI operations).

---

## Table of Contents
- [1. Baselines & Reference Benchmarks](#1-baselines--reference-benchmarks)
- [2. Mechanism Re-engineering vs. sunny-rework (v1.0.0 Foundations)](#2-mechanism-re-engineering-vs-sunny-rework-v100-foundations)
- [3. Lineage vs. Legacy SPM (0.4.0 / 0.5.0delta): Inheritance, Enhancements, and Deletions](#3-lineage-vs-legacy-spm-040--050delta-inheritance-enhancements-and-deletions)
- [4. v1.0.0 Research Phases & Falsified Hypotheses](#4-v100-research-phases--falsified-hypotheses)
- [5. Engineering Lessons & The BranchD Dead-Code Postmortem](#5-engineering-lessons--the-branchd-dead-code-postmortem)
- [6. v1.1.0 Architecture Overhaul: Motivations & Core Breakthroughs](#6-v110-architecture-overhaul-motivations--core-breakthroughs)
- [7. v1.1.0 Research Phases: Axiom Fixes & Segment-Level Validation](#7-v110-research-phases-axiom-fixes--segment-level-validation)
- [8. v1.1.1 Incremental Upgrade: Re-calibration on Updated Benchmarks](#8-v111-incremental-upgrade-re-calibration-on-updated-benchmarks)
- [9. Scientific Methodology & Engineering Discipline](#9-scientific-methodology--engineering-discipline)
- [10. Historical Performance Landscape](#10-historical-performance-landscape)
- [11. Future Directions & Action Items](#11-future-directions--action-items)

---

## 1. Baselines & Reference Benchmarks

Throughout development, the algorithm was evaluated against three external baselines:

1. **[sunny-rework](https://github.com/sunnyxxy/Star-Rating-Rebirth) Original (Untuned)**: Public release; single total output without separating jacks and holds. 7K Spearman correlation: **0.97129**; 4K: **0.97911**.
2. **sunny-rework Tuned**: Identical architecture, but recalibrated on a 339-chart playtest set (5 adjustments: removed jack nerf, jack aggregation power $5 \to 4$, doubled stream booster, column asymmetry cubic $5 \to 10$, recalibrated release index and 26 final constants). 7K reached **0.97921**, but because it was never tuned on 4K, 4K dropped to **0.97134**.
3. **Legacy SPM (0.4.0 / 0.5.0delta)**: Based on sunny-rework with modifications (Vbar inverse/guide, Sbar shield, Ebar leaky integrator), most notably an external 9-14 feature chart-level scalar correction layer. 7K reached **0.98373** (with home-field calibration advantages), and up to 0.98674 on specific overfit sets.

**Generational Milestones of SPM Rating**:
- **v1.0.0 Release**: 7K **0.98449**, 4K **0.98535** (exceeding strongest baselines by +0.0053 and +0.0062); on an un-tuned 6K dataset, zero-shot reached **0.95796** (exceeding sunny original 0.95515).
- **v1.1.0 Release**: Fully resolved underlying physiological axioms and stress tests, reaching 7K **0.98490**, 4K **0.98531**, 6K zero-shot **0.95595**; crucially, **segment-level error narrowed by nearly 10x**, establishing theoretical consistency from whole charts down to individual sections.

---

## 2. Mechanism Re-engineering vs. sunny-rework (v1.0.0 Foundations)

The conventional pipeline of sunny-rework is: parse chart → lay corner grid → compute 6 channels (jack, cross, stream, asymmetry, irregularity, release) → combine into total difficulty → percentile aggregation → postprocessing.

SPM Rating v1.0.0 retained the skeleton while fundamentally re-engineering every channel and introducing four new mechanisms:

1. **Three-way Jack Split**:
   - Replaced single jack channel with isolated minijacks ($J_m$), chordjacks ($J_c$), and long anchors ($J_a$) channels, each with independent physical saturation;
   - Added Triple Gain and Fast Jack (FJ) compensation.
2. **Second-Generation Long Note Coordination (Cbar v2 / LNCoord)**:
   - Replaced scalar hold count modulation with an **anatomical cross-column cost matrix**, identifying exactly which fingers are pinned to capture straddle patterns.
3. **Hybrid LN Components**:
   - Addressed 7K mixed and LN chart gaps: release churn, holdage accumulation, chord2 density, and recovery discount.
4. **Visual Regularity (EyeR)**:
   - Quantified alignment entropy with uniform subdivisions. Discovered that visually regular charts are systematically under-predicted by pure motor models, entering as a positive gated modifier.
5. **Robustness & Engineering Enhancements**:
   - Decomposed release into per-tail, sequence, and order-triple stages;
   - Directional inward/outward split on cross-column alternation;
   - Vectorized tensor execution across the entire dataset;
   - 5 continuous smoothing switches to resist 1ms timing jitter.

---

## 3. Lineage vs. Legacy SPM (0.4.0 / 0.5.0delta): Inheritance, Enhancements, and Deletions

### 3.1 Inherited & Enhanced
- Inherited the continuous time-series flow skeleton and magnitude scales;
- Enhanced shield from circular head exponentials into localized pairwise geometric overlaps in `lncoord_v2`;
- Restructured release into continuous positive-power dynamics;
- Enhanced cross-column alternation with boundary chains and inward/outward splits.

### 3.2 Deleted & Rejected
- **Complete Deletion of the Chart-Level Scalar Correction Layer (S10 / F14 / Defense)**:
  - First Principle: **Difficulty must originate solely from local temporal structures**. Legacy SPM computed 10-14 global scalars (speed, burst, chord ratios) and added a linear shift to final stars;
  - This destroyed section-level local additivity and failed on sliced segments. In v1.0.0/v1.1.0, this entire layer was eliminated, re-implementing all genuine physical phenomena directly into the time-series curves.
- **Complete Elimination of the Stamina Leaky Integrator (Ebar)**:
  - Screened 23 fatigue kernels; all failed cyclical shift null hypotheses, proving to be disguised global density channels.
- **Complete Elimination of Inverse & Release Guidance (Vbar)**:
  - Consisted of a "guide dip" (reducing release difficulty if a note appeared ~83ms after a release tail) and an "inverse spike". Ablations confirmed it failed cross-validation and introduced noise; completely removed.
- **Unified 4K/7K Geometry & Abolished 4K External Affine Patches**:
  - Removed `fit_delta_affine.json` and dual-regime compression divisors in favor of unified geometry.

---

## 4. v1.0.0 Research Phases & Falsified Hypotheses

### 4.1 Tuning Protocol & Testbed
- **Objective**: Pairwise ranking loss + auxiliary classification loss + deadzone loss on 7K, jointly optimized across 4K and 7K;
- **Optimizer**: Block coordinate descent with inner Nelder-Mead simplex search;
- **Discipline**: Shared backbone parameters were only tuned on joint 4K/7K objectives, never on a single keycount in isolation.

### 4.2 The Four Phases of v1.0.0
- **Phase 1 (Architecture Establishment)**: Established keycount split, visual regularity, peak envelope, hybrid LN, and 3-way jack split (+0.005 over strongest baseline). Falsified LN tail truncation.
- **Phase 2 (Wiring Repairs)**: Reconnected 4 dead parameters and harmonized blocks; 4-fold cross-validation confirmed robust superiority across all folds.
- **Phase 3 (Cross-Column Completion)**: Calibrated the omitted cross-column block (+0.002 on 7K), introduced inward/outward splits, and broke ties with 6K zero-shot.
- **Phase 4 (Smoothing & Robustness)**: Activated all 5 smoothing switches under identical computation budgets. 6K zero-shot reached a record 0.95796, with timing perturbation checks substantially improved to finalize v1.0.0.

### 4.3 Falsified Mechanisms Asset Register
1. **LN Tail Truncation**: 3 tested doses showed strictly monotonic harm. Paper length is the true physical metric; release timing itself is an essential skill.
2. **Lock-Mass Saturation (`cbv_lock_pow`)**: Grid scans confirmed no generalizable benefit.
3. **Max vs. Power-Mean Combinations for Visuals**: Failed ablations.
4. **Seven Low-Correlation Visual Features**: All dropped after screening.
5. **Conditional Lock Gating (`lockgate`)**: Failed cross-validation.
6. **Stamina Leaky Integrators**: Screened 23 kernels; 7K fully rejected. 4K beat random permutations but lost to the **smooth cyclical shift null hypothesis**, exposing it as a pseudo-temporal channel.
7. **Hold-Dominant Row Gating**: Sharp transitions broke gradients.
8. **4K Overrides for Hybrid LN**: Induced negative transfer.
9. **Independent Long-Jack Merged Channel**: Redundant with minijack/chordjack; weights decayed to zero.
10. **Pairwise Cross Coverage Extensions**: Exponential complexity without gain.

---

## 5. Engineering Lessons & The BranchD Dead-Code Postmortem

During the analysis of prior experimental lines (such as BranchD), a critical engineering failure was uncovered:

### 5.1 The Accident
Across 14 tuning waves in BranchD (v0.7.3 to v0.99.0), documentation claimed four components were "adopted and tuned":
- Cbar family;
- ivbar family;
- dshape family;
- Rbar-v2 release order matrix family.

In reality, their gating switches were hardcoded to 0 (e.g. `use_ivbar=0`, `use_dshape=0`), or completely disconnected from the computation graph. Optimizers pushed their downstream weights against numerical noise, mistaking random fluctuations for algorithmic improvements.

### 5.2 Enacted Discipline
- **Strict Gating Assertions**: Unit tests must assert `curve(on) != curve(off)`.
- **Single Source of Truth**: Monitored by `test_params_sync.py` to ensure exact equivalence between embedded python dictionaries and external JSON files.

---

## 6. v1.1.0 Architecture Overhaul: Motivations & Core Breakthroughs

Following v1.0.0 deployment, rigorous axiomatic testing and user-provided slice data exposed structural vulnerabilities:

### 6.1 Two Axiom Inversions
1. **A1 Superset Monotonicity Inversion**:
   - In chord-dense sections, adding simultaneous notes occasionally decreased calculated difficulty;
   - Cause: Excessive dilution factor `mj_dilute` (0.237) over-discounted short jacks when chords were present.
2. **A4 Two-Hand Parallelism Inversion**:
   - Compact one-hand 3-chords (e.g. left 0, 1, 2) were priced cheaper than split 2+1 chords (e.g. 0, 2, 4);
   - Cause: $X_{\mathrm{bar}}$ priced purely on column distance (smaller within one hand) and was canceled in chord sections by negative mix terms.

### 6.2 Stress Test Vulnerabilities (T1-T4)
v1.0.0's single-stage percentile / Sigmoid bisection aggregation (`agg_mode=0/2`) failed under synthetic stress tests:
- **T1 Scatter Dilution**: Adding 3 isolated notes 30s after the music depressed difficulty by **-0.309 stars**;
- **T2 Copy-Paste Inflation**: Repeating a chart twice inflated difficulty by **+0.343 stars**;
- **T3 Long Rest Dilution**: Inserting a 400s silent pause collapsed an 8-star chart by **-0.141 stars**;
- **Real-World Slices**: User-provided segment evaluations on `seishun no hacking` collapsed with an error of **-3.83 stars**.

---

## 7. v1.1.0 Research Phases: Axiom Fixes & Segment-Level Validation

### Phase 1: Continuous Gesture Chord Pricing
- Eliminated $10^{-9}\mathrm{s}$ discrete cliffs using triangular cosine kernels ($\tau_c = 28.0\mathrm{ms}$);
- Native parametric pricing for 2-chords through 7-chords (`p_chw2` ~ `p_chw7`).

### Phase 2: A1 & A4 Axiom Repairs
- **A1 Fix**: Activated `chord_no_dilute = 1.0` and reduced `mj_dilute` from 0.237 to **0.080**, ensuring monotonic non-decreasing difficulty when adding notes;
- **A4 Fix**: Routed the **Hand Constraint Fee** directly into the stream backbone:

$$
\mathrm{HCd} = \sum_{h \in \{L, R\}} \max(0, u_h - 1.0)^{1.5}
$$

  with $d_{\mathrm{hc}} = 0.400$. Concentrating $\ge 2$ keys on one hand incurs steep super-additive fees, eliminating the inversion.

### Phase 3: Two-Level Mass-Weighted Sectioned Aggregation (`agg_mode=4`)
- Completely replaced Sigmoid bisection and 30-bucket sorting;
- **Level 1**: Within ~6.9s sliding windows, computes mass power-mean with $p_1 = 2.1366$;
- **Level 2**: Across windows, evaluates peak envelope bisection with rest decay ($\tau = 4000.57\mathrm{ms}$, 26 iterations);
- **Stress Test Convergence**:
  - T1 Scatter: reduced from 0.309 to **0.0035** (~90x improvement);
  - T2 Repetition: reduced from 0.343 to **0.0043** (true copy-paste invariance);
  - T3 Rest: reduced from 0.141 to **0.0123** (rest-immune).

### Phase 4: Segment Validation & Full-Dataset Calibration
- Evaluated on 12 user-annotated slices of `seishun no hacking`:
  - v1.0.0 mean residual: $-3.83$ stars;
  - v1.1.0 mean residual: **$-0.03$ stars** (>100x accuracy improvement);
- Full Benchmark: 7K **0.98490**, 4K **0.98531**, 6K zero-shot **0.95595**.

---

## 8. v1.1.1 Incremental Upgrade: Re-calibration on Updated Benchmarks

With v1.1.0 having established solid structural foundations (two-level mass aggregation mode 4, continuous gesture chord pricing, and fundamental axiom repairs), the team conducted a comprehensive hyperparameter re-calibration **using updated and expanded benchmark datasets**.

### 8.1 Key Physical Channel Refinements
1. **Long-Note Coordination & Release Dynamics**:
   - Long-note release scaling ($a_r$) was adjusted from 59.95 to **71.93**, and damping ($a_c$) from 10.72 to **13.38**, accurately reflecting physiological resistance when executing consecutive releases;
   - Multi-finger chord suppression ($a_{cbc}$) was optimized down from 10.31 to **4.55**, eliminating over-penalization in complex parallel long-notes while yielding over 1.6% correlation gains on extreme long-note charts with deadzone loss reduced by nearly 30%.
2. **High-Speed Stream & Jack Balance**:
   - Refined combination weights $c_{jm}$ (0.489 $\to$ 0.545), $c_{jc}$ (0.060 $\to$ 0.018), and $c_s$ (0.412 $\to$ 0.393), providing cleaner separation between sustained streams and single-column bursts.
3. **Multi-Key Decoupling**:
   - Independent 4K long-note parameters ($a_{cb\_4k} = 10.33, a_{p\_4k} = 1.03$) prevent multi-key physical parameters from interfering with one another.
4. **Strict Stress-Test Invariance**:
   - With $pp\_n0 = 0.0$ strictly preserved, repeated section inflation (T2) maximum deviation was pushed down to **0.00856** (fully passing the strict $<0.010$ ceiling), while T1 (0.0029), T3 (0.0135), and T4 (0.0567) maintained 100% robustness.

---

## 9. Scientific Methodology & Engineering Discipline

SPM Rating strictly adheres to five methodological tenets:

1. **The 5-Rung Verification Ladder**:
   - Step 1: Defeat uniform random permutations;
   - Step 2: Defeat **smooth cyclical shift null hypotheses** (rules out disguised global scalars);
   - Step 3: Low-cost local block re-fits;
   - Step 4: Matched-budget joint cross-keycount ablations;
   - Step 5: $\pm 1\mathrm{ms}$ timing jitter perturbation checks (median drift must remain $\le 0.05$).
2. **Pre-Declared Acceptance Criteria**: Quantitative thresholds are locked prior to experiments.
3. **6K Zero-Shot Blind Arbiter**: The 6K dataset receives zero training gradients and acts strictly as an uncorrupted validation oracle.
4. **4-Fold CV & Flat Basin Verification**: Core parameters undergo $\pm 5\%$ perturbations; worst-case changes must remain $< 0.001$, proving convergence into a flat basin.
5. **Elimination Discipline**: Disproven mechanisms are completely removed, and dead parameters are explicitly named to prevent "ghost tuning."

---

## 10. Historical Performance Landscape

| Algorithm Version | 7K Spearman | 4K Spearman | 6K Zero-Shot | T1 Scatter Error | T2 Copy-Paste Error | T3 Rest Error | Core Aggregation Engine |
|---|---|---|---|---|---|---|---|
| **sunny-rework Original** | 0.97129 | 0.97911 | 0.95515 | ~0.45 | ~0.50 | >0.60 | Global Percentile Cutoff |
| **sunny-rework Tuned** | 0.97921 | 0.97134 | 0.94820 | ~0.42 | ~0.48 | >0.55 | Global Percentile Cutoff |
| **Legacy SPM (0.5.0delta)** | 0.98373 | 0.98120 | 0.93210 | 0.380 | 0.410 | 0.320 | 30-Bucket Sigmoid + Scalars |
| **SPM Rating v1.0.0** | 0.98449 | 0.98535 | **0.95796** | 0.309 | 0.343 | 0.141 | Sectioned Power Mean (mode 0) |
| **SPM Rating v1.1.0** | **0.98490** | 0.98531 | 0.95595 | 0.0035 | 0.0043 | **0.0123** | Two-Level Mass Aggregation (mode 4) |
| **SPM Rating v1.1.1** | **0.97899**★<br>(legacy set 0.98437) | **0.98573** | 0.95417 | **0.0029** | **0.00096**<br>(max 0.0086) | 0.0135 | **Two-Level Mass Aggregation + Retuned on Expanded Data (276 params)** |

> ★ Note: v1.1.1 7K score (0.97899) is measured on the updated and expanded benchmark set with broader diversity and difficulty spectrum; deadzone loss $dz$ decreases by 5.6% and MAE drops to 0.2922. On the legacy subset, it retains 0.98437 within the statistical noise floor.

---

## 11. Future Directions & Action Items

### 11.1 Re-Verification of Original User-Designed Mechanisms
Certain mechanisms originally designed by the author (rather than generated by AI) demonstrated significant empirical advantages on edge-case difficulty profiles and are formally scheduled for re-evaluation on updated datasets:
1. **v0.5.0delta Accuracy-Difficulty Curve & Sigmoid Bisection Aggregation**: Empirical findings indicate that power-mean aggregation suffers substantial deviation on charts with extreme difficulty distributions (e.g., intense localized bursts combined with long idle gaps), whereas the sigmoid bisection solver adapts naturally across such distributions via pass probability curves. Future work will investigate hybrid architectures merging local time-windowing with the bisection solver.
2. **Microscopic LN Release Dynamics**: Re-evaluating inverse spikes, guide dips, and concurrent release drag on dedicated high-density long-note benchmarks with repaired mathematical bounds.

### 11.2 Red Lines (Strictly Prohibited)
1. **Prohibit chart-level scalar correction layers** (destroys locality and slice validity);
2. **Prohibit training on the 6K dataset** (destroys the sole blind evaluation oracle);
3. **Prohibit reviving disproven non-saturating stamina integrators or tail truncation**;
4. **Prohibit un-gated dead-code parameter tuning**.
