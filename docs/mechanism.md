# SPM Rating v1.1.1 Algorithm Mechanism Specification (Full Reproducibility Standard)

> This document serves as the **exhaustive specification** for SPM Rating v1.1.1. Any developer or researcher can completely reproduce the algorithm from scratch with exact numerical parity (ulp-level consistency) across any programming language solely by following the equations and procedures detailed herein.
>
> Single-source-of-truth parameter file: [params/spm-rating-1.1.1.json](../params/spm-rating-1.1.1.json) (276 independent parameters).
> Accompanying technical reports: [History & Findings](history.md) (evolution and ablation studies from v0.5.0/v1.0.0 to v1.1.0/v1.1.1), [Maintenance Guide](maintenance.md) (engineering benchmarks and CI gates).

---

## Table of Contents
- [0. Core Design Principles & Notation](#0-core-design-principles--notation)
- [1. Input Parsing & Geometry Embedding (Parser)](#1-input-parsing--geometry-embedding-parser)
- [2. Row Grid & Feature Tensor Construction (Grid & Struct)](#2-row-grid--feature-tensor-construction-grid--struct)
- [3. Hit Leniency & OD Mapping (Leniency)](#3-hit-leniency--od-mapping-leniency)
- [4. Jack & Repeated Tap Channels: Jm, Jc, Ja (Jacks)](#4-jack--repeated-tap-channels-jm-jc-ja-jacks)
- [5. Stream & Gesture Chord Pricing Channels: Pbar, Xbar, Abar (Stream)](#5-stream--gesture-chord-pricing-channels-pbar-xbar-abar-stream)
- [6. Long Note Release Dynamics: Rbar (Release)](#6-long-note-release-dynamics-rbar-release)
- [7. Long Note Coordination Geometry Matrix: Cbar v2 (LNCoord)](#7-long-note-coordination-geometry-matrix-cbar-v2-lncoord)
- [8. Biomechanical Constraints & Auxiliary Channels: HCd, SHd, EyeR, Recov](#8-biomechanical-constraints--auxiliary-channels-hcd-shd-eyer-recov)
- [9. Instantaneous Local Difficulty Synthesis: D(t) (Combine)](#9-instantaneous-local-difficulty-synthesis-dt-combine)
- [10. Two-Level Mass-Weighted Sectioned Aggregation: agg_mode=4 (Aggregation)](#10-two-level-mass-weighted-sectioned-aggregation-agg_mode4-aggregation)
- [11. Chart-Level Post-Processing & Calibration (Postprocess)](#11-chart-level-post-processing--calibration-postprocess)
- [Appendix A: Hand Topology & Anatomical Groups](#appendix-a-hand-topology--anatomical-groups)
- [Appendix B: Full Release Parameter Manifest (272 Items)](#appendix-b-full-release-parameter-manifest-272-items)

---

## 0. Core Design Principles & Notation

1. **Strict Temporal Causality & Spatial Locality**: Difficulty must originate solely from the local temporal structure of the chart. Global beatmap scalar correction layers are strictly prohibited.
2. **Section-Level Robustness & Strict Invariance**: The algorithm is mathematically invariant under chart slicing, silent pauses, and section repetition, eliminating rest dilution and repetition inflation.
3. **Continuity & Perturbation Immunity**: Machine-precision threshold gates (e.g. $10^{-9}\mathrm{s}$) and piecewise-linear sharp cutoffs are abolished. All channels must remain continuously differentiable under $\pm 1\mathrm{ms}$ timing jitter.
4. **Unified Dimensional Units**:
   - Timestamps $t$, intervals $\Delta t$, and sliding window spans are stored and computed in **milliseconds (ms)**;
   - Physical velocities (notes per second) and decay rates use **seconds (s)**, denoted $\delta = \Delta t / 1000$.

---

## 1. Input Parsing & Geometry Embedding (Parser)

### 1.1 `.osu` Parsing
The parser extracts fields from `[General]`, `[Difficulty]`, and `[HitObjects]`:
- `Mode`: Must equal `3` (osu!mania);
- `CircleSize`: Native key count $K_{\mathrm{orig}}$ (4K, 6K, 7K);
- `OverallDifficulty`: Hit window rating $\mathrm{OD} \in [0.0, 10.0]$;
- Speed rate scalar $\mathrm{rate}$ ($1.0$ for standard, $1.5$ for DoubleTime).

Each HitObject yields `(t_raw, c_raw, t_end_raw, is_ln)`:
- **Native column**: clamped to $[0, K_{\mathrm{orig}}-1]$:

$$
c_{\mathrm{raw}} = \left\lfloor \frac{x \cdot K_{\mathrm{orig}}}{512} \right\rfloor
$$

- **Onset time**: $t = t_{\mathrm{raw}} / \mathrm{rate}$ (ms);
- **Object classification**: If bit 7 of the type field is set (`type & 128`) and end timestamp satisfies `t_end_raw > t_raw`, classified as Long Note (LN) with end timestamp `t_end = t_end_raw / rate` and `is_ln = True`; otherwise, regular tap note with `t_end = t` and `is_ln = False`.

### 1.2 Unified 7-Track Geometry Embedding
All calculations operate within a canonical **7-column physical frame** (columns $k \in \{0, 1, 2, 3, 4, 5, 6\}$):
- **7K Native**: Direct mapping $c = c_{\mathrm{raw}}$;
- **4K Symmetric Embedding**:

$$
\text{4K native } [0, 1, 2, 3] \longrightarrow \text{7-track frame } [1, 2, 4, 5]
$$

  Columns 0, 3, 6 remain unoccupied; left hand strikes tracks 1 and 2, right hand strikes tracks 4 and 5, leaving center thumb track 3 empty. This preserves exact physical finger distances and cross-hand transitions between 4K and 7K.
- **6K Embedding**: Maps to $(0, 1, 2, 4, 5, 6)$ with thumb column 3 empty.

Objects are sorted in ascending order by $t$, breaking ties by ascending $c$.

---

## 2. Row Grid & Feature Tensor Construction (Grid & Struct)

Rather than using event-driven scheduling, charts are projected onto a **Row Grid** of strictly increasing unique timestamps:

$$
t_0 < t_1 < \dots < t_{R-1}, \quad r \in [0, R-1]
$$

- **Row time delta**:

$$
\Delta t_r = t_{r+1} - t_r \quad (\mathrm{ms}), \quad \Delta t_{R-1} = 0, \quad \delta_r = \frac{\Delta t_r}{1000} \quad (\mathrm{s})
$$

- **7-bit column mask**:

$$
\mathrm{mask}_r = \sum_{k \in \mathrm{notes}(r)} 2^k \in [1, 127]
$$

- **Row chord size**:

$$
\mathrm{size}_r = \mathrm{popcount}(\mathrm{mask}_r) \in [1, 7]
$$

- **Held LN state**: An LN covering $[t_{\mathrm{head}}, t_{\mathrm{tail}}]$ is actively *held* at row $r$ if $t_{\mathrm{head}} < t_r < t_{\mathrm{tail}}$. We record the 7-bit mask $\mathrm{heldmask}_r$ with held count:

$$
\mathrm{held}_r = \mathrm{popcount}(\mathrm{heldmask}_r)
$$

- **Local note mass count $C_r$**: Total notes falling within a $\pm 500\mathrm{ms}$ symmetric window centered at $t_r$:

$$
C_r = \sum_{j: |t_j - t_r| \le 500\mathrm{ms}} \mathrm{size}_j
$$

- Active column count $K_{s, r} = 4.0$ for 4K and $7.0$ for 7K.

---

## 3. Hit Leniency & OD Mapping (Leniency)

Hit window rating $\mathrm{OD}$ maps to timing leniency scalar $x$ (seconds):
1. Rate-adjusted OD: When $\mathrm{rate} \ne 1.0$, effective OD is adjusted such that hit window divided by rate remains constant, yielding $\mathrm{OD}_{\mathrm{eff}}$ (variable `eff_od`);
2. Core leniency equations:

$$
\begin{aligned}
q &= \frac{64.5 - \lceil 3 \cdot \mathrm{OD}_{\mathrm{eff}} \rceil}{500.0} \\\\
x_0 &= 0.3 \cdot \sqrt{\max(q, 10^{-6})} \\\\
x &= \min\big(x_0,\; 0.6 \cdot (x_0 - 0.09) + 0.09\big)
\end{aligned}
$$

   For $\mathrm{OD} = 8.0$, $x \approx 0.0834\mathrm{s}$.
3. **Channel Hit Leniency Transforms**:
   - Jack channel: $x_{\mathrm{jm}} = x^{0.750426}$ (parameter `od_mult_jm`);
   - Stream channel: $x_p = x^{1.126485}$ (parameter `od_mult_p`);
   - Cross channel: $x_x = x^{1.165131}$ (parameter `od_mult_x`);
   - Release channel: $x_r = x^{1.064324}$ (parameter `od_mult_r`).

---

## 4. Jack & Repeated Tap Channels: Jm, Jc, Ja (Jacks)

Jacks are decomposed into three independent physiological channels:

### 4.1 Column Jack Extraction
For each column $k \in [0, 6]$, extract successive notes $(t_{k, j}, t_{k, j+1})$ with gap $\Delta t_j = t_{k, j+1} - t_{k, j}$ (ms) and $g_j = \Delta t_j / 1000$ (s):
- Same-column jack kernel:

$$
\mathrm{kern}_j = \max(g_j, x_{\mathrm{jm}})^{-2.0}
$$

- **Dilution suppression (A1 Superset Monotonicity Fix)**:
  Let $\mathrm{other}_j$ denote the notes on other tracks between this pair, and $\mathrm{nrows}_j$ the row count.

$$
\mathrm{dil}_j = \frac{1.0}{1.0 + 0.080 \cdot \left(\frac{\mathrm{other}_j}{\mathrm{nrows}_j}\right)^{0.9615}}
$$

  With parameters `mj_dilute = 0.080` and `mj_dilute_exp = 0.9615`. When a chord is struck simultaneously, parameter `chord_no_dilute = 1.0` overrides $\mathrm{dil}_j = 1.0$, preventing chords from improperly discounting jack costs.
- **Triple gain**:
  If a 3-note same-column jack spans $\le 135.97\mathrm{ms}$ (parameter `j_triple_tau`):

$$
\mathrm{gain}_j = 1.0 + \gamma_{\mathrm{trip}} \cdot \mathrm{trip}_j, \quad \gamma_{\mathrm{trip}} = 0.0193 \quad (0.5356 \text{ in 4K})
$$

### 4.2 Three Sub-Channel Formulations
1. **Minijack ($J_m$)**:

$$
v_{\mathrm{jm}, j} = \mathrm{kern}_j \cdot \mathrm{dil}_j \cdot \mathrm{gain}_j
$$

   Fast same-column compensation: $J_m \leftarrow J_m + c_{\mathrm{fj}} \cdot \mathrm{FJ}$ (parameter `c_fj` is $1.1224$, $1.6385$ in 4K).
2. **Chordjack ($J_c$)**:

$$
v_{\mathrm{jc}, j} = \mathrm{kern}_j \cdot \left(\frac{\max(\mathrm{load}_j, 10^{-6})}{1.9758}\right)^{1.0152}
$$

   where parameters are `cj_norm = 1.9758`, `cj_size_exp = 1.0152`.
3. **Anchor / Long-jack ($J_a$)**:

$$
\begin{aligned}
e_j &= \max(\mathrm{runlen}_j + 2.0 - 4.0742,\; 0.0) \\\\
\mathrm{runf}_j &= \left(\frac{e_j}{e_j + 5.9814}\right)^{1.0048} \\\\
v_{\mathrm{ja}, j} &= \mathrm{kern}_j \cdot \mathrm{runf}_j \cdot \left(1.0 + \frac{\mathrm{other}_j}{\mathrm{nrows}_j}\right)^{0.6808}
\end{aligned}
$$

   where parameters are `ja_norm = 5.9814`, `ja_len_exp = 1.0048`, `ja_len_floor = 4.0742`, `ja_coord_exp = 0.6808`.

### 4.3 Across-Column Power Mean
Scatter back to rows, smooth with moving-window filter (parameter `w_jm = 842.9ms`), and aggregate across tracks using power mean with exponent $p_{\mathrm{agg}} = 4.2205$ (parameter `j_agg_pow`) with weights $w_j = 1 / g_j$:

$$
J = \left( \frac{\sum_k w_k \cdot \mathrm{smooth}(v_k)^{p_{\mathrm{agg}}}}{\sum_k w_k} \right)^{1 / p_{\mathrm{agg}}}
$$

---

## 5. Stream & Gesture Chord Pricing Channels: Pbar, Xbar, Abar (Stream)

### 5.1 Global Flow Density $P_{\mathrm{bar}}$
Given row delta $\delta_r = \Delta t_r / 1000$:
1. **Continuous Gesture Windowing**:
   Within kernel width $\tau_c = 28.0\mathrm{ms}$ (parameter `chord_tau_c`), compute left/right hand triangular-window note mass $u_L, u_R$, yielding continuous effective chord size $u = u_L + u_R$.
2. **Gesture Chord Pricing**:
   Parametric size weights for 2-chord through 7-chord:

$$
\{p_{\mathrm{chw2}}: 0.700,\; p_{\mathrm{chw3}}: 1.471,\; p_{\mathrm{chw4}}: 0.821,\; p_{\mathrm{chw5}}: 0.868,\; p_{\mathrm{chw6}}: 0.747,\; p_{\mathrm{chw7}}: 1.033\}
$$

   Linearly interpolated at continuous size $u$ to obtain pricing factor $\mathrm{chw}_u$.
3. **Impulse Intensity**:

$$
\begin{aligned}
q &= \min\left(\delta - \frac{x_p}{2.0},\; \frac{x_p}{6.0}\right) \\\\
f_p &= \left( \max\left( \frac{p_{\mathrm{scale}}}{x_p} \cdot \left(1.0 - \frac{p_{\mathrm{lam3}}}{x_p} \cdot q^2\right),\; 0.0 \right) \right)^{0.25}
\end{aligned}
$$

   where parameters are `p_scale = 0.05862`, `p_lam3 = 27.838`.
4. **Bursts & Row Accumulation**:

$$
\mathrm{inc} = \frac{1}{\delta} \cdot f_p \cdot \mathrm{chw}_u \cdot \mathrm{mix} + 0.0575 \cdot \mathrm{cnt}_{100} \cdot 10.0 + 0.0416 \cdot \mathrm{cnt}_{150} \cdot 6.6667
$$

   Smoothed via window filter (parameter `w_p = 806.2ms`) to output $P_{\mathrm{bar}}$, with burst parameters `p_burst3 = 0.0575`, `p_burst4 = 0.0416`.

### 5.2 Alternation & Directional Jump $X_{\mathrm{bar}}$
1. **Boundary Event Chains & Inward/Outward Split**:
   Across 8 adjacent track boundaries, movements directed toward the center thumb track (inward) are weighted by parameter `x_din = -0.5959`; movements directed outward are weighted by parameter `x_dout = 0.9859`:

$$
X_b = x_{\mathrm{amp}} \cdot \max(g, x_x)^{-2.0} \cdot \big(1.0 + x_{\mathrm{din}} \cdot \max(dd, 0) + x_{\mathrm{dout}} \cdot \max(-dd, 0)\big)
$$

   where amplitude parameter `x_amp = 0.1478`.
2. **Fast Cross Geometric Coupling**: Coupled adjacent boundary velocities take geometric mean: $\sqrt{\mathrm{fc}_b \cdot \mathrm{fc}_{b+1}}$.
3. **Geometric Jump Channel**: Incorporates cross-hand multi-column leap term with parameter `x_jump_w = -0.8523`.
4. Smoothed with window filter (parameter `w_x = 939.3ms`) to form $X_{\mathrm{bar}}$.

### 5.3 Rhythm Irregularity $A_{\mathrm{bar}}$
1. For adjacent active columns $k_0, k_1$, delta difference $\Delta d = |d_{k0} - d_{k1}|$ and $mx = \max(d_{k0}, d_{k1})$;
2. **10ms Rhythm Weld & Soft Boundary** (parameter `abar_weld = 1.0`):

$$
dd = \Delta d + w_{\mathrm{mx}} \cdot \max(0, mx - 0.0886)
$$

$$
A_{\mathrm{val}} = \begin{cases}
\min(c_0 + w_{\mathrm{mx}} \cdot mx,\; 1.0), & dd < 0.0168 \\\\
\min(c_0 + k \cdot (dd - 0.0168) + w_{\mathrm{mx}} \cdot mx,\; 1.0), & 0.0168 \le dd < 0.0945 \\\\
1.0, & dd \ge 0.0945
\end{cases}
$$

   where parameters are $c_0 = 0.7299$, $k = 5.8693$, `thr_lo = 0.0168s`, `thr_hi = 0.0945s`, `abar_mx_w = 0.0716`, `abar_mx_thr = 0.0886`.
3. Smoothed with window filter (parameter `w_a = 60.79ms`) and scaled by parameter `abar_scale = 1.00986`, producing $A \in [0.65, 1.0]$.

---

## 6. Long Note Release Dynamics: Rbar (Release)

### 6.1 Per-Tail Release
For each LN tail at $t_i$, find next event $t_{\mathrm{next}}$ with $dtr = \max((t_{\mathrm{next}} - t_i)/1000.0,\; dtr_{\min})$, where $dtr_{\min} = 0.015\mathrm{s}$ (15ms):
- **Base Release Cost**:

$$
r_v = \frac{r_{\mathrm{tail}}}{\sqrt{dtr}} \cdot x_r^{-1.0} \cdot (1.0 + w_I \cdot I)
$$

  where $r_{\mathrm{tail}} = 0.028985$, and weight parameter $w_I = 1.0611$ (parameter `r_I_w`).
- **Release Index $I$**:

$$
I = \frac{2.0}{2.0 + \exp\big(-s_I \cdot (|dur - 80|/1000/x - o_I)\big)}
$$

- **Positive Power Hand Coordination**:

$$
r_v \leftarrow r_v \cdot \mathrm{cw}^{e_{\mathrm{coord}} \cdot e_{\mathrm{tt}}}
$$

  where $\mathrm{cw} \in \{0.982 \text{ (same)}, 0.425 \text{ (hand)}, 0.366 \text{ (thumb)}, 0.690 \text{ (cross)}\}$.
- **Short LN Reduction**: Lines $< 440\mathrm{ms}$ scale down linearly to $0.109$.
- **Lock Penalty & Straddle AIN Matrix**:
  Multiplied by $1.0 + r_{\mathrm{lock}} \cdot \mathrm{lock}$ ($r_{\mathrm{lock}} = 0.1775$) and scaled through finger groups `r_ain0` ~ `r_ain3`.

### 6.2 Tail-to-Tail Sequence
For consecutive release events:

$$
s_v = \frac{r_{\mathrm{seq}}}{\sqrt{dtr}} \cdot x_r^{-1.0} \cdot \big(1.0 + w_I(I_1 + I_2)\big) \cdot \mathrm{cw}^{e_{\mathrm{coord}}}, \quad r_{\mathrm{seq}} = 0.06596
$$

### 6.3 Release-Order Triple
For three successive same-hand releases in span $\le r_{\mathrm{tau}} = 404.3\mathrm{ms}$:

$$
\mathrm{excess} = \max\Big(|c_1 - c_2| + |c_2 - c_3| - \big(\max(c_1,c_2,c_3) - \min(c_1,c_2,c_3)\big),\; 0.0\Big)
$$

Weighted into release channel with penalty weight `r_order_pen = 0.04983`.

Smoothed with window filter (parameter `w_r = 1349.4ms`) to output $R_{\mathrm{bar}}$.

---

## 7. Long Note Coordination Geometry Matrix: Cbar v2 (LNCoord)

### 7.1 Anatomical Cross-Column Cost Matrix $M$
For column pairs $(c_1, c_2)$, lookup cost by biomechanical relation:
- Same hand adjacent (`sh_adj`): $0.9898$;
- Same hand split (`sh_split`): $0.6943$;
- Thumb bridge (`1h_split`): $0.6670$;
- Cross hand adjacent (`2h_split`): $0.6285$;
- Cross hand far (`cross`): $0.6384$.

### 7.2 The 7 Geometric Sub-Terms
1. **Overlap (`cbv_overlap` = 0.9864)**: Concurrently held LNs;
2. **Shield (`cbv_shield` = 1.1201)**: Taps arriving immediately before LN heads;
3. **Lockstack (`cbv_lockstack` = 0.5382)**: Same-hand repeated taps during held LNs;
4. **Lockanchor (`cbv_lockanchor` = 1.3024)**: Long jack runs during held LNs;
5. **Holdage (`cbv_holdage` = 3.4071)**: Extended duration hold strain;
6. **Churn (`cbv_churn` = 2.7804)**: Rapid hold/release transitions;
7. **Holdtap (`cbv_holdtap` = -1.0227)**: High-density stream bursts while holding LNs.

Summed and smoothed with window filter (parameter `w_cb = 1716.9ms`) to produce $C_{\mathrm{bar}}$.

---

## 8. Biomechanical Constraints & Auxiliary Channels: HCd, SHd, EyeR, Recov

### 8.1 A4 Two-Hand Parallelism: Hand Constraint Fee (HCd)
To eliminate inversion where single-hand 3-chords were computed cheaper than split 2+1 chords:
1. Compute left-hand mass $u_L$ and right-hand mass $u_R$ with window $\tau_c = 28\mathrm{ms}$;
2. Apply super-additive exponent (parameter `hc_e = 1.500`):

$$
\mathrm{HCd} = \max(0, u_L - 1.0)^{1.5} + \max(0, u_R - 1.0)^{1.5}
$$

3. Routed directly into stream with weight $d_{\mathrm{hc}} = 0.400$. Chords requiring $\ge 2$ keys on one hand receive steep super-additive pricing.

### 8.2 Visual Regularity (EyeR)
Measures alignment entropy between note timings and uniform subdivisions. Entered as a positive density-gated modifier:

$$
E_{\mathrm{term}} = d_{\mathrm{eye}} \cdot P_{\mathrm{bar}} \cdot \mathrm{EyeR}, \quad d_{\mathrm{eye}} = 0.08187 \ (\text{4K: } 0.01401)
$$

### 8.3 Recovery Discount (Recov)
Depresses transient difficulty following dense stamina sections using $d_{\mathrm{rec}} = 0.78125$.

---

## 9. Instantaneous Local Difficulty Synthesis: D(t) (Combine)

### 9.1 Branch Non-linear Ceiling

$$
\mathrm{branch}(v, \mathrm{mult}) = A^{\mathrm{mult} / K_s} \cdot \min\big(\max(v, 0),\; \mathrm{cap}_a + \mathrm{cap}_b \cdot \max(v, 0)\big)
$$

where saturation cutoff parameters are $\mathrm{cap}_a = 10.8467, \; \mathrm{cap}_b = 0.7670$. Saturated branches are computed with exponent parameter $a_j = 3.4083$:

### 9.2 Stream Branch

$$
\mathrm{stream} = A^{a_p} \cdot \left( a_p \cdot P_{\mathrm{bar}} + \frac{a_r \cdot R_{\mathrm{bar}}}{C + a_c} + \frac{a_{cb} \cdot C_{\mathrm{bar}}}{C + a_{cbc}} + d_{\mathrm{sh}} \cdot \mathrm{SHd} + d_{\mathrm{hc}} \cdot \mathrm{HCd} + E_{\mathrm{term}} \right)
$$

7K constants: $a_p = 0.4571, a_r = 59.955, a_{cb} = 4.000, a_c = 10.717, a_{cbc} = 10.311$.
4K overrides: $a_{p, 4\mathrm{K}} = 0.9672, \; a_{r, 4\mathrm{K}} = 0.0, \; a_{cb, 4\mathrm{K}} = 11.357$.

### 9.3 Generalized Power Sum $S$
Using generalized power sum exponent $s_p = 1.1397$ (parameter `s_p`):

$$
S = \left( c_{\mathrm{jm}} \cdot J_{m, \mathrm{sat}}^{s_p} + c_{\mathrm{jc}} \cdot J_{c, \mathrm{sat}}^{s_p} + c_{\mathrm{ja}} \cdot J_{a, \mathrm{sat}}^{s_p} + c_s \cdot \mathrm{stream}^{s_p} \right)^{1 / s_p}
$$

Weights: $c_{\mathrm{jm}} = 0.5453, c_{\mathrm{jc}} = 0.01839, c_{\mathrm{ja}} = -0.00125, c_s = 0.3931$.

### 9.4 Twist Factor $T$ and Total Instantaneous $D$

$$
\begin{aligned}
X_t &= \max(X_{\mathrm{bar}}, 0) + t_{\mathrm{jm}} \cdot J_{m, \mathrm{sat}}, \quad t_{\mathrm{jm}} = -0.3555 \\\\
T &= A^{a_t / K_s} \cdot \frac{X_t}{X_t + S + 0.4820}, \quad a_t = 5.7043 \\\\
D(t) &= d_{b1} \cdot S^{d_{\mathrm{ds}}} \cdot \max(T, 10^{-9})^{d_{\mathrm{dt}}} + d_{b2} \cdot S
\end{aligned}
$$

Constants: $d_{b1} = 1.4529, d_{b2} = 0.4343, d_{\mathrm{ds}} = 0.5437, d_{\mathrm{dt}} = 1.3838$, and offset `t_s_off = 0.4820`.

---

## 10. Two-Level Mass-Weighted Sectioned Aggregation: agg_mode=4 (Aggregation)

### 10.1 Engagement Mass $m_r$

$$
m_r = \big(\max(C_r, 0)\big)^{1.155} \cdot \min(\Delta t_r,\; 175.549)^{0.966}
$$

Parameters: `aggC_wc = 1.15498`, `aggC_gc = 175.549ms`, `aggC_gw = 0.96625`.

### 10.2 Sectioning & Sliding Grid
1. **Silence Reset**: If $\Delta t_r \ge 1854.5\mathrm{ms}$ (parameter `aggC_gap`), reset section boundary. Silent pauses contribute exactly 0 weight;
2. **Window Framing**: Window length $W_{\mathrm{ms}} = 6902.95\mathrm{ms}$ (~6.9s, parameter `aggC_ms`), stepped across the section.

### 10.3 Level 1: Intra-Window Mass Power Mean

$$
\begin{aligned}
M_w &= \sum_{r \in w} m_r \\\\
d_1[w] &= \left( \frac{\sum_{r \in w} m_r \cdot D_r^{p_1}}{\max(M_w, 10^{-12})} \right)^{1 / p_1}, \quad p_1 = 2.13656
\end{aligned}
$$

### 10.4 Level 2: Inter-Window Peak Envelope Bisection
1. Decay between windows:

$$
\mathrm{DEC}_w = \exp\left(-\frac{\Delta t_{\mathrm{gap}, w}}{\tau}\right), \quad \tau = 4000.57\mathrm{ms}
$$
2. Bisection search over $[0, \max(d_1) + 1.0]$ for exactly **26 iterations**. Testing root candidate $\mathrm{mid}$:

$$
\begin{aligned}
F_w &= \max\big(0.0,\; F_{w-1} \cdot \mathrm{DEC}_w + M_w \cdot (d_1[w] - \mathrm{mid})\big) \\\\
\mathrm{peak} &= \max_w F_w
\end{aligned}
$$

   Feasible if $\mathrm{peak} \le \kappa$ (convergence threshold parameter `aggC_kap = 0.81587`);
3. Returns midpoint after 26 iterations as $\mathrm{SR}_{\mathrm{agg}}$.

---

## 11. Chart-Level Post-Processing & Calibration (Postprocess)

1. **Note Count Normalization** (parameter `pp_n0 = 2.8826`):

$$
\mathrm{SR}_1 = \mathrm{SR}_{\mathrm{agg}} \cdot \frac{N_{\mathrm{notes}}}{N_{\mathrm{notes}} + 2.8826} 
$$

2. **High-SR Compression**:

$$
\mathrm{SR}_2 = \begin{cases}
\mathrm{SR}_1, & \mathrm{SR}_1 \le 8.3100 \\\\
8.3100 + \frac{\mathrm{SR}_1 - 8.3100}{1.5281}, & \mathrm{SR}_1 > 8.3100
\end{cases}
$$

   where parameters are `pp_thr = 8.3100`, `pp_div = 1.5281`.
3. **Linear Affine Calibration**:

$$
\mathrm{SR} = \max\big(0.0,\; \mathrm{SR}_2 \cdot 0.98662 \cdot 0.98771 + 0.68240\big)
$$

   where parameters are `pp_scale = 0.97524`, `calib_a = 0.98747`, `calib_b = 0.77542`.

Output scalar $\mathrm{SR}$ is the final SPM Rating v1.1.1 Star Rating.

---

## Appendix A: Hand Topology & Anatomical Groups

| 7-Track Column | Finger (7K) | Finger (4K Embedded) | Anatomical AIN Group | Hand Side HAND |
|---|---|---|---|---|
| Track 0 | Left Ring | (Empty) | 0 (Ring) | 0 (Left) |
| Track 1 | Left Middle | Left Middle | 1 (Middle) | 0 (Left) |
| Track 2 | Left Index | Left Index | 2 (Index) | 0 (Left) |
| Track 3 | Thumb | (Empty) | 3 (Thumb) | 1 (Thumb) |
| Track 4 | Right Index | Right Index | 2 (Index) | 2 (Right) |
| Track 5 | Right Middle | Right Middle | 1 (Middle) | 2 (Right) |
| Track 6 | Right Ring | (Empty) | 0 (Ring) | 2 (Right) |

---

## Appendix B: Full Release Parameter Manifest (272 Items)

All parameters conform to `params/spm-rating-1.1.1.json`, certified by unit tests:

```json
{
{
  "a_c": 13.380682264603966,
  "a_cb": 3.290737504274206,
  "a_cb_4k": 10.329032565184669,
  "a_cbc": 4.546298626916172,
  "a_p": 0.457050886903945,
  "a_p_4k": 1.0271932840768654,
  "a_r": 71.93325028558291,
  "a_r_4k": 0.0,
  "abar_active": 0.02531712913365081,
  "abar_c0": 0.729854838334696,
  "abar_c1": 0.6730069593477146,
  "abar_k": 5.869324658227848,
  "abar_mx": 0.48830826958952744,
  "abar_mx_thr": 0.10636359112398937,
  "abar_mx_w": 0.45817570312814426,
  "abar_scale": 1.009862473386292,
  "abar_thr_hi": 0.09447563280275387,
  "abar_thr_lo": 0.01682395583948343,
  "abar_weld": 1.0,
  "aggC_gap": 1919.100392694976,
  "aggC_gap_4k": 0.0,
  "aggC_gc": 206.24663914625867,
  "aggC_gw": 1.0018741768102066,
  "aggC_hw": 1.0,
  "aggC_kap": 0.8250936661252044,
  "aggC_ms": 6903.358465661703,
  "aggC_ms_4k": 0.0,
  "aggC_pw1": 2.1462221289544194,
  "aggC_pw2": 5.549120840435838,
  "aggC_tau": 4027.709861084976,
  "aggC_wc": 1.1528146438243094,
  "agg_C": 5.7765188650680015,
  "agg_gamma": 0.3142734186203886,
  "agg_gap_w": 1.0873927743990874,
  "agg_k": 3.923290274033005,
  "agg_mode": 4.0,
  "agg_nseg": 30.0,
  "agg_wc": 1.145022164188944,
  "aggw_gap_cap": 0.0,
  "aggw_src": 0.0,
  "aj": 3.4082562260373974,
  "an_a0": 0.17043768459177114,
  "an_a1": 0.34850214310770977,
  "an_cubic": 6.806495193662656,
  "an_on_p": -0.3268168035336469,
  "ap": 1.039645612096299,
  "at": 5.704296617075204,
  "c_chord2": 0.0,
  "c_fj": 1.1223698076832407,
  "c_fj_4k": 1.6384853006440867,
  "c_ja": -0.001250731346377573,
  "c_jc": 0.018393884652087668,
  "c_jm": 0.5452603821867452,
  "c_s": 0.3931375724076219,
  "c_stam": 0.0,
  "calib_a": 0.9877134644832144,
  "calib_b": 0.6823998306913243,
  "cap_a": 10.84672199638674,
  "cap_b": 0.7669642744433789,
  "cb_holdtap": 0.0,
  "cb_holdtap_pow": 1.0,
  "cb_lock_pow": 1.0,
  "cb_lockanchor": 0.0,
  "cb_lockjack": 0.0,
  "cb_par": 0.0,
  "cb_shield": 0.0,
  "cb_shield_lock": 0.8062,
  "cb_shield_tau": 56.24,
  "cbv_churn": 2.7804415879691047,
  "cbv_churn_4k": 0.0018149727082640878,
  "cbv_holdage": 3.4070743387277163,
  "cbv_holdage_4k": 0.0017655285135421454,
  "cbv_holdtap": -1.0227403453455164,
  "cbv_lock_pow": 0.9662307730974806,
  "cbv_lockanchor": 1.3023578337914123,
  "cbv_lockstack": 0.5382064568725327,
  "cbv_ls_1h_split": 0.6669564049783417,
  "cbv_ls_2h_split": 0.628481439284875,
  "cbv_ls_cross": 0.6384201937709314,
  "cbv_ls_sh_adj": 0.9897529278156161,
  "cbv_ls_sh_split": 0.694320421215397,
  "cbv_ov_1h_split": 0.8912801595852077,
  "cbv_ov_2h_split": 0.628660224600841,
  "cbv_ov_cap": 2451.2427577736403,
  "cbv_ov_cross": 0.43216200109373926,
  "cbv_ov_dt_max": 1511.5378347371243,
  "cbv_ov_sh_adj": 1.2272220519138874,
  "cbv_ov_sh_split": 1.2153000673016874,
  "cbv_overlap": 0.9863986916717904,
  "cbv_sh_dt_max": 256.0406131304308,
  "cbv_sh_ln_tau": 145.443782024068,
  "cbv_sh_ln_w": 1.0790072249120857,
  "cbv_sh_lock": 1.0870780764260457,
  "cbv_sh_tau": 75.33463318028997,
  "cbv_shield": 1.1200999548581216,
  "cbv_str_1h_split": 0.5323308514011785,
  "cbv_str_2h_split": 0.17479688357773077,
  "cbv_str_bal": 1.141728630652899,
  "cbv_str_cross": 0.09922357503493356,
  "cbv_str_h0": 0.8793037998512705,
  "cbv_str_h1": 0.9113686383081036,
  "cbv_str_h2": 0.8557205717895041,
  "cbv_str_h3": 0.5028456653288493,
  "cbv_str_maxd": 2.048176307602251,
  "cbv_str_sh_adj": 0.19036255960279166,
  "cbv_str_sh_split": 0.9209759560022021,
  "cbv_straddle": 0.45854924271224995,
  "cbv_w": 1140.2267181326677,
  "chord2_hard": 1.0,
  "chord2_w": 998.3068456460403,
  "chord_no_dilute": 1.0,
  "chord_soft_att": 1.0,
  "chord_tau_c": 28.00006103515625,
  "churn_event_rate": 0.0,
  "churn_sat": 14.712127862753169,
  "churn_w": 1022.5551356347157,
  "cj_norm": 1.9757525590733565,
  "cj_size_exp": 1.0152415724691892,
  "d_b1": 1.4529112152056358,
  "d_b2": 0.4343400978218659,
  "d_ds": 0.5436886594030598,
  "d_dt": 1.3838263378115971,
  "d_eye": 0.0818717181241305,
  "d_eye_4k": 0.014013206098578592,
  "d_hc": 0.4,
  "d_read": 0.0,
  "d_read_4k": 0.0,
  "d_rec": 0.78125,
  "d_sh": 0.0,
  "env_ms": 395.28032205483703,
  "env_w": 0.0033333333333333335,
  "eye_tau": 0.2,
  "eye_w_ref": 4000.0,
  "eye_w_s": 500.0,
  "fj_bounded": 0.0,
  "ga_alpha": 0.0,
  "ga_gam": 1.0,
  "ga_pw": 1.0,
  "ga_q": 1.0,
  "hc_e": 1.5,
  "holdage_cap": 0.9712508526439506,
  "holdage_th": 0.25073103183213685,
  "hw_a": -4.833333333333334,
  "hw_half": 51.932695284873475,
  "j_agg_pow": 4.220517945172583,
  "j_c1": 0.11994305455701491,
  "j_nerf_a": -2.266651862989442e-05,
  "j_nerf_c": 0.07443739821757173,
  "j_nerf_off": 0.15369105717534054,
  "j_triple_gain": 0.01930565188193239,
  "j_triple_gain_4k": 0.5356436393950172,
  "j_triple_pow": 1.005976267076746,
  "j_triple_tau": 135.97491132645644,
  "ja_coord_exp": -9.268498548764114e-05,
  "ja_gap_thr": 253.00356478769385,
  "ja_len_exp": 1.0047596122947928,
  "ja_len_floor": 4.074244535477681,
  "ja_norm": 5.981391018054386,
  "ln_eff_tail_ms": 0.0,
  "ln_note_level": 0.0,
  "mean_pow": 6.465267031930503,
  "mj_dilute": 0.08,
  "mj_dilute_exp": 0.9615171298692908,
  "note_chord": 1.0,
  "od_mult_ja": -1.0,
  "od_mult_jc": -0.37109374999999944,
  "od_mult_jm": 0.7504257038347726,
  "od_mult_p": 1.1264849156945338,
  "od_mult_r": 1.064323992348447,
  "od_mult_x": 1.1651306736110387,
  "out_a": 0.8686939253627315,
  "out_b": -0.9591165705858606,
  "p83_c": 2.667686959362696,
  "p93_c": 2.505172683584436,
  "p_boost": -1e-05,
  "p_boost_hi": 401.3793072521453,
  "p_boost_lo": 186.3466709871835,
  "p_burst3": 0.05749725160072005,
  "p_burst4": 0.041571581046874806,
  "p_bv_max": 1.023169107328028,
  "p_chw2": 0.6997350782823039,
  "p_chw3": 1.4710731528472405,
  "p_chw4": 0.82149995851254,
  "p_chw5": 0.8676070127257947,
  "p_chw6": 0.7474127532548505,
  "p_chw7": 1.03284160083206,
  "p_lam2": 0.006084380903675146,
  "p_lam3": 27.838293996532915,
  "p_sat_a": 10.690152688495372,
  "p_sat_b": 2.0349325289971496,
  "p_scale": 0.05862260059258437,
  "p_v_norm": -0.005007331887365899,
  "pp_div": 1.559992046815954,
  "pp_n0": 0.0,
  "pp_scale": 0.9866220368284715,
  "pp_thr": 8.416405509249257,
  "r_I_off": 1.679847001293981,
  "r_I_steep": 3.8155719025104933,
  "r_I_w": 1.0611113361746765,
  "r_ain0": 1.402277599248007,
  "r_ain1": 0.8487725298502042,
  "r_ain2": 0.9889530992577216,
  "r_ain3": 1.4808396303580251,
  "r_coord_e": 0.6807568539174652,
  "r_cw_cross": 0.6899948287076584,
  "r_cw_hand": 0.42489506847315023,
  "r_cw_same": 0.9824539263359018,
  "r_cw_thumb": 0.3660514384223894,
  "r_dt_max": 5985.197267885601,
  "r_dtr_min": 0.015,
  "r_lock": 0.1775454477373913,
  "r_order_pen": 0.04982726415201301,
  "r_order_tau": 404.3308885287425,
  "r_same_col": 0.39414011153048933,
  "r_seq": 0.06596340070980779,
  "r_short_red": 0.10936100877002214,
  "r_short_thr": 440.09681230961536,
  "r_sim_tau": 10.0,
  "r_simul": 0.08534094291675064,
  "r_soft_edges": 1.0,
  "r_straddle": -0.11540999412449468,
  "r_tail": 0.02898542543998874,
  "r_tt": 3.5417544104085192,
  "read_clip_hi": 1.5,
  "read_clip_lo": 0.5,
  "read_n_win": 16.0,
  "read_tol": 0.18,
  "rec_half": 0.4886690846205094,
  "recov_w": 989.9492044110127,
  "recov_wl": 21003.231873446508,
  "s_p": 1.1396559261442123,
  "shd_w": 1071.1117107781824,
  "stam_grp": 30.0,
  "stam_tau": 8000.0,
  "t_jack_mix": -0.35552500051452174,
  "t_s_off": 0.4819735114526039,
  "use_cbar_v2": 1.0,
  "use_cbv_holdtap": 1.0,
  "use_cbv_lockanchor": 1.0,
  "use_cbv_lockstack": 1.0,
  "use_cbv_overlap": 1.0,
  "use_cbv_shield": 1.0,
  "use_cbv_straddle": 1.0,
  "use_eye": 1.0,
  "use_hb": 1.0,
  "use_read": 0.0,
  "w83": 0.4815139944602175,
  "w93": 0.3605157897072774,
  "w_a": 60.794584215917084,
  "w_cb": 1716.856156378797,
  "w_ja": 0.0,
  "w_jc": -0.4781249999999999,
  "w_jm": 842.9060364136417,
  "w_p": 806.2241563795392,
  "w_r": 1349.4219390385326,
  "w_x": 939.2935347185421,
  "wmean": 2.1613377064577026,
  "x_amp": 0.14776064699735425,
  "x_cw0": 0.1460113252705887,
  "x_cw1": 0.3535388218642357,
  "x_cw2": 0.3389326726836199,
  "x_cw3": 0.04918990446028124,
  "x_din": -0.595854461464495,
  "x_dir_in": 0.2845860534263308,
  "x_dir_out": -0.016749922744166222,
  "x_dout": 0.9858637139438629,
  "x_fc_a": 0.541953977488367,
  "x_fc_floor": 0.04577642898224647,
  "x_fc_off": 60.317605380705,
  "x_fc_w": 1.6631718126735746,
  "x_jd_e": 1.0765985987293114,
  "x_jd_p": 0.0456784859552089,
  "x_jh_e": 1.6812589290379765,
  "x_jh_p": 1.0261744953668672,
  "x_jt": 0.5592172136082718,
  "x_jump_w": -0.852278272911138
}
```
