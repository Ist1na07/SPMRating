# How the model works

[简体中文](model.zh-CN.md) · [Usage](../README.md) · [Evaluation](evaluation.md)

## From objects to a score

The parser reads press times, columns, long-note tails, key count, and overall difficulty, embedding 4-key and 6-key layouts in a symmetric seven-column hand frame. Exact duplicate objects on the same column are merged before scoring; simultaneous objects on one key with conflicting tails raise an error. Overlapping long-note intervals on a single finger are merged when measuring finger occupation.

`spm_rating/v130_rev2.py` evaluates native row-grid curves for repeated-finger jack load, shared hand contacts, sustained finger occupation, chord pressure, cross-column transitions, positive release difficulty (`R_POSITIVE`), and nonnegative long-note coordination (`CbarV2` and `hb`). A positive 100 ms burst term prices clustered attacks, and local exponent scaling adapts continuously to active lane counts (`Ks`).

- **`detail` (`V13-D1`)** fits **78** continuous parameters across downstream curves and upstream extraction/coordination geometry (`U1–U3`). It evaluates both the canonical orientation and its horizontal mirror and averages the resulting difficulty fields after the nonlinear forward pass, guaranteeing exact `0.0` mirror asymmetry.
- **`compact` (`V13-C`)** fits **70** continuous parameters in a single canonical orientation pass, omitting `lockanchor` and `shield` from `CbarV2` while holding six secondary shape/hinge parameters at fixed defaults.
- Both profiles apply a frozen, strictly increasing piecewise-linear calibration (`reference_calibration`) per key count to produce `reference_score` (`dan`, `lv`, `dan_sr`).

## Independent fingers and shared hand contacts

Repeated presses carry evidence about both an individual finger's jack workload and a possible shared hand action. `hand.py` inspects consecutive same-finger contacts; intervening hits from the same hand weaken the evidence for an isolated single-finger continuation, while smooth coactivation weights track nearby participating fingers.

A shared contact budget is distributed across eligible repeated contacts. Per-finger projection yields independent repeat-rate fields, and shared projection yields a hand-rate field. On each hand, the excess cost equals the positive excess of shared rate over its busiest independent finger, combined across both hands by their bottleneck. Alternative spacebar-thumb assignments combine symmetrically via root-mean-square aggregation.

`hand_cost.py` applies the Sunny-style rate cost:

$$
F(r) = \frac{r^2}{1 + k r},
$$

where `r` is the effective press rate per second and `k` comes from the timing-precision constants.

## Actions after sustained finger occupation

`role.py` models the extra burden placed on a finger that was recently occupied by long notes. At each head, let `u` be the fraction of the preceding local window spent holding on that finger, and let `g` be the nearest previous or next same-finger gap in seconds (capped at the local window width, with a 0.1 ms floor). The action budget is

$$
w = \frac{u^2}{g + k}.
$$

The quadratic term expresses a convex prior: concentrating hold occupation onto one finger increases additional workload faster than alternating across fingers. Each budget is spread with a unit-integral cosine-squared pulse whose value and derivative join continuously at zero, accumulated per finger, and reduced by the busiest finger into movement `S`, separate from the bare jack context $J$.

Long-note coordination combines strictly nonnegative row-grid atoms—churn, hold age, cross-hand hold-taps, extra same-hand hold-taps, lockstack, positive straddle, positive overlap, and (in `detail`) lockanchor and shield—with a fixed physical distance matrix that prices same-hand occupation above cross-hand work. Release difficulty uses a single density-gated positive curve (`ln_gate`).

On **4K charts**, an additional positive pressure gate `ln_gate_4k` scales excess long-note pressure (`R` and `cb`) by $C / (C + \mathrm{ln\_gate\_4k})$, alongside 4K-specific stream weights and a 4K triple-jack scaling term, preventing overestimation on sparse/low-Dan 4K long-note patterns without affecting tap-only charts.

## Combining movement and technique ($p=3$ Hill gate)

With bare jack context $J$, total movement workload $S$, and positive cross-column crossing demand $X$, the model applies a cubic ($p=3$) Hill gate:

$$
T = A^{a_t / K_s} \cdot \frac{X}{X + S + o} \cdot \frac{X^3}{X^3 + (k_j J)^3 + 10^{-12}},
$$

where $o = t_{s\_\text{off}}$ and $k_j$ switches cleanly between 6K/7K (`k_jt`) and 4K (`k_jt_4k`). The cubic Hill gate suppresses spurious crossing bonuses inside jack-dominated patterns while remaining smooth and strictly positive everywhere. Base row demand combines movement and technique via

$$
\mathrm{base}_j = b_1 S_j^{d_s} T_j^{d_t} + b_2 S_j + c_{\mathrm{floor}} H_{\mathrm{floor}, j},
$$

with explicit support for static chords consisting purely of held tails.

## Stamina, pure Row Sigmoid aggregation, and time

### Bounded causal stamina

Causal exposure $E_j$ tracks exponentially decaying load above threshold `stam_theta` across inter-row intervals $\Delta t$, dissipating during rests with decay constant `tau_stam`. The positive multiplier

$$
D_j = \mathrm{base}_j \left(1 + c_{\mathrm{stam}} \frac{E_j}{E_j + \mathrm{stam\_half}}\right)
$$

is bounded above by $1 + c_{\mathrm{stam}}$ and excludes any subtractive recovery term.

### Engagement-mass-weighted Row Sigmoid aggregation

Rather than slicing charts into discrete time windows with two-level window power means, v1.3.0 aggregates directly over native row demands blended with a local envelope:

$$
z_j = (1 - w) D_j + w \, e_j.
$$

Each row receives a physical engagement mass $M_j$ combining active note count $c_j = \max(C_j, 10^{-6})$, inter-row time span $q_j = \min(\max(\mathrm{gp}_j, 0),\, 387.12377)$, and simultaneous held-tail attenuation:

$$
M_j = \frac{c_j^2}{c_j^2 + 4} \, c_j^{p_{\mathrm{mass}}} \, q_j^{p_{\mathrm{gap}}} \exp\!\left(-\frac{h \cdot \mathrm{held}_j}{\mathrm{held}_j + 51.9327}\right).
$$

Given steepness $\beta$ (`8.959` in `detail`, `9.893` in `compact`) and target accuracy threshold $\gamma$ (`0.9640` in `detail`, `0.9510` in `compact`), the chart score $u > 0$ is the unique root of

$$
F(u) = \frac{\sum_j M_j \, \sigma\!\left(\beta \ln\frac{u}{z_j} - \ln\frac{1-\gamma}{\gamma}\right)}{\sum_j M_j} - \gamma = 0,
$$

where $\sigma(x) = (1 + e^{-x})^{-1}$, equivalently interpreting each row's per-row pass accuracy at challenge level $u$ as

$$
\mathrm{acc}(z_j, u) = \frac{u^\beta}{u^\beta + \frac{1-\gamma}{\gamma} z_j^\beta}.
$$

Safeguarded log-space Newton-Raphson root finding converges deterministically in microsecond-to-submillisecond time per chart. Because aggregation operates directly on exact row masses without sliding window partitions, **aggregation-origin phase jump is identically `0.0`**, while preserving strict first-order scale homogeneity ($z_j \mapsto c z_j \implies u \mapsto c u$) and strict monotonicity.

Finally, $u$ applies fixed short-chart length attenuation ($n / (n + 7.6)$), an upper compression hinge, and affine scaling.

### Clock phasing and segments

Fractional original-clock timestamps (common in offset-shifted edits) are handled separately in `clock.py` via shared continuous rounding-phase integration. Integer-millisecond `.osu` files and standard speed multipliers (`--rate`) run in a single native pass. `context` mode queries the full-chart demand field at selected rows, whereas `isolated` rebuilds the sliced segment independently. See [Time handling](time.md).
