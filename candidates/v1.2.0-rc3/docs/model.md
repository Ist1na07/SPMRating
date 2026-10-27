# How the model works

[简体中文](model.zh-CN.md) · [Usage](../README.md) · [Evaluation](evaluation.md)

## From objects to a score

The parser reads press times, columns, long-note ends, key count, and overall difficulty. It embeds 4-key and 6-key layouts in a symmetric seven-column frame. Exact duplicate objects are merged before scoring. Overlapping long-note intervals on a finger are merged when measuring that finger's occupation.

The numerical core builds several local curves: repeated-finger demand, nearby chord and anchor costs, stream demand, cross-column transitions, long-note coordination, releases, and visual context. Each component has its own established window or smoothing operation. The named fields in `_core/model.py` expose these intermediate quantities for source inspection.

`engine.py` evaluates the chart and its mirror. It shares the average release field, combines each view's local demands, and averages the resulting difficulty fields. The chart score then uses note-mass-weighted, two-level section aggregation. A final monotone output alignment provides the reference score for each key count.

## Independent fingers and shared hand contacts

Repeated presses give evidence about both a finger's workload and a possible shared hand action. `hand.py` examines consecutive same-finger contacts. Intervening actions from the same hand reduce the evidence for one continuing contact sequence. Smooth coactivation weights describe which fingers participate near a contact.

A shared contact budget is allocated among eligible repeated contacts. Per-finger projection produces independent repeat-rate fields; shared projection produces a hand-rate field. For each hand, the cost uses the positive excess of shared rate over its busiest independent finger. The two hands combine by their bottleneck. Two spacebar-thumb assignments contribute symmetrically through a root-mean-square combination.

`hand_cost.py` applies the Sunny-style rate cost

\[
F(r)=\frac{r^2}{1+kr}.
\]

Here `r` is a rate per second and `k` comes from the existing timing-precision parameters. This represents additional action demand. General staircase-jack and switching sequences still require further motion comparisons to establish how well this detector captures their differences.

## Actions after sustained finger occupation

`role.py` adds an action-conditioned cost for a finger recently occupied by long notes. At a head, let `u` be the fraction of the preceding local window for which that finger was held. Let `g` be the nearest previous or next same-finger gap in seconds, capped by that existing local window. A 0.1 ms floor handles extremely close repeats. The action budget is

\[
w=\frac{u^2}{g+k}.
\]

The square expresses a convex prior: concentrating occupation on one finger increases its additional workload faster than sharing it. Each budget is spread with a normalized compact cosine-squared pulse. Pulses add per finger, and the busiest finger supplies the local additional demand. The pulse and its first derivative meet continuously at its support boundary.

For steady cadence `r`, with gaps inside the local window and constant occupation `u`, the mean demand is

\[
rw=u^2\frac{r^2}{1+kr}.
\]

This shares units with the existing repeated-finger cost. The component reuses the established local window and timing coefficient, adding one price: approximately 0.10220 in `detail` and 0.11428 in `compact`. Calibration kept this price at least 0.1, expressing the finger-role distinction observed in the included P/Q controls.

The detailed model also keeps the earlier conditional hold-duration curve. The compact model sets its two amplitudes to zero; its threshold and cap then become inactive. Other coordination terms remain available because they describe additional local context.

## Combining movement and context

Additional hand and finger-role costs increase the movement input. The original repeated-finger context remains separate when evaluating the residual technique term. This gives the combination a useful interpretation: a new movement requirement can increase movement demand without simultaneously explaining away more technique.

For fixed other fields, the relevant combination has the form

\[
D(S)=b_1S^a\left(\frac{KX}{X+S+o}\right)^b+b_2S,
\]

where movement `S` and technique context `X` are nonnegative. In the current parameter family, `K` lies between zero and one. Although the first term can eventually decline when `a<b`, its negative derivative is bounded. The positive linear term dominates that bound; the present parameters give a conservative derivative lower bound of about 0.625. This supports the direction of the added movement costs while other context stays fixed.

Adding an object also changes context, coactivation, and aggregation weights. Evaluating those changes remains a separate modeling question.

## Recovery, aggregation, and time

`recovery.py` measures short and long head-count context with continuous window integrals. Signed transition budgets in `geometry.py` use a cosine cumulative integral before their final nonnegative clamp. These interfaces keep the intended temporal quantity explicit.

The aggregation in `_core/model.py` constructs sections and local windows, weights their difficulty by note mass, combines windows within sections, and combines sections into a chart score. Both models preserve this mature aggregation structure and its parameterization.

Fractional original timestamps use the shared-phase extension in `clock.py`. Integer original times follow the native path. [Time handling](time.md) describes its formula, segment contract, and computational cost.

## Extending the model

Useful follow-up comparisons include staircase-jack sequences, switching sequences with matched timing and note counts, and sustained occupation distributed across different fingers. Holding speed, chord size, and mirror orientation under control helps identify which action a proposed cost measures. Some comparisons can then be reserved for evaluating the completed design.

The parameter files retain a broad inherited feature set. A simpler model can be explored by removing components individually and checking the resulting motion behavior, local scores, overall ordering, and runtime together.
