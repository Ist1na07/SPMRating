# Fractional chart times

[简体中文](time.zh-CN.md) · [Model](model.md)

## Shared-phase extension

The native model operates on integer original-clock milliseconds. Charts edited with fractional timestamps can cross discrete simultaneous-group boundaries. `clock.py` extends the score between integer inputs by averaging a shared rounding phase.

Let `F` be the native scorer, `r` the speed rate, and `t` a time already divided by `r`. For one phase `U`, every head and long-note end uses

\[
Q_U(t)=\frac{\lfloor tr+U\rfloor}{r},\qquad 0\le U<1.
\]

The reported raw score is

\[
F_{\mathrm{extended}}=\int_0^1 F(Q_U(\mathrm{chart}))\,dU.
\]

All objects share `U`, preserving simultaneous relationships. Breakpoints occur at one minus the fractional parts of original-clock times. Between breakpoints, the rounded chart stays constant, so interval lengths give an exact finite weighted sum. The implementation snaps values within floating-point roundoff of integers.

Integer original-clock input follows the native function directly. This also covers ordinary speed changes applied to integer charts, because multiplying the parsed times by the speed rate restores the original clock.

## Continuity and domain

With object identities and judgment metadata fixed, moving one time by `delta` milliseconds changes its rounded value over at most `abs(delta)` of the phase interval. For finitely many objects and bounded native scores on the neighboring supported states, the change in the expected score tends to zero with the perturbation. Shared rounding and positive averaging also retain mirror symmetry and the conditional direction of added movement costs.

This reasoning applies where neighboring rounded charts remain supported. Changes in object membership, conflicting same-key objects, and judgment metadata require their own treatment. A small finite displacement such as 1 ms may still have a noticeable score effect.

In a 15-chart development check, column-dependent shifts of one millionth of a millisecond gave maximum relative score changes of about 1.84e-8 and 1.87e-8 for the two models. At 1 ms, the maxima were about 1.84% and 1.87%.

## Segments and curves

Segment membership is chosen on the observed chart before rounding. Selected objects and clipped long-note ends then use the shared phase. Context mode reads each phase's full-chart field; isolated mode rebuilds that phase's selected chart.

The scalar averages scores after each phase has completed nonlinear aggregation. The returned curve averages the corresponding local fields. Applying the nonlinear aggregator once to that averaged curve can produce a different scalar.

## Cost

With `K` distinct fractional breakpoints, scoring can require `K+1` phase passes and one pass for interface metadata. Integer original times use the direct native path. States are generated one at a time to limit memory, and cancellation is checked between passes.
