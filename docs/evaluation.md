# Evaluation and model tradeoffs

[简体中文](evaluation.zh-CN.md) · [Usage](../README.md)

## What the ranking numbers measure

Spearman rank correlation measures agreement between the model's ordering and the label ordering. A higher value indicates closer ordering on that collection. Tied values receive average ranks.

The weighted development summary is

\[
0.5\rho_{7K}+\frac{5}{18}\rho_{4K}+\frac{2}{9}\rho_{\mathrm{jack}}.
\]

The long-note and 6-key collections are reported separately. These collections were used repeatedly during development; fresh collections would help assess transfer beyond this setting. Machine-readable results are in [benchmark.json](benchmark.json).

| Collection | Scored charts | Summary weight |
|---|---:|---:|
| 7-key | 393 | 0.5 |
| 4-key | 741 | 5/18 |
| Jack-focused | 292 | 2/9 |
| Long-note-focused | 40 | Separate |
| 6-key extension | 2343 | Separate |

Each candidate was evaluated on all 3809 supported inputs. One additional 6-key input contained simultaneous same-key objects with conflicting tails and was excluded under the input contract.

## Results

| Measure | v1.1.1 reference | Detailed | Compact |
|---|---:|---:|---:|
| Weighted ranking correlation | 0.981346 | **0.988334** | 0.988271 |
| 7-key ranking correlation | 0.979041 | **0.988976** | 0.988759 |
| 4-key ranking correlation | 0.985726 | 0.988330 | **0.988452** |
| Jack-focused ranking correlation | 0.981059 | 0.986895 | **0.986950** |
| Long-note ranking correlation | 0.883076 | **0.904663** | 0.899101 |
| Local 12-segment ranking correlation | 0.514097 | **0.866219** | **0.866219** |
| 7-key disagreements beyond summed label-error bounds | 293 | 103 | **99** |
| 6-key ranking correlation | — | **0.957222** | 0.956659 |

The v1.1.1 reference uses the common development comparison pipeline and collections. This table's scope differs from older version summaries that used different chart subsets.

The local panel consists of 12 selected 7-key excerpts already contained in the 7-key collection; nine involve long notes. It highlights local behavior within this development panel. Staircase-jack and switching patterns need a broader dedicated panel.

For 7-key labels, the annotator supplied an estimated maximum error. A pair is counted in the error-bound disagreement measure when its label gap exceeds the sum of the two error estimates and the model reverses that order. In these results the detailed and compact models disagree on 103 and 99 such pairs out of 58614 eligible pairs. This complements the full ranking statistic by focusing on more clearly separated labels.

## Interpreting the pair

The detailed model has slightly higher weighted and long-note correlation. The compact model has fewer error-bound disagreements and a simpler duration-cost structure. The two local correlations coincide, while individual scores can differ.

Both retain a substantial inherited feature set. Their files contain 291 configuration fields plus one finger-role price; several fields describe fixed settings or inactive branches. Disabling the compact model's duration component removes four continuous values from active use. Each key-count's output alignment adds a scale and exponent, which preserve ordering within that key count.

The role-price floor of 0.1 expresses a modeling choice supported by the P/Q playtest distinction. An earlier version assigned those controls the same score, while a player found Q clearly harder at both 120 and 180 beats per minute. These controls were used in design. The completed detailed model rates Q about 2.9% above P; the compact model gives about 3.3%. Additional players and reserved controls would help assess the size and generality of this effect.

## Verification and runtime

The publication runtime was extracted from the completed scorer into a self-contained package. Whole-chart outputs were compared against the archived completed scores on all 3809 supported charts for each model. Forty-five whole-chart and local cases per model also checked the public interface. The included small tests exercise the generated controls, segment contracts, time extension, and duplicate-object handling.

Runtime depends on chart length, local windows, and timestamp precision. Initial calls include compilation. Integer original-clock inputs use the native path; many distinct fractional timestamps can require multiple phase passes. The two models share most computations, so the compact model's structural reduction brings little speed difference in the present implementation.

Useful next evaluations include new chart collections, player comparisons of matched staircase and switching patterns, and local segments with different distributions of finger occupation. Such evidence can guide both further simplification and changes to the motion model.
