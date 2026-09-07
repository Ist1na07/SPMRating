# SPM Rating v1.0.0 Maintenance and Upgrade Guide

This document is for readers who use and maintain the algorithm inside this repository, covering: environment setup, everyday chart rating, regression checks, the new-release process, and the parts that must not be changed. Two companion documents exist: the [mechanism document](mechanism.md) describes the algorithm itself; the [history document](history.md) covers the differences from predecessors, the development process, and the conclusions.

The tuning toolchain (tuning scripts, experiment records, validation datasets) is maintained in the development repository, not here. Retuning must be done there; Section 5 of this guide keeps only the binding, validated principles.

## 1. Environment

The runtime is Python 3.10 or newer, with two dependencies, numpy and scipy:

```
pip install -r requirements.txt
```

All commands run with the repository root as the working directory.

## 2. Everyday Rating

Single-chart rating goes through the standard interface, i.e. the spm_interface module under interface/python. The input can be a chart file path or its text content, and the return value is the star rating; on failure it returns an object carrying the error information and never throws. Batch rating goes through the core pipeline: parse and batch, build row structures, compute curves, merge and aggregate — see [Section 2 of the mechanism document](mechanism.md#2-computation-pipeline). New charts need no special handling; the parser embeds them into the 7-column frame automatically. 4-key charts use their own set of output coefficients, because the two label sets are not on the same scale.

## 3. Regression Checks

[tests/cases/](../tests/cases/) holds ten built-in charts, and [tests/expected.json](../tests/expected.json) records the expected star rating for each one. Run

```
python tests/compliance_check.py
```

It recomputes all charts with the current code and compares them against the expected values (tolerance 0.001), and additionally checks that the interface runs from an independent directory, that all output fields are present, and that invalid inputs are handled per the contract — five checks in total, reported as 5/5 when all pass. The tests can be run two ways: `python -m pytest tests/ -q` runs everything (test_lncoord_v2.py is the unit test for the LN coordination channel, test_params_sync.py guards the consistency between the parameters embedded in the bundled engine and the release parameter file); each test file also runs standalone, e.g. `python tests/test_lncoord_v2.py`, which needs only the requirements.txt dependencies. Both suites must be fully green before a release. The repository runs the same two suites automatically on GitHub Actions under Python 3.10 through 3.13 (see .github/workflows/ci.yml); test dependencies are listed in requirements-dev.txt.

Known state at release: benchmark scores 0.98449 on 7-key, 0.98535 on 4-key, 0.95796 on 6-key zero-shot; timestamp-perturbation median errors 0.178 (7-key) and 0.039 (4-key), against a target of both under 0.05 — 4-key passes, 7-key recorded as is; four-fold cross-validation gaps about 0.001 on 7-key and about 0.0006 on 4-key. The perturbation and cross-validation scripts live in the development repository; see [Section 5 of the history document](history.md#5-methodology) for the meaning and provenance of these numbers.

## 4. Release Process

Parameter files are named spm-rating plus the version number. Versioning rules: architecture or pipeline changes bump the major version, mechanism additions/removals or retunes bump the minor version, pure parameter retunes bump the patch number. Each release updates four places together: the parameter file's own description field, the parameter values in the [mechanism document](mechanism.md), the score table in the [history document](history.md), and the version log at the end of this guide. The release commit message must state the three numbers relative to the previous release, the mechanism changes, and the decision basis.

## 5. Principles for Parameter Retuning

Retuning happens in the development repository, not here. This section lists only the validated, binding principles; see [Section 5 of the history document](history.md#5-methodology) for the full methods and evidence.

The adoption criteria are written down before the experiment starts: the objective must fall by at least four times the noise floor, neither key count's score may regress, and 6-key may not regress. Reject if the scores pass but the objective does not; ties are decided by 6-key.

The shared backbone is only ever adjusted on the joint two-key-count objective, never on single-key-count data alone. 6-key labels are coarse and pre-screened, so they serve as a zero-shot arbiter only, never as a training signal.

Candidate mechanisms walk the validation ladder: null-hypothesis calibration must include the smoothness-preserving cyclic-shift kind; then a low-cost block refit checks redundancy; finally a same-budget joint ablation concludes. A result tuned on only one side is not evidence.

New mechanisms start with a frozen dose scan; only past the null-hypothesis bar do they enter joint tuning, and joint tuning must have a same-budget control running first. The first round always starts from the current release, never from a cold start on defaults. Parameters affecting data structures are handled separately by grid plus inner search, never inside ordinary parameter blocks.

## 6. Standard Flow for New Mechanisms

Implement the new curve under core/components first — only row-local time-series quantities are allowed, whole-chart statistics are banned; that is this algorithm's design principle. Then add defaults and block membership to the parameter registry, with comments stating the semantics and the meaning of zero. Defaults must reproduce old behavior bit-for-bit: old version numbers unchanged, all tests green. Then walk the validation ladder in Section 5. After landing, update the matching sections of all three documents; rejected mechanisms are recorded as well.

## 7. Common Mistakes

Retuning dead parameters is the most common waste. A few parameters are never read by the model — perturb before tuning, and a parameter whose curve difference is exactly zero is dead. The semantics of zero is easy to misuse: a zero key-count override means inherit, a negative judgment degree means inherit, a non-positive smoothing window means inherit; close channels with gate switches or zero weights, and keep the two apart. Mojibake in Chinese-containing files is the third recurring problem: such files must always be rewritten whole, never read-modify-written.

## 8. What Must Not Be Changed

The reading branch (the read module under core/components) is disabled as a whole, kept only as an ablation control. The group of legacy LN coordination coefficients starting with cb_ in the parameter table are all zero, kept only as ablation controls. The near-zero long-run merge weight is a redundancy finding, not a tuning failure; the curve must be kept for the locked-anchor term. The near-zero envelope weight is essentially off, kept as a fallback against narrow peaks being drowned. Two structural-level switches remain at zero — their screening failed, and old behavior is preserved. One group of coefficients in the output affine transform is never read by the model; it is historical residue and may be deleted during cleanup.

## 9. Version Log

spm-rating-1.0.0: scores 0.98449 (7-key), 0.98535 (4-key), 0.95796 (6-key zero-shot). Gate status: regression checks pass; perturbation check below the ideal on 7-key, recorded as is; 6-key passes; cross-validation gap the smallest of any generation. Continue in this format: parameter file, the three numbers, change summary, gate status.
