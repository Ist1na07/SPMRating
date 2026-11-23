# Changelog

[简体中文](CHANGELOG.zh-CN.md) · [Usage](README.md) · [Algorithm Mechanism](docs/mechanism.md)

---

## v1.3.0 (2026-11-24)

### Core Algorithm and Aggregation
- **Pure engagement-mass-weighted Row Sigmoid aggregation (`Row_LogSig`)**: Replaced two-level sliding-window power means with an implicit log-space Log-Logistic root finder directly over blended row demand and physical engagement mass. Eliminates window-origin phase drift (**exact `0.0` phase jump**), guarantees first-order scale homogeneity and row-level monotonicity, and boosts external BMS `version_matched` weighted Spearman $\rho$ from `0.8209` to **`0.8852`** (`detail`) and **`0.8778`** (`compact`).
- **Cubic ($p=3$) Hill gate and nonnegative dual-profile kernels ([v130_rev2.py](spm_rating/v130_rev2.py))**: Separated bare jack context from shared hand/role surcharges, replacing linear subtractive jack penalties and cancelling chord tables with a strictly positive, smooth cubic Hill gate.
- **Nonnegative LN coordination and positive release**: Integrated row-grid cross-column locking (`CbarV2`: `churn`, `holdage`, `xh_holdtap`, `sh_holdtap_extra`, `lockstack`, `lockanchor`, `shield`, `straddle`, `overlap`), positive release (`R_POSITIVE`), and the 4K-specific `ln_gate_4k` gate plus 4K triple-jack positive increment to curb low-Dan 4K LN overestimation.
- **Causal bounded stamina**: Replaced the subtractive recovery channel with bounded causal fatigue accumulation (`c_stam ≈ 0.16`), cutting violations across 91 synthetic physical controls from `120` in `v1.2.0-rc3` to **`0`** while raising 7K 2-second local replay mean $\rho$ from `0.3962` to **`0.5618`** (partial $\rho$ `0.2739 → 0.3780`).
- **Joint parameter audit hygiene**: Pruned all dead, zero-effect, and negative fitted weights cleanly (`78` active continuous parameters in `detail` with exact dual-mirror post-averaging; `70` active continuous parameters in `compact`), lowering maximum OOD sensitivity ratio from `85.90` to `15.12` / `12.26`.

### Reference Calibration, API, and Engineering
- **Piecewise-linear reference calibration & `detail-4k`**: Updated `4K` (`dan`), `6K` (`lv`), and `7K` (`dan_sr`) to independently frozen strictly increasing piecewise-linear maps. `detail-4k` drops the legacy power override and pairs `detail` directly with the v1.3.0 4K Dan map, lowering holdout 4K Dan MAE from `0.5737` to **`0.2413` (-58%)**.
- **Self-contained production runtime & speedups**: Removed all external research dependencies from the package and public `unittest` suite. On 64 parsed 7K charts without I/O, warm median evaluation runs in `73.9 ms` for `detail` (~12% faster) and `36.4 ms` for `compact` (~2.3× faster than RC3 `detail`).
- **Cross-platform LF identity normalization**: Normalized line endings when hashing packaged JSON identities and added `.gitattributes` to prevent `windows-latest` CRLF checkout mismatches on `detail-4k`.

> **Compatibility note**: v1.3.0 preserves CLI, Python API, fractional original-clock phase integration, and `context` / `isolated` segment contracts verbatim, while `raw_score` and `reference_score` scales are not directly interchangeable with `v1.2.0-rc3` or earlier v1.x releases.

---

## v1.2.0-rc3 (2026-10-27)

- Upgraded scoring to vectorized row-grid geometry kernels, shared-phase continuous clock extension (`clock.py`), shared hand contact accounting (`hand.py`), sustained finger role costs (`role.py`), and P/Q playtest controls.
- Consolidated package layout under `spm_rating` at the repository root with `detail`/`compact` profiles and `context`/`isolated` segment APIs.

## v1.1.1 (2026-09-16)

- Finalized initial multi-scale feature release with `spm_interface` CLI and Python bindings.

## v1.1.0 (2026-09-14) & v1.0.0 (2026-09-07)

- Initial parameter tuning and public algorithm documentation built atop the sunny-rework coordinate frame.
