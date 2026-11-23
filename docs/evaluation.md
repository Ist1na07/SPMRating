# Evaluation and model tradeoffs

[简体中文](evaluation.zh-CN.md) · [Usage](../README.md) · [Model](model.md) · [Full Mechanism](mechanism.md)

## Evaluation datasets and protocol

SPM Rating is evaluated across four complementary datasets covering community difficulty labels, fine-grained player replay errors, external BMS difficulty tables, and synthetic physical stress tests:

1. **Multi-key community difficulty panels (development & family-isolated OOF)**:
   - **7K curated reference panel (~393 charts)**: Broad-spectrum 7K charts spanning rice/tap, chordstream/HB, and long-note patterns (`dan_sr` unit), plus a supplementary high-range 7K reference panel (~292 charts).
   - **4K Dan panels (1,210 charts across two distinct distributions)**: A wide-span 4K Dan reference panel (~741 charts covering beginner through high-Dan grades, including 102 LN charts) paired with a supplemental high-Dan 4K cohort (~469 charts, of which **95.3%** lie tightly inside `13.8–14.2 Dan`). Because an unstratified rank loss over `1,210` charts lets that narrow 0.4-Dan band absorb over half of the Spearman gradient, v1.3.0 decouples 4K training (primary ordering on the wide-span panel plus auxiliary Dan error regularization on the tight high-Dan cohort), fits 4K Dan calibration exclusively on the wide-span panel, and reports all three cohorts separately (`4K wide`, `4K tight`, `4K pooled`).
   - **6K reference panel (~2,343 charts, `lv` unit)** and **LN40 subset**.
   - All cross-validation (**OOF**) metrics strictly isolate chart families across 5 folds; **Full** metrics evaluate the shipped fit on the development collections. Neither is an out-of-domain benchmark.
2. **Disjoint held-out replay consensus and local replay windows**:
   - Constructed from real `CL`/Lazer replays on charts with zero chart-ID, MD5, or chart-family overlap against the development set: **282** charts (154 7K, 99 4K, 29 6K) for cross-player global consensus (`B1` ridge-regularized and `B2` truncated accuracy models) and paired speed-rate transfer (`1.5× DT`, `0.75× HT`), plus **150** held-out 7K charts (**11,442** 2-second windows; **148** charts / **20,210** 1-second windows) for fine-grained local error prediction and partial Spearman $\rho$ controlling for NPS and chord counts.
3. **External BMS 8-table benchmark**: **1,099** paired charts (**1,080** strictly `version_matched` rows) spanning `Normal`, `Normal2`, `Insane`, `Insane2`, `Starlight`, `Satellite`, `Stella`, and `Overjoy`.
4. **External 4K holdout panel (139 records, 124 chart families)** & **Synthetic control / OOD checks**:
   - Family-disjoint 4K holdout covering `jack` (46), `speed` (22), `stamina` (38), and `tech` (33), evaluated at `1.0×` using the frozen development calibration without test-set refitting.
   - **91 synthetic control conditions** verifying monotonicity (chord size, speed rate, density, non-negative long-note interactions) plus single-parameter finite-difference OOD leverage-to-curvature audits over 693 development charts and 91 OOD conditions.

Spearman $\rho$ measures rank agreement (ties receive average ranks), not raw error magnitude. External tracks were evaluated once after freezing the released parameters and calibrations without tuning on external data *(note: an earlier internal build had been checked once on the same external benchmark suite prior to finalizing the row-level Sigmoid aggregator)*. Uncertainty intervals below are paired chart-level percentile bootstrap 95% intervals (2,000 replicates) comparing each v1.3.0 profile directly against `v1.2.0-rc3`.

## External benchmark results

| Measure | RC3 `detail` / `compact` | v1.3.0 `detail` / `compact` | v1.3.0 Δ vs. RC3 [95% CI] (`detail` / `compact`) |
|---|---|---|---|
| **BMS `version_matched` weighted ρ** (1,080 rows) | 0.8209 / 0.8180 | **0.8852** / **0.8778** | **+0.0643 [+0.0487, +0.0816]** / **+0.0598 [+0.0431, +0.0783]** |
| **BMS `version_matched` 8-table average ρ** | 0.8045 / 0.8018 | **0.8675** / **0.8592** | **+0.0630** / **+0.0574** |
| **BMS `all` weighted ρ** (1,099 rows / 1,083 charts) | 0.8155 / 0.8126 | **0.8798** / **0.8725** | **+0.0643 [+0.0495, +0.0806]** / **+0.0599 [+0.0439, +0.0769]** |
| **7K 2 s local replay mean ρ** (150 ch, 11,442 win) | 0.3962 / 0.3919 | **0.5618** / **0.5526** | **+0.1655 [+0.1371, +0.1960]** / **+0.1608 [+0.1333, +0.1902]** |
| 7K 2 s local replay **median ρ** | 0.4656 / 0.4535 | **0.6273** / **0.6142** | Pure-tap median: 0.491 → **0.706 / 0.715** |
| **7K 2 s partial ρ** (controlling NPS & chords) | 0.2739 / 0.2701 | **0.3780** / **0.3751** | **+0.1041 [+0.0793, +0.1294]** / **+0.1050 [+0.0808, +0.1300]** |
| **7K 1 s local replay mean ρ** (148 ch, 20,210 win) | 0.3622 / 0.3593 | **0.5272** / **0.5205** | **+0.1650 [+0.1354, +0.1956]** / **+0.1612 [+0.1325, +0.1902]** |
| **7K 1 s partial ρ** (146 ch) | 0.2676 / 0.2646 | **0.3601** / **0.3588** | **+0.0925 [+0.0720, +0.1150]** / **+0.0942 [+0.0737, +0.1168]** |
| **7K global consensus `B1` ρ** (154 ch) | **0.6314** / **0.6333** | 0.6230 / 0.6233 | −0.0084 [−0.0242, +0.0069] (pure-tap: 0.659 → **0.693**) |
| **7K global consensus `B2` ρ** (154 ch) | **0.8697** / **0.8708** | 0.8688 / 0.8682 | −0.0009 [−0.0112, +0.0092] (pure-tap **0.907**, heavy-LN **0.933**) |
| 7K paired rate **`1.5× DT`** Δρ (79 ch) | 0.3097 / 0.3076 | **0.3380** / **0.3348** | +0.0283 [−0.0072, +0.0735] / +0.0272 [−0.0110, +0.0746] |
| 7K paired rate **`0.75× HT`** Δρ (44 ch) | **0.4337** / **0.4351** | 0.4178 / 0.4104 | −0.0159 [−0.1025, +0.0715] / −0.0247 [−0.0909, +0.0371] |
| **4K external holdout raw rank ρ** (139 rec / 124 fam) | **0.4947** / **0.4913** | 0.4725 / 0.4539 | −0.0222 [−0.0998, +0.0465] / −0.0374 [−0.1112, +0.0323] |
| **4K external holdout frozen Dan MAE** (lower is better) | 0.5737 / 0.5810 | **0.2413** / **0.2869** | **−0.3324 (−58%) [−0.3828, −0.2771]** / **−0.2941 [−0.3418, −0.2457]** |

### BMS 8-table breakdown (`version_matched`, 1,080 rows)

Mass-weighted row Sigmoid aggregation eliminates rigid time-window power distortion (`mass_pow ≈ 0.95–0.98`), yielding strong improvements across entry, mid-tier, and high-tier BMS tables while retaining a small deficit on `Stella`:

| Table | `n` | RC3 `detail` | v1.3.0 `detail` | v1.3.0 `compact` | `detail` Δ vs. RC3 [95% CI] |
|---|---:|---:|---:|---:|---|
| **Starlight** | 181 | 0.6810 | **0.8389** | **0.8328** | **+0.1579 [+0.1041, +0.2270]** |
| **Normal2** | 171 | 0.7438 | **0.8715** | **0.8623** | **+0.1277 [+0.0831, +0.1795]** |
| **Normal** | 60 | 0.7638 | **0.8870** | **0.8808** | **+0.1232 [+0.0485, +0.2271]** |
| **Insane** | 164 | 0.8470 | **0.8866** | **0.8858** | **+0.0396 [+0.0020, +0.0868]** |
| **Insane2** | 119 | 0.9042 | **0.9349** | **0.9330** | **+0.0307 [+0.0129, +0.0544]** |
| **Satellite** | 176 | 0.9306 | **0.9465** | **0.9441** | **+0.0159 [+0.0045, +0.0290]** |
| **Overjoy** | 43 | 0.6497 | **0.6726** | **0.6520** | +0.0230 [−0.0321, +0.0862] |
| **Stella** | 166 | **0.9161** | 0.9021 | 0.8825 | −0.0140 [−0.0327, +0.0028] (`compact` −0.0319 [−0.0583, −0.0095]) |

### 4K holdout subsets & sparse replay panels

- **4K external holdout (139 records)**: All four pattern classes reduce Dan MAE sharply (`jack` 0.535 → **0.180**, `speed` 0.637 → **0.298**, `stamina` 0.367 → **0.197**, `tech` 0.824 → **0.340**; overall median bias **−0.021 Dan** vs. RC3 **−0.540 Dan**). Native ordering improves substantially on `jack` (`0.5165 → 0.6079` `detail` / `0.6320` `compact`), crosses zero on `speed` (`0.6359 → 0.6113`) and `tech` (`0.4515 → 0.3988`), and trails RC3 on **`stamina`** (38 records: `0.6611 → 0.5368`, `[−0.2691, −0.0026]`).
- **4K & 6K replay consensus**: On 99 held-out 4K charts (sparse bipartite links), `B1` is `0.0892 → 0.0467` (`[−0.0798, −0.0107]`) and `B2` is `0.4801 → 0.4659` (`[−0.0499, +0.0152]`). On 29 held-out 6K charts, `B1` is `0.3158 → 0.3143` and `B2` is `0.4084 → 0.3877` (`1.5× DT` on 15 charts: `0.5679 → 0.5857` `detail` / `0.6214` `compact`).
- **Residual fatigue in long charts**: Even with causal stamina (`c_stam ≈ 0.16`), local error regressions on 7K 2 s windows retain a positive elapsed-time slope (`t ≈ 7.68` vs. `8.17` in RC3), reflecting unmodeled late-chart accuracy breakdown under sustained extreme load.

## Development panels, physical controls, and runtime

The composite development score balances $0.5\,\rho_{7K} + (5/18)\,\rho_{4K} + (2/9)\,\rho_{\mathrm{Apo}}$. The historical composite `H` uses the wide-span 4K panel; `D_pooled` replaces 4K with all 1,210 4K charts (including the 95.3%-crowded 13.8–14.2 Dan cohort).

| Measure (OOF / Full fit) | RC3 `detail` | v1.3.0 `detail` (OOF / Full) | v1.3.0 `compact` (OOF / Full) |
|---|---:|---:|---:|
| **`H` weighted ρ** (wide-span 4K panel) | **0.9883** | 0.9811 / **0.9849** | 0.9804 / **0.9848** |
| **`D_clean` weighted ρ** (7K + wide-span 4K D + Apo, Full) | — | **0.9853** | **0.9851** |
| **`D_pooled` weighted ρ** (with 469 tight-band 4K) | **0.9740** | 0.9665 / **0.9717** | 0.9659 / **0.9712** |
| **7K reference panel (393) ρ / inversions** | **0.9888** / 110 | 0.9797 / **0.9849** (149) | 0.9781 / **0.9848** (150) |
| **7K `dan_sr` MAE** (OOF / Full) | — | **0.273 / 0.240** | **0.291 / 0.247** |
| **4K wide-span panel (741) unweighted / weighted ρ** | **0.9883 / 0.9887** | 0.9841 / **0.9855** (`0.9861`) | 0.9836 / **0.9850** (`0.9855`) |
| **4K tight-band cohort (469) ρ / Dan MAE** | — | 0.525 (0.271) / **0.564 (0.213)** | 0.535 (0.253) / **0.548 (0.256)** |
| **4K pooled (1,210) ρ** | **0.9376** | 0.9308 / **0.9373** | 0.9304 / **0.9358** |
| **4K wide-span LN subset (102) Dan MAE** | 0.736 | **0.704** (bias −0.001) | **0.664** (bias −0.056) |
| **Apo high-range panel (292) ρ** | **0.9861** | 0.9805 / **0.9842** | 0.9816 / **0.9846** |
| **6K panel (2,343) ρ / Lv MAE** | — | **0.9606 / 0.798** (Full **0.9632 / 0.775**) | **0.9594 / 0.799** (Full **0.9626 / 0.782**) |
| **`LN40` / 7K 2 s Dev ρ** | 0.9047 / 0.4211 | **0.9071** / **0.5798** | 0.8974 / **0.5753** |
| **Violations on 91 synthetic conditions** | 120 | **0** | **0** |
| **Active fitted continuous coords (empirical)** | 206 (196) | **78 (78)** | **70 (70)** |
| **Dead / zero-effect / negative fitted weights** | 10 / 0 / multiple | **0 / 0 / 0** | **0 / 0 / 0** |
| **OOD leverage-to-curvature ratio (med / max)** | 1.880 / 85.902 | **1.310 / 15.124** | **1.271 / 12.255** |
| **Aggregation-origin phase spread (raw)** | 0.1689 | **0.0** | **0.0** |
| **Parsed-note latency (64 7K charts median)** | 84.29 ms (1.00×) | **73.93 ms (0.88×)** | **36.43 ms (0.43×, ~2.3× faster)** |

### Known limitations and tradeoffs

Compared with RC3 (which relied on 206 active continuous coordinates, multiple negative compensating channel weights, and subtractive recovery terms that triggered 120 synthetic stress violations and `85.902` max OOD leverage-to-curvature danger), v1.3.0 trades slightly lower middle-band ordering on the 393-chart 7K panel (`0.9849` vs. `0.9888`, primarily in `5–9` mid-range charts where v1.3.0 simultaneously improves extreme `<5` band ordering `0.664 → 0.761` and reduces `9–11` MAE `0.277 → 0.176`), `4K holdout stamina` ordering, and `BMS Stella` in exchange for **zero synthetic violations**, **~6× lower OOD instability**, **+0.064 BMS weighted ρ**, **+0.166 7K 2 s local replay ρ**, **−58% 4K Dan MAE**, and **exact `0.0` aggregation-origin invariance**.

Structured v1.3.0 metrics together with the historical RC3 development reference tables are stored in [benchmark.json](benchmark.json).
