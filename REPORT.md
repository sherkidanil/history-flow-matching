# Flow-Matching Geological Inversion — Report

This report contains only measured results and explicitly labelled assumptions.

## Blockers and assumptions

The central unresolved blocker is PUNQ-S3 forward fidelity. The available
official truth deck runs successfully in OPM Flow 2026.04 but its 6025-day
`FOPT` is 46.1131% below the published value. The PUNQ forward-coverage gate,
ES-MDA comparisons, and literature table are therefore intentionally absent;
reporting them would turn a failed benchmark validation into a false result.

The completed Egg study uses explicitly assumed choices where the benchmark
does not prescribe an inversion protocol: realization 100 as held-out truth,
the first 1,800 of 3,600 days as history, 90-day observations, independent 5%
rate error with a 1 m3/day floor, four ES-MDA assimilations with alpha 4, and
the 0.65 training-ensemble permeability quantile as the shared connectivity
threshold. These choices are fixed across compared parameterizations. The
source-ablation and resolution experiments are controlled comparisons, not a
claim that the resulting synthetic augmentation prior is unique or fully
geological.

## Environment and M0

Local preparation, analysis, tests, and publication outputs use `uv 0.12.13`,
CPython 3.11.16, and macOS 26.3.1 on arm64. Heavy training and OPM runs use a
64-CPU, 503 GiB Linux 5.15 x86_64 host with eight NVIDIA A100 80GB PCIe GPUs;
controller containers use PyTorch 2.4.1 with CUDA 12.1. All project Python
commands use the same locked `uv` environment or its declared dependencies in
the controller image.

On 2026-09-15, SPE1 completed on the Linux cluster with OPM Flow 2026.04 in the
official `openporousmedia/opmreleases:latest` image. The run produced a
1,820-byte `.SMSPEC` and a 33,840-byte `.UNSMRY`, no restart output, and a final
`FOPT` of 45,898,384 field units. Runtime was 2.218 seconds. The transient
Docker filesystem had 32,324,386,816 free bytes before the run.

## PUNQ-S3 prior and blocker

A deterministic 256-member reduced object-based ensemble passed the
pre-forward validation: all three truth sand fractions lay inside the prior
2.5–97.5% intervals, the normalized log-permeability Wasserstein distance was
0.15012 (threshold 0.5), and the truth variogram lay inside the prior envelope
for 11 of 12 tested axis/lag pairs (coverage 0.9167; threshold 0.70).

The official truth deck completes with Flow 2026.04, but at 6025 days produces
`FOPT = 2,085,424.375 m3` versus the published `3.87e6 m3`, a measured relative
error of 46.1131%. This is not accepted. The PUNQ forward-coverage gate and all
downstream PUNQ assimilation remain paused; no benchmark claim is made from
that diagnostic run.

## Egg training-data strategies

All strategies use 5,000 fields, the same ten-realization held-out split, a
772,241-parameter 3D U-Net, batch size 16, and 5,008 optimizer steps. The two
earlier completed fits had terminal/minimum losses of 0.18188/0.13227 for
augmentation and 0.27906/0.18294 for procedural generation. The MPS fit had
terminal/minimum losses of 0.29491/0.20995.

MPS used `mps_snesim_tree` built from official MPSlib commit
`a47718fc0e2c7c6f3411de429e51f1267b5d7f7c`. Generation of 5,000 full Egg
fields took 5,576.82 seconds according to the launch PID-file and final HDF5
timestamps. The artifact has shape `(5000, 7, 60, 60)`, size 295,252,077 bytes,
and SHA-256 `359acde845f756df0c332fcfcb94ca17b4392a9be322db55a9ec384bd74d0062`.

The shared 1,000-sample FM evaluation gave:

| Strategy | Round trip | KS | Vario X | Vario Y | Spanning error | Bimodality error |
|---|---:|---:|---:|---:|---:|---:|
| Augmentation | 0.004199 | 0.01691 | 0.10475 | 0.11083 | 0.04531 | 0.02598 |
| MPS | 0.000804 | 0.05351 | 0.21708 | 0.19866 | 0.04844 | 0.00248 |
| Procedural | 0.000681 | 0.16366 | 0.65002 | 0.33432 | 0.90000 | 0.03381 |

The acceptance limits were fixed only after all three first pilots and are
therefore labelled a retrospective engineering gate, not a confirmatory
pre-registration. Under that gate augmentation and MPS pass all six criteria;
procedural generation fails KS, both variograms, and spanning connectivity.
No costly procedural inversion is run.

## Egg held-out inversion

Realization 100 is held out as truth. Its restart-free OPM run completed in
approximately 32 seconds from measured file timestamps, with terminal
`FOPT = 505,246.84375 m3`. The first 1,800 of 3,600 days are assimilated at
90-day intervals using four producer WOPR and four producer WWPR series (160
observations). The 90-day cadence, 5% independent rate error, and 1 m3/day
error floor are explicitly marked assumptions in `configs/egg/inversion.yaml`.

Raw, PCA, and FM ES-MDA each completed all 500 forward models without a single
failure. PCA retained 85 components explaining 0.95286 of prior variance. The
measured final comparison is:

| Parameterization | Final normalized misfit | FOPT P10 | FOPT P50 | FOPT P90 | Covers truth | Source report | Run commit |
|---|---:|---:|---:|---:|:---:|---|---|
| Raw `ln k` | 2.00665 | 504,517.84 | 505,500.72 | 506,619.82 | yes | `results/raw/egg_inversion_augmentation_raw.json` | `ccc14b09e1323023f3f1fd7ec689c2770980b88d` |
| PCA | 1.88520 | 503,833.58 | 504,820.61 | 505,850.00 | yes | `results/raw/egg_inversion_augmentation_pca.json` | `8feec05e63599aa6b413edc332d2625828f92214` |
| FM latent | 61.92531 | 495,305.02 | 500,046.95 | 504,164.75 | no | `results/raw/egg_inversion_augmentation_fm.json` | `ccc14b09e1323023f3f1fd7ec689c2770980b88d` |

Relative to truth, the mean absolute error across 32 injector–producer
connectivity probabilities is 0.20281 for the prior, 0.13250 for raw, 0.11938
for PCA, and 0.16844 for FM. Bimodality is 0.48308 in the prior, 0.31896 after
raw inversion, 0.32432 after PCA, and 0.48133 after FM. The FM value differs
from its own prior by only 0.00175 while its final misfit remains 61.92531.
The median fraction of sand in the largest connected component similarly moves
from 0.78547 in the prior to 0.70308 for FM, compared with 0.57181 for raw and
0.47605 for PCA (truth: 0.92273). These numbers do not establish an FM
geological advantage: the apparently preserved morphology cannot be separated
from the fact that the FM ensemble barely assimilated the observations.

A stagewise diagnostic does not yet provide a genuinely equal-misfit control.
The stored raw and PCA stages closest to the final FM misfit of 61.92531 are
both stage 1, with misfits 12.09767 and 11.93388 and absolute gaps 49.82764 and
49.99143. They already fit the data much better than FM. Consequently the
existing stage grid cannot separate geological preservation from FM
under-assimilation, and no equal-misfit advantage is claimed from this control.
The complete trajectories and explicit diagnostic limitation are in
`results/tables/egg_stagewise.csv` and `results/tables/egg_matched_misfit.csv`.
The assimilation-step remedy below resolves this particular comparison gap
with new raw, PCA, and FM trajectories whose stages bracket the improved FM
misfit; it does not retroactively make the original four-step comparison fair.

The ensemble-score view gives the same ranking. Raw/PCA/FM FOPT CRPS values
are 217.80/286.25/3,518.04 m3, and their P10–P90 widths are
2,101.98/2,016.42/8,859.73 m3. Normalizing every production observation by
its declared sigma gives multivariate energy scores of 6.6000, 6.3030, and
45.7577 respectively (prior: 57.0439). PCA has the best multivariate
production score, while raw has the best scalar FOPT CRPS; FM remains much
less calibrated than either baseline.

Raw and PCA posterior median breakthrough times equal the truth at all four
producers at the 30-day simulator output resolution. FM medians equal truth at
PROD3 and PROD4, are 60 days early at PROD1, and 30 days early at PROD2. The
complete values, artifact SHA-256 hashes, and deriving commit are in
`results/tables/egg_connectivity.csv`, `egg_bimodality.csv`,
`egg_cluster_sizes.csv`, `egg_breakthrough.csv`, and
`egg_inversion_summary.csv`. The central SVG compares selected fields,
ensemble histograms, and cluster-size survival distributions; a second SVG
shows water-cut uncertainty through the held-out forecast. Deterministic vector
PDF counterparts are stored beside both SVG files.

## Parameterization subspaces

PCA retained 85 components at an ensemble size of 100 and explained 0.95286
of the prior variance. Its subspace therefore consumes most of the maximum 99
ensemble-anomaly directions available to raw ES-MDA. The near-equal final raw
and PCA misfits, 2.00665 and 1.88520, are not evidence from two independent
baselines: both updates operate in nearly the same sampled subspace. The
ambient raw and FM vectors each contain 18,553 active cells, but that dimension
does not determine the update rank; for every parameterization the ES-MDA
update lies in an anomaly subspace of rank at most `N - 1 = 99`.

For a symmetric comparison, every stage-0 regression uses rank 79, the largest
rank supported by all 80-member training folds. The available ranks are 99 for
raw/FM and 85 for PCA. At this common rank, in-sample `R²_lin` is `0.93327`
for raw, `0.95894` for PCA, and `0.78165` for FM. A deterministic five-fold
test fits on 80 members and predicts 20 held-out members. Its stage-0
`R²_oos` mean and fold standard deviation are `0.3093 ± 0.0414` for raw,
`-0.1528 ± 0.1448` for PCA, and `-0.4885 ± 0.2309` for FM. FM has the
poorest local linear transfer at the prior, but PCA is also below zero, so the
result is not an FM-only nonlinearity test. At the final stage all three values
are negative (`-2.8843`, `-1.9936`, and `-2.1813`).

A separate decoder test reaches a compatible but narrower result: over 200
fixed member pairs, FM midpoint errors have P10/P50/P90
`0.07577 / 0.08427 / 0.09332`, versus exactly zero for raw and below
`1.62e-8` for PCA. Across the four original FM updates, relative latent shifts
decline from 0.73244 to 0.45606, but field shifts are only 0.11565 to 0.09029;
the field-to-latent ratios are 0.15789–0.19798. The out-of-sample result and
decoder attenuation establish nonlinearity and compression, but do not alone
prove that either caused the final high misfit. Complete fold scores, old
in-sample values, and hashes remain in `results/tables/f2_linearity.csv` and
`results/raw/f2_linearity.json`.

## Assimilation-step remedy

Eight equal ES-MDA steps improve every parameterization without changing the
truth, observations, prior, or noise seed. The gain is highly non-uniform. The
clean CPU FM rerun falls from a four-step misfit of 61.92531 to 7.65762, an
8.09-fold reduction;
raw falls from 2.00665 to 1.57134 (21.7%), and PCA from 1.88520 to 1.48103
(21.4%). Relative to FM Na=8, sixteen steps worsen the final misfit by 2.71
times and the bounded 200-member run worsens it by 4.66 times. All runs completed
without failed simulations:

| Variant | N_a | N | Localization | Latent rank | Simulations | Failures | Final misfit | Field spread | Prior-relative field shift | Effective members |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| Raw N_a=8 | 8 | 100 | no | — | 900 | 0 | 1.57134 | 0.39222 | 0.13154 | 36.71 |
| PCA N_a=8 | 8 | 100 | no | — | 900 | 0 | 1.48103 | 0.37643 | 0.12975 | 34.20 |
| FM N_a=8 | 8 | 100 | no | — | 900 | 0 | 7.65762 | 0.36576 | 0.15206 | 29.61 |
| FM N_a=16 | 16 | 100 | no | — | 1,700 | 0 | 20.78570 | 0.32378 | 0.15426 | 22.43 |
| FM N_a=8, N=200 | 8 | 200 | no | — | 1,800 | 0 | 35.65546 | 0.55669 | 0.14054 | 66.45 |

| Variant | FOPT P10 | FOPT P50 | FOPT P90 | Covers truth | FOPT CRPS | Normalized energy | Connectivity MAE | Bimodality | Largest-component P50 |
|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| Raw N_a=8 | 504,573.80 | 505,468.00 | 506,416.41 | yes | 214.23 | 5.7630 | 0.13719 | 0.31735 | 0.59646 |
| PCA N_a=8 | 503,644.94 | 504,499.12 | 505,326.43 | yes | 487.13 | 5.8189 | 0.12563 | 0.31975 | 0.51045 |
| FM N_a=8 | 503,640.24 | 505,892.44 | 507,804.69 | yes | 469.09 | 13.1565 | 0.11281 | 0.49075 | 0.47277 |
| FM N_a=16 | 501,638.82 | 505,392.39 | 507,816.81 | yes | 686.66 | 19.7246 | 0.11906 | 0.47737 | 0.49021 |
| FM N_a=8, N=200 | 499,374.77 | 503,498.59 | 507,215.42 | yes | 1,060.93 | 21.3028 | 0.17438 | 0.48913 | 0.69378 |

The complete 33-column table, including artifact/report hashes and run commits,
is `results/tables/f2_remedies.csv`; all 53 stages are in
`results/tables/egg_stagewise_remedies.csv`.

The improved FM run permits a direct but diagnostic stage-matched geology
comparison. At misfit 7.65762 its bimodality coefficient is 0.49075. The two
bracketing raw stages have misfits 12.37065 and 5.29334 with bimodality
0.35300 and 0.32838; the PCA brackets are 12.07456 and 5.05434 with
bimodality 0.34409 and 0.32321. FM bimodality changes little over its full
trajectory (0.4838 at stage 0 and 0.4908 at stage 8), so the high value is
primarily a stable property of the trained decoder output and is not evidence
that assimilation preserved geological structure.

FM connectivity MAE is 0.11281, below the nearer lower-misfit raw/PCA stage
values 0.13781/0.13688 and final PCA's 0.12563. This association is not an
independent matched-protocol comparison: final PCA has a much lower misfit of
1.48103, and only one truth and seed are used. FM's median
largest-component fraction is 0.47277, below raw/PCA at the nearer
lower-misfit stages (0.52459/0.51195) and far below the truth 0.92273.
Intermediate ES-MDA stages are not equivalent to completed runs with fewer
assimilations, so these five rows are descriptive trajectory diagnostics,
not independent matched-protocol experiments. The exact rows and this scope
restriction are in `results/tables/egg_matched_misfit_v2.csv`.

G1 therefore establishes a decoder-stable bimodality signature and a
stage-specific connectivity value, but no robust inversion advantage. The FM
run does not close the data-fit gap: 7.65762 remains 4.87 times the raw
misfit and 5.17 times the PCA misfit after eight steps. The comparison uses
one truth, one observation-noise seed, and one assimilation seed; differences
at the third decimal place are not treated as statistically resolved. The
full trajectories are shown in
`results/figures/egg_misfit_geology_tradeoff.svg` and its deterministic vector
PDF counterpart.

The anomalous resource controls are configuration-clean. All four FM runs use
the same prior, truth, observation seed, checkpoint, and FM configuration; all
inflation schedules satisfy `sum(1/alpha_i)=1`. The first 96 members of the
200-member run match the 100-member run exactly. Only members 96--99 differ at
the final inference-batch boundary, with stage-0 field relative Frobenius
difference `5.71e-8`. Prior-relative field distance saturates near 0.15 in all
runs. In `fm_na16`, misfit oscillation after stage 9 coincides with effective
membership falling from 33.96 to 22.43 and covariance-system conditioning
rebounding, consistent with noisy updates after decoder-limited displacement.
This is an association, not a causal proof. The bounded FM N=200 control
retains 66.45 effective members (0.332 of N) and 0.797 ensemble-explained
normalized data variance, yet ends worse than N=100 at 35.65546 versus
7.65762. The 40 stagewise diagnostic rows are
in `results/tables/g2_anomaly_diagnostics.csv`, and the unresolved inference is
recorded in `BLOCKERS.md`.

## G4 bound, localization, and ensemble-size controls

G4 separates three effects: decoded-field bounds, fixed-radius localization,
and ensemble size. All controls use eight ES-MDA steps. Bounded runs clamp the
decoded active-cell `log(k)` to `[2, 11]`; the exact effective field is both
stored and passed to OPM Flow. This nonlinear, non-invertible decoder operation
is not a constrained ES-MDA update. Localized runs use the predeclared 64 m
Gaspari--Cohn radius; PCA receives the squared-loading projection of the same
cell-space taper used by Raw and FM. The bound was introduced after the first
unbounded localized Raw attempt reached
`[-67.6, 83.6]` at stage 2, causing 30 simulator failures and 39 timeouts. A
two-member representative replay with the bound succeeded; it was not a replay
of all 69 failed/timeout members. Every complete control below has zero failed
simulations.

| Variant | N | Bounds | Localization | Final misfit | Field spread | Effective members | Effective fraction | Cells clipped | FOPT covers truth |
|---|---:|:---:|:---:|---:|---:|---:|---:|---:|:---:|
| Raw unbounded | 100 | none | no | 1.57134 | 0.39222 | 36.71 | 0.367 | n/a | yes |
| PCA unbounded | 100 | none | no | 1.48103 | 0.37643 | 34.20 | 0.342 | n/a | yes |
| FM unbounded | 100 | none | no | 7.65762 | 0.36576 | 29.61 | 0.296 | n/a | yes |
| Raw bounded | 100 | [2, 11] | no | 1.57153 | 0.39222 | 36.71 | 0.367 | 0.0000% | yes |
| PCA bounded | 100 | [2, 11] | no | 1.48103 | 0.37643 | 34.20 | 0.342 | 0.0000% | yes |
| FM bounded | 100 | [2, 11] | no | 7.65762 | 0.36576 | 29.61 | 0.296 | 0.0000% | yes |
| Raw bounded + localized | 100 | [2, 11] | 64 m | 468.90823 | 0.85047 | 18.44 | 0.184 | 5.1766% | no |
| PCA bounded + localized | 100 | [2, 11] | 64 m | 64.72718 | 0.67490 | 63.41 | 0.634 | 0.0000% | yes |
| FM bounded + localized | 100 | [2, 11] | 64 m | 3613.37879 | 0.44302 | 41.58 | 0.416 | 41.9547% | no |
| Raw bounded, N=200 | 200 | [2, 11] | no | 1.67878 | 0.54360 | 85.58 | 0.428 | 0.0002% | yes |
| FM bounded, N=200 | 200 | [2, 11] | no | 35.65546 | 0.55669 | 66.45 | 0.332 | 0.0000% | yes |

At N=100 the bounded and unbounded results are numerically identical for PCA
and FM and differ by only `0.00019` in Raw misfit; no active cells are clipped.
The bounds therefore do not explain the localized deterioration. With the same
bound, localization raises final misfit from 1.57153 to 468.90823 for Raw,
from 1.48103 to 64.72718 for PCA, and from 7.65762 to 3613.37879 for FM. PCA
activates no bound, independently confirming that the fixed 64 m localization
itself can degrade the fit. The severity remains parameterization-dependent,
and the single radius is rejected only for this protocol, not localization in
general.

The N=200 comparison is likewise parameterization-dependent. Bounded Raw stays
near N=100 (1.67878 versus 1.57153), whereas bounded FM worsens from 7.65762 to
35.65546. Larger N alone is therefore insufficient to cause the anomaly, but
the single-seed result does not identify a causal FM mechanism. Full
provenance and geological metrics are
in `results/tables/g4_bounded_controls.csv`; stage trajectories and clipping
are shown in `results/figures/g4_bounded_controls.svg` and its vector PDF.

## Matérn source ablation (source-quality stage)

The controlled source comparison used the same augmentation data, U-Net,
initialization seed, 5,008 optimizer steps, batch order, and optimizer for
white, fitted-Matérn, and correlation-length-times-three Matérn sources. Each
was evaluated after epochs 4, 8, and 16 using 10, 20, and 50 Heun steps, with
128 identically seeded samples per cell of the 27-run matrix.

The result contradicts the motivating expectation. At epoch 16 and 50 ODE
steps, white noise gives KS 0.01520, X/Y variogram NRMSE 0.04036/0.01399,
spanning error 0.02969, and bimodality error 0.02053. Fitted Matérn gives
0.01908, 0.10516/0.10696, 0.05313, and 0.02651; misspecified Matérn gives
0.04717, 0.11995/0.11549, 0.03750, and 0.02870. White also leads the same
distributional metrics at epochs 4 and 8. Conversely, final raw FM losses are
0.62268 for white, 0.18419 for fitted Matérn, and 0.09440 for misspecified
Matérn. These losses are not directly comparable across source measures
because the target velocity scale changes with the source covariance; in this
experiment lower loss does not imply better generated geology.

All 27 rows, checkpoint/sample/report hashes, and the deriving commit are in
`results/tables/egg_source_ablation.csv`; the corresponding convergence and
integration-budget plot is available as both SVG and vector PDF in
`results/figures/`.

The downstream controlled inversions each completed 500/500 OPM simulations
without failure. Their final comparison is:

| Source | Final misfit | FOPT P10 | FOPT P50 | FOPT P90 | CRPS | Normalized energy score | Connectivity MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| White | 12.9160 | 502,290.84 | 504,827.62 | 506,684.66 | 499.42 | 13.0683 | 0.15219 |
| Fitted Matérn | 36.2578 | 500,855.33 | 504,138.42 | 507,658.08 | 795.16 | 22.0823 | 0.13031 |
| Misspecified Matérn | 26.1943 | 497,431.53 | 502,821.73 | 507,127.10 | 1,535.59 | 24.1837 | 0.22469 |

All three P10--P90 intervals cover the truth FOPT of 505,246.84 m3. White is
best on final data misfit, scalar FOPT CRPS, and the normalized multivariate
production energy score. Fitted Matérn is best only on the reported
connectivity MAE; its bimodality absolute error is 0.03143 versus 0.02178 for
white. The misspecified source has the best bimodality and largest-cluster
fraction errors (0.00385 and 0.18301), but the worst CRPS, energy score, and
connectivity. Thus this inversion does not rescue the fixed-resolution
Matérn hypothesis: covariance choice changes which geological property is
preserved, while white gives the strongest overall history-match and forecast
calibration in this protocol. The exact three-row comparison and deterministic
SVG/PDF are `results/tables/egg_source_inversions.csv` and
`results/figures/egg_source_inversions.*`.

## Cross-resolution transfer

The Egg augmentation ensemble was deterministically average-pooled from
7x60x60 to 7x30x30 using only active cells inside each horizontal 2x2 block.
This leaves 4,827 coarse active cells from 18,553 fine active cells. Matérn
correlation lengths were preserved in physical units: `[1,4,8]` fine-grid
cells became `[1,2,4]` coarse-grid cells and were converted back to `[1,4,8]`
for transferred sampling. Matched white/Matérn U-Net and UNO models used the
same seed, 5,008 optimizer steps, and optimizer settings within each
architecture. The evaluation matrix contains coarse-to-coarse,
coarse-to-full, and direct-full-to-full runs for both sources and both
architectures, each with 128 samples and 50 Heun steps.

For the resolution-flexible UNO, the fitted Matérn source transfers better
than white on the main distributional and spatial metrics: coarse-to-full KS
is 0.06224 versus 0.10829, while X/Y variogram NRMSE is 0.35835/0.33946 versus
0.46990/0.49205. Bimodality error also improves from 0.07632 to 0.04908.
This advantage is not universal: connectivity MAE is 0.12368 for Matérn and
0.11592 for white, and spanning error is 0.10313 versus 0.07656. The U-Net
does not show the expected Matérn transfer advantage: white coarse-to-full has
KS 0.06109 and X/Y variogram NRMSE 0.24341/0.19050, whereas Matérn gives
0.14397 and 0.52352/0.51243. Direct full-resolution training remains better
than transfer for both architectures and sources.

Thus the experiment supports the narrower claim that a physically scaled
Matérn source helps the spectral UNO transfer selected distributional and
variogram statistics, but it does not establish resolution invariance for all
geological diagnostics. The complete 12-row matrix is
`results/tables/ablation_resolution.csv`; the physical-lag variograms are in
`results/figures/ablation_resolution.svg` and its vector PDF counterpart.

## Negative results and limitations

- The motivating expectation that fitted Matérn would dominate white noise at
  fixed resolution is rejected by both the controlled U-Net sample-quality
  matrix and the overall inversion/forecast scores; fitted Matérn improves
  only the selected connectivity diagnostic.
- The original four-step FM inversion has much worse data fit and FOPT
  calibration than raw and PCA. The eight-step trajectory has decoder-stable
  bimodality and a lower connectivity MAE at its final stage, but this is a
  diagnostic comparison rather than an independent matched protocol; its
  final misfit remains nearly five times both baselines.
- More assimilation steps and a larger ensemble are not monotonically better:
  `fm_na16` and `fm_na8_n200` finish worse than `fm_na8`. Saturation and
  collapse are consistent with the late `N_a=16` oscillations. Stable Raw
  N=100 and N=200 controls rule out ensemble size alone, but the FM-specific
  N=200 mechanism is unresolved and must not be generalized.
- Fixed 64 m Gaspari--Cohn localization is a negative result: bounded Raw,
  PCA, and FM finish at misfits 468.91, 64.73, and 3613.38. PCA clips no cells,
  while Raw/FM clip 5.18%/41.95%; the bounded-unlocalized controls show that
  bounds are not the cause. This rejects that radius and implementation for
  the present setup, not localization as a class.
- Matérn improves selected UNO coarse-to-full statistics, but not connectivity
  or spanning error, and it does not improve U-Net transfer in this experiment.
- Every FM model reproduces the supplied synthetic training distribution; it
  cannot be more geologically trustworthy than that generator without new
  conditioning evidence.
- The held-out Egg comparison uses one truth and one declared observation-noise
  seed. It measures this protocol, not frequentist coverage over many truths.
- PUNQ literature values from Floris et al. (2001), Gu and Oliver (2005), and
  Gao et al. (2007) are not numerically compared because the forward gate
  failed and the regenerated prior would not match the participant ensembles
  even after the deck discrepancy was resolved.

## What next

G4 is complete. The next publication-strength step is to repeat selected
`N_a=8` Raw, PCA, and FM comparisons over multiple assimilation seeds,
observation-noise seeds, and held-out truths. This is needed to distinguish a
repeatable FM-specific failure mode from one non-monotone stochastic ES-MDA
trajectory. A localization-radius sweep may be useful only as an explicitly
exploratory or preregistered follow-up; tuning a radius against this one truth
would not establish robustness. Decoder architecture and latent-rank remedies
should likewise wait for replication. In parallel, PUNQ remains gated on
reproducing the published truth forecast with an explicit OPM-compatible
well-cutback schedule.
