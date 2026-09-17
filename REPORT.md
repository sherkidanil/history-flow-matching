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

The stage-0 in-sample truncated-SVD regression gives `R²_lin = 1.0000` for raw,
`0.98830` for PCA, and `1.0000` for FM. These values are degenerate as a
linearity diagnostic: raw and FM retain all 99 available anomaly directions,
so the same ensemble used to fit and score the regression can be interpolated.
A deterministic five-fold test instead fits on 80 members and predicts 20
held-out members. Its stage-0 `R²_oos` mean and fold standard deviation are
`0.3093 ± 0.0414` for raw, `0.3066 ± 0.0589` for PCA, and
`-0.4889 ± 0.2309` for FM. The negative FM value means that the fitted linear
map predicts held-out responses worse than the training-fold response mean;
this supports substantially poorer local linear transfer for the composed FM
map at the prior. At the final stage all three values are negative
(`-2.8843`, `-1.9936`, and `-2.1300`), so final-ensemble nonlinearity is not
specific to FM.

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
truth, observations, prior, or noise seed. The gain is highly non-uniform. FM
falls from a four-step misfit of 61.92531 to 6.62827, a 9.34-fold reduction;
raw falls from 2.00665 to 1.57134 (21.7%), and PCA from 1.88520 to 1.48103
(21.4%). Sixteen steps and a 200-member FM ensemble do not improve further.
All runs completed without failed simulations:

| Variant | N_a | N | Localization | Latent rank | Simulations | Failures | Final misfit | Field spread | Prior-relative field shift | Effective members |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| Raw N_a=8 | 8 | 100 | no | — | 900 | 0 | 1.57134 | 0.39222 | 0.13154 | 36.71 |
| PCA N_a=8 | 8 | 100 | no | — | 900 | 0 | 1.48103 | 0.37643 | 0.12975 | 34.20 |
| FM N_a=8 | 8 | 100 | no | — | 900 | 0 | 6.62827 | 0.37770 | 0.15249 | 27.83 |
| FM N_a=16 | 16 | 100 | no | — | 1,700 | 0 | 20.78570 | 0.32378 | 0.15426 | 22.43 |
| FM N_a=8, N=200 | 8 | 200 | no | — | 1,800 | 0 | 32.10739 | 0.55989 | 0.14120 | 64.63 |

| Variant | FOPT P10 | FOPT P50 | FOPT P90 | Covers truth | FOPT CRPS | Normalized energy | Connectivity MAE | Bimodality | Largest-component P50 |
|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| Raw N_a=8 | 504,573.80 | 505,468.00 | 506,416.41 | yes | 214.23 | 5.7630 | 0.13719 | 0.31735 | 0.59646 |
| PCA N_a=8 | 503,644.91 | 504,499.23 | 505,326.43 | yes | 487.13 | 5.8189 | 0.12563 | 0.31975 | 0.51045 |
| FM N_a=8 | 503,483.88 | 505,804.69 | 508,290.79 | yes | 548.09 | 10.0060 | 0.12938 | 0.49293 | 0.49019 |
| FM N_a=16 | 501,638.82 | 505,392.39 | 507,816.81 | yes | 686.66 | 19.7246 | 0.11906 | 0.47737 | 0.49021 |
| FM N_a=8, N=200 | 499,284.82 | 504,080.69 | 507,171.81 | yes | 940.47 | 22.1642 | 0.16328 | 0.49072 | 0.68506 |

The complete 33-column table, including artifact/report hashes and run commits,
is `results/tables/f2_remedies.csv`; all 53 stages are in
`results/tables/egg_stagewise_remedies.csv`.

The improved FM run finally permits a non-interpolated matched-misfit geology
comparison. At misfit 6.62827 its bimodality coefficient is 0.49293. The two
bracketing raw stages have misfits 12.37065 and 5.29334 with bimodality
0.35300 and 0.32838; the PCA brackets are 12.07456 and 5.05434 with
bimodality 0.34409 and 0.32321. FM is therefore substantially more bimodal on
both sides of the target, and this preservation can no longer be explained by
under-assimilation alone. Its connectivity MAE, 0.12938, is also lower than
the nearer lower-misfit raw/PCA values 0.13781/0.13688. The advantage is not
universal: FM's median largest-component fraction is 0.49019, below raw/PCA
at the nearer lower-misfit stages (0.52459/0.51195) and far below the truth
0.92273. The exact five comparison rows are in
`results/tables/egg_matched_misfit_v2.csv`.

Thus G1 changes the earlier negative conclusion to a metric-specific positive
result: FM preserves permeability bimodality and competitive connectivity at
comparable history-match quality, but not the largest connected component.
It also does not close the data-fit gap: 6.62827 remains 4.22 times the raw
misfit and 4.48 times the PCA misfit after eight steps. The full trajectories
are shown in `results/figures/egg_misfit_geology_tradeoff.svg` and its
deterministic vector PDF counterpart.

The anomalous resource controls are configuration-clean. All four FM runs use
the same prior, truth, observation seed, checkpoint, and FM configuration; all
inflation schedules satisfy `sum(1/alpha_i)=1`. The first 96 members of the
200-member run match the 100-member run exactly. Only members 96--99 differ at
the final inference-batch boundary, with stage-0 field relative Frobenius
difference `5.71e-8`. Prior-relative field distance saturates near 0.15 in all
runs. In `fm_na16`, misfit oscillation after stage 9 coincides with effective
membership falling from 33.96 to 22.43 and covariance-system conditioning
rebounding, consistent with noisy updates after decoder-limited displacement.
This is an association, not a causal proof. The N=200 anomaly remains open:
it retains 64.63 effective members and 0.770 ensemble-explained normalized
data variance, yet ends worse than N=100. The 40 stagewise diagnostic rows are
in `results/tables/g2_anomaly_diagnostics.csv`, and the unresolved inference is
recorded in `BLOCKERS.md`.

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
  calibration than raw and PCA. The eight-step control establishes a real
  matched-misfit advantage for bimodality and connectivity, but not for the
  largest connected component, and its final misfit remains more than four
  times both baselines.
- More assimilation steps and a larger ensemble are not monotonically better:
  `fm_na16` and `fm_na8_n200` finish worse than `fm_na8`. Saturation and
  collapse are consistent with the late `N_a=16` oscillations, but the
  200-member anomaly is unexplained and must not be generalized.
- No first-session inversion used localization despite 18,553 parameters and
  only 100 ensemble members. The raw baseline may therefore overfit spurious
  long-range correlations; its very low misfit and narrow forecast interval
  are a potentially optimistic upper benchmark until the localized control is
  complete.
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

The immediate Egg control is the predeclared `N_a=8` localization matrix: raw
and FM with the fixed 64 m Gaspari--Cohn radius, plus the unlocalized
200-member raw control and explicitly unlocalized PCA. This tests whether the
matched-misfit FM advantage survives a treatment designed to suppress
spurious long-range covariance. It must be completed before treating the
present positive result as robust.

Next, repeat selected `N_a=8` comparisons over multiple assimilation seeds,
observation-noise seeds, and held-out truths. A replicated ensemble-size study
is needed to distinguish the unexplained N=200 trajectory from a systematic
effect. Only then should decoder architecture or latent-rank remedies be
tuned. In parallel, PUNQ remains gated on reproducing the published truth
forecast with an explicit OPM-compatible well-cutback schedule. Source and
cross-resolution extensions are secondary to these inversion-validity checks.
