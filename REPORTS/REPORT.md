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

The geological diagnostics expose the central trade-off. Relative to truth,
the mean absolute error across 32 injector–producer connectivity probabilities
is 0.20281 for the prior, 0.13250 for raw, 0.11938 for PCA, and 0.16844 for FM.
Raw and PCA therefore recover connectivity better in this run, but blur the
permeability distribution: bimodality falls from 0.48308 in the prior to
0.31896 and 0.32432. FM retains 0.48133, close to both the prior and the truth
value 0.45291. The median fraction of sand belonging to the largest connected
component is 0.78547 before inversion, 0.57181 for raw, 0.47605 for PCA,
0.70308 for FM, and 0.92273 in the held-out truth. Thus FM preserves the prior
geological morphology better, but that preservation comes with substantially
worse history matching and a biased-low FOPT forecast in this pilot.

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
- The original augmentation FM inversion has much worse data fit and FOPT
  calibration than raw and PCA despite preserving bimodality and connected
  morphology better.
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

First resolve the PUNQ schedule/aquifer discrepancy against a trustworthy deck
and rerun the coverage gate. For Egg, repeat held-out inversion across several
truths and noise seeds, then investigate why the fitted Matérn U-Net has lower
training loss but worse fixed-grid samples. For resolution transfer, tune UNO
capacity and spectral modes under a validation-only budget and test whether
the Matérn advantage persists for connectivity-aware objectives.
