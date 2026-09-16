# Reproduction guide

This guide will contain the commands, inputs, expected artifacts, and measured
runtimes needed to reproduce each accepted milestone from a clean checkout.

## Current status

The cross-platform `uv` environment and M0 Flow smoke test are implemented.
On a Linux host with Docker and at least 20 GiB free on the Docker filesystem:

```bash
uv sync --frozen
git clone --depth 1 https://github.com/OPM/opm-tests.git scratch/opm-tests
docker pull openporousmedia/opmreleases:latest
uv run python scripts/m0_smoke_test.py \
  --flow-command "$PWD/scripts/flow_docker.sh" \
  --spe1-deck scratch/opm-tests/spe1/SPE1CASE1.DATA \
  --work-dir scratch/m0 \
  --disk-check-path /path/on/docker/filesystem \
  --environment-file ENVIRONMENT.md
```

The accepted cluster run used OPM tests commit
`7486607ca47722cf6be73e914bb80cd10088a508`. The smoke script removes active
restart requests, adds `FOPT` to `SUMMARY`, requires `.SMSPEC` and `.UNSMRY`,
loads `FOPT` through ResData, rejects any `.UNRST`/`.X####`, and records
measured sizes and runtime.

## Egg training data

Augmentation and procedural datasets use the same 5,000-sample budget and the
same ten-realization held-out split:

```bash
uv run python scripts/m8_egg_case.py \
  --config configs/egg/augmentation.yaml \
  --realizations-dir data/egg/Egg_Model_Data_Files_v2/Permeability_Realizations \
  --actnum data/egg/Egg_Model_Data_Files_v2/Eclipse/ACTIVE.INC \
  --output artifacts/egg_augmentation_5000.h5 \
  --manifest artifacts/MANIFEST.json \
  --report results/raw/egg_augmentation_5000.json

uv run python scripts/m8_egg_case.py \
  --config configs/egg/procedural.yaml \
  --realizations-dir data/egg/Egg_Model_Data_Files_v2/Permeability_Realizations \
  --actnum data/egg/Egg_Model_Data_Files_v2/Eclipse/ACTIVE.INC \
  --output artifacts/egg_procedural_5000.h5 \
  --manifest artifacts/MANIFEST.json \
  --report results/raw/egg_procedural_5000.json
```

MPS runs only on Linux with a source-built executable. The commit is pinned so
the broken wheel binary is never accepted accidentally:

```bash
git clone https://github.com/ergosimulation/mpslib.git scratch/mpslib
git -C scratch/mpslib checkout a47718fc0e2c7c6f3411de429e51f1267b5d7f7c
make -C scratch/mpslib

PYTHONPATH=src uv run --no-project \
  --with 'numpy<2' --with 'scipy<2' --with 'pydantic<3' \
  --with pyyaml --with h5py --with scikit-mps \
  python scripts/m8_egg_case.py \
  --config configs/egg/mps.yaml \
  --realizations-dir data/egg/Egg_Model_Data_Files_v2/Permeability_Realizations \
  --actnum data/egg/Egg_Model_Data_Files_v2/Eclipse/ACTIVE.INC \
  --mpslib-executable-dir scratch/mpslib \
  --git-commit "$(git rev-parse HEAD)" \
  --output artifacts/egg_mps_5000.h5 \
  --manifest artifacts/MANIFEST.json \
  --report results/raw/egg_mps_5000.json
```

## Egg flow models and generator gate

Each strategy is trained with exactly the same tracked configuration. Replace
`STRATEGY` and `DATA` together with one of `augmentation` / `mps` /
`procedural` and its corresponding `artifacts/egg_*_5000.h5` file:

```bash
uv run python scripts/m8_train_egg.py \
  --config configs/egg/fm_train.yaml \
  --strategy STRATEGY \
  --data DATA \
  --output-dir artifacts/egg_models/STRATEGY \
  --manifest artifacts/STRATEGY_training_manifest.json \
  --report results/raw/egg_STRATEGY_training.json \
  --device auto

uv run python scripts/m8_evaluate_egg.py \
  --config configs/egg/fm_train.yaml \
  --strategy-config configs/egg/STRATEGY.yaml \
  --metric-config configs/egg/evaluation.yaml \
  --checkpoint artifacts/egg_models/STRATEGY/ema.pt \
  --training-data DATA \
  --realizations-dir data/egg/Egg_Model_Data_Files_v2/Permeability_Realizations \
  --output artifacts/egg_fm_samples_STRATEGY_1000.h5 \
  --manifest artifacts/STRATEGY_evaluation_manifest.json \
  --report results/raw/egg_STRATEGY_evaluation.json \
  --device auto
```

The retrospective generator table is always rebuilt from all three immutable
raw reports; no values are entered by hand:

```bash
uv run python scripts/m8_build_egg_generator_table.py \
  --config configs/egg/generator_acceptance.yaml \
  --report results/raw/egg_augmentation_evaluation.json \
  --report results/raw/egg_mps_evaluation.json \
  --report results/raw/egg_procedural_evaluation.json \
  --output results/tables/egg_generator_validation.csv
```

## Egg held-out inversion

Prepare and run realization 100 once to materialize the compact truth summary:

```bash
uv run python scripts/m8_prepare_egg.py \
  --eclipse-dir data/egg/Egg_Model_Data_Files_v2/Eclipse \
  --permeability data/egg/Egg_Model_Data_Files_v2/Permeability_Realizations/PERM100_ECL.INC \
  --output-dir scratch/egg_truth_perm100

(cd scratch/egg_truth_perm100 && ../../scripts/flow_docker.sh EGG.DATA)
```

`scripts/m8_invert_egg.py` runs one complete raw, PCA, or FM comparison and
persists every stage in one HDF5 file. It requires a Linux CUDA environment for
`--method fm`; raw and PCA are CPU-only. All methods use the same arguments
below. FM additionally requires `--fm-config`, `--checkpoint`, and
`--training-data`.

```bash
uv run python scripts/m8_invert_egg.py \
  --method METHOD \
  --strategy augmentation \
  --config configs/egg/inversion.yaml \
  --prior-fields artifacts/egg_fm_samples_augmentation_1000.h5 \
  --truth-case scratch/egg_truth_perm100/EGG \
  --template-dir scratch/egg_truth_perm100 \
  --flow-command scripts/flow_docker.sh \
  --simulator-id 'OPM Flow 2026.04 / openporousmedia/opmreleases:latest' \
  --work-root /path/on/docker/filesystem/fmgeo/egg_work \
  --cache-dir scratch/egg_cache \
  --output artifacts/egg_inversion_augmentation_METHOD.h5 \
  --manifest artifacts/augmentation_METHOD_inversion_manifest.json \
  --report results/raw/egg_inversion_augmentation_METHOD.json \
  --device auto
```

The work root must be on a filesystem with at least the configured 20 GiB
reserve. The cluster run mounted that path and the repository at identical host
and controller-container paths so nested storage-safe OPM containers could bind
their isolated cases correctly.

After all three methods finish, derive the required publication tables and SVG
directly from the immutable HDF5/JSON outputs:

```bash
uv run python scripts/m8_build_egg_inversion_tables.py \
  --config configs/egg/inversion.yaml \
  --deck data/egg/Egg_Model_Data_Files_v2/Eclipse/Egg_Model_ECL.DATA \
  --truth-case scratch/egg_truth_perm100/EGG \
  --realizations-dir data/egg/Egg_Model_Data_Files_v2/Permeability_Realizations \
  --active-source artifacts/egg_augmentation_5000.h5 \
  --evaluation-report results/raw/egg_augmentation_evaluation.json \
  --inversion artifacts/egg_inversion_augmentation_raw.h5 \
  --inversion artifacts/egg_inversion_augmentation_pca.h5 \
  --inversion artifacts/egg_inversion_augmentation_fm.h5 \
  --inversion-report results/raw/egg_inversion_augmentation_raw.json \
  --inversion-report results/raw/egg_inversion_augmentation_pca.json \
  --inversion-report results/raw/egg_inversion_augmentation_fm.json \
  --connectivity-output results/tables/egg_connectivity.csv \
  --bimodality-output results/tables/egg_bimodality.csv \
  --cluster-output results/tables/egg_cluster_sizes.csv \
  --breakthrough-output results/tables/egg_breakthrough.csv \
  --summary-output results/tables/egg_inversion_summary.csv

uv run python scripts/m8_plot_egg_inversion.py \
  --config configs/egg/inversion.yaml \
  --truth-case scratch/egg_truth_perm100/EGG \
  --realizations-dir data/egg/Egg_Model_Data_Files_v2/Permeability_Realizations \
  --active-source artifacts/egg_augmentation_5000.h5 \
  --evaluation-report results/raw/egg_augmentation_evaluation.json \
  --inversion artifacts/egg_inversion_augmentation_raw.h5 \
  --inversion artifacts/egg_inversion_augmentation_pca.h5 \
  --inversion artifacts/egg_inversion_augmentation_fm.h5 \
  --output results/figures/egg_posterior_comparison.svg \
  --water-cut-output results/figures/egg_water_cut_forecast.svg
```

## Matérn source ablation

Train the three strictly controlled source variants. Each run retains only EMA
snapshots at epochs 4, 8, and 16, the final EMA, and one resume checkpoint:

```bash
for VARIANT in white matern matern_misspec; do
  uv run python scripts/m8_train_egg.py \
    --config "configs/ablation/egg_source_${VARIANT}.yaml" \
    --strategy augmentation \
    --data artifacts/egg_augmentation_5000.h5 \
    --output-dir "artifacts/egg_models/ablation_${VARIANT}" \
    --manifest "artifacts/ablation_${VARIANT}_training_manifest.json" \
    --report "results/raw/egg_ablation_${VARIANT}_training.json" \
    --device auto
done
```

Evaluate the complete epoch × ODE-step matrix with the same 128-sample budget:

```bash
for VARIANT in white matern matern_misspec; do
  for EPOCH in 4 8 16; do
    PADDED_EPOCH=$(printf '%04d' "$EPOCH")
    for STEPS in 10 20 50; do
      uv run python scripts/m8_evaluate_egg.py \
        --config "configs/ablation/egg_source_${VARIANT}.yaml" \
        --strategy-config configs/egg/augmentation.yaml \
        --metric-config configs/egg/evaluation.yaml \
        --checkpoint "artifacts/egg_models/ablation_${VARIANT}/ema-epoch-${PADDED_EPOCH}.pt" \
        --training-data artifacts/egg_augmentation_5000.h5 \
        --realizations-dir data/egg/Egg_Model_Data_Files_v2/Permeability_Realizations \
        --output "artifacts/egg_ablation_samples_${VARIANT}_e${EPOCH}_s${STEPS}.h5" \
        --manifest "artifacts/ablation_${VARIANT}_evaluation_manifest.json" \
        --report "results/raw/egg_ablation_${VARIANT}_e${EPOCH}_s${STEPS}.json" \
        --integration-steps "$STEPS" --sample-count 128 --device auto
    done
  done
done
```

The table builder refuses incomplete 27-run matrices:

```bash
EVALUATION_ARGS=()
for REPORT in results/raw/egg_ablation_*_e*_s*.json; do
  EVALUATION_ARGS+=(--evaluation-report "$REPORT")
done
uv run python scripts/m9_build_source_ablation.py \
  --variant-config white=configs/ablation/egg_source_white.yaml \
  --variant-config matern=configs/ablation/egg_source_matern.yaml \
  --variant-config matern_misspec=configs/ablation/egg_source_matern_misspec.yaml \
  --training-report white=results/raw/egg_ablation_white_training.json \
  --training-report matern=results/raw/egg_ablation_matern_training.json \
  --training-report matern_misspec=results/raw/egg_ablation_matern_misspec_training.json \
  "${EVALUATION_ARGS[@]}" \
  --table-output results/tables/egg_source_ablation.csv \
  --figure-output results/figures/egg_source_ablation.svg
```

Run the three final source-inversion comparisons with the same held-out truth,
initial physical ensemble, ES-MDA schedule, and simulator budget:

```bash
for VARIANT in white matern matern_misspec; do
  uv run python scripts/m8_invert_egg.py \
    --method fm --parameterization-label "$VARIANT" --strategy augmentation \
    --config configs/egg/inversion.yaml \
    --fm-config "configs/ablation/egg_source_${VARIANT}.yaml" \
    --checkpoint "artifacts/egg_models/ablation_${VARIANT}/ema-epoch-0016.pt" \
    --training-data artifacts/egg_augmentation_5000.h5 \
    --prior-fields artifacts/egg_fm_samples_augmentation_1000.h5 \
    --truth-case scratch/egg_truth_perm100/EGG \
    --template-dir scratch/egg_truth_perm100 \
    --flow-command scripts/flow_docker.sh \
    --simulator-id 'OPM Flow 2026.04 / openporousmedia/opmreleases:latest' \
    --work-root "/path/on/docker/filesystem/fmgeo/egg_work_ablation_${VARIANT}" \
    --cache-dir scratch/egg_cache \
    --output "artifacts/egg_inversion_ablation_${VARIANT}_fm.h5" \
    --manifest "artifacts/ablation_${VARIANT}_inversion_manifest.json" \
    --report "results/raw/egg_inversion_ablation_${VARIANT}_fm.json" \
    --device auto
done
```

Build the provenance-complete source-inversion comparison:

```bash
uv run python scripts/m9_build_source_inversions.py \
  --config configs/egg/inversion.yaml \
  --deck data/egg/Egg_Model_Data_Files_v2/Eclipse/Egg_Model_ECL.DATA \
  --realizations-dir data/egg/Egg_Model_Data_Files_v2/Permeability_Realizations \
  --active-source artifacts/egg_fm_samples_augmentation_1000.h5 \
  --evaluation-report results/raw/egg_evaluation_augmentation.json \
  --inversion white=artifacts/egg_inversion_ablation_white_fm.h5 \
  --inversion matern=artifacts/egg_inversion_ablation_matern_fm.h5 \
  --inversion matern_misspec=artifacts/egg_inversion_ablation_matern_misspec_fm.h5 \
  --inversion-report white=results/raw/egg_inversion_ablation_white_fm.json \
  --inversion-report matern=results/raw/egg_inversion_ablation_matern_fm.json \
  --inversion-report matern_misspec=results/raw/egg_inversion_ablation_matern_misspec_fm.json \
  --table-output results/tables/egg_source_inversions.csv \
  --figure-output results/figures/egg_source_inversions.svg
```
