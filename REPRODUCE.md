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
  --output artifacts/egg_mps_5000.h5 \
  --manifest artifacts/MANIFEST.json \
  --report results/raw/egg_mps_5000.json
```
