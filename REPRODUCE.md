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
