# History Flow Matching

Research code for evaluating flow-matching parameterizations of geological
models in ensemble data assimilation. The experiments compare direct log
permeability, PCA, and flow-matching latent variables on the PUNQ-S3 and Egg
reservoir benchmarks.

The project prioritizes reproducibility: experiment parameters are configured
in YAML, stochastic components use explicit seeds, and reported values are
derived from provenance-bearing raw artifacts.

## Status

The repository is under active development. The cross-platform `uv`
environment, benchmark validation, numerical foundations, and a storage-safe
OPM Flow runner are implemented. M0 is verified against SPE1 with Flow 2026.04;
measured details are in `ENVIRONMENT.md`.

## Development

Python 3.11 or newer and [`uv`](https://docs.astral.sh/uv/) are required.
Setup and experiment commands will be maintained in `REPRODUCE.md` as each
milestone is validated.
