# History Flow Matching

Research code for evaluating flow-matching parameterizations of geological
models in ensemble data assimilation. The experiments compare direct log
permeability, PCA, and flow-matching latent variables on the PUNQ-S3 and Egg
reservoir benchmarks.

The project prioritizes reproducibility: experiment parameters are configured
in YAML, stochastic components use explicit seeds, and reported values are
derived from provenance-bearing raw artifacts.

## Status

The repository is under active development. The first milestone establishes a
cross-platform `uv` environment and validates the available OPM Flow runtime.

## Development

Python 3.11 or newer and [`uv`](https://docs.astral.sh/uv/) are required.
Setup and experiment commands will be maintained in `REPRODUCE.md` as each
milestone is validated.

