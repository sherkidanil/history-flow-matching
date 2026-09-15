"""Create the central Egg posterior slice and bimodality comparison figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
from m8_egg_case import load_egg_ensemble

from fmgeo.artifacts import sha256_file
from fmgeo.inverse.egg import load_egg_inversion_config


def _last_stage(handle: h5py.File) -> str:
    indices = sorted(
        int(name.removeprefix("stage_")) for name in handle if name.startswith("stage_")
    )
    if not indices:
        raise ValueError("inversion artifact contains no assimilation stages")
    return f"stage_{indices[-1]}"


def _best_member(handle: h5py.File, stage: str) -> np.ndarray:
    group = handle[stage]
    simulated = np.asarray(group["simulated_data"], dtype=np.float64)
    observation = np.asarray(handle["observation"], dtype=np.float64)
    sigma = np.asarray(handle["sigma"], dtype=np.float64)
    misfit = np.mean(((simulated - observation[None, :]) / sigma[None, :]) ** 2, axis=1)
    return np.asarray(group["logk"][int(np.argmin(misfit))], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--realizations-dir", type=Path, required=True)
    parser.add_argument("--active-source", type=Path, required=True)
    parser.add_argument("--inversion", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_egg_inversion_config(args.config)
    with h5py.File(args.active_source) as handle:
        active = np.asarray(handle["active_mask"], dtype=bool)
    truth = load_egg_ensemble(args.realizations_dir)[config.truth_realization - 1]
    fields: dict[str, np.ndarray] = {"truth": truth}
    sources: dict[str, str] = {
        "truth": sha256_file(
            args.realizations_dir / f"PERM{config.truth_realization}_ECL.INC"
        )
    }
    for path in args.inversion:
        with h5py.File(path) as handle:
            method = str(handle.attrs["method"])
            fields[method] = _best_member(handle, _last_stage(handle))
            if method == "raw":
                fields["prior"] = _best_member(handle, "stage_0")
        sources[method] = sha256_file(path)
        if method == "raw":
            sources["prior"] = sources[method]
    categories = ("truth", "prior", "raw", "pca", "fm")
    if set(fields) != set(categories):
        raise ValueError("figure requires complete truth/prior/raw/PCA/FM inputs")

    active_values = np.concatenate([fields[name][active] for name in categories])
    value_range = np.quantile(active_values, [0.005, 0.995])
    bins = np.linspace(value_range[0], value_range[1], 55)
    layers = (0, 3, 6)
    plt.rcParams["svg.hashsalt"] = "fmgeo-egg-inversion"
    figure, axes = plt.subplots(4, 5, figsize=(14.5, 10.5), constrained_layout=True)
    image = None
    for column, category in enumerate(categories):
        field = fields[category]
        for row, layer in enumerate(layers):
            shown = np.where(active[layer], field[layer], np.nan)
            image = axes[row, column].imshow(
                shown,
                origin="lower",
                cmap="viridis",
                vmin=value_range[0],
                vmax=value_range[1],
                interpolation="nearest",
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if column == 0:
                axes[row, column].set_ylabel(f"Layer {layer + 1}")
        axes[0, column].set_title(category.upper())
        axes[3, column].hist(
            field[active], bins=bins, density=True, color="#31688e", alpha=0.8
        )
        axes[3, column].hist(
            truth[active], bins=bins, density=True, histtype="step", color="black", linewidth=1
        )
        axes[3, column].set_xlabel("ln permeability")
        if column == 0:
            axes[3, column].set_ylabel("Density")
    if image is not None:
        figure.colorbar(image, ax=axes[:3, :], label="ln permeability", shrink=0.8)
    figure.suptitle("Egg held-out inversion: best-misfit ensemble members")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.output,
        format="svg",
        metadata={
            "Creator": "fmgeo m8_plot_egg_inversion.py",
            "Date": None,
            "Description": json.dumps(sources, sort_keys=True),
        },
    )
    plt.close(figure)
    print(json.dumps({"output": str(args.output), "source_hashes": sources}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
