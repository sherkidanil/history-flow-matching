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
from fmgeo.forward.egg import EGG_PRODUCERS, extract_egg_observations
from fmgeo.inverse.egg import load_egg_inversion_config
from fmgeo.metrics.geology import connected_component_sizes


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


def _metadata(handle: h5py.File, stage: str) -> list[dict[str, object]]:
    raw = np.asarray(handle[stage]["metadata_json"])
    return [json.loads(item.decode() if isinstance(item, bytes) else str(item)) for item in raw]


def _water_cut_quantiles(
    metadata: list[dict[str, object]], producer: str
) -> tuple[np.ndarray, np.ndarray]:
    if not metadata:
        raise ValueError("water-cut metadata must contain at least one member")
    times = np.asarray(metadata[0]["forecast_times_days"], dtype=np.float64)
    curves: list[np.ndarray] = []
    for item in metadata:
        member_times = np.asarray(item["forecast_times_days"], dtype=np.float64)
        if not np.array_equal(member_times, times):
            raise ValueError("all ensemble members must share forecast times")
        water_cut = item["water_cut"]
        if not isinstance(water_cut, dict) or producer not in water_cut:
            raise ValueError(f"missing water-cut series for {producer}")
        curve = np.asarray(water_cut[producer], dtype=np.float64)
        if curve.shape != times.shape or not np.all(np.isfinite(curve)):
            raise ValueError("water-cut curves must be finite and match forecast times")
        curves.append(curve)
    return times, np.quantile(np.stack(curves), (0.1, 0.5, 0.9), axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--truth-case", type=Path, required=True)
    parser.add_argument("--realizations-dir", type=Path, required=True)
    parser.add_argument("--active-source", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--inversion", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--water-cut-output", type=Path, required=True)
    args = parser.parse_args()

    config = load_egg_inversion_config(args.config)
    evaluation = json.loads(args.evaluation_report.read_text(encoding="utf-8"))
    threshold = float(evaluation["high_permeability_threshold_logk"])
    with h5py.File(args.active_source) as handle:
        active = np.asarray(handle["active_mask"], dtype=bool)
    truth = load_egg_ensemble(args.realizations_dir)[config.truth_realization - 1]
    fields: dict[str, np.ndarray] = {"truth": truth}
    ensembles: dict[str, np.ndarray] = {"truth": truth[None]}
    metadata: dict[str, list[dict[str, object]]] = {}
    sources: dict[str, str] = {
        "truth": sha256_file(
            args.realizations_dir / f"PERM{config.truth_realization}_ECL.INC"
        )
    }
    for path in args.inversion:
        with h5py.File(path) as handle:
            method = str(handle.attrs["method"])
            final_stage = _last_stage(handle)
            fields[method] = _best_member(handle, final_stage)
            ensembles[method] = np.asarray(handle[final_stage]["logk"], dtype=np.float64)
            metadata[method] = _metadata(handle, final_stage)
            if method == "raw":
                fields["prior"] = _best_member(handle, "stage_0")
                ensembles["prior"] = np.asarray(
                    handle["stage_0"]["logk"], dtype=np.float64
                )
                metadata["prior"] = _metadata(handle, "stage_0")
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
    figure, axes = plt.subplots(5, 5, figsize=(14.5, 13), constrained_layout=True)
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
            ensembles[category][:, active].ravel(),
            bins=bins,
            density=True,
            color="#31688e",
            alpha=0.8,
        )
        axes[3, column].hist(
            truth[active], bins=bins, density=True, histtype="step", color="black", linewidth=1
        )
        axes[3, column].set_xlabel("ln permeability")
        if column == 0:
            axes[3, column].set_ylabel("Density")
        sizes = np.concatenate(
            [
                connected_component_sizes(sample >= threshold)
                for sample in ensembles[category]
            ]
        )
        sizes = np.sort(sizes)
        survival = (len(sizes) - np.arange(len(sizes))) / len(sizes)
        axes[4, column].step(sizes, survival, where="post", color="#35b779")
        axes[4, column].set_xscale("log")
        axes[4, column].set_yscale("log")
        axes[4, column].set_xlabel("Cluster size, cells")
        if column == 0:
            axes[4, column].set_ylabel("Survival probability")
    if image is not None:
        figure.colorbar(image, ax=axes[:3, :], label="ln permeability", shrink=0.8)
    figure.suptitle("Egg held-out inversion: selected fields and ensemble geology")
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

    truth_summary = extract_egg_observations(
        args.truth_case,
        history_end_day=config.history_end_day,
        observation_interval_days=config.observation_interval_days,
        oil_rate_relative_sigma=config.oil_rate_relative_sigma,
        water_rate_relative_sigma=config.water_rate_relative_sigma,
        rate_sigma_floor=config.rate_sigma_floor,
        water_breakthrough_fraction=config.water_breakthrough_fraction,
    )
    truth_metadata = truth_summary["metadata"]
    truth_times = np.asarray(truth_metadata["forecast_times_days"], dtype=np.float64)
    truth_water_cut = truth_metadata["water_cut"]
    colors = {"prior": "#7f7f7f", "raw": "#440154", "pca": "#21918c", "fm": "#fde725"}
    water_figure, water_axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, producer in zip(water_axes.flat, EGG_PRODUCERS, strict=True):
        axis.axvspan(
            config.history_end_day,
            config.forecast_end_day,
            color="#eeeeee",
            label="held-out forecast" if producer == EGG_PRODUCERS[0] else None,
        )
        for category in ("prior", "raw", "pca", "fm"):
            times, quantiles = _water_cut_quantiles(metadata[category], producer)
            axis.fill_between(times, quantiles[0], quantiles[2], color=colors[category], alpha=0.12)
            axis.plot(times, quantiles[1], color=colors[category], label=category.upper())
        axis.plot(
            truth_times,
            np.asarray(truth_water_cut[producer], dtype=np.float64),
            color="black",
            linewidth=1.5,
            label="Truth",
        )
        axis.axvline(config.history_end_day, color="black", linestyle="--", linewidth=0.8)
        axis.set(title=producer, xlabel="Time, days", ylabel="Water cut", ylim=(0.0, 1.0))
    water_figure.suptitle("Egg water-cut history and held-out forecast")
    handles, labels = water_axes.flat[0].get_legend_handles_labels()
    water_figure.legend(handles, labels, loc="outside lower center", ncol=6)
    args.water_cut_output.parent.mkdir(parents=True, exist_ok=True)
    water_figure.savefig(
        args.water_cut_output,
        format="svg",
        metadata={
            "Creator": "fmgeo m8_plot_egg_inversion.py",
            "Date": None,
            "Description": json.dumps(sources, sort_keys=True),
        },
    )
    plt.close(water_figure)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "water_cut_output": str(args.water_cut_output),
                "source_hashes": sources,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
