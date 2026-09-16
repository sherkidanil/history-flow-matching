"""Build the controlled FM source-ablation table and convergence figure."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from m8_train_egg import EggFMConfig, load_fm_config

from fmgeo.artifacts import canonical_config_hash, sha256_file


def _loss_at_epoch(
    history: list[float], *, epoch: int, training_samples: int, batch_size: int
) -> float:
    batches = math.ceil(training_samples / batch_size)
    index = epoch * batches - 1
    if epoch < 1 or index >= len(history):
        raise ValueError("requested epoch is absent from the loss history")
    return float(history[index])


def _labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(raw_path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty ablation table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _checkpoint_record(report: dict[str, Any], checkpoint: str) -> dict[str, Any]:
    matches = [item for item in report["artifacts"] if item["path"] == checkpoint]
    if len(matches) != 1:
        raise ValueError(f"training report does not identify checkpoint {checkpoint}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-config", action="append", type=_labeled_path, required=True)
    parser.add_argument("--training-report", action="append", type=_labeled_path, required=True)
    parser.add_argument("--evaluation-report", action="append", type=Path, required=True)
    parser.add_argument("--table-output", type=Path, required=True)
    parser.add_argument("--figure-output", type=Path, required=True)
    args = parser.parse_args()

    configs: dict[str, tuple[Path, EggFMConfig]] = {
        label: (path, load_fm_config(path)) for label, path in args.variant_config
    }
    training: dict[str, tuple[Path, dict[str, Any]]] = {
        label: (path, json.loads(path.read_text(encoding="utf-8")))
        for label, path in args.training_report
    }
    if set(configs) != set(training) or len(configs) != 3:
        raise ValueError("source ablation requires the same three configs and reports")
    hash_to_label: dict[str, str] = {}
    for label, (_, config) in configs.items():
        digest = canonical_config_hash(config)
        if training[label][1]["config_hash"] != digest:
            raise ValueError(f"training report/config mismatch for {label}")
        hash_to_label[digest] = label

    rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    derived_git_commit = _git_commit()
    for report_path in args.evaluation_report:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        label = hash_to_label[str(report["training_config_hash"])]
        config = configs[label][1]
        training_path, training_report = training[label]
        epoch = int(report["checkpoint_epoch"])
        checkpoint = str(report["checkpoint"])
        checkpoint_record = _checkpoint_record(training_report, checkpoint)
        metrics = report["metrics"]
        rows.append(
            {
                "variant": label,
                "source_kind": config.source.kind,
                "corr_len_scale": config.source.corr_len_scale,
                "checkpoint_epoch": epoch,
                "optimizer_steps": epoch
                * math.ceil(config.data.training_samples / config.training.batch_size),
                "integration_steps": int(report["integration_steps"]),
                "sample_count": int(report["sample_count"]),
                "training_loss": _loss_at_epoch(
                    training_report["loss"]["history"],
                    epoch=epoch,
                    training_samples=config.data.training_samples,
                    batch_size=config.training.batch_size,
                ),
                "roundtrip_relative_error": report["roundtrip_relative_error"],
                "marginal_ks": metrics["marginal_ks"],
                "variogram_x_nrmse": metrics["variogram_x_nrmse"],
                "variogram_y_nrmse": metrics["variogram_y_nrmse"],
                "spanning_fraction_abs_error": metrics["spanning_fraction_abs_error"],
                "bimodality_abs_error": metrics["bimodality_abs_error"],
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_record["sha256"],
                "source_artifact": report["artifact"]["path"],
                "source_artifact_sha256": report["artifact"]["sha256"],
                "source_git_commit": report["artifact"]["git_commit"],
                "training_report": training_path.as_posix(),
                "training_report_sha256": sha256_file(training_path),
                "evaluation_report": report_path.as_posix(),
                "evaluation_report_sha256": sha256_file(report_path),
                "derived_git_commit": derived_git_commit,
            }
        )
        source_hashes[report_path.as_posix()] = sha256_file(report_path)
    rows.sort(
        key=lambda row: (
            str(row["variant"]),
            int(row["checkpoint_epoch"]),
            int(row["integration_steps"]),
        )
    )
    expected = {
        (label, epoch, steps)
        for label in configs
        for epoch in configs[label][1].training.evaluation_epochs
        for steps in (10, 20, 50)
    }
    actual = {
        (str(row["variant"]), int(row["checkpoint_epoch"]), int(row["integration_steps"]))
        for row in rows
    }
    if actual != expected:
        raise ValueError("evaluation reports do not form the complete controlled matrix")
    _write_csv(args.table_output, rows)

    plt.rcParams["svg.hashsalt"] = "fmgeo-source-ablation"
    figure, axes = plt.subplots(2, 4, figsize=(14, 7.5), constrained_layout=True)
    colors = {"white": "#440154", "matern": "#21918c", "matern_misspec": "#fde725"}
    for label, (_, report) in training.items():
        losses = report["loss"]["history"]
        axes[0, 0].plot(
            range(1, len(losses) + 1), losses, color=colors[label], label=label
        )
    axes[0, 0].set(title="Training loss", xlabel="Optimizer step", ylabel="FM loss", yscale="log")
    metric_axes = (
        ("roundtrip_relative_error", "Round-trip error"),
        ("marginal_ks", "Marginal KS"),
        ("variogram_x_nrmse", "Variogram X NRMSE"),
        ("variogram_y_nrmse", "Variogram Y NRMSE"),
        ("spanning_fraction_abs_error", "Spanning error"),
        ("bimodality_abs_error", "Bimodality error"),
    )
    for axis, (metric, title) in zip(axes.flat[1:7], metric_axes, strict=True):
        for label in configs:
            for steps, linestyle in ((10, ":"), (20, "--"), (50, "-")):
                selected = [
                    row
                    for row in rows
                    if row["variant"] == label and row["integration_steps"] == steps
                ]
                axis.plot(
                    [int(row["optimizer_steps"]) for row in selected],
                    [float(row[metric]) for row in selected],
                    color=colors[label],
                    linestyle=linestyle,
                    marker="o",
                    label=f"{label}, {steps} ODE",
                )
        axis.set(title=title, xlabel="Optimizer step", ylabel=title)
    axes.flat[7].axis("off")
    handles, labels = axes.flat[1].get_legend_handles_labels()
    axes.flat[7].legend(handles, labels, loc="center", fontsize="small")
    figure.suptitle("Egg FM source ablation: training and integration budgets")
    args.figure_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.figure_output,
        format="svg",
        metadata={
            "Creator": "fmgeo m9_build_source_ablation.py",
            "Date": None,
            "Description": json.dumps(source_hashes, sort_keys=True),
        },
    )
    plt.close(figure)
    print(json.dumps({"rows": len(rows), "variants": sorted(configs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
