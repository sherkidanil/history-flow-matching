from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np
import pytest
import torch
import yaml


def test_repository_egg_fm_config_is_valid() -> None:
    repository = Path(__file__).parents[1]
    sys.path.insert(0, str(repository / "scripts"))
    try:
        module = importlib.import_module("m8_train_egg")
        config = module.load_fm_config(repository / "configs/egg/fm_train.yaml")
    finally:
        sys.path.pop(0)

    assert config.data.training_samples == 5000
    assert config.training.epochs == 16


def test_egg_fm_config_accepts_controlled_white_source(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    payload = yaml.safe_load((repository / "configs/egg/fm_train.yaml").read_text())
    payload["source"] = {
        "kind": "white",
        "nu": 1.5,
        "corr_len_cells_zyx": [1.0, 4.0, 8.0],
        "corr_len_scale": 1.0,
    }
    path = tmp_path / "white.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    sys.path.insert(0, str(repository / "scripts"))
    try:
        module = importlib.import_module("m8_train_egg")
        config = module.load_fm_config(path)
    finally:
        sys.path.pop(0)

    assert config.source.kind == "white"
    assert config.source.corr_len_scale == 1.0


def test_repository_source_ablation_configs_vary_only_source_measure() -> None:
    repository = Path(__file__).parents[1]
    sys.path.insert(0, str(repository / "scripts"))
    try:
        module = importlib.import_module("m8_train_egg")
        configs = {
            name: module.load_fm_config(
                repository / f"configs/ablation/egg_source_{name}.yaml"
            )
            for name in ("white", "matern", "matern_misspec")
        }
    finally:
        sys.path.pop(0)

    controlled = {
        name: {key: value for key, value in config.model_dump().items() if key != "source"}
        for name, config in configs.items()
    }
    assert controlled["white"] == controlled["matern"] == controlled["matern_misspec"]
    assert configs["white"].source.kind == "white"
    assert configs["matern"].source.corr_len_scale == 1.0
    assert configs["matern_misspec"].source.corr_len_scale == 3.0


def test_resolution_configs_keep_architecture_budgets_and_physical_lengths() -> None:
    repository = Path(__file__).parents[1]
    sys.path.insert(0, str(repository / "scripts"))
    try:
        module = importlib.import_module("m8_train_egg")
        configs = {
            (architecture, source, resolution): module.load_fm_config(
                repository
                / f"configs/ablation/egg_resolution_{architecture}_{source}_{resolution}.yaml"
            )
            for architecture in ("unet", "uno")
            for source in ("white", "matern")
            for resolution in (("coarse",) if architecture == "unet" else ("coarse", "full"))
        }
    finally:
        sys.path.pop(0)

    for architecture in ("unet", "uno"):
        white = configs[(architecture, "white", "coarse")]
        matern = configs[(architecture, "matern", "coarse")]
        assert white.training == matern.training
        assert white.model == matern.model
        assert white.data == matern.data
        assert white.source.kind == "white"
        assert matern.source.kind == "matern"
        assert matern.source.corr_len_cells_zyx == (1.0, 2.0, 4.0)
    for source in ("white", "matern"):
        coarse = configs[("uno", source, "coarse")]
        full = configs[("uno", source, "full")]
        assert coarse.training == full.training
        assert coarse.model == full.model
        assert coarse.data.shape_zyx == (7, 30, 30)
        assert full.data.shape_zyx == (7, 60, 60)
        assert tuple(2 * value for value in coarse.source.corr_len_cells_zyx[1:]) == (
            full.source.corr_len_cells_zyx[1:]
        )


def test_evaluation_budget_overrides_are_explicit_and_positive() -> None:
    repository = Path(__file__).parents[1]
    sys.path.insert(0, str(repository / "scripts"))
    try:
        training = importlib.import_module("m8_train_egg")
        evaluation = importlib.import_module("m8_evaluate_egg")
        config = training.load_fm_config(
            repository / "configs/ablation/egg_source_white.yaml"
        )
    finally:
        sys.path.pop(0)

    assert evaluation._resolve_evaluation_budget(
        config, integration_steps=10, sample_count=128
    ) == (10, 128)
    assert evaluation._resolve_evaluation_budget(
        config, integration_steps=None, sample_count=None
    ) == (50, 1000)
    with pytest.raises(ValueError, match="positive"):
        evaluation._resolve_evaluation_budget(
            config, integration_steps=0, sample_count=128
        )
    assert evaluation._checkpoint_epoch(Path("ema-epoch-0004.pt"), {}, 16) == 4
    assert evaluation._checkpoint_epoch(Path("ema.pt"), {}, 16) == 16


def test_evaluation_reference_is_pooled_to_declared_target_mask() -> None:
    repository = Path(__file__).parents[1]
    sys.path.insert(0, str(repository / "scripts"))
    try:
        evaluation = importlib.import_module("m8_evaluate_egg")
    finally:
        sys.path.pop(0)
    fields = np.asarray(
        [[[[1.0, 3.0, 10.0, 14.0], [5.0, 7.0, 18.0, 22.0]]]], dtype=np.float32
    )
    full_active = np.asarray([[[True, True, True, False], [True, True, False, False]]])
    target_active = np.asarray([[[True, True]]])

    pooled = evaluation._match_reference_resolution(
        fields, full_active=full_active, target_active=target_active
    )

    np.testing.assert_allclose(pooled, [[[[4.0, 10.0]]]])


def test_egg_training_script_enforces_budget_and_writes_checkpoint(tmp_path: Path) -> None:
    data = tmp_path / "tiny.h5"
    with h5py.File(data, "w") as handle:
        handle.create_dataset(
            "logk", data=np.random.default_rng(7).normal(size=(4, 2, 4, 4)).astype(np.float32)
        )
        handle.create_dataset("active_mask", data=np.ones((2, 4, 4), dtype=bool))
        handle.attrs["strategy"] = "augmentation"
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "seed": 7,
                "strategies": ["augmentation", "mps", "procedural"],
                "data": {
                    "field": "logk",
                    "shape_zyx": [2, 4, 4],
                    "training_samples": 4,
                    "remove_layer_trend": True,
                },
                "source": {
                    "kind": "matern",
                    "nu": 1.5,
                    "corr_len_cells_zyx": [1.0, 1.0, 1.0],
                },
                "model": {
                    "kind": "unet3d",
                    "in_channels": 1,
                    "base_channels": 2,
                    "time_dim": 8,
                    "coarse_attention_only": True,
                },
                "training": {
                    "batch_size": 4,
                    "epochs": 1,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "gradient_clip_norm": 1.0,
                    "ema_decay": 0.9,
                    "checkpoint_policy": "ema_snapshots_and_latest_resume",
                    "evaluation_epochs": [1],
                },
                "integration": {"method": "heun", "steps": 2},
                "evaluation": {
                    "reference": "official_heldout",
                    "sample_count": 4,
                    "metrics": ["marginal_ks"],
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "model"
    manifest = tmp_path / "MANIFEST.json"
    report = tmp_path / "report.json"
    repository = Path(__file__).parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/m8_train_egg.py"),
            "--config",
            str(config),
            "--strategy",
            "augmentation",
            "--data",
            str(data),
            "--output-dir",
            str(output),
            "--manifest",
            str(manifest),
            "--report",
            str(report),
            "--device",
            "cpu",
            "--git-commit",
            "0123456789abcdef",
        ],
        check=False,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    assert {path.name for path in output.glob("*.pt")} == {
        "ema.pt",
        "ema-epoch-0001.pt",
        "resume.pt",
    }
    assert torch.load(output / "ema-epoch-0001.pt", weights_only=True)["completed_epochs"] == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["optimization_steps"] == 1
    assert payload["evaluation_epochs"] == [1]
    assert payload["parameter_count"] > 0
