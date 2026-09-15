from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np
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
                    "checkpoint_policy": "ema_and_latest_resume_only",
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

    assert {path.name for path in output.glob("*.pt")} == {"ema.pt", "resume.pt"}
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["optimization_steps"] == 1
    assert payload["parameter_count"] > 0
