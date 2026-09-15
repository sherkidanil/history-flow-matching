from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from fmgeo.param.flowmatching.model_unet3d import (
    SelfAttention3D,
    SinusoidalTimeEmbedding,
    UNet3D,
)
from fmgeo.param.flowmatching.model_uno3d import UNO3D
from fmgeo.param.flowmatching.sample import FlowTransform, integrate_ode
from fmgeo.param.flowmatching.train import (
    LayerTrendNormalizer,
    MaternSourceSampler,
    masked_flow_matching_loss,
    save_checkpoint_policy,
    seeded_batch_indices,
)


class ZeroVelocity(nn.Module):
    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del time
        return torch.zeros_like(x)


class LinearVelocity(nn.Module):
    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del time
        return 0.25 * x


class GlobalMeanVelocity(nn.Module):
    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del time
        return torch.ones_like(x) * x.mean()


def test_masked_loss_ignores_inactive_and_padded_cells() -> None:
    model = ZeroVelocity()
    x0 = torch.zeros(1, 1, 2, 2, 2)
    x1 = torch.ones_like(x0)
    x1[..., 0, 0, 0] = 100.0
    mask = torch.ones(1, 1, 2, 2, 2, dtype=torch.bool)
    mask[..., 0, 0, 0] = False

    loss = masked_flow_matching_loss(
        model, x0=x0, x1=x1, time=torch.tensor([0.5]), mask=mask
    )

    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_inactive_values_cannot_leak_into_active_velocity() -> None:
    model = GlobalMeanVelocity()
    x0 = torch.zeros(1, 1, 2, 2, 2)
    clean = torch.ones_like(x0)
    noisy = clean.clone()
    noisy[..., 0, 0, 0] = 100.0
    mask = torch.ones_like(x0, dtype=torch.bool)
    mask[..., 0, 0, 0] = False
    time = torch.tensor([0.5])

    clean_loss = masked_flow_matching_loss(model, x0=x0, x1=clean, time=time, mask=mask)
    noisy_loss = masked_flow_matching_loss(model, x0=x0, x1=noisy, time=time, mask=mask)

    torch.testing.assert_close(noisy_loss, clean_loss)


def test_time_embedding_and_models_preserve_punq_shape() -> None:
    embedding = SinusoidalTimeEmbedding(16)(torch.tensor([0.0, 0.5]))
    assert embedding.shape == (2, 16)
    x = torch.randn(2, 1, 5, 28, 19)
    time = torch.tensor([0.2, 0.8])

    unet = UNet3D(in_channels=1, base_channels=4, time_dim=16)
    uno = UNO3D(in_channels=1, hidden_channels=4, time_dim=16, modes=(3, 4, 4))

    assert unet(x, time).shape == x.shape
    assert uno(x, time).shape == x.shape


def test_unet_attention_is_restricted_to_coarsest_resolution() -> None:
    model = UNet3D(in_channels=1, base_channels=4, time_dim=16)

    attention_blocks = [module for module in model.modules() if isinstance(module, SelfAttention3D)]

    assert len(attention_blocks) == 1


def test_seeded_batches_repeat_and_change_by_epoch() -> None:
    first = seeded_batch_indices(11, batch_size=4, seed=73, epoch=2)
    second = seeded_batch_indices(11, batch_size=4, seed=73, epoch=2)
    next_epoch = seeded_batch_indices(11, batch_size=4, seed=73, epoch=3)

    assert first == second
    assert first != next_epoch
    assert sorted(index for batch in first for index in batch) == list(range(11))


def test_matern_source_sampler_is_seeded_and_shape_correct() -> None:
    sampler = MaternSourceSampler(shape=(3, 5, 7), corr_len=(0.8, 1.5, 2.0))
    first_generator = torch.Generator().manual_seed(79)
    second_generator = torch.Generator().manual_seed(79)

    first = sampler(2, first_generator, torch.device("cpu"), torch.float32)
    second = sampler(2, second_generator, torch.device("cpu"), torch.float32)

    assert first.shape == (2, 1, 3, 5, 7)
    torch.testing.assert_close(first, second)


def test_layer_trend_normalization_is_invertible() -> None:
    values = torch.randn(4, 1, 5, 3, 2)
    values[:, :, 2] += 7.0
    mask = torch.ones(1, 1, 5, 3, 2, dtype=torch.bool)

    normalizer = LayerTrendNormalizer.fit(values, mask)
    reconstructed = normalizer.inverse(normalizer.transform(values))

    torch.testing.assert_close(reconstructed, values)


def test_euler_and_heun_solve_constant_velocity() -> None:
    initial = torch.zeros(2, 1, 2, 2, 2)

    def velocity(x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del time
        return torch.ones_like(x) * 3.0

    for method in ("euler", "heun"):
        result = integrate_ode(velocity, initial, steps=7, method=method)
        torch.testing.assert_close(result, torch.full_like(initial, 3.0))


def test_bidirectional_transform_round_trips_controlled_velocity() -> None:
    transform = FlowTransform(LinearVelocity(), steps=40, method="heun")
    source = torch.randn(3, 1, 3, 4, 5)

    reconstructed = transform.inverse(transform.forward(source))

    relative_error = torch.linalg.vector_norm(reconstructed - source) / torch.linalg.vector_norm(
        source
    )
    assert float(relative_error) < 1e-2


def test_checkpoint_policy_keeps_ema_and_one_resume_file(tmp_path: Path) -> None:
    (tmp_path / "resume-step-1.pt").write_bytes(b"old")
    (tmp_path / "ema-step-1.pt").write_bytes(b"old")

    save_checkpoint_policy(
        tmp_path,
        ema_state={"weight": torch.tensor([1.0])},
        resume_state={"step": 2},
    )

    assert {path.name for path in tmp_path.glob("*.pt")} == {"ema.pt", "resume.pt"}
    assert torch.load(tmp_path / "resume.pt", weights_only=True)["step"] == 2
