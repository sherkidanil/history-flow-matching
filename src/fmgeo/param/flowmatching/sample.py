"""Shared bidirectional ODE integration for learned geological transforms."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

Velocity = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Integrator = Literal["euler", "heun"]


def integrate_ode(
    velocity: Velocity,
    initial: torch.Tensor,
    *,
    steps: int,
    method: Integrator,
    start_time: float = 0.0,
    end_time: float = 1.0,
) -> torch.Tensor:
    """Integrate dx/dt=v(x,t) in either time direction."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if method not in ("euler", "heun"):
        raise ValueError(f"unknown integration method: {method}")
    state = initial
    step_size = (end_time - start_time) / steps
    batch = initial.shape[0]
    for index in range(steps):
        current_value = start_time + index * step_size
        current_time = torch.full(
            (batch,), current_value, device=state.device, dtype=state.dtype
        )
        first = velocity(state, current_time)
        if method == "euler":
            state = state + step_size * first
        else:
            proposed = state + step_size * first
            next_time = torch.full(
                (batch,), current_value + step_size, device=state.device, dtype=state.dtype
            )
            second = velocity(proposed, next_time)
            state = state + 0.5 * step_size * (first + second)
    return state


@dataclass(frozen=True)
class FlowTransform:
    """One model exposed as consistent latent-to-field and inverse transforms."""

    model: nn.Module
    steps: int = 50
    method: Integrator = "heun"
    mask: torch.Tensor | None = None

    def _integrate(
        self, values: torch.Tensor, *, start_time: float, end_time: float
    ) -> torch.Tensor:
        if self.mask is None:
            velocity: Velocity = self.model
            initial = values
        else:
            try:
                active = torch.broadcast_to(
                    self.mask.to(device=values.device, dtype=torch.bool), values.shape
                )
            except RuntimeError as error:
                raise ValueError("mask must broadcast to transform inputs") from error
            initial = torch.where(active, values, torch.zeros_like(values))

            def velocity(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
                masked_state = torch.where(active, state, torch.zeros_like(state))
                prediction = self.model(masked_state, time)
                return torch.where(active, prediction, torch.zeros_like(prediction))

        return integrate_ode(
            velocity,
            initial,
            steps=self.steps,
            method=self.method,
            start_time=start_time,
            end_time=end_time,
        )

    @torch.no_grad()
    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return self._integrate(source, start_time=0.0, end_time=1.0)

    @torch.no_grad()
    def inverse(self, target: torch.Tensor) -> torch.Tensor:
        return self._integrate(target, start_time=1.0, end_time=0.0)
