"""
Placeholder implementations for diffusion inference solvers.

These will be extended as we add offline sampling support. The world model trainer does not rely
on them yet, but keeping the scaffolding here clarifies where future Euler or Heun solvers will
live.
"""

from dataclasses import dataclass
from typing import Protocol

import torch


class DiffusionSolver(Protocol):
    def sample(self, model, latents: torch.Tensor) -> torch.Tensor:
        ...


@dataclass
class EulerSolver:
    step_size: float = 1.0

    def sample(self, model, latents: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("EulerSolver is a placeholder; implement sampling before use.")
