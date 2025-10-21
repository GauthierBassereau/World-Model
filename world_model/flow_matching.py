from dataclasses import dataclass

import torch
import math
from typing import Protocol


@dataclass
class DiffusionConfig:
    min_signal: float = 0.0
    max_signal: float = 1.0
    base_dimension: int = 4_096
    noise_mean: float = 0.0
    noise_std: float = 1.0

    def validate(self) -> None:
        if not 0.0 <= self.min_signal < self.max_signal <= 1.0:
            raise ValueError("Expected 0.0 <= min_signal < max_signal <= 1.0 for flow matching.")
        if self.base_dimension <= 0:
            raise ValueError("diffusion.base_dimension must be strictly positive.")
        std_tensor = torch.as_tensor(self.noise_std, dtype=torch.float32)
        if torch.any(std_tensor <= 0):
            raise ValueError("diffusion.noise_std must be strictly positive.")
        
class DimensionShiftedUniformScheduler:
    """
    Uniform signal sampler with the dimension-dependent shift introduced in DiT-RAE.

    The scheduler draws baseline values from U(0, 1) and shifts them according to the effective
    latent dimensionality before mapping to [min_signal, max_signal].
    """

    def __init__(self, config: DiffusionConfig) -> None:
        self.config = config
        self._min_signal = torch.tensor(config.min_signal, dtype=torch.float32)
        self._max_signal = torch.tensor(config.max_signal, dtype=torch.float32)

    def sample(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 4:
            raise ValueError("Expected latents to have shape [B, T, tokens, dim].")

        batch, steps, tokens, dim = latents.shape
        if tokens <= 0 or dim <= 0:
            raise ValueError("Latents must contain at least one token with non-zero dimension.")

        device = latents.device
        dtype = latents.dtype

        base = torch.rand((batch, steps), device=device, dtype=torch.float32)

        effective_dim = float(tokens * dim)
        alpha = math.sqrt(effective_dim / float(self.config.base_dimension))
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError(
                f"Invalid dimension shift factor derived from tokens={tokens}, dim={dim}, "
                f"base_dimension={self.config.base_dimension}."
            )
        if not math.isclose(alpha, 1.0):
            base = (alpha * base) / (1.0 + (alpha - 1.0) * base)

        min_signal = self._min_signal.to(device=device, dtype=base.dtype)
        max_signal = self._max_signal.to(device=device, dtype=base.dtype)
        shifted = base * (max_signal - min_signal) + min_signal
        return shifted.to(dtype=dtype)


def sample_base_noise(latents: torch.Tensor, config: DiffusionConfig) -> torch.Tensor:
    """Sample the base noise prior used for flow matching."""
    mean = torch.as_tensor(
        config.noise_mean,
        device=latents.device,
        dtype=latents.dtype,
    )
    std = torch.as_tensor(
        config.noise_std,
        device=latents.device,
        dtype=latents.dtype,
    )
    noise = torch.randn_like(latents)
    return noise * std + mean


class DiffusionSolver(Protocol):
    def sample(self, model, latents: torch.Tensor) -> torch.Tensor:
        ...


@dataclass
class EulerSolver:
    step_size: float = 1.0

    def sample(self, model, latents: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("EulerSolver is a placeholder; implement sampling before use.")
