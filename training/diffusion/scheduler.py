import math

import torch

from .config import DiffusionConfig


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
