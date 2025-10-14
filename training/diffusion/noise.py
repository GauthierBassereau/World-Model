import torch

from .config import DiffusionConfig


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
