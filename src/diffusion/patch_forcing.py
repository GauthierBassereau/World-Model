from typing import Optional

import torch


def sample_ltg_patch_signal_levels(
    base_signal_levels: torch.Tensor,
    num_tokens: int,
    ltg_std: float = 0.6,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample patch signal levels below each frame-level base signal with LTG."""
    if num_tokens < 1:
        raise ValueError("num_tokens must be positive.")
    if ltg_std <= 0.0:
        raise ValueError("ltg_std must be positive.")

    output_dtype = base_signal_levels.dtype
    base = base_signal_levels.float()
    std = torch.minimum(
        base / 2.0,
        torch.full_like(base, ltg_std),
    )
    noise = torch.randn(
        (*base.shape, num_tokens),
        device=base.device,
        dtype=base.dtype,
        generator=generator,
    ).abs()
    patch_signal_levels = base.unsqueeze(-1) - noise * std.unsqueeze(-1)
    fallback = torch.rand(
        patch_signal_levels.shape,
        device=base.device,
        dtype=base.dtype,
        generator=generator,
    ) * base.unsqueeze(-1)
    patch_signal_levels = torch.where(patch_signal_levels < 0.0, fallback, patch_signal_levels)
    return patch_signal_levels.to(dtype=output_dtype)
