from typing import Any, Optional

import torch


def unwrap_world_model(model: Any) -> Any:
    """Unwrap common containers around the world model."""
    unwrapped = model
    seen: set[int] = set()
    while True:
        model_id = id(unwrapped)
        if model_id in seen:
            return unwrapped
        seen.add(model_id)

        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
            continue
        if hasattr(unwrapped, "_orig_mod"):
            unwrapped = unwrapped._orig_mod
            continue
        return unwrapped


def model_uses_patch_signal_conditioning(model: Any) -> bool:
    config = getattr(unwrap_world_model(model), "config", None)
    return bool(getattr(config, "patch_signal_conditioning", False))


def expand_signal_levels_for_model(
    model: Any,
    signal_levels: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    if model_uses_patch_signal_conditioning(model) and signal_levels.ndim == 2:
        return signal_levels.unsqueeze(-1).expand(-1, -1, num_tokens)
    return signal_levels


def global_signal_levels_from_signal(signal_levels: torch.Tensor) -> Optional[torch.Tensor]:
    if signal_levels.ndim == 2:
        return signal_levels
    if signal_levels.ndim == 3:
        return signal_levels.amax(dim=-1)
    return None
