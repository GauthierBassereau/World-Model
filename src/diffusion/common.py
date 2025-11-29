import torch
import torch.nn.functional as F


def calculate_velocity_1_to_2(
    latent_1: torch.Tensor,
    latent_2: torch.Tensor,
    signal_levels_1: torch.Tensor,
    signal_levels_2: torch.Tensor,
    min_denom: float = 0.05,
) -> torch.Tensor:
    denom = (signal_levels_2 - signal_levels_1).clamp_min(min_denom)
    denom = denom.unsqueeze(-1).unsqueeze(-1)
    return (latent_2 - latent_1) / denom