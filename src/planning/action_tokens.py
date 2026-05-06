from dataclasses import dataclass
from math import pi
from typing import Optional

import torch

from src.dataset.common import get_action_delta_stats


UR5_ACTION_MODES = {
    "ur5_relative_local_ee_normalized",
    "ur5_relative_base_ee_normalized",
    "ur5_ee_state_relative_base_ee_normalized",
}


@dataclass
class ActionTokenBuilder:
    action_mode: str
    action_dim: int
    device: torch.device
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.action_mode not in UR5_ACTION_MODES:
            raise ValueError(
                f"Planning currently supports UR5 action modes {sorted(UR5_ACTION_MODES)}, "
                f"got {self.action_mode!r}."
            )

        stats = get_action_delta_stats(self.action_mode)
        if stats is None:
            raise ValueError(f"Missing delta statistics for action mode {self.action_mode!r}.")

        self.delta_mean = torch.tensor(stats["mean"], device=self.device, dtype=self.dtype)
        self.delta_std = torch.tensor(stats["std"], device=self.device, dtype=self.dtype)
        self.delta_dim = int(self.delta_mean.numel())
        self.has_state_token = self.action_mode == "ur5_ee_state_relative_base_ee_normalized"
        self.required_token_dim = self.delta_dim * 2 if self.has_state_token else self.delta_dim

        if self.action_dim < self.required_token_dim:
            raise ValueError(
                f"action_dim={self.action_dim} is too small for {self.action_mode}; "
                f"need at least {self.required_token_dim}."
            )

    def normalize_delta(self, deltas: torch.Tensor) -> torch.Tensor:
        mean = self.delta_mean.to(device=deltas.device, dtype=deltas.dtype)
        std = self.delta_std.to(device=deltas.device, dtype=deltas.dtype)
        return torch.clamp((deltas - mean) / (std + 1e-8), -10.0, 10.0)

    def denormalize_delta(self, normalized_deltas: torch.Tensor) -> torch.Tensor:
        mean = self.delta_mean.to(device=normalized_deltas.device, dtype=normalized_deltas.dtype)
        std = self.delta_std.to(device=normalized_deltas.device, dtype=normalized_deltas.dtype)
        return normalized_deltas * (std + 1e-8) + mean

    def extract_ee_state_token(self, observation_state: torch.Tensor) -> torch.Tensor:
        if observation_state.shape[-1] >= 13:
            ee_state = observation_state[..., 6:13].clone()
        elif observation_state.shape[-1] == self.delta_dim:
            ee_state = observation_state.clone()
        else:
            raise ValueError(
                f"Expected observation state with last dimension 7 or >=13, got {tuple(observation_state.shape)}."
            )
        ee_state[..., -1] = ee_state[..., -1] / 100.0 * (2.0 * pi) - pi
        return ee_state.to(device=self.device, dtype=self.dtype)

    def state_from_action_tokens(self, action_tokens: torch.Tensor, action_index: int) -> Optional[torch.Tensor]:
        if not self.has_state_token:
            return None
        if action_index < 0 or action_index >= action_tokens.shape[0]:
            raise IndexError(f"action_index={action_index} outside action token sequence length {action_tokens.shape[0]}.")
        return action_tokens[action_index, : self.delta_dim].to(device=self.device, dtype=self.dtype)

    def delta_from_action_tokens(self, action_tokens: torch.Tensor) -> torch.Tensor:
        if self.has_state_token:
            normalized = action_tokens[..., self.delta_dim : self.delta_dim * 2]
        else:
            normalized = action_tokens[..., : self.delta_dim]
        return self.denormalize_delta(normalized.to(device=self.device, dtype=self.dtype))

    def integrate_states(self, initial_state: Optional[torch.Tensor], deltas: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.has_state_token:
            return None
        if initial_state is None:
            raise ValueError(f"{self.action_mode} needs an initial EE state to build action tokens.")
        if initial_state.ndim == 1:
            initial_state = initial_state.unsqueeze(0)
        if deltas.ndim != 3:
            raise ValueError(f"Expected deltas with shape [B, H, D], got {tuple(deltas.shape)}.")
        batch_size, horizon, delta_dim = deltas.shape
        if delta_dim != self.delta_dim:
            raise ValueError(f"Expected delta_dim={self.delta_dim}, got {delta_dim}.")
        initial_state = initial_state.to(device=deltas.device, dtype=deltas.dtype)
        if initial_state.shape[0] == 1 and batch_size > 1:
            initial_state = initial_state.expand(batch_size, -1)
        if initial_state.shape != (batch_size, self.delta_dim):
            raise ValueError(
                f"Initial state shape {tuple(initial_state.shape)} incompatible with "
                f"batch_size={batch_size}, delta_dim={self.delta_dim}."
            )
        cumulative = torch.cumsum(deltas, dim=1)
        states_after_step = initial_state.unsqueeze(1) + cumulative
        return torch.cat([initial_state.unsqueeze(1), states_after_step], dim=1)

    def build_future_tokens(
        self,
        deltas_denormalized: torch.Tensor,
        initial_state: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if deltas_denormalized.ndim != 3:
            raise ValueError(
                f"Expected denormalized deltas with shape [B, H, D], got {tuple(deltas_denormalized.shape)}."
            )

        batch_size, horizon, delta_dim = deltas_denormalized.shape
        if delta_dim != self.delta_dim:
            raise ValueError(f"Expected delta_dim={self.delta_dim}, got {delta_dim}.")

        normalized_deltas = self.normalize_delta(deltas_denormalized)
        if self.has_state_token:
            state_sequence = self.integrate_states(initial_state, deltas_denormalized)
            assert state_sequence is not None
            step_states = state_sequence[:, :horizon]
            unpadded = torch.cat([step_states, normalized_deltas], dim=-1)
        else:
            unpadded = normalized_deltas

        if self.action_dim == unpadded.shape[-1]:
            return unpadded

        padding = torch.zeros(
            batch_size,
            horizon,
            self.action_dim - unpadded.shape[-1],
            device=unpadded.device,
            dtype=unpadded.dtype,
        )
        return torch.cat([unpadded, padding], dim=-1)

