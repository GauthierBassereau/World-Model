from dataclasses import dataclass
from math import pi
from typing import Optional

import torch

from src.dataset.common import get_action_delta_stats


UR5_ACTION_MODES = {
    "ur5_delta_absg",
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
        ee_state = self.extract_ee_pose(observation_state)
        ee_state[..., -1] = ee_state[..., -1] / 100.0 * (2.0 * pi) - pi
        return ee_state

    def extract_ee_pose(self, observation_state: torch.Tensor) -> torch.Tensor:
        if observation_state.shape[-1] >= 13:
            ee_state = observation_state[..., 6:13].clone()
        elif observation_state.shape[-1] == self.delta_dim:
            ee_state = observation_state.clone()
        else:
            raise ValueError(
                f"Expected observation state with last dimension 7 or >=13, got {tuple(observation_state.shape)}."
            )
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
        states = self.integrate_trajectory_poses(initial_state, deltas)
        if states is None:
            raise RuntimeError("State-token integration unexpectedly returned None.")
        return states

    def integrate_trajectory_poses(
        self,
        initial_pose: Optional[torch.Tensor],
        deltas: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if initial_pose is None:
            return None
        if initial_pose.ndim == 1:
            initial_pose = initial_pose.unsqueeze(0)
        if deltas.ndim != 3:
            raise ValueError(f"Expected deltas with shape [B, H, D], got {tuple(deltas.shape)}.")

        batch_size, _, delta_dim = deltas.shape
        if delta_dim != self.delta_dim:
            raise ValueError(f"Expected delta_dim={self.delta_dim}, got {delta_dim}.")

        initial_pose = initial_pose.to(device=deltas.device, dtype=deltas.dtype)
        if initial_pose.shape[0] == 1 and batch_size > 1:
            initial_pose = initial_pose.expand(batch_size, -1)
        if initial_pose.shape != (batch_size, self.delta_dim):
            raise ValueError(
                f"Initial pose shape {tuple(initial_pose.shape)} incompatible with "
                f"batch_size={batch_size}, delta_dim={self.delta_dim}."
            )

        current_position = initial_pose[:, :3]
        current_rotation = _rotvec_to_matrix(initial_pose[:, 3:6])
        current_gripper = initial_pose[:, 6:7]
        poses = [initial_pose]

        for step in range(deltas.shape[1]):
            delta = deltas[:, step]
            delta_translation = delta[:, :3]
            delta_rotation = _rotvec_to_matrix(delta[:, 3:6])

            if self.action_mode == "ur5_relative_local_ee_normalized":
                current_position = current_position + torch.matmul(
                    current_rotation,
                    delta_translation.unsqueeze(-1),
                ).squeeze(-1)
                current_rotation = current_rotation @ delta_rotation
            else:
                current_position = current_position + delta_translation
                current_rotation = delta_rotation @ current_rotation

            if self.action_mode == "ur5_delta_absg":
                current_gripper = delta[:, 6:7]
            else:
                current_gripper = current_gripper + delta[:, 6:7]

            poses.append(
                torch.cat(
                    [
                        current_position,
                        _matrix_to_rotvec(current_rotation),
                        current_gripper,
                    ],
                    dim=-1,
                )
            )

        return torch.stack(poses, dim=1)

    def rotation_geodesic_distance(self, rotvec_a: torch.Tensor, rotvec_b: torch.Tensor) -> torch.Tensor:
        rotation_a = _rotvec_to_matrix(rotvec_a)
        rotation_b = _rotvec_to_matrix(rotvec_b)
        relative_rotation = rotation_b.transpose(-1, -2) @ rotation_a
        trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        cosine = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
        return torch.acos(cosine)

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


def _rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    theta_sq = theta.square()
    skew = _skew_matrix(rotvec)
    identity = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).expand(*rotvec.shape[:-1], 3, 3)
    a = torch.where(
        theta.abs() < 1e-4,
        1.0 - theta_sq / 6.0 + theta_sq.square() / 120.0,
        torch.sin(theta) / theta,
    )
    b = torch.where(
        theta.abs() < 1e-4,
        0.5 - theta_sq / 24.0 + theta_sq.square() / 720.0,
        (1.0 - torch.cos(theta)) / theta_sq,
    )
    return identity + a.unsqueeze(-1) * skew + b.unsqueeze(-1) * (skew @ skew)


def _matrix_to_rotvec(matrix: torch.Tensor) -> torch.Tensor:
    trace = matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    theta = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))
    skew = torch.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        dim=-1,
    )
    sin_theta = torch.sin(theta)
    scale = torch.where(
        theta.abs() < 1e-4,
        0.5 + theta.square() / 12.0,
        theta / (2.0 * sin_theta),
    )
    return scale.unsqueeze(-1) * skew


def _skew_matrix(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ],
        dim=-2,
    )
