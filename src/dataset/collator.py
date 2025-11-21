import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import torch
from .configs import DatasetConfig

@dataclass
class WorldModelBatch:
    sequence_frames: torch.Tensor
    sequence_actions: torch.Tensor
    independent_frames_mask: torch.Tensor
    actions_mask: torch.Tensor
    frames_valid_mask: torch.Tensor


class LeRobotSequenceCollator:
    def __init__(
        self,
        dataset_cfg: DatasetConfig,
    ) -> None:
        self.cfg = dataset_cfg

        if not self.cfg.cameras:
            raise ValueError("DatasetConfig.cameras must contain at least one camera key.")
        if not self.cfg.action_keys:
            raise ValueError("DatasetConfig.action_keys must contain at least one key.")
        if not 0.0 <= self.cfg.independent_frames_probability <= 1.0:
            raise ValueError("DatasetConfig.independent_frames_probability must be between 0 and 1.")
        if not 0.0 <= self.cfg.drop_action_probability <= 1.0:
            raise ValueError("DatasetConfig.drop_action_probability must be between 0 and 1.")

        self.max_sequence_length = max(
            int(length) for length in self.cfg.sequence_length_distribution.keys()
        )
        if self.max_sequence_length < 2:
            raise ValueError("Resolved sequence length must be >= 2.")

        if self.cfg.camera_probabilities:
            weights = [
                self.cfg.camera_probabilities.get(camera, 0.0) for camera in self.cfg.cameras
            ]
            if not any(weight > 0 for weight in weights):
                raise ValueError(
                    "camera_probabilities must include a positive weight for at least one camera."
                )
        else:
            weights = [1.0] * len(self.cfg.cameras)

        total = sum(weights)
        self.camera_weights = [weight / total for weight in weights]
        self.camera_keys = list(self.cfg.cameras)
        self.sequence_length_choices, self.sequence_length_probs = (
            self._build_sequence_length_distribution()
        )
        self._normalization_dim: Optional[int] = None
        self._normalization_params: Optional[Dict[str, torch.Tensor]] = None

    def __call__(self, samples: Iterable[Dict[str, torch.Tensor]]) -> WorldModelBatch:
        target_length = self._sample_sequence_length()

        frame_sequences: List[torch.Tensor] = []
        action_sequences: List[torch.Tensor] = []
        independent_frames_mask: List[torch.Tensor] = []
        actions_mask: List[torch.Tensor] = []
        frames_valid_masks: List[torch.Tensor] = []

        for sample in samples:
            camera_key = random.choices(self.camera_keys, weights=self.camera_weights, k=1)[0]
            frames = self._prepare_frames(sample, camera_key, target_length)
            validity_mask = self._prepare_valid_mask(sample, camera_key, target_length)
            actions = self._prepare_actions(sample, target_length)

            use_independent_frame = random.random() < self.cfg.independent_frames_probability
            drop_actions = random.random() < self.cfg.drop_action_probability

            if use_independent_frame or drop_actions:
                actions.zero_()

            frame_sequences.append(frames)
            action_sequences.append(actions)
            independent_frames_mask.append(
                torch.tensor(use_independent_frame, dtype=torch.bool)
            )
            base_action_mask = torch.full(
                (target_length,),
                not (use_independent_frame or drop_actions),
                dtype=torch.bool,
            )
            actions_mask.append(base_action_mask & validity_mask)
            frames_valid_masks.append(validity_mask)

        return WorldModelBatch(
            sequence_frames=torch.stack(frame_sequences, dim=0),
            sequence_actions=torch.stack(action_sequences, dim=0),
            independent_frames_mask=torch.stack(independent_frames_mask, dim=0),
            actions_mask=torch.stack(actions_mask, dim=0),
            frames_valid_mask=torch.stack(frames_valid_masks, dim=0),
        )

    def _prepare_frames(
        self,
        sample: Dict[str, torch.Tensor],
        camera_key: str,
        target_length: int,
    ) -> torch.Tensor:
        frames = sample[camera_key]
        if frames.ndim != 4:
            raise ValueError(f"Expected frames of shape [T, C, H, W], got {tuple(frames.shape)}.")
        if frames.shape[0] != self.max_sequence_length:
            raise ValueError(
                f"Expected frame sequence of length {self.max_sequence_length}, got {frames.shape[0]}."
            )

        frames = frames.to(torch.float32)
        # Heuristic to detect if images are in [0, 255] or [0, 1]
        if frames.max() > 1.5:
            frames = frames / 255.0
        frames = frames.clamp(0.0, 1.0)
        return frames[:target_length].contiguous()

    def _prepare_valid_mask(
        self,
        sample: Dict[str, torch.Tensor],
        camera_key: str,
        target_length: int,
    ) -> torch.Tensor:
        pad_key = f"{camera_key}_is_pad"
        if pad_key in sample:
            pad_tensor = sample[pad_key].to(dtype=torch.bool)
            if pad_tensor.ndim != 1:
                raise ValueError(
                    f"Expected padding mask '{pad_key}' to have 1 dimension, got {pad_tensor.ndim}."
                )
            if pad_tensor.shape[0] != self.max_sequence_length:
                raise ValueError(
                    f"Padding mask '{pad_key}' length {pad_tensor.shape[0]} does not match sequence length {self.max_sequence_length}."
                )
        else:
            pad_tensor = torch.zeros(
                (self.max_sequence_length,), dtype=torch.bool
            )
        valid = (~pad_tensor).to(dtype=torch.bool)
        return valid[:target_length].contiguous()

    def _prepare_actions(
        self,
        sample: Dict[str, torch.Tensor],
        target_length: int,
    ) -> torch.Tensor:
        action_parts: List[torch.Tensor] = []
        for key in self.cfg.action_keys:
            if key not in sample:
                raise KeyError(f"Sample is missing required action key '{key}'.")
            part = sample[key]
            if part.ndim == 1:
                part = part.unsqueeze(-1)
            if part.ndim != 2:
                raise ValueError(
                    f"Expected action '{key}' to have shape [T, D], got {tuple(part.shape)}."
                )
            if part.shape[0] != self.max_sequence_length:
                raise ValueError(
                    f"Action '{key}' has length {part.shape[0]}, expected {self.max_sequence_length}."
                )
            action_parts.append(part.to(torch.float32))

        if not action_parts:
            raise ValueError("No action data collected; check DatasetConfig.action_keys.")

        actions = torch.cat(action_parts, dim=-1)[:target_length].contiguous()
        if self.cfg.action_representation == "delta":
            deltas = actions[1:] - actions[:-1]
            result = torch.zeros(
                (target_length, deltas.shape[-1]),
                dtype=deltas.dtype,
                device=deltas.device,
            )
            result[1:] = deltas
            actions = result
        if self.cfg.action_normalization:
            actions = self._normalize_actions(actions)
        return actions

    def _normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.cfg.action_normalization:
            return actions
        self._ensure_normalization_tensors(actions.shape[-1])
        if not self._normalization_params:
            return actions
        offset = self._normalization_params["offset"].to(
            device=actions.device, dtype=actions.dtype
        )
        scale = self._normalization_params["scale"].to(
            device=actions.device, dtype=actions.dtype
        )
        return (actions - offset) / scale

    def _ensure_normalization_tensors(self, feature_dim: int) -> None:
        if self._normalization_params is not None:
            if self._normalization_dim != feature_dim:
                raise ValueError(
                    "Configured action normalization dimension does not match the action tensor."
                )
            return
        params = self.cfg.action_normalization_params
        if params is None:
            raise ValueError("action_normalization_params are required for normalization.")
        norm_type = self.cfg.action_normalization
        if norm_type == "min_max":
            raw_min = torch.as_tensor(params["min"], dtype=torch.float32)
            raw_max = torch.as_tensor(params["max"], dtype=torch.float32)
            if raw_min.ndim != 1 or raw_max.ndim != 1:
                raise ValueError("min and max normalization parameters must be 1D sequences.")
            if raw_min.shape[0] != feature_dim or raw_max.shape[0] != feature_dim:
                raise ValueError("Normalization parameter length does not match action dimension.")
            scale = raw_max - raw_min
            if torch.any(scale <= 0):
                raise ValueError("For min-max normalization, each max must be greater than min.")
            offset = raw_min
        elif norm_type == "mean_std":
            raw_mean = torch.as_tensor(params["mean"], dtype=torch.float32)
            raw_std = torch.as_tensor(params["std"], dtype=torch.float32)
            if raw_mean.ndim != 1 or raw_std.ndim != 1:
                raise ValueError("mean and std normalization parameters must be 1D sequences.")
            if raw_mean.shape[0] != feature_dim or raw_std.shape[0] != feature_dim:
                raise ValueError("Normalization parameter length does not match action dimension.")
            if torch.any(raw_std <= 0):
                raise ValueError("Standard deviation entries must be strictly positive.")
            offset = raw_mean
            scale = raw_std
        else:
            raise ValueError("Unsupported action normalization type provided.")
        self._normalization_dim = feature_dim
        self._normalization_params = {"offset": offset, "scale": scale}

    def _build_sequence_length_distribution(self) -> Tuple[List[int], List[float]]:
        choices: List[int] = []
        weights: List[float] = []
        for length, weight in self.cfg.sequence_length_distribution.items():
            if length < 2:
                raise ValueError("sequence_length_distribution keys must be >= 2.")
            if length > self.max_sequence_length:
                raise ValueError(
                    f"sequence_length_distribution contains length {length} which exceeds "
                    f"the resolved maximum sequence length ({self.max_sequence_length})."
                )
            if weight <= 0:
                raise ValueError("sequence_length_distribution probabilities must be positive.")
            choices.append(int(length))
            weights.append(float(weight))

        total = sum(weights)
        if total <= 0:
            raise ValueError("sequence_length_distribution weights must sum to a positive value.")
        normalized = [weight / total for weight in weights]
        return choices, normalized

    def _sample_sequence_length(self) -> int:
            return random.choices(
                self.sequence_length_choices, weights=self.sequence_length_probs, k=1
            )[0]
