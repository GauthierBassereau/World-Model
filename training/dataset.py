from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Literal

import random

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

ActionRepresentation = Literal["delta", "position"]


@dataclass
class DataloaderConfig:
    batch_size: int = 2
    shuffle: bool = False
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class DatasetConfig:
    repo_id: str = "lerobot/droid_1.0.1"
    use_streaming: bool = False
    cameras: Sequence[str] = (
        "observation.images.exterior_1_left",
        "observation.images.exterior_2_left",
    )
    camera_probabilities: Optional[Dict[str, float]] = None
    action_keys: Sequence[str] = (
        "observation.state.cartesian_position",
        "observation.state.gripper_position",
    )
    action_representation: ActionRepresentation = "delta"
    episodes: Optional[Sequence[int]] = None # None -> all
    sequence_length_distribution: Dict[int, float] = field(default_factory=lambda: {4: 1.0})
    frame_delta_seconds: float | str = 5.0 / 15.0
    independant_frames_probability: float = 0.0
    drop_action_probability: float = 0.0

    def __post_init__(self) -> None:
        self.frame_delta_seconds = _coerce_frame_delta(self.frame_delta_seconds)
        if not self.sequence_length_distribution:
            raise ValueError(
                "DatasetConfig.sequence_length_distribution must contain at least one entry."
            )
        self.sequence_length_distribution = {
            int(length): float(weight)
            for length, weight in self.sequence_length_distribution.items()
        }
        keys: Sequence[str]
        if isinstance(self.action_keys, str):
            keys = (self.action_keys,)
        else:
            keys = tuple(self.action_keys)
        if not keys:
            raise ValueError("DatasetConfig.action_keys must contain at least one key.")
        self.action_keys = keys
        if self.action_representation not in {"delta", "position"}:
            raise ValueError(
                "DatasetConfig.action_representation must be either 'delta' or 'position'."
            )


@dataclass
class WorldModelBatch:
    sequence_frames: torch.Tensor
    sequence_actions: torch.Tensor
    independant_frames_mask: torch.Tensor
    actions_mask: torch.Tensor


def _coerce_frame_delta(value: float | str) -> float:
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            result = float(numerator) / float(denominator)
        else:
            result = float(text)
    else:
        raise TypeError(
            f"frame_delta_seconds must be a float or string, received {type(value).__name__}."
        )
    if result <= 0:
        raise ValueError("frame_delta_seconds must be strictly positive.")
    return result


def _ensure_delta_timestamps(
    dataset_cfg: DatasetConfig,
    metadata: LeRobotDatasetMetadata,
) -> Dict[str, Sequence[float]]:
    """Ensure delta timestamps are provided for all cameras and action key."""
    max_length = max(int(length) for length in dataset_cfg.sequence_length_distribution.keys())
    step = dataset_cfg.frame_delta_seconds
    if step <= 0:
        step = 1.0 / metadata.fps
    offsets = [-step * i for i in range(max_length - 1, -1, -1)]
    delta = {camera: list(offsets) for camera in dataset_cfg.cameras}

    for key in dataset_cfg.action_keys:
        delta[key] = list(offsets)
    return delta


class LeRobotSequenceCollator:
    def __init__(
        self,
        dataset_cfg: DatasetConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.cfg = dataset_cfg
        self.device = device or torch.device("cpu")

        if not self.cfg.cameras:
            raise ValueError("DatasetConfig.cameras must contain at least one camera key.")
        if not self.cfg.action_keys:
            raise ValueError("DatasetConfig.action_keys must contain at least one key.")
        if not 0.0 <= self.cfg.independant_frames_probability <= 1.0:
            raise ValueError("DatasetConfig.independant_frames_probability must be between 0 and 1.")
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

    def __call__(self, samples: Iterable[Dict[str, torch.Tensor]]) -> WorldModelBatch:
        target_length = self._sample_sequence_length()

        frame_sequences: List[torch.Tensor] = []
        action_sequences: List[torch.Tensor] = []
        independant_frames_mask: List[torch.Tensor] = []
        actions_mask: List[torch.Tensor] = []

        for sample in samples:
            camera_key = random.choices(self.camera_keys, weights=self.camera_weights, k=1)[0]
            frames = self._prepare_frames(sample, camera_key, target_length)
            actions = self._prepare_actions(sample, target_length)

            use_independant_frame = random.random() < self.cfg.independant_frames_probability
            drop_actions = random.random() < self.cfg.drop_action_probability

            if use_independant_frame:
                frame_index = random.randrange(target_length)
                static_frame = frames[frame_index].unsqueeze(0)
                frames = static_frame.repeat(target_length, 1, 1, 1)

            if use_independant_frame or drop_actions:
                actions.zero_()

            frame_sequences.append(frames)
            action_sequences.append(actions)
            independant_frames_mask.append(
                torch.tensor(use_independant_frame, dtype=torch.bool, device=self.device)
            )
            actions_mask.append(
                torch.full(
                    (target_length,),
                    not (use_independant_frame or drop_actions),
                    dtype=torch.bool,
                    device=self.device,
                )
            )

        return WorldModelBatch(
            sequence_frames=torch.stack(frame_sequences, dim=0),
            sequence_actions=torch.stack(action_sequences, dim=0),
            independant_frames_mask=torch.stack(independant_frames_mask, dim=0),
            actions_mask=torch.stack(actions_mask, dim=0),
        )

    def _prepare_frames(
        self,
        sample: Dict[str, torch.Tensor],
        camera_key: str,
        target_length: int,
    ) -> torch.Tensor:
        frames = sample[camera_key].to(self.device)
        if frames.ndim != 4:
            raise ValueError(f"Expected frames of shape [T, C, H, W], got {tuple(frames.shape)}.")
        if frames.shape[0] != self.max_sequence_length:
            raise ValueError(
                f"Expected frame sequence of length {self.max_sequence_length}, got {frames.shape[0]}."
            )

        frames = frames.to(torch.float32)
        if frames.max() > 1.5:
            frames = frames / 255.0
        frames = frames.clamp(0.0, 1.0)
        return frames[-target_length:].contiguous()

    def _prepare_actions(
        self,
        sample: Dict[str, torch.Tensor],
        target_length: int,
    ) -> torch.Tensor:
        action_parts: List[torch.Tensor] = []
        for key in self.cfg.action_keys:
            if key not in sample:
                raise KeyError(f"Sample is missing required action key '{key}'.")
            part = sample[key].to(self.device)
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

        actions = torch.cat(action_parts, dim=-1)[-target_length:].contiguous()
        if self.cfg.action_representation == "delta":
            deltas = actions[1:] - actions[:-1]
            result = torch.zeros(
                (target_length, deltas.shape[-1]),
                dtype=deltas.dtype,
                device=deltas.device,
            )
            result[1:] = deltas
            return result
        return actions

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


def build_world_model_dataloader(
    dataset_cfg: DatasetConfig,
    dataloader_cfg: DataloaderConfig,
    device: Optional[torch.device] = None,
) -> DataLoader:
    device = device or torch.device("cpu")
    metadata = LeRobotDatasetMetadata(dataset_cfg.repo_id)
    delta_timestamps = _ensure_delta_timestamps(dataset_cfg, metadata)
    if dataset_cfg.use_streaming:
        from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

        dataset = StreamingLeRobotDataset(
            dataset_cfg.repo_id,
            episodes=None,
            delta_timestamps=delta_timestamps,
            shuffle=dataloader_cfg.shuffle,
            tolerance_s=0.01
        )
        shuffle = False
    else:
        dataset = LeRobotDataset(
            dataset_cfg.repo_id,
            episodes=list(dataset_cfg.episodes) if dataset_cfg.episodes else None,
            delta_timestamps=delta_timestamps,
            tolerance_s=0.01
        )
        shuffle = dataloader_cfg.shuffle

    collate = LeRobotSequenceCollator(dataset_cfg, device=device)
    return DataLoader(
        dataset,
        batch_size=dataloader_cfg.batch_size,
        shuffle=shuffle,
        num_workers=dataloader_cfg.num_workers,
        pin_memory=dataloader_cfg.pin_memory,
        collate_fn=collate,
    )
