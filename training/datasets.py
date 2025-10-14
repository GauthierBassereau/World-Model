"""
Dataset utilities for turning LeRobot clips into DINO latents consumed by the
world model trainer.

Each batch contains:
    sequence_latents : [B, T, tokens, dim]
        (DINO latents for each timestep; target frame kept in sequence.)
    sequence_actions : [B, T, action_dim] (delta between successive states with a zero dummy at step 0)

Config keys consumed here:
    dataset.repo_id
    dataset.cameras
    dataset.action_key
    dataset.sequence_length_distribution
    dataset.frame_delta_seconds
    dataset.single_frame_probability
    dataset.drop_action_probability
    dataloader.batch_size
    dataloader.shuffle
    dataloader.num_workers
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from vision.dino_v3 import DinoV3Embedder


@dataclass
class DataloaderConfig:
    batch_size: int = 2
    shuffle: bool = False
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class DatasetConfig:
    repo_id: str = "lerobot/droid_1.0.1"
    cameras: Sequence[str] = (
        "observation.images.exterior_1_left",
        "observation.images.exterior_2_left",
    )
    camera_probabilities: Optional[Dict[str, float]] = None
    action_key: Optional[str] = "observation.state.cartesian_position"
    episodes: Optional[Sequence[int]] = None
    sequence_length_distribution: Dict[int, float] = field(default_factory=lambda: {4: 1.0})
    frame_delta_seconds: float | str = 5.0 / 15.0  # 15 FPS
    single_frame_probability: float = 0.0
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


@dataclass
class WorldModelBatch:
    sequence_latents: torch.Tensor
    sequence_actions: torch.Tensor
    single_frame_mask: Optional[torch.Tensor] = None
    actions_dropped_mask: Optional[torch.Tensor] = None
    profiling: Optional[Dict[str, float]] = None


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
    max_length = max(int(length) for length in dataset_cfg.sequence_length_distribution.keys())
    step = dataset_cfg.frame_delta_seconds
    if step <= 0:
        step = 1.0 / metadata.fps
    offsets = [-step * i for i in range(max_length - 1, -1, -1)]
    delta = {camera: list(offsets) for camera in dataset_cfg.cameras}

    if dataset_cfg.action_key:
        delta[dataset_cfg.action_key] = list(offsets)
    return delta


class LeRobotSequenceCollator:
    def __init__(
        self,
        dataset_cfg: DatasetConfig,
        embedder: DinoV3Embedder,
        device: torch.device,
    ) -> None:
        self.cfg = dataset_cfg
        self.embedder = embedder
        self.device = device
        if not self.cfg.sequence_length_distribution:
            raise ValueError("DatasetConfig.sequence_length_distribution must contain at least one entry.")
        self.max_sequence_length = max(int(length) for length in self.cfg.sequence_length_distribution.keys())
        # The embedder runs inside the collate function, so stick to num_workers=0.

        if not self.cfg.cameras:
            raise ValueError("DatasetConfig.cameras must contain at least one camera key.")
        if not self.cfg.action_key:
            raise ValueError("DatasetConfig.action_key must be provided to compute action deltas.")
        if self.max_sequence_length < 2:
            raise ValueError("Resolved sequence length must be >= 2.")
        if not 0.0 <= self.cfg.single_frame_probability <= 1.0:
            raise ValueError("DatasetConfig.single_frame_probability must be between 0 and 1.")
        if not 0.0 <= self.cfg.drop_action_probability <= 1.0:
            raise ValueError("DatasetConfig.drop_action_probability must be between 0 and 1.")

        if self.cfg.camera_probabilities:
            weights = [self.cfg.camera_probabilities.get(camera, 0.0) for camera in self.cfg.cameras]
            if not any(weight > 0 for weight in weights):
                raise ValueError("camera_probabilities must include a positive weight for at least one camera.")
        else:
            weights = [1.0] * len(self.cfg.cameras)

        total = sum(weights)
        self.camera_weights = [weight / total for weight in weights]
        self.camera_keys = list(self.cfg.cameras)
        self.sequence_length_choices, self.sequence_length_probs = self._build_sequence_length_distribution()

    @torch.no_grad()
    def __call__(self, samples: Iterable[Dict[str, torch.Tensor]]) -> WorldModelBatch:
        target_length = self._sample_sequence_length()
        # Every item in the batch is truncated to the same target length so tensors still stack cleanly.

        sequence_latents: List[torch.Tensor] = []
        sequence_actions: List[torch.Tensor] = []
        single_frame_flags: List[torch.Tensor] = []
        actions_dropped_flags: List[torch.Tensor] = []
        collate_start = time.perf_counter()
        embed_time_total = 0.0

        for sample in samples:
            camera_key = random.choices(self.camera_keys, weights=self.camera_weights, k=1)[0]
            embed_start = time.perf_counter()
            latent_sequence = self._encode_sample(sample, camera_key)
            embed_time_total += time.perf_counter() - embed_start

            if latent_sequence.shape[0] != self.max_sequence_length:
                raise ValueError(
                    f"Expected {self.max_sequence_length} latent steps, got {latent_sequence.shape[0]}."
                )
            if target_length > latent_sequence.shape[0]:
                raise ValueError(
                    f"Requested target sequence length {target_length} but only "
                    f"{latent_sequence.shape[0]} frames were available."
                )
            latent_sequence = latent_sequence[-target_length:]

            action_seq = sample[self.cfg.action_key].to(self.device)
            if action_seq.ndim == 1:
                action_seq = action_seq.unsqueeze(-1)
            if action_seq.shape[0] < 2:
                raise ValueError("Action sequence must contain at least two steps to compute deltas.")
            if action_seq.shape[0] != self.max_sequence_length:
                raise ValueError(
                    f"Expected action sequence of length {self.max_sequence_length}, got {action_seq.shape[0]}."
                )
            action_seq = action_seq[-target_length:]
            delta_actions = action_seq[1:] - action_seq[:-1]

            use_single_frame = random.random() < self.cfg.single_frame_probability
            drop_actions = random.random() < self.cfg.drop_action_probability

            if use_single_frame or drop_actions:
                delta_actions = torch.zeros_like(delta_actions)

            actions_for_frames = torch.zeros(
                (target_length, delta_actions.shape[-1]),
                device=self.device,
                dtype=delta_actions.dtype,
            )
            actions_for_frames[1:] = delta_actions

            if use_single_frame:
                latent_sequence = torch.cat(
                    [
                        torch.zeros_like(latent_sequence[:-1]),
                        latent_sequence[-1:].clone(),
                    ],
                    dim=0,
                )

            sequence_latents.append(latent_sequence)
            sequence_actions.append(actions_for_frames)
            single_frame_flags.append(
                torch.tensor(use_single_frame, dtype=torch.bool, device=self.device)
            )
            actions_dropped_flags.append(
                torch.tensor(use_single_frame or drop_actions, dtype=torch.bool, device=self.device)
            )

        sequence_latent_tensor = torch.stack(sequence_latents, dim=0)
        action_tensor = torch.stack(sequence_actions, dim=0)
        single_frame_mask = torch.stack(single_frame_flags, dim=0) if single_frame_flags else None
        actions_dropped_mask = (
            torch.stack(actions_dropped_flags, dim=0) if actions_dropped_flags else None
        )
        collate_time = time.perf_counter() - collate_start
        return WorldModelBatch(
            sequence_latents=sequence_latent_tensor,
            sequence_actions=action_tensor,
            single_frame_mask=single_frame_mask,
            actions_dropped_mask=actions_dropped_mask,
            profiling={
                "collate_time": float(collate_time),
                "embed_time": float(embed_time_total),
                "target_length": float(target_length),
                "batch_size": float(sequence_latent_tensor.shape[0]),
            },
        )

    def _encode_sample(self, sample: Dict[str, torch.Tensor], camera_key: str) -> torch.Tensor:
        frames = sample[camera_key].to(self.device)
        return self.embedder(frames)

    def _build_sequence_length_distribution(self) -> tuple[List[int], List[float]]:
        choices: List[int] = []
        weights: List[float] = []
        max_length = self.max_sequence_length

        for length, weight in self.cfg.sequence_length_distribution.items():
            if length < 2:
                raise ValueError("sequence_length_distribution keys must be >= 2.")
            if length > max_length:
                raise ValueError(
                    f"sequence_length_distribution contains length {length} which exceeds "
                    f"the resolved maximum sequence length ({max_length})."
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
        return random.choices(self.sequence_length_choices, weights=self.sequence_length_probs, k=1)[0]


def build_world_model_dataloader(
    dataset_cfg: DatasetConfig,
    dataloader_cfg: DataloaderConfig,
    embedder: DinoV3Embedder,
    device: Optional[torch.device] = None,
) -> DataLoader:
    device = device or torch.device("cpu")
    metadata = LeRobotDatasetMetadata(dataset_cfg.repo_id)
    delta_timestamps = _ensure_delta_timestamps(dataset_cfg, metadata)
    dataset = LeRobotDataset(
        dataset_cfg.repo_id,
        episodes=list(dataset_cfg.episodes) if dataset_cfg.episodes else None,
        delta_timestamps=delta_timestamps,
    )

    collate = LeRobotSequenceCollator(dataset_cfg, embedder, device=device)
    return DataLoader(
        dataset,
        batch_size=dataloader_cfg.batch_size,
        shuffle=dataloader_cfg.shuffle,
        num_workers=dataloader_cfg.num_workers,
        pin_memory=dataloader_cfg.pin_memory,
        collate_fn=collate,
    )


if __name__ == "__main__":
    dataset_cfg = DatasetConfig(episodes=[0])
    dataloader_cfg = DataloaderConfig(batch_size=8, shuffle=False)

    embedder = DinoV3Embedder()
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    embedder.to(device)

    loader = build_world_model_dataloader(
        dataset_cfg=dataset_cfg,
        dataloader_cfg=dataloader_cfg,
        embedder=embedder,
        device=device,
    )

    iterator = iter(loader)
    batches: List[WorldModelBatch] = []
    timings: List[Dict[str, float]] = []
    for _ in range(5):
        iter_start = time.perf_counter()
        try:
            current_batch = next(iterator)
        except StopIteration:
            break
        total_time = time.perf_counter() - iter_start
        profile = current_batch.profiling or {}
        collate_time = float(profile.get("collate_time", float("nan")))
        embed_time = float(profile.get("embed_time", float("nan")))
        target_length = float(profile.get("target_length", float("nan")))
        fetch_time = max(total_time - collate_time, 0.0) if not math.isnan(collate_time) else float("nan")
        timings.append(
            {
                "total": total_time,
                "collate": collate_time,
                "embed": embed_time,
                "fetch": fetch_time,
                "target_length": target_length,
            }
        )
        batches.append(current_batch)

    if not batches:
        raise RuntimeError("Failed to retrieve any batches from the dataloader for profiling.")

    first_batch = batches[0]
    print(f"sequence_latents: {tuple(first_batch.sequence_latents.shape)}")
    print(f"sequence_actions: {tuple(first_batch.sequence_actions.shape)}")
    print(f"single_frame_mask: {first_batch.single_frame_mask}")
    print(f"actions_dropped_mask: {first_batch.actions_dropped_mask}")

    for idx, timing in enumerate(timings, start=1):
        print(
            f"iteration_{idx}: total={timing['total']:.4f}s "
            f"fetch≈{timing['fetch']:.4f}s collate={timing['collate']:.4f}s "
            f"embed={timing['embed']:.4f}s target_length≈{timing['target_length']:.1f}"
        )
