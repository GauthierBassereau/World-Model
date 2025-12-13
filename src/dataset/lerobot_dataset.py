import torch
import random
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Any, Union
import numpy as np

from src.training.logger import WorldModelLogger
from lerobot.datasets.lerobot_dataset import LeRobotDataset as LeRobotDatasetBackend
from .common import WorldBatch, RESIZE_CROP_TRANSFORM_224
from src.dataset.common import get_delta_timestamps, get_actions

@dataclass
class LeRobotDatasetConfig:
    repo_id: str = "aractingi/droid_1.0.1"
    cameras: Dict[str, float] = field(
        default_factory=lambda: {
            "observation.images.exterior_1_left": 0.25,
            "observation.images.exterior_2_left": 0.25,
            "observation.images.wrist_left": 0.5,
        }
    )
    action_mode: str = "soar_ee_relative_normalized" # {soar_relative_ee_normalized, soar_relative_ee}
    episodes: Optional[List[int]] = None
    excluded_episodes: Optional[List[int]] = None
    episode_midpoint_only: bool = False
    sequence_length: int = 15
    fps: float = 3.0
    independent_frames_probability: float = 0.0
    use_action_probability: float = 1.0
    action_dim: int = 7

    def __post_init__(self) -> None:
        self.episodes = self._get_list(self.episodes)
        self.excluded_episodes = self._get_list(self.excluded_episodes)

    @staticmethod
    def _get_list(indices: Optional[Sequence[int]]) -> Optional[List[int]]:
        if indices is None:
            return None
        return [int(idx) for idx in indices]


class LeRobotDataset(Dataset):
    def __init__(self, cfg: LeRobotDatasetConfig, logger: WorldModelLogger):
        self.cfg = cfg
        self.fps = cfg.fps
        self.sequence_length = cfg.sequence_length
        self.logger = logger

        delta_timestamps = get_delta_timestamps(
            action_mode=self.cfg.action_mode,
            fps=self.fps,
            sequence_length=self.sequence_length,
            camera_keys=list(self.cfg.cameras.keys())
        )

        # Initialize backend with episodes=None to avoid loading all data into RAM, and because splitting the dataset is too expensive...
        self.backend = LeRobotDatasetBackend(
            repo_id=cfg.repo_id,
            episodes=None,
            image_transforms=RESIZE_CROP_TRANSFORM_224,
            delta_timestamps=delta_timestamps,
            tolerance_s=1e-3,
            download_videos=True,
        )
        
        self.stats = self.backend.meta.stats
        self.indices = self._build_indices()
        self.camera_keys = list(self.cfg.cameras.keys())
        self.camera_probs = list(self.cfg.cameras.values())

    def _build_indices(self) -> List[int]:
        indices = []

        total_episodes = self.backend.meta.total_episodes
        available_episodes = set(range(total_episodes))

        if self.cfg.episodes is not None:
            available_episodes = available_episodes.intersection(set(self.cfg.episodes))

        if self.cfg.excluded_episodes is not None:
            available_episodes = available_episodes - set(self.cfg.excluded_episodes)

        sorted_episodes = sorted(list(available_episodes))

        for ep_idx in sorted_episodes:
            ep_meta = self.backend.meta.episodes[ep_idx]
            start_idx = ep_meta["dataset_from_index"]
            end_idx = ep_meta["dataset_to_index"]
            if self.cfg.episode_midpoint_only:
                midpoint = (start_idx + end_idx) // 2
                indices.append(midpoint)
            else:
                indices.extend(range(start_idx, end_idx))

        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> WorldBatch:
        global_idx = self.indices[index]
        item = self.backend[global_idx]
        cam_key = random.choices(self.camera_keys, weights=self.camera_probs, k=1)[0]
        sequence_frames = item[cam_key]

        sequence_actions, validity_mask = get_actions(self.cfg.action_mode, item, self.stats)

        if sequence_actions.shape[-1] < self.cfg.action_dim:
            padding = torch.zeros(
                (*sequence_actions.shape[:-1], self.cfg.action_dim - sequence_actions.shape[-1]),
                device=sequence_actions.device,
                dtype=sequence_actions.dtype
            )
            sequence_actions = torch.cat([sequence_actions, padding], dim=-1)

        # Determine frames valid mask (handling padded frames)
        pad_key = f"{cam_key}_is_pad"
        if pad_key in item:
            padded_frames = item[pad_key]
        else:
            padded_frames = torch.zeros((self.sequence_length,), dtype=torch.bool)

        independent_frames_mask = torch.rand(self.sequence_length) < self.cfg.independent_frames_probability
        # only dependent frames can have actions
        actions_mask = torch.zeros(self.sequence_length, dtype=torch.bool)
        dependent_mask = ~independent_frames_mask
        num_dependent = dependent_mask.sum().item()
        
        if num_dependent > 0:
            actions_mask[dependent_mask] = torch.rand(num_dependent) < self.cfg.use_action_probability
            actions_mask[0] = False
            actions_mask[padded_frames] = False
            actions_mask = actions_mask & validity_mask

        episode_index = item["episode_index"]
        if isinstance(episode_index, torch.Tensor):
            episode_index = episode_index.item()

        return WorldBatch(
            sequence_frames=sequence_frames,
            sequence_actions=sequence_actions,
            independent_frames=independent_frames_mask,
            use_actions=actions_mask,
            padded_frames=padded_frames,
            dataset_indices=torch.tensor(-1, dtype=torch.long), # defined by worlddataset
            dataset_names=torch.tensor(-1, dtype=torch.long), # defined by worlddataset
            episode_ids=torch.tensor(episode_index, dtype=torch.long),
        )