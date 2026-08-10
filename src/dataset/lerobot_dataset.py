import torch
import random
import json
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Any, Union
from pathlib import Path
import numpy as np

from src.training.logger import WorldModelLogger
from lerobot.datasets.lerobot_dataset import LeRobotDataset as LeRobotDatasetBackend
from .common import WorldBatch, build_resize_crop_transform
from src.dataset.common import get_delta_timestamps, get_actions

def _flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

@dataclass
class LeRobotDatasetConfig:
    repo_id: str = "aractingi/droid_1.0.1"
    cameras: Dict[str, Any] = field(
        default_factory=lambda: {
            "observation.images.exterior_1_left": 0.25,
            "observation.images.exterior_2_left": 0.25,
            "observation.images.wrist_left": 0.5,
        }
    )
    action_mode: Optional[str] = "soar_relative_ee_normalized"  # None for no actions (pretraining)
    episodes: Optional[List[int]] = None
    excluded_episodes: Optional[List[int]] = None
    episode_startpoint_only: bool = False
    sequence_length: int = 15
    fps: float = 3.0
    independent_frames_probability: float = 0.0
    use_action_probability: float = 1.0
    action_dim: int = 7
    image_size: int = 224

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")
        self.cameras = _flatten_dict(self.cameras)
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

        # Initialize backend with episodes=None to avoid loading all data into RAM, and because splitting the dataset is too expensive... :(
        self.backend = LeRobotDatasetBackend(
            repo_id=cfg.repo_id,
            episodes=None,
            image_transforms=build_resize_crop_transform(cfg.image_size),
            delta_timestamps=delta_timestamps,
            tolerance_s=1e-3,
            download_videos=True,
        )
        
        self.stats = self.backend.meta.stats
        self.selected_episodes = self._get_selected_episodes()
        self.native_fps = self._resolve_native_fps()
        self.available_frame_count = self._count_available_frames()
        self.duration_seconds = self.available_frame_count / self.native_fps if self.available_frame_count > 0 else 0.0
        self.indices = self._build_indices()
        self.camera_keys = list(self.cfg.cameras.keys())
        self.camera_probs = list(self.cfg.cameras.values())

    def _get_selected_episodes(self) -> List[int]:
        total_episodes = self.backend.meta.total_episodes
        available_episodes = set(range(total_episodes))

        if self.cfg.episodes is not None:
            available_episodes = available_episodes.intersection(set(self.cfg.episodes))

        if self.cfg.excluded_episodes is not None:
            available_episodes = available_episodes - set(self.cfg.excluded_episodes)

        return sorted(list(available_episodes))

    def _build_indices(self) -> List[int]:
        indices = []

        for ep_idx in self.selected_episodes:
            ep_meta = self.backend.meta.episodes[ep_idx]
            start_idx = ep_meta["dataset_from_index"]
            end_idx = ep_meta["dataset_to_index"]
            if self.cfg.episode_startpoint_only:
                indices.append(start_idx)
            else:
                indices.extend(range(start_idx, end_idx))

        return indices

    def _count_available_frames(self) -> int:
        frame_count = 0

        for ep_idx in self.selected_episodes:
            ep_meta = self.backend.meta.episodes[ep_idx]
            start_idx = int(ep_meta["dataset_from_index"])
            end_idx = int(ep_meta["dataset_to_index"])
            frame_count += max(0, end_idx - start_idx)

        return frame_count

    def effective_length(self, target_fps: float) -> float:
        if self.cfg.episode_startpoint_only:
            return float(len(self.indices))
        return self.duration_seconds * float(target_fps)

    def _resolve_native_fps(self) -> float:
        root = getattr(self.backend, "root", None)
        if root is None:
            raise RuntimeError(f"[LeRobotDataset] | backend.root is missing for {self.cfg.repo_id}.")

        info_path = Path(root) / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"[LeRobotDataset] | Missing dataset metadata file: {info_path}")

        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)

        try:
            fps = float(info["fps"])
        except KeyError as e:
            raise KeyError(f"[LeRobotDataset] | Missing 'fps' in dataset metadata: {info_path}") from e
        except (TypeError, ValueError) as e:
            raise ValueError(f"[LeRobotDataset] | Invalid 'fps' value in dataset metadata: {info_path}") from e

        if fps <= 0:
            raise ValueError(f"[LeRobotDataset] | Dataset fps must be > 0 in metadata: {info_path}")

        return fps

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> WorldBatch:
        return self.get_item(index)

    def get_item(
        self,
        index: int,
        *,
        python_rng: Optional[random.Random] = None,
        torch_generator: Optional[torch.Generator] = None,
    ) -> WorldBatch:
        global_idx = self.indices[index]
        item = self.backend[global_idx]
        rng = python_rng if python_rng is not None else random
        cam_key = rng.choices(self.camera_keys, weights=self.camera_probs, k=1)[0]
        sequence_frames = item[cam_key]

        sequence_actions = get_actions(
            self.cfg.action_mode, item,
            sequence_length=self.sequence_length,
            action_dim=self.cfg.action_dim,
            stats=self.stats,
        )

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

        independent_frames_mask = (
            torch.rand(self.sequence_length, generator=torch_generator)
            < self.cfg.independent_frames_probability
        )
        # only dependent frames can have actions
        actions_mask = torch.zeros(self.sequence_length, dtype=torch.bool)
        dependent_mask = ~independent_frames_mask
        num_dependent = dependent_mask.sum().item()
        
        if num_dependent > 0:
            actions_mask[dependent_mask] = (
                torch.rand(num_dependent, generator=torch_generator)
                < self.cfg.use_action_probability
            )
            actions_mask[0] = False
            actions_mask[padded_frames] = False

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
