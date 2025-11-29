import random
import bisect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Any

import torch
from torch.utils.data import Dataset

from .common import WorldBatch
from lerobot.datasets.lerobot_dataset import LeRobotDataset as LeRobotDatasetLib
from .common import RESIZE_CROP_TRANSFORM_224


@dataclass
class LeRobotDatasetConfig:
    repo_id: str = "aractingi/droid_1.0.1"
    cameras: Tuple[str, ...] = (
        "observation.images.exterior_1_left",
        "observation.images.exterior_2_left",
        "observation.images.wrist_left",
    )
    camera_probabilities: Optional[Dict[str, float]] = None
    action_keys: Tuple[str, ...] = (
        "observation.state",
    )
    action_representation: str = "position"
    action_normalization: Optional[str] = "mean_std"
    action_normalization_params: Optional[Dict[str, List[float]]] = field(default_factory=lambda: {
        "mean": [0.01428347627459065, 0.23987777195473095, -0.014661363646208965, -2.027954954645529, -0.035476306435143545, 2.3233678805030076, 0.08326671319745274, 0.36085284714413735],
        "std": [0.3244343844169754, 0.5158281965633522, 0.2901322476548548, 0.5021871776809874, 0.5331921797732538, 0.46856794632856424, 0.7446577730524244, 0.40315882970033534],
    })
    episodes: Optional[List[int]] = None
    excluded_episodes: Optional[List[int]] = None
    episode_midpoint_only: bool = False
    sequence_length: int = 15
    fps: float = 3.0
    independent_frames_probability: float = 0.0
    drop_action_probability: float = 0.0

    def __post_init__(self) -> None:
        self.episodes = self._get_list(self.episodes)
        self.excluded_episodes = self._get_list(self.excluded_episodes)

    @staticmethod
    def _get_list(indices: Optional[Sequence[int]]) -> Optional[List[int]]:
        if indices is None:
            return None
        return [int(idx) for idx in indices]


def _compute_episode_midpoints(
    dataset: Any,
    episodes: Optional[List[int]],
) -> Tuple[List[int], List[int]]:
    """
    Return a single middle-frame index for each target episode (global index) and the episode ids used.

    When the dataset is initialized with a subset of episodes, the metadata still contains offsets for
    the full dataset. We therefore compute midpoints relative to the actually loaded subset ordering.
    """
    meta_eps = getattr(getattr(dataset, "meta", None), "episodes", None)
    if meta_eps is None:
        raise ValueError("Dataset does not expose episode metadata required for midpoint sampling.")

    target_episode_ids = list(episodes) if episodes is not None else list(range(len(meta_eps)))

    loaded_episodes = getattr(dataset, "episodes", None)
    loaded_episodes = list(loaded_episodes) if loaded_episodes is not None else list(range(len(meta_eps)))

    midpoint_map: Dict[int, int] = {}
    current_idx = 0
    for ep_id in loaded_episodes:
        if ep_id < 0 or ep_id >= len(meta_eps):
            raise ValueError(
                f"Episode id {ep_id} requested but metadata only contains {len(meta_eps)} episodes."
            )
        ep = meta_eps[ep_id]
        length = ep.get("length")
        if length is None:
            raise ValueError(f"Episode {ep_id} is missing a length field required for midpoint calculation.")
        if ep_id in target_episode_ids:
            midpoint_map[ep_id] = current_idx + int(length) // 2
        current_idx += int(length)

    missing = [ep_id for ep_id in target_episode_ids if ep_id not in midpoint_map]
    if missing:
        raise ValueError(
            "Failed to compute midpoint indices for requested episodes (not present in loaded subset): "
            + ", ".join(str(ep) for ep in missing)
        )

    midpoint_indices = [midpoint_map[ep_id] for ep_id in target_episode_ids]
    return midpoint_indices, target_episode_ids


def _create_delta_timestamps(
    dataset_cfg: "LeRobotDatasetConfig",
    sequence_length: int = None,
) -> Dict[str, Sequence[float]]:
    """Ensure delta timestamps are provided for all cameras and action key."""
    max_length = sequence_length if sequence_length is not None else dataset_cfg.sequence_length
    frame_delta_seconds = 1.0 / dataset_cfg.fps
    # Generate offsets starting from 0
    offsets = [frame_delta_seconds * i for i in range(max_length)]
    delta = {camera: list(offsets) for camera in dataset_cfg.cameras}

    for key in dataset_cfg.action_keys:
        delta[key] = list(offsets)
    return delta


class LeRobotDataset(Dataset):
    def __init__(self, cfg: LeRobotDatasetConfig):
        self.cfg = cfg
        
        if not self.cfg.cameras:
            raise ValueError("LeRobotDatasetConfig.cameras must contain at least one camera key.")
        if not self.cfg.action_keys:
            raise ValueError("LeRobotDatasetConfig.action_keys must contain at least one key.")
        
        self.sequence_length = cfg.sequence_length
        delta_timestamps = _create_delta_timestamps(cfg, self.sequence_length)
        
        episodes = cfg.episodes
        if episodes is not None and cfg.excluded_episodes is not None:
            excluded_set = set(cfg.excluded_episodes)
            episodes = [e for e in episodes if e not in excluded_set]

        self.dataset = LeRobotDatasetLib(
            cfg.repo_id,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=RESIZE_CROP_TRANSFORM_224,
            tolerance_s=0.001,
        )

        self.custom_index_map = False
        self.midpoint_indices = None
        self.midpoint_episode_ids = None

        if cfg.episode_midpoint_only:
            self.custom_index_map = True
            self.midpoint_indices, self.midpoint_episode_ids = _compute_episode_midpoints(
                self.dataset, cfg.episodes
            )
        elif cfg.episodes is None and cfg.excluded_episodes is not None:
            self.custom_index_map = True
            self.valid_ranges = []
            self.cumulative_lengths = [0]

            all_episodes = self.dataset.meta.episodes
            excluded_set = set(cfg.excluded_episodes)

            current_cum_len = 0
            for ep in all_episodes:
                ep_idx = ep["episode_index"]
                if isinstance(ep_idx, list):
                    ep_idx = ep_idx[0]

                if ep_idx in excluded_set:
                    continue

                start = ep["dataset_from_index"]
                end = ep["dataset_to_index"]
                if isinstance(start, list):
                    start = start[0]
                if isinstance(end, list):
                    end = end[0]

                length = end - start
                if length > 0:
                    self.valid_ranges.append((start, end))
                    current_cum_len += length
                    self.cumulative_lengths.append(current_cum_len)
        
        self.max_sequence_length = self.sequence_length
        
        if self.cfg.camera_probabilities:
            weights = [
                self.cfg.camera_probabilities.get(camera, 0.0) for camera in self.cfg.cameras
            ]
        else:
            weights = [1.0] * len(self.cfg.cameras)

        total = sum(weights)
        self.camera_weights = [weight / total for weight in weights]
        self.camera_keys = list(self.cfg.cameras)
        self._normalization_dim: Optional[int] = None
        self._normalization_params: Optional[Dict[str, torch.Tensor]] = None

    def __len__(self) -> int:
        if self.custom_index_map:
            if self.cfg.episode_midpoint_only:
                return len(self.midpoint_indices)
            return self.cumulative_lengths[-1]
        return len(self.dataset)

    def __getitem__(self, index: int) -> WorldBatch:
        episode_id = -1
        if self.custom_index_map:
            if self.cfg.episode_midpoint_only:
                real_index = self.midpoint_indices[index]
                episode_id = self.midpoint_episode_ids[index]
                sample = self.dataset[real_index]
            else:
                if index < 0 or index >= self.cumulative_lengths[-1]:
                    raise IndexError(f"Index {index} out of range")
                k = bisect.bisect_right(self.cumulative_lengths, index) - 1
                offset = index - self.cumulative_lengths[k]
                real_start = self.valid_ranges[k][0]
                real_index = real_start + offset
                sample = self.dataset[real_index]
                # Try to infer episode_id if possible, though not strictly required for non-midpoint
                # But for consistency, we might want to leave it as -1 or try to find it.
                # For now, leaving as -1 for non-midpoint to avoid expensive lookups unless needed.
        else:
            sample = self.dataset[index]
        
        # Always return max length sequences - collator will crop them
        target_length = self.max_sequence_length
        camera_key = random.choices(self.camera_keys, weights=self.camera_weights, k=1)[0]
        
        frames = self._prepare_frames(sample, camera_key, target_length)
        validity_mask = self._prepare_valid_mask(sample, camera_key, target_length)
        actions = self._prepare_actions(sample, target_length)

        use_independent_frame = random.random() < self.cfg.independent_frames_probability
        drop_actions = random.random() < self.cfg.drop_action_probability

        if use_independent_frame or drop_actions:
            actions.zero_()

        independent_frames_mask = torch.tensor(use_independent_frame, dtype=torch.bool)
        
        base_action_mask = torch.full(
            (target_length,),
            not (use_independent_frame or drop_actions),
            dtype=torch.bool,
        )
        actions_mask = base_action_mask & validity_mask
        
        # dataset_indices will be set by WorldDataset
        return WorldBatch(
            sequence_frames=frames,
            sequence_actions=actions,
            independent_frames_mask=independent_frames_mask,
            actions_mask=actions_mask,
            frames_valid_mask=validity_mask,
            dataset_indices=torch.tensor(-1, dtype=torch.long),
            dataset_names=torch.tensor(-1, dtype=torch.long),
            episode_ids=torch.tensor(episode_id, dtype=torch.long),
        )

    def _prepare_frames(self, sample: Dict[str, torch.Tensor], camera_key: str, target_length: int) -> torch.Tensor:
        frames = sample[camera_key]
        frames = frames.to(torch.uint8)
        return frames[:target_length].contiguous()

    def _prepare_valid_mask(self, sample: Dict[str, torch.Tensor], camera_key: str, target_length: int) -> torch.Tensor:
        pad_key = f"{camera_key}_is_pad"
        if pad_key in sample:
            pad_tensor = sample[pad_key].to(dtype=torch.bool)
        else:
            pad_tensor = torch.zeros((self.max_sequence_length,), dtype=torch.bool)
        valid = (~pad_tensor).to(dtype=torch.bool)
        return valid[:target_length].contiguous()

    def _prepare_actions(self, sample: Dict[str, torch.Tensor], target_length: int) -> torch.Tensor:
        action_parts: List[torch.Tensor] = []
        for key in self.cfg.action_keys:
            part = sample[key]
            if part.ndim == 1:
                part = part.unsqueeze(-1)
            action_parts.append(part.to(torch.float32))

        actions = torch.cat(action_parts, dim=-1)[:target_length].contiguous()
        if self.cfg.action_representation == "delta":
            deltas = actions[1:] - actions[:-1]
            result = torch.zeros((target_length, deltas.shape[-1]), dtype=deltas.dtype, device=deltas.device)
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
        offset = self._normalization_params["offset"].to(device=actions.device, dtype=actions.dtype)
        scale = self._normalization_params["scale"].to(device=actions.device, dtype=actions.dtype)
        return (actions - offset) / scale

    def _ensure_normalization_tensors(self, feature_dim: int) -> None:
        if self._normalization_params is not None:
            if self._normalization_dim != feature_dim:
                raise ValueError("Configured action normalization dimension does not match the action tensor.")
            return
        params = self.cfg.action_normalization_params
        if params is None:
            raise ValueError("action_normalization_params are required for normalization.")
        norm_type = self.cfg.action_normalization
        if norm_type == "min_max":
            raw_min = torch.as_tensor(params["min"], dtype=torch.float32)
            raw_max = torch.as_tensor(params["max"], dtype=torch.float32)
            scale = raw_max - raw_min
            offset = raw_min
        elif norm_type == "mean_std":
            raw_mean = torch.as_tensor(params["mean"], dtype=torch.float32)
            raw_std = torch.as_tensor(params["std"], dtype=torch.float32)
            offset = raw_mean
            scale = raw_std
        else:
            raise ValueError("Unsupported action normalization type provided.")
        self._normalization_dim = feature_dim
        self._normalization_params = {"offset": offset, "scale": scale}


