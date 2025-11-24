import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any

import torch
from torch.utils.data import Dataset

from .batch import WorldBatch
from .resilient_dataset import ResilientLeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

def _coerce_frame_delta(value: Union[float, str]) -> float:
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

@dataclass
class DroidDatasetConfig:
    repo_id: str = "aractingi/droid_1.0.1"
    decoder_retry_attempts: int = 5
    cameras: Tuple[str, ...] = (
        "observation.images.exterior_1_left",
        "observation.images.exterior_2_left",
        "observation.images.wrist_left",
    )
    camera_probabilities: Optional[Dict[str, float]] = None
    action_keys: Tuple[str, ...] = (
        "observation.state",
    )
    action_representation: str = "delta"
    action_normalization: Optional[str] = None
    action_normalization_params: Optional[Dict[str, List[float]]] = None

    episodes: Optional[List[int]] = None
    episode_midpoint_only: bool = False
    sequence_length_distribution: Dict[int, float] = field(default_factory=lambda: {4: 1.0})
    frame_delta_seconds: Union[float, str] = 5.0 / 15.0
    independent_frames_probability: float = 0.0
    drop_action_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.decoder_retry_attempts < 1:
            raise ValueError("decoder_retry_attempts must be >= 1.")
        self.frame_delta_seconds = _coerce_frame_delta(self.frame_delta_seconds)
        if not self.sequence_length_distribution:
            raise ValueError(
                "DroidDatasetConfig.sequence_length_distribution must contain at least one entry."
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
            raise ValueError("DroidDatasetConfig.action_keys must contain at least one key.")
        self.action_keys = keys
        if self.action_representation not in {"delta", "position"}:
            raise ValueError(
                "DroidDatasetConfig.action_representation must be either 'delta' or 'position'."
            )

        if self.action_normalization not in {None, "min_max", "mean_std"}:
            raise ValueError(
                "DroidDatasetConfig.action_normalization must be one of None, 'min_max', or 'mean_std'."
            )
        if self.action_normalization is None:
            if self.action_normalization_params is not None:
                raise ValueError(
                    "action_normalization_params provided without a corresponding action_normalization."
                )
        else:
            if self.action_normalization_params is None:
                raise ValueError(
                    "action_normalization_params must be provided when action_normalization is set."
                )
            expected_keys = {"min", "max"} if self.action_normalization == "min_max" else {"mean", "std"}
            provided_keys = set(self.action_normalization_params.keys())
            missing = expected_keys - provided_keys
            extra = provided_keys - expected_keys
            if missing:
                raise ValueError(
                    "action_normalization_params missing required entries: " + ", ".join(sorted(missing))
                )
            if extra:
                raise ValueError(
                    "action_normalization_params contains unexpected entries: " + ", ".join(sorted(extra))
                )
            first_key = next(iter(expected_keys))
            first_value = self.action_normalization_params[first_key]
            if not isinstance(first_value, Sequence):
                raise TypeError(
                    f"action_normalization_params['{first_key}'] must be a sequence of floats."
                )
            reference_length = len(first_value)
            for key in expected_keys:
                value = self.action_normalization_params[key]
                if not isinstance(value, Sequence):
                    raise TypeError(
                        f"action_normalization_params['{key}'] must be a sequence of floats."
                    )
                if len(value) != reference_length:
                    raise ValueError(
                        "All action_normalization_params sequences must have the same length."
                    )
        self.episodes = self._normalize_episode_indices(self.episodes)

    @staticmethod
    def _normalize_episode_indices(indices: Optional[Sequence[int]]) -> Optional[List[int]]:
        if indices is None:
            return None
        normalized = [int(idx) for idx in indices]
        return normalized or None


def _compute_episode_midpoints(
    dataset: ResilientLeRobotDataset,
    episodes: Optional[List[int]],
) -> Tuple[List[int], List[int]]:
    """
    Return a single middle-frame index for each target episode (global index) and the episode ids used.

    When the dataset is initialized with a subset of episodes, the metadata still contains offsets for
    the full dataset. We therefore compute midpoints relative to the actually loaded subset ordering.
    """
    meta_eps = getattr(getattr(dataset, "meta", None), "episodes", None)
    if meta_eps is None:
        raise AttributeError("Dataset does not expose episode metadata required for midpoint sampling.")

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


def _ensure_delta_timestamps(
    dataset_cfg: "DroidDatasetConfig",
    metadata: LeRobotDatasetMetadata,
) -> Dict[str, Sequence[float]]:
    """Ensure delta timestamps are provided for all cameras and action key."""
    max_length = max(int(length) for length in dataset_cfg.sequence_length_distribution.keys())
    step = dataset_cfg.frame_delta_seconds
    if step <= 0:
        step = 1.0 / metadata.fps
    # Generate offsets symmetric around 0
    offsets = [step * (i - (max_length - 1) / 2) for i in range(max_length)]
    delta = {camera: list(offsets) for camera in dataset_cfg.cameras}

    for key in dataset_cfg.action_keys:
        delta[key] = list(offsets)
    return delta


class DroidDataset(Dataset):
    def __init__(self, dataset: Any, cfg: DroidDatasetConfig):
        self.dataset = dataset
        self.cfg = cfg
        
        if not self.cfg.cameras:
            raise ValueError("DroidDatasetConfig.cameras must contain at least one camera key.")
        if not self.cfg.action_keys:
            raise ValueError("DroidDatasetConfig.action_keys must contain at least one key.")
        
        self.max_sequence_length = max(
            int(length) for length in self.cfg.sequence_length_distribution.keys()
        )
        
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

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> WorldBatch:
        sample = self.dataset[index]
        
        target_length = self._sample_sequence_length()
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
            dataset_indices=torch.tensor(-1, dtype=torch.long) 
        )

    def _prepare_frames(self, sample: Dict[str, torch.Tensor], camera_key: str, target_length: int) -> torch.Tensor:
        frames = sample[camera_key]
        # Frames are already [T, C, H, W] from LeRobotDataset
        frames = frames.to(torch.float32)
        if frames.max() > 1.5:
            frames = frames / 255.0
        frames = frames.clamp(0.0, 1.0)
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

    def _build_sequence_length_distribution(self) -> Tuple[List[int], List[float]]:
        choices: List[int] = []
        weights: List[float] = []
        for length, weight in self.cfg.sequence_length_distribution.items():
            choices.append(int(length))
            weights.append(float(weight))
        total = sum(weights)
        normalized = [weight / total for weight in weights]
        return choices, normalized

    def _sample_sequence_length(self) -> int:
        return random.choices(self.sequence_length_choices, weights=self.sequence_length_probs, k=1)[0]
