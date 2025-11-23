import random
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch.utils.data import Dataset
from torchvision.datasets import Kinetics, ImageFolder

from .configs import LeRobotDatasetConfig, KineticsDatasetConfig, ImageNetDatasetConfig
from .batch import WorldModelBatch

class LeRobotDatasetWrapper(Dataset):
    def __init__(self, dataset: Any, cfg: LeRobotDatasetConfig):
        self.dataset = dataset
        self.cfg = cfg
        
        if not self.cfg.cameras:
            raise ValueError("LeRobotDatasetConfig.cameras must contain at least one camera key.")
        if not self.cfg.action_keys:
            raise ValueError("LeRobotDatasetConfig.action_keys must contain at least one key.")
        
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

    def __getitem__(self, index: int) -> WorldModelBatch:
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
        
        # dataset_indices will be set by WorldModelDataset
        return WorldModelBatch(
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


class KineticsDatasetWrapper(Dataset):
    def __init__(
        self,
        dataset: Kinetics,
        cfg: KineticsDatasetConfig,
        action_dim: int,
        transform: Optional[Any] = None
    ):
        self.dataset = dataset
        self.cfg = cfg
        self.action_dim = action_dim
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> WorldModelBatch:
        video, _, _ = self.dataset[index]
        # video is [C, T, H, W] (uint8)
        video = video.permute(1, 0, 2, 3) # [T, C, H, W]
        video = video.to(torch.float32) / 255.0
        
        if self.transform:
            # Apply transform to each frame or the whole video?
            # Transforms usually expect [C, H, W] or [T, C, H, W] depending on the transform.
            # _DROID_RESIZE_CROP_TRANSFORM expects [C, H, W] or batch.
            # Let's apply to the whole tensor if possible, or loop.
            # torchvision v2 transforms can handle [T, C, H, W]
            video = self.transform(video)

        T = video.shape[0]
        
        actions = torch.zeros((T, self.action_dim), dtype=torch.float32)
        independent_frames_mask = torch.tensor(False, dtype=torch.bool)
        actions_mask = torch.zeros((T,), dtype=torch.bool)
        frames_valid_mask = torch.ones((T,), dtype=torch.bool)

        return WorldModelBatch(
            sequence_frames=video,
            sequence_actions=actions,
            independent_frames_mask=independent_frames_mask,
            actions_mask=actions_mask,
            frames_valid_mask=frames_valid_mask,
            dataset_indices=torch.tensor(-1, dtype=torch.long)
        )


class ImageNetDatasetWrapper(Dataset):
    def __init__(
        self,
        dataset: ImageFolder,
        cfg: ImageNetDatasetConfig,
        action_dim: int,
        sequence_length: int = 16,
        transform: Optional[Any] = None
    ):
        self.dataset = dataset
        self.cfg = cfg
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> WorldModelBatch:
        # Sample T images
        images = []
        
        # First image from index
        img, _ = self.dataset[index]
        images.append(img)
        
        # Sample T-1 other images
        for _ in range(self.sequence_length - 1):
            rand_idx = random.randint(0, len(self.dataset) - 1)
            img, _ = self.dataset[rand_idx]
            images.append(img)
            
        # Stack images
        # img is PIL Image or Tensor? ImageFolder default loader is PIL.
        # We need to convert to tensor and apply transform.
        
        processed_images = []
        for img in images:
            if self.transform:
                img = self.transform(img)
            else:
                # Basic to tensor if no transform provided (should not happen in practice)
                import torchvision.transforms.functional as F
                img = F.to_tensor(img)
            processed_images.append(img)

        frames = torch.stack(processed_images, dim=0) # [T, C, H, W]
        
        T = self.sequence_length
        actions = torch.zeros((T, self.action_dim), dtype=torch.float32)
        independent_frames_mask = torch.tensor(True, dtype=torch.bool)
        actions_mask = torch.zeros((T,), dtype=torch.bool)
        frames_valid_mask = torch.ones((T,), dtype=torch.bool)

        return WorldModelBatch(
            sequence_frames=frames,
            sequence_actions=actions,
            independent_frames_mask=independent_frames_mask,
            actions_mask=actions_mask,
            frames_valid_mask=frames_valid_mask,
            dataset_indices=torch.tensor(-1, dtype=torch.long)
        )
