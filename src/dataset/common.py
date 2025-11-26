from typing import Dict
from dataclasses import dataclass
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as transforms_v2

@dataclass
class WorldBatch:
    sequence_frames: torch.Tensor
    sequence_actions: torch.Tensor
    independent_frames_mask: torch.Tensor
    actions_mask: torch.Tensor
    frames_valid_mask: torch.Tensor
    dataset_indices: torch.Tensor
    dataset_names: Dict[int, str]
    episode_ids: torch.Tensor


RESIZE_CROP_TRANSFORM_224 = transforms_v2.Compose(
    [
        transforms_v2.Resize(
            size=224,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms_v2.CenterCrop(224),
    ]
)