from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from torchvision.datasets import Kinetics

from .batch import WorldBatch
from .utils import IMAGE_RESIZE_CROP_TRANSFORM_224

@dataclass
class KineticsDatasetConfig:
    root: str
    frames_per_clip: int = 16
    step_between_clips: int = 1

class KineticsDataset(Dataset):
    def __init__(
        self,
        dataset: Kinetics,
        cfg: KineticsDatasetConfig,
        action_dim: int,
    ):
        self.dataset = dataset
        self.cfg = cfg
        self.action_dim = action_dim

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> WorldBatch:
        video, _, _ = self.dataset[index]
        # video is [C, T, H, W] (uint8)
        video = video.permute(1, 0, 2, 3) # [T, C, H, W]
        video = video.to(torch.float32) / 255.0
        
        # Apply standard transform
        video = IMAGE_RESIZE_CROP_TRANSFORM_224(video)

        T = video.shape[0]
        
        actions = torch.zeros((T, self.action_dim), dtype=torch.float32)
        independent_frames_mask = torch.tensor(False, dtype=torch.bool)
        actions_mask = torch.zeros((T,), dtype=torch.bool)
        frames_valid_mask = torch.ones((T,), dtype=torch.bool)

        return WorldBatch(
            sequence_frames=video,
            sequence_actions=actions,
            independent_frames_mask=independent_frames_mask,
            actions_mask=actions_mask,
            frames_valid_mask=frames_valid_mask,
            dataset_indices=torch.tensor(-1, dtype=torch.long)
        )
