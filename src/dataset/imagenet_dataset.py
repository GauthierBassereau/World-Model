import random
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from .batch import WorldBatch
from .utils import IMAGE_RESIZE_CROP_TRANSFORM_224

@dataclass
class ImageNetDatasetConfig:
    root: str # TODO shoul be also using the same transform as kinetics and droid

class ImageNetDataset(Dataset):
    def __init__(
        self,
        dataset: ImageFolder,
        cfg: ImageNetDatasetConfig,
        action_dim: int,
        sequence_length: int = 16,
    ):
        self.dataset = dataset
        self.cfg = cfg
        self.action_dim = action_dim
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> WorldBatch:
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
        processed_images = []
        for img in images:
            # Apply standard transform
            img = IMAGE_RESIZE_CROP_TRANSFORM_224(img)
            processed_images.append(img)

        frames = torch.stack(processed_images, dim=0) # [T, C, H, W]
        
        T = self.sequence_length
        actions = torch.zeros((T, self.action_dim), dtype=torch.float32)
        independent_frames_mask = torch.tensor(True, dtype=torch.bool)
        actions_mask = torch.zeros((T,), dtype=torch.bool)
        frames_valid_mask = torch.ones((T,), dtype=torch.bool)

        return WorldBatch(
            sequence_frames=frames,
            sequence_actions=actions,
            independent_frames_mask=independent_frames_mask,
            actions_mask=actions_mask,
            frames_valid_mask=frames_valid_mask,
            dataset_indices=torch.tensor(-1, dtype=torch.long)
        )
