import random
from dataclasses import dataclass
from pathlib import Path
from typing import List
import torch
from torch.utils.data import Dataset
from PIL import Image

from .batch import WorldBatch
from .utils import IMAGE_RESIZE_CROP_TRANSFORM_224

@dataclass
class OpenImagesDatasetConfig:
    root: str  # Path to the OpenImages v7 dataset directory
    split: str = "train"  # train, validation, or test

class OpenImagesDataset(Dataset):
    """
    Dataset wrapper for OpenImages v7 downloaded via FiftyOne.
    
    OpenImages is better suited for world modeling than ImageNet because:
    - More diverse, real-world images (9M images vs 1.2M)
    - Images from Flickr with natural scene diversity
    - Better representation of everyday objects and scenarios
    - More suitable for learning general visual representations
    """
    
    def __init__(
        self,
        cfg: OpenImagesDatasetConfig,
        action_dim: int,
        sequence_length: int = 16,
    ):
        self.cfg = cfg
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        
        # Build list of image paths
        # FiftyOne downloads images to: {root}/data/{image_id}.jpg
        data_dir = Path(cfg.root) / "data"
        
        if not data_dir.exists():
            raise ValueError(
                f"OpenImages data directory not found at {data_dir}. "
                f"Make sure the dataset has been downloaded via FiftyOne."
            )
        
        # Collect all image paths
        self.image_paths: List[Path] = sorted(list(data_dir.glob("*.jpg")))
        
        if len(self.image_paths) == 0:
            raise ValueError(
                f"No images found in {data_dir}. "
                f"The download might still be in progress or failed."
            )
        
        print(f"Loaded OpenImages {cfg.split} split with {len(self.image_paths)} images")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> WorldBatch:
        # Sample T images
        images = []
        
        # First image from index
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert("RGB")
        images.append(img)
        
        # Sample T-1 other images randomly
        for _ in range(self.sequence_length - 1):
            rand_idx = random.randint(0, len(self.image_paths) - 1)
            img_path = self.image_paths[rand_idx]
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            
        # Stack images
        processed_images = []
        for img in images:
            # Apply standard transform (resize and crop to 224x224)
            img = IMAGE_RESIZE_CROP_TRANSFORM_224(img)
            processed_images.append(img)

        frames = torch.stack(processed_images, dim=0)  # [T, C, H, W]
        
        T = self.sequence_length
        # Zero actions since this is an image-only dataset
        actions = torch.zeros((T, self.action_dim), dtype=torch.float32)
        # Mark frames as independent (not from a video sequence)
        independent_frames_mask = torch.tensor(True, dtype=torch.bool)
        # No action supervision
        actions_mask = torch.zeros((T,), dtype=torch.bool)
        # All frames are valid
        frames_valid_mask = torch.ones((T,), dtype=torch.bool)

        return WorldBatch(
            sequence_frames=frames,
            sequence_actions=actions,
            independent_frames_mask=independent_frames_mask,
            actions_mask=actions_mask,
            frames_valid_mask=frames_valid_mask,
            dataset_indices=torch.tensor(-1, dtype=torch.long)
        )
