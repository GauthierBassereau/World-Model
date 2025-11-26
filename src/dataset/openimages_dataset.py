import random
from dataclasses import dataclass
from pathlib import Path
from typing import List
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from .common import WorldBatch
from .common import RESIZE_CROP_TRANSFORM_224

@dataclass
class OpenImagesDatasetConfig:
    root: str
    sequence_length: int = 15

class OpenImagesDataset(Dataset):
    def __init__(
        self,
        cfg: OpenImagesDatasetConfig,
        action_dim: int,
    ):
        self.cfg = cfg
        self.action_dim = action_dim
        
        self.sequence_length = cfg.sequence_length
        self.max_sequence_length = self.sequence_length
        
        root_path = Path(cfg.root)
        cache_file = root_path / ".image_list.txt"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                self.image_paths = [root_path / line.strip() for line in f if line.strip()]
        else:
            print(f"Scanning directory for images: {root_path}")
            print("This may take a while for the first time...")
            
            self.image_paths = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                self.image_paths.extend(root_path.rglob(ext))
            
            self.image_paths = sorted(self.image_paths)
            
            if len(self.image_paths) == 0:
                raise ValueError(
                    f"No images found in {root_path}. "
                    f"Make sure the images have been downloaded."
                )
            
            print(f"Saving {len(self.image_paths)} image paths to cache: {cache_file}")
            with open(cache_file, 'w') as f:
                for img_path in self.image_paths:
                    rel_path = img_path.relative_to(root_path)
                    f.write(f"{rel_path}\n")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> WorldBatch:
        images = []
        
        img_path = self.image_paths[index]
        img = self._load_image(img_path)
        images.append(img)
        
        for _ in range(self.max_sequence_length - 1):
            rand_idx = random.randint(0, len(self.image_paths) - 1)
            img_path = self.image_paths[rand_idx]
            img = self._load_image(img_path)
            images.append(img)

        frames = torch.stack(images, dim=0)
        
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        
        T = self.max_sequence_length
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
            dataset_indices=torch.tensor(-1, dtype=torch.long),
            dataset_names=torch.tensor(-1, dtype=torch.long),
            episode_ids=torch.tensor(-1, dtype=torch.long),
        )

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1)
        img = RESIZE_CROP_TRANSFORM_224(img)
        return img
