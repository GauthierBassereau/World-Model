from typing import Dict, List, Tuple, Union, Any, Optional
import random
import math
import logging
import torch
from torch.utils.data import Dataset
from torchvision.datasets import Kinetics
from lerobot.datasets.lerobot_dataset import LeRobotDataset as RawLeRobotDataset, LeRobotDatasetMetadata

from .batch import WorldBatch
from .configs import WorldDatasetConfig
from .droid_dataset import DroidDataset, DroidDatasetConfig, _ensure_delta_timestamps
from .kinetics_dataset import KineticsDataset, KineticsDatasetConfig
from .openimages_dataset import OpenImagesDataset, OpenImagesDatasetConfig
from .utils import IMAGE_RESIZE_CROP_TRANSFORM_224

logger = logging.getLogger(__name__)

class WorldDataset(Dataset):
    def __init__(
        self,
        cfg: WorldDatasetConfig,
        tolerance: int = 10
    ):
        self.cfg = cfg
        self.weights = cfg.weights
        self.tolerance = tolerance
        self.action_dim = cfg.action_dim
        
        self.datasets: Dict[str, Dataset] = {}
        
        # Initialize datasets
        for name, ds_cfg in self.cfg.datasets.items():
            if isinstance(ds_cfg, DroidDatasetConfig):
                metadata = LeRobotDatasetMetadata(ds_cfg.repo_id)
                delta_timestamps = _ensure_delta_timestamps(ds_cfg, metadata)
                
                dataset = RawLeRobotDataset(
                    ds_cfg.repo_id,
                    episodes=ds_cfg.episodes,
                    delta_timestamps=delta_timestamps,
                    image_transforms=IMAGE_RESIZE_CROP_TRANSFORM_224,
                )
                
                self.datasets[name] = DroidDataset(dataset, ds_cfg)
                
            elif isinstance(ds_cfg, KineticsDatasetConfig):
                dataset = Kinetics(
                    root=ds_cfg.root,
                    frames_per_clip=ds_cfg.frames_per_clip,
                    step_between_clips=ds_cfg.step_between_clips,
                )
                self.datasets[name] = KineticsDataset(
                    dataset, 
                    ds_cfg, 
                    action_dim=self.action_dim,
                )
                
            elif isinstance(ds_cfg, OpenImagesDatasetConfig):
                self.datasets[name] = OpenImagesDataset(
                    ds_cfg, 
                    action_dim=self.action_dim,
                )

        self.dataset_names = sorted(list(self.datasets.keys()))
        self.dataset_to_idx = {name: i for i, name in enumerate(self.dataset_names)}
        
        dataset_lengths = {name: len(ds) for name, ds in self.datasets.items()}
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.normalized_weights = {k: v / total_weight for k, v in self.weights.items()}
        
        # 1. Determine Virtual Epoch Length
        # We calculate the total length of the virtual epoch such that every dataset is visited 
        # at least once on average, respecting the specified weights.
        # N = max(|D_i| / w_i)
        max_required_total = 0
        for name, length in dataset_lengths.items():
            w = self.normalized_weights[name]
            if w > 0:
                req = length / w
                if req > max_required_total:
                    max_required_total = req
        
        self.total_length = int(max_required_total)
        
        # 2. Allocate Samples per Dataset
        # Calculate how many samples each dataset contributes to the virtual epoch based on weights.
        self.samples_per_dataset = {
            name: int(self.total_length * w) for name, w in self.normalized_weights.items()
        }
        
        # Adjust to match total_length exactly by adding remainder to the first dataset
        current_total = sum(self.samples_per_dataset.values())
        diff = self.total_length - current_total
        if diff != 0 and self.dataset_names:
             self.samples_per_dataset[self.dataset_names[0]] += diff

        # 3. Build Virtual Map (Global Index -> Dataset Index)
        # Create a mapping where each index in the virtual epoch points to a specific dataset.
        # We create blocks of indices for each dataset. The DataLoader's shuffling will mix these.
        self.virtual_map = []
        for name in self.dataset_names:
            count = self.samples_per_dataset[name]
            idx = self.dataset_to_idx[name]
            self.virtual_map.extend([idx] * count)
        
        # 4. Build Indices Map (Global Index -> Inner Dataset Index)
        # Assign specific indices from the underlying datasets to the virtual slots.
        # We iterate sequentially (modulo dataset length) to ensure full coverage of each dataset.
        self.indices_map = [0] * self.total_length
        
        cursor = 0
        for name in self.dataset_names:
            count = self.samples_per_dataset[name]
            ds_len = dataset_lengths[name]
            # Generate indices: 0, 1, ... ds_len-1, 0, 1...
            indices = [k % ds_len for k in range(count)]
            # Assign to the corresponding slots in the map
            for k in range(count):
                self.indices_map[cursor + k] = indices[k]
            cursor += count

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, index: int) -> WorldBatch:
        dataset_idx = self.virtual_map[index]
        dataset_name = self.dataset_names[dataset_idx]
        inner_index = self.indices_map[index]
        
        dataset = self.datasets[dataset_name]
        
        # Resilience loop, sometimes some images are corrupted in droid...
        for attempt in range(self.tolerance):
            try:
                batch = dataset[inner_index]
                batch.dataset_indices = torch.tensor(dataset_idx, dtype=torch.long)
                return batch
            except Exception as e:
                logger.warning(f"Failed to load sample from {dataset_name} at index {inner_index} (attempt {attempt+1}/{self.tolerance}): {e}")
                # Pick a random index from the same dataset
                inner_index = random.randint(0, len(dataset) - 1)
        
        raise RuntimeError(f"Failed to load sample from {dataset_name} after {self.tolerance} attempts.")
