from typing import Dict, List, Tuple, Union, Any, Optional
from dataclasses import dataclass, field
import random
import math
import logging
import torch
from torch.utils.data import Dataset
import dacite

from .lerobot_dataset import LeRobotDataset, LeRobotDatasetConfig
from .video_dataset import VideoDataset, VideoDatasetConfig
from .image_dataset import ImageDataset, ImageDatasetConfig

from src.training.logger import WorldModelLogger
from .common import WorldBatch


@dataclass
class WorldDatasetConfig:
    weights: Dict[str, float]
    datasets: Dict[str, Any]
    reference_dataset: Optional[str] = None
    action_dim: int = 8
    sequence_length_distribution: Dict[int, float] = field(default_factory=lambda: {15: 1.0})
    fps: float = 3.0

    def __post_init__(self) -> None:
        if set(self.datasets.keys()) != set(self.weights.keys()):
            raise ValueError(f"Keys in datasets {list(self.datasets.keys())} and weights {list(self.weights.keys())} must match.")
        
        if not self.sequence_length_distribution:
            raise ValueError("WorldDatasetConfig.sequence_length_distribution must contain at least one entry.")

        self.sequence_length_distribution = {
            int(length): float(weight)
            for length, weight in self.sequence_length_distribution.items()
        }
        
        if self.reference_dataset is None and self.weights:
            # Default to the first dataset in weights if not provided
            self.reference_dataset = list(self.weights.keys())[0]
        
        if self.reference_dataset and self.reference_dataset not in self.datasets:
             raise ValueError(f"Reference dataset {self.reference_dataset} not found in datasets.")

class WorldDataset(Dataset):
    def __init__(
        self,
        cfg: WorldDatasetConfig,
        logger: WorldModelLogger,
        tolerance: int = 10
    ):
        self.cfg = cfg
        self.weights = cfg.weights
        self.tolerance = tolerance
        self.action_dim = cfg.action_dim
        self.logger = logger

        if not self.cfg.datasets:
            raise ValueError("WorldDatasetConfig.datasets must contain at least one dataset.")
        
        self.datasets: Dict[str, Dataset] = {}
        
        max_sequence_length = max(self.cfg.sequence_length_distribution.keys())
        
        for name, ds_conf_dict in self.cfg.datasets.items():
            ds_type = ds_conf_dict.get("type")
            if not ds_type:
                raise ValueError(f"Dataset {name} missing 'type' field.")
            
            # Prepare config dict: remove 'type' and inject global settings
            ds_conf_dict = ds_conf_dict.copy()
            del ds_conf_dict["type"]
            
            # Inject global settings if not present (or override? User said "pass the rest of the config")
            # Usually we want the global config to enforce these, but individual configs might need them.
            # The previous code overrode them.
            ds_conf_dict["sequence_length"] = max_sequence_length
            ds_conf_dict["fps"] = self.cfg.fps
            
            if ds_type == "lerobot":
                ds_cfg = dacite.from_dict(data_class=LeRobotDatasetConfig, data=ds_conf_dict)
                self.datasets[name] = LeRobotDataset(ds_cfg)
            elif ds_type == "video":
                ds_cfg = dacite.from_dict(data_class=VideoDatasetConfig, data=ds_conf_dict)
                self.datasets[name] = VideoDataset(ds_cfg, action_dim=self.action_dim)
            elif ds_type == "image":
                ds_cfg = dacite.from_dict(data_class=ImageDatasetConfig, data=ds_conf_dict)
                self.datasets[name] = ImageDataset(ds_cfg, action_dim=self.action_dim)
            else:
                raise ValueError(f"Unknown dataset type: {ds_type}")

        self.dataset_names = sorted(list(self.datasets.keys()))
        self.dataset_to_idx = {name: i for i, name in enumerate(self.dataset_names)}
        self.idx_to_dataset = {i: name for name, i in self.dataset_to_idx.items()}
        
        dataset_lengths = {name: len(ds) for name, ds in self.datasets.items()}
        
        total_weight = sum(self.weights.values())
        self.normalized_weights = {k: v / total_weight for k, v in self.weights.items()}

        ref_len = dataset_lengths[self.cfg.reference_dataset]
        ref_weight = self.normalized_weights[self.cfg.reference_dataset]
        theoretical_total_length = int(ref_len / ref_weight)
        
        self.samples_per_dataset = {
            name: int(theoretical_total_length * w) for name, w in self.normalized_weights.items()
        }
        self.total_length = sum(self.samples_per_dataset.values())

        # virtual_map: maps virtual_index -> dataset_index
        # indices_map: maps virtual_index -> inner_dataset_index
        self.virtual_map = []
        self.indices_map = []
        
        for name in self.dataset_names:
            count = self.samples_per_dataset[name]
            dataset_idx = self.dataset_to_idx[name]
            ds_len = dataset_lengths[name]
            # Generate indices for this dataset
            # If count > ds_len, we repeat the dataset (shuffled) multiple times
            # If count < ds_len, we take a random subset
            indices = []
            while len(indices) < count:
                indices.extend(torch.randperm(ds_len).tolist())
            indices = indices[:count]
            
            self.virtual_map.extend([dataset_idx] * count)
            self.indices_map.extend(indices)

    def __repr__(self) -> str:
        return f"[WorldDataset] total_length={self.total_length}, samples_per_dataset={self.samples_per_dataset})"

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, index: int) -> WorldBatch:
        dataset_idx = self.virtual_map[index]
        dataset_name = self.dataset_names[dataset_idx]
        inner_index = self.indices_map[index]
        
        dataset = self.datasets[dataset_name]
        
        # Resilience loop, sometimes some videos are corrupted in droid...
        for attempt in range(self.tolerance):
            try:
                batch = dataset[inner_index]
                batch.dataset_indices = torch.tensor(dataset_idx, dtype=torch.long)
                batch.dataset_names = self.idx_to_dataset
                return batch
            except Exception as e:
                self.logger.warning(f"Failed to load sample from {dataset_name} at index {inner_index} (attempt {attempt+1}/{self.tolerance}): {e}")
                inner_index = random.randint(0, len(dataset) - 1) # just picking a random index from the same dataset
        
        raise RuntimeError(f"Failed to load sample from {dataset_name} after {self.tolerance} attempts.")
