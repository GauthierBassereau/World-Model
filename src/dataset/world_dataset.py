from typing import Dict, List, Tuple, Union, Any, Optional
from dataclasses import dataclass, field
import random
import math
import logging
import torch
from torch.utils.data import Dataset
import pyrallis

from .lerobot_dataset import LeRobotDataset, LeRobotDatasetConfig

from src.training.logger import WorldModelLogger
from .common import WorldBatch



def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@dataclass
class WorldDatasetConfig:
    weights: Dict[str, float]
    datasets: Dict[str, Any]
    reference_dataset: Optional[str] = None
    action_dim: int = 8
    sequence_length_distribution: Dict[Union[str, int], float] = field(default_factory=lambda: {15: 1.0})
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
        tolerance: int = 10,
        seed: Optional[int] = None,
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
            
            # Inject global settings if not present else override
            ds_conf_dict["sequence_length"] = max_sequence_length
            ds_conf_dict["fps"] = self.cfg.fps
            ds_conf_dict["action_dim"] = self.action_dim
            
            # This is kind of a hack... not good code
            if "cameras" in ds_conf_dict and isinstance(ds_conf_dict["cameras"], dict):
                ds_conf_dict["cameras"] = _flatten_dict(ds_conf_dict["cameras"])
            
            if ds_type == "lerobot":
                ds_cfg = pyrallis.decode(LeRobotDatasetConfig, ds_conf_dict)
                self.datasets[name] = LeRobotDataset(ds_cfg, logger=self.logger)
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

        self.create_virtual_map(seed=seed)

        self.logger.info(f"[WorldDataset] | World dataset initialized.")

    def create_virtual_map(self, generator: Optional[torch.Generator] = None, seed: Optional[int] = None) -> None:
        if generator is None and seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        # virtual_map: maps virtual_index -> dataset_index
        # indices_map: maps virtual_index -> inner_dataset_index
        self.virtual_map = []
        self.indices_map = []
        
        dataset_lengths = {name: len(ds) for name, ds in self.datasets.items()}
        
        for name in self.dataset_names:
            count = self.samples_per_dataset[name]
            dataset_idx = self.dataset_to_idx[name]
            ds_len = dataset_lengths[name]
            # Generate indices for this dataset
            # If count > ds_len, we repeat the dataset (shuffled) multiple times
            # If count < ds_len, we take a random subset
            indices = []
            while len(indices) < count:
                indices.extend(torch.randperm(ds_len, generator=generator).tolist())
            indices = indices[:count]
            
            self.virtual_map.extend([dataset_idx] * count)
            self.indices_map.extend(indices)

    def reshuffle_virtual_map(self, seed: Optional[int] = None) -> None:
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        self.create_virtual_map(generator)

    def __repr__(self) -> str:
        return f"[WorldDataset] total_length={self.total_length}, samples_per_dataset={self.samples_per_dataset})"

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, index: int) -> WorldBatch:
        dataset_idx = self.virtual_map[index]
        dataset_name = self.dataset_names[dataset_idx]
        inner_index = self.indices_map[index]
        
        dataset = self.datasets[dataset_name]
        
        # Resilience loop, sometimes some videos can be corrupted in droid... so we just pick a random index from the same dataset
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
