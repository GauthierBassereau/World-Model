import random
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Union, Any
import math
import logging

from .batch import WorldModelBatch

logger = logging.getLogger(__name__)

class WorldModelDataset(Dataset):
    def __init__(
        self,
        datasets: Dict[str, Dataset],
        weights: Dict[str, float],
        tolerance: int = 10
    ):
        self.datasets = datasets
        self.weights = weights
        self.tolerance = tolerance
        
        self.dataset_names = sorted(list(datasets.keys()))
        self.dataset_to_idx = {name: i for i, name in enumerate(self.dataset_names)}
        
        # Calculate total length and samples per dataset
        # Strategy: Length is determined by the dataset that requires the most samples 
        # to satisfy its weight, relative to its natural size?
        # Or simply sum of lengths?
        # User wants specific weights in batch.
        # If we use sum of lengths, the natural distribution might violate weights.
        # So we construct a virtual dataset.
        # We set total length such that the dataset with the highest (size / weight) ratio is fully covered?
        # No, usually we want to cover the largest dataset.
        # Let's use N = max(|D_i| / w_i). This ensures every dataset is visited at least once (on average) per epoch 
        # if we sample with replacement.
        # Actually, if we want to support "infinite" streaming, we just need __getitem__ to sample according to weights.
        # But map-style dataset needs __len__.
        
        dataset_lengths = {name: len(ds) for name, ds in datasets.items()}
        
        # Normalize weights
        total_weight = sum(weights.values())
        self.normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # Determine virtual epoch length
        # We want to visit every sample of every dataset at least once?
        # Or just the largest one?
        # Let's take the max required length to cover each dataset at least once.
        # required_total_for_i = len(i) / weight_i
        max_required_total = 0
        for name, length in dataset_lengths.items():
            w = self.normalized_weights[name]
            if w > 0:
                req = length / w
                if req > max_required_total:
                    max_required_total = req
        
        self.total_length = int(max_required_total)
        
        # Build the mapping
        # We will create a list of (dataset_name, index)
        # But this list would be huge (millions).
        # Instead, we can just sample randomly in __getitem__?
        # But map-style dataset usually expects deterministic indexing for reproducibility with same seed.
        # And DistributedSampler expects a fixed length.
        
        # If we want exact weights in the epoch, we can pre-generate the assignment of dataset source for each virtual index.
        # But we don't need to pre-generate the exact inner index if we sample randomly from the sub-dataset.
        # However, it's better to iterate over sub-datasets sequentially (shuffled) to avoid random access overhead if possible?
        # But we are mixing them.
        
        # Let's pre-calculate the number of samples for each dataset in this virtual epoch.
        self.samples_per_dataset = {
            name: int(self.total_length * w) for name, w in self.normalized_weights.items()
        }
        
        # Adjust to match total_length exactly
        current_total = sum(self.samples_per_dataset.values())
        diff = self.total_length - current_total
        # Add diff to the first dataset (arbitrary)
        if diff != 0 and self.dataset_names:
             self.samples_per_dataset[self.dataset_names[0]] += diff

        # Create a flat list of dataset indices
        # [0, 0, ..., 1, 1, ..., 2, 2, ...]
        # Then we can shuffle this list deterministically or just let it be and rely on DataLoader shuffle.
        # DataLoader shuffle will shuffle the indices passed to __getitem__.
        # So __getitem__(i) maps to a specific dataset source.
        
        self.virtual_map = []
        for name in self.dataset_names:
            count = self.samples_per_dataset[name]
            idx = self.dataset_to_idx[name]
            self.virtual_map.extend([idx] * count)
        
        # Now self.virtual_map[i] tells us which dataset to use for global index i.
        # But which sample within that dataset?
        # We can use (i % len(dataset))?
        # But since we have multiple blocks of same dataset, this might be biased if not shuffled.
        # But DataLoader shuffles indices.
        # So if we use (i % len(dataset)), and i is random, we access random element.
        # But wait, if we have [0, 0, 0, 1, 1], and we access index 3 (which is 1), we want a sample from dataset 1.
        # Which sample? We can maintain a counter or just hash i?
        # Or we can pre-assign inner indices.
        # To ensure we cover the dataset, we should assign inner indices [0, 1, ... len, 0, 1...] to the slots allocated for that dataset.
        
        self.indices_map = [0] * self.total_length
        
        # Fill indices_map
        # We need to distribute the inner indices for each dataset across the slots in virtual_map.
        # Since virtual_map is grouped by dataset (sorted), we can just fill sequentially.
        
        cursor = 0
        for name in self.dataset_names:
            count = self.samples_per_dataset[name]
            ds_len = dataset_lengths[name]
            # Generate indices: 0, 1, ... ds_len-1, 0, 1...
            indices = [k % ds_len for k in range(count)]
            # Assign
            for k in range(count):
                self.indices_map[cursor + k] = indices[k]
            cursor += count
            
        # Now self.virtual_map and self.indices_map are aligned.
        # self.virtual_map is sorted by dataset index.
        # This is fine because DataLoader with shuffle=True will access random global indices.

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, index: int) -> WorldModelBatch:
        dataset_idx = self.virtual_map[index]
        dataset_name = self.dataset_names[dataset_idx]
        inner_index = self.indices_map[index]
        
        dataset = self.datasets[dataset_name]
        
        # Resilience loop
        for attempt in range(self.tolerance):
            try:
                batch = dataset[inner_index]
                # Set dataset index
                batch.dataset_indices = torch.tensor(dataset_idx, dtype=torch.long)
                return batch
            except Exception as e:
                logger.warning(f"Failed to load sample from {dataset_name} at index {inner_index} (attempt {attempt+1}/{self.tolerance}): {e}")
                # Pick a random index from the same dataset
                inner_index = random.randint(0, len(dataset) - 1)
        
        raise RuntimeError(f"Failed to load sample from {dataset_name} after {self.tolerance} attempts.")
