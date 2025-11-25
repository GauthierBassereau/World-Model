from typing import Optional, Dict, Any
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from .world_dataset import WorldDataset, WorldDatasetConfig
from .collator import StackCollator

@dataclass
class DataloaderConfig:
    batch_size: int = 2
    shuffle: bool = False
    num_workers: int = 0
    pin_memory: bool = True

def build_world_dataloader(
    dataset_cfg: WorldDatasetConfig,
    dataloader_cfg: DataloaderConfig,
    grad_accum_steps: int = 1,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> DataLoader:
    effective_world_size = world_size or 1
    distributed = effective_world_size > 1

    if dataloader_cfg.batch_size % grad_accum_steps != 0:
        raise ValueError("Global batch size must be divisible by grad_accum_steps.")
    global_micro_batch = dataloader_cfg.batch_size // grad_accum_steps
    if distributed and global_micro_batch % effective_world_size != 0:
        raise ValueError("Global micro-batch size must be divisible by world_size.")
    micro_batch_size = (
        global_micro_batch // effective_world_size if distributed else global_micro_batch
    )
    dataloader_batch_size = micro_batch_size

    if rank is None or rank == 0:
        print("[ ] Building world dataset...")

    world_dataset = WorldDataset(dataset_cfg)
    
    if rank is None or rank == 0:
        print(f"[x] World dataset created with {len(world_dataset.datasets)} sub-datasets. Total virtual length: {len(world_dataset)}")

    sampler: Optional[DistributedSampler] = None
    if distributed:
        sampler = DistributedSampler(
            world_dataset,
            num_replicas=world_size,
            rank=rank or 0,
            shuffle=dataloader_cfg.shuffle,
            drop_last=True,
            seed=seed or 0,
        )

    collate = StackCollator(
        shuffle=True,
        sequence_length_distribution=dataset_cfg.sequence_length_distribution
    )

    dataloader = DataLoader(
        world_dataset,
        batch_size=dataloader_batch_size,
        shuffle=dataloader_cfg.shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=dataloader_cfg.num_workers,
        pin_memory=dataloader_cfg.pin_memory,
        collate_fn=collate,
    )
    
    if rank is None or rank == 0:
        print(f"[x] Dataloader built. Batch size: {dataloader_batch_size} (local), {dataloader_cfg.batch_size} (global).")
        
    return dataloader
