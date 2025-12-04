from typing import Optional, Dict, Any
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from .world_dataset import WorldDataset
from .collator import StackCollator

@dataclass
class DataloaderConfig:
    batch_size: int = 2
    shuffle: bool = False
    num_workers: int = 0
    pin_memory: bool = True

def build_world_dataloader(
    dataset: WorldDataset,
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

    sampler: Optional[DistributedSampler] = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank or 0,
            shuffle=dataloader_cfg.shuffle,
            drop_last=True,
            seed=seed or 0,
        )

    collate = StackCollator(sequence_length_distribution=dataset.cfg.sequence_length_distribution)

    dataloader = DataLoader(
        dataset,
        batch_size=dataloader_batch_size,
        shuffle=dataloader_cfg.shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=dataloader_cfg.num_workers,
        pin_memory=dataloader_cfg.pin_memory,
        collate_fn=collate,
        persistent_workers=True if dataloader_cfg.num_workers > 0 else False,
    )
    
    return dataloader
