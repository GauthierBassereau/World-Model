from typing import Iterator, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .world_dataset import WorldDataset
from .collator import StackCollator


@dataclass
class DataloaderConfig:
    batch_size: int = 2
    shuffle: bool = False
    num_workers: int = 0
    pin_memory: bool = True


class ResumableDistributedSampler(DistributedSampler):
    """A deterministic distributed sampler that can resume within an epoch."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.start_index = 0

    def set_start_index(self, start_index: int) -> None:
        start_index = int(start_index)
        if start_index < 0 or start_index > self.num_samples:
            raise ValueError(
                f"Sampler start index must be in [0, {self.num_samples}], got {start_index}."
            )
        self.start_index = start_index

    def __iter__(self) -> Iterator[int]:
        indices = list(super().__iter__())
        return iter(indices[self.start_index:])

    def __len__(self) -> int:
        return self.num_samples - self.start_index


def build_world_dataloader(
    dataset: WorldDataset,
    dataloader_cfg: DataloaderConfig,
    grad_accum_steps: int = 1,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> DataLoader:
    effective_world_size = world_size or 1

    if dataloader_cfg.batch_size % grad_accum_steps != 0:
        raise ValueError("Global batch size must be divisible by grad_accum_steps.")
    global_micro_batch = dataloader_cfg.batch_size // grad_accum_steps
    if global_micro_batch % effective_world_size != 0:
        raise ValueError("Global micro-batch size must be divisible by world_size.")
    dataloader_batch_size = global_micro_batch // effective_world_size

    # Use the same sampler implementation for one and many ranks. Its order is
    # then a pure function of (seed, epoch), and its cursor can be restored
    # without decoding all already-consumed samples.
    sampler = ResumableDistributedSampler(
        dataset,
        num_replicas=effective_world_size,
        rank=rank or 0,
        shuffle=dataloader_cfg.shuffle,
        drop_last=True,
        seed=0 if seed is None else seed,
    )

    collate = StackCollator(sequence_length_distribution=dataset.cfg.sequence_length_distribution)
    worker_generator = torch.Generator()
    worker_generator.manual_seed(0 if seed is None else seed)

    dataloader = DataLoader(
        dataset,
        batch_size=dataloader_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=dataloader_cfg.num_workers,
        pin_memory=dataloader_cfg.pin_memory,
        collate_fn=collate,
        persistent_workers=True if dataloader_cfg.num_workers > 0 else False,
        drop_last=True,
        generator=worker_generator,
    )
    
    return dataloader
