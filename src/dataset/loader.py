from typing import Optional
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from .configs import DatasetConfig, DataloaderConfig
from .resilient_dataset import ResilientLeRobotDataset
from .utils import _compute_episode_midpoints, _ensure_delta_timestamps, _DROID_RESIZE_CROP_TRANSFORM
from .collator import LeRobotSequenceCollator

def build_world_model_dataloader(
    dataset_cfg: DatasetConfig,
    dataloader_cfg: DataloaderConfig,
    grad_accum_steps: int = 1,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> DataLoader:
    metadata = LeRobotDatasetMetadata(dataset_cfg.repo_id)
    delta_timestamps = _ensure_delta_timestamps(dataset_cfg, metadata)

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

    if rank is None or rank == 0:
        print("[ ] Building dataset and dataloader...")
    dataset = ResilientLeRobotDataset(
        dataset_cfg.repo_id,
        episodes=dataset_cfg.episodes,
        delta_timestamps=delta_timestamps,
        tolerance_s=0.01,
        max_decode_failures=dataset_cfg.decoder_retry_attempts,
        image_transforms=_DROID_RESIZE_CROP_TRANSFORM,
    )
    if rank is None or rank == 0:
        print(f"[x] Dataset created with {len(dataset_cfg.episodes) if dataset_cfg.episodes else 'all'} episodes, with length {len(dataset)}.")

    if dataset_cfg.episode_midpoint_only:
        midpoint_indices, episode_ids_used = _compute_episode_midpoints(dataset, dataset_cfg.episodes)
        dataset = Subset(dataset, midpoint_indices)
        # Carry episode ids alongside subset order for downstream consumers (e.g., video logging)
        setattr(dataset, "episode_ids", episode_ids_used)
        if rank is None or rank == 0:
            print(
                f"[x] Using midpoint-only sampling; {len(midpoint_indices)} episodes mapped to {len(dataset)} samples."
            )

    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank or 0,
            shuffle=dataloader_cfg.shuffle,
            drop_last=True,
            seed=seed or 0,
        )

    collate = LeRobotSequenceCollator(dataset_cfg)

    dataloader = DataLoader(
        dataset,
        batch_size=dataloader_batch_size,
        shuffle=dataloader_cfg.shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=dataloader_cfg.num_workers,
        pin_memory=dataloader_cfg.pin_memory,
        collate_fn=collate,
    )
    if rank is None or rank == 0:
        print(f"[x] Dataloader built with {len(dataset_cfg.episodes) if dataset_cfg.episodes else 'all'} episodes, with length {len(dataloader)}.")
    return dataloader
