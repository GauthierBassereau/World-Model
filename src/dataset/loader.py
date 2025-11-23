from typing import Optional, Dict, Any
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from torchvision.datasets import Kinetics, ImageFolder

from .configs import (
    WorldModelDatasetConfig, 
    DataloaderConfig, 
    LeRobotDatasetConfig, 
    KineticsDatasetConfig, 
    ImageNetDatasetConfig
)
from .wrappers import LeRobotDatasetWrapper, KineticsDatasetWrapper, ImageNetDatasetWrapper
from .combined import WorldModelDataset
from .collator import StackCollator
from .utils import _ensure_delta_timestamps, _DROID_RESIZE_CROP_TRANSFORM

def build_world_model_dataloader(
    dataset_cfg: WorldModelDatasetConfig,
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
        print("[ ] Building datasets...")

    datasets: Dict[str, Dataset] = {}
    action_dim: Optional[int] = None

    # First pass: Initialize LeRobot datasets to determine action_dim
    for name, cfg in dataset_cfg.datasets.items():
        if isinstance(cfg, LeRobotDatasetConfig):
            metadata = LeRobotDatasetMetadata(cfg.repo_id)
            delta_timestamps = _ensure_delta_timestamps(cfg, metadata)
            
            dataset = LeRobotDataset(
                cfg.repo_id,
                episodes=cfg.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=_DROID_RESIZE_CROP_TRANSFORM,
            )
            
            wrapper = LeRobotDatasetWrapper(dataset, cfg)
            datasets[name] = wrapper
            
            # Determine action dim from a sample
            # We assume all LeRobot datasets have compatible action dims if multiple exist
            if action_dim is None:
                # Sample one item to check action dim
                # wrapper[0] returns WorldModelBatch
                # We need to be careful not to trigger random failures if dataset is empty or broken
                try:
                    sample_batch = wrapper[0]
                    action_dim = sample_batch.sequence_actions.shape[-1]
                    if rank is None or rank == 0:
                        print(f"[x] Determined action dimension: {action_dim} from dataset {name}")
                except Exception as e:
                    print(f"[!] Failed to determine action dim from {name}: {e}")
                    # Fallback or error?
                    pass

    if action_dim is None:
        # If no LeRobot dataset or failed to determine, set a default or raise error?
        # If only ImageNet/Kinetics, we need an action dim.
        # Let's default to 1 if not found (though this might be wrong for the model)
        # Or maybe the user should provide it.
        # For now, warn and default to 1.
        if rank is None or rank == 0:
            print("[!] Could not determine action dimension from LeRobot datasets. Defaulting to 1.")
        action_dim = 1

    # Second pass: Initialize other datasets
    for name, cfg in dataset_cfg.datasets.items():
        if name in datasets:
            continue # Already initialized
            
        if isinstance(cfg, KineticsDatasetConfig):
            dataset = Kinetics(
                root=cfg.root,
                frames_per_clip=cfg.frames_per_clip,
                step_between_clips=cfg.step_between_clips,
                # transform is handled in wrapper
            )
            wrapper = KineticsDatasetWrapper(
                dataset, 
                cfg, 
                action_dim=action_dim,
                transform=_DROID_RESIZE_CROP_TRANSFORM # Apply same transform?
            )
            datasets[name] = wrapper
            
        elif isinstance(cfg, ImageNetDatasetConfig):
            dataset = ImageFolder(root=cfg.root)
            wrapper = ImageNetDatasetWrapper(
                dataset, 
                cfg, 
                action_dim=action_dim,
                transform=_DROID_RESIZE_CROP_TRANSFORM
            )
            datasets[name] = wrapper

    # Create combined dataset
    combined_dataset = WorldModelDataset(datasets, dataset_cfg.weights)
    
    if rank is None or rank == 0:
        print(f"[x] WorldModelDataset created with {len(datasets)} sub-datasets. Total virtual length: {len(combined_dataset)}")

    sampler: Optional[DistributedSampler] = None
    if distributed:
        sampler = DistributedSampler(
            combined_dataset,
            num_replicas=world_size,
            rank=rank or 0,
            shuffle=dataloader_cfg.shuffle,
            drop_last=True,
            seed=seed or 0,
        )

    collate = StackCollator(shuffle=True)

    dataloader = DataLoader(
        combined_dataset,
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
