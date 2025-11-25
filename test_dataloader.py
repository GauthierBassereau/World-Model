#!/usr/bin/env python3
"""
Test script to verify dataset and dataloader creation.
This mimics how world_trainer.py creates the dataloader.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from src.dataset.droid_dataset import DroidDatasetConfig
from src.dataset.kinetics_dataset import KineticsDatasetConfig
from src.dataset.openimages_dataset import OpenImagesDatasetConfig
from src.dataset.world_dataset import WorldDatasetConfig
from src.dataset.loader import build_world_dataloader, DataloaderConfig


def main():
    print("=" * 80)
    print("Testing Dataset and DataLoader Creation")
    print("=" * 80)
    
    # Create dataset configs matching the YAML structure
    droid_cfg = DroidDatasetConfig(
        repo_id="aractingi/droid_1.0.1",
        episodes=None,
        excluded_episodes=[0, 1, 2, 3, 4, 5, 6, 7],
        cameras=(
            "observation.images.exterior_1_left",
            "observation.images.exterior_2_left",
            "observation.images.wrist_left",
        ),
        action_keys=("observation.state",),
        action_representation="position",
        action_normalization="mean_std",
        action_normalization_params={
            "mean": [0.01428347627459065, 0.23987777195473095, -0.014661363646208965, 
                     -2.027954954645529, -0.035476306435143545, 2.3233678805030076, 
                     0.08326671319745274, 0.36085284714413735],
            "std": [0.3244343844169754, 0.5158281965633522, 0.2901322476548548, 
                    0.5021871776809874, 0.5331921797732538, 0.46856794632856424, 
                    0.7446577730524244, 0.40315882970033534],
        },
        fps=3.0,
        sequence_length=15,
        independent_frames_probability=0.3,
        drop_action_probability=0.75,
    )
    
    # kinetics_cfg = KineticsDatasetConfig(
    #     root="/gpfs/helios/home/gauthierbernarda/data/kinetics",
    #     split="train",
    #     fps=3.0,
    #     sequence_length=15,
    #     step_between_clips=1,
    # )
    
    openimages_cfg = OpenImagesDatasetConfig(
        root="/gpfs/helios/home/gauthierbernarda/data/open_images_v7",
        sequence_length=15,
    )
    
    # Create WorldDatasetConfig
    world_dataset_cfg = WorldDatasetConfig(
        datasets={
            "droid": droid_cfg,
            # "kinetics": kinetics_cfg,
            "openimages": openimages_cfg,
        },
        weights={
            "droid": 0.3,
            # "kinetics": 0.4,
            "openimages": 0.3,
        },
        action_dim=8,
        sequence_length_distribution={
            7: 0.8,
            15: 0.2,
        },
        fps=3.0,
    )
    
    print("\n✓ WorldDatasetConfig created successfully")
    print(f"  - Datasets: {list(world_dataset_cfg.datasets.keys())}")
    print(f"  - Weights: {world_dataset_cfg.weights}")
    print(f"  - Action dim: {world_dataset_cfg.action_dim}")
    print(f"  - Sequence length distribution: {world_dataset_cfg.sequence_length_distribution}")
    print(f"  - FPS: {world_dataset_cfg.fps}")
    
    # Create DataloaderConfig
    dataloader_cfg = DataloaderConfig(
        batch_size=8,  # Global batch size
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    print("\n✓ DataloaderConfig created successfully")
    print(f"  - Batch size: {dataloader_cfg.batch_size}")
    print(f"  - Shuffle: {dataloader_cfg.shuffle}")
    print(f"  - Num workers: {dataloader_cfg.num_workers}")
    
    # Build dataloader (mimicking trainer)
    print("\n" + "=" * 80)
    print("Building DataLoader...")
    print("=" * 80)
    
    try:
        dataloader = build_world_dataloader(
            dataset_cfg=world_dataset_cfg,
            dataloader_cfg=dataloader_cfg,
            grad_accum_steps=1,
            seed=3,
            rank=0,
            world_size=1,
        )
        
        print("\n✓ DataLoader created successfully!")
        print(f"  - Total length: {len(dataloader)}")
        print(f"  - Batch size: {dataloader.batch_size}")
        
        # Try to get one batch
        print("\n" + "=" * 80)
        print("Testing batch retrieval...")
        print("=" * 80)
        
        batch_iter = iter(dataloader)
        batch = next(batch_iter)
        
        print("\n✓ Successfully retrieved a batch!")
        print(f"  - Batch type: {type(batch)}")
        print(f"  - sequence_frames shape: {batch.sequence_frames.shape}")
        print(f"  - sequence_actions shape: {batch.sequence_actions.shape}")
        print(f"  - independent_frames_mask shape: {batch.independent_frames_mask.shape}")
        print(f"  - actions_mask shape: {batch.actions_mask.shape}")
        print(f"  - frames_valid_mask shape: {batch.frames_valid_mask.shape}")
        print(f"  - dataset_indices shape: {batch.dataset_indices.shape}")
        
        print("\n" + "=" * 80)
        print("✅ All tests passed!")
        print("=" * 80)

        # Save one video per dataset
        print("\n" + "=" * 80)
        print("Saving videos...")
        print("=" * 80)
        
        try:
            import torchvision.io as io
            
            # Get dataset names mapping
            dataset_names = sorted(list(world_dataset_cfg.datasets.keys()))
            saved_datasets = set()
            
            # Iterate through batch
            batch_size = batch.sequence_frames.shape[0]
            for i in range(batch_size):
                dataset_idx = batch.dataset_indices[i].item()
                dataset_name = dataset_names[dataset_idx]
                
                if dataset_name not in saved_datasets:
                    print(f"Saving video for dataset: {dataset_name}")
                    
                    # Get frames: [T, C, H, W] -> [T, H, W, C]
                    frames = batch.sequence_frames[i]
                    frames = (frames * 255).to(torch.uint8)
                    frames = frames.permute(0, 2, 3, 1)
                    
                    output_filename = f"test_video_{dataset_name}.mp4"
                    io.write_video(output_filename, frames, fps=3)
                    print(f"  -> Saved to {output_filename}")
                    
                    saved_datasets.add(dataset_name)
                    
                if len(saved_datasets) == len(dataset_names):
                    break
            
            if len(saved_datasets) < len(dataset_names):
                print(f"\nWarning: Could not find examples for all datasets in the first batch.")
                print(f"Missing: {set(dataset_names) - saved_datasets}")
                
        except ImportError:
            print("Could not import torchvision.io. Skipping video saving.")
        except Exception as e:
            print(f"Error saving videos: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"\n❌ Error creating dataloader or retrieving batch:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
