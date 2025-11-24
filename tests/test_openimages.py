#!/usr/bin/env python3
"""
Test script to verify OpenImages dataset loading.
Run this after the OpenImages v7 download completes.
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.openimages_dataset import OpenImagesDataset, OpenImagesDatasetConfig

def main():
    # Configuration
    cfg = OpenImagesDatasetConfig(
        root="/gpfs/helios/home/gauthierbernarda/data/open_images_v7",
        split="train"
    )
    
    print("Initializing OpenImages dataset...")
    try:
        dataset = OpenImagesDataset(
            cfg=cfg,
            action_dim=8,
            sequence_length=16
        )
        
        print(f"✓ Dataset initialized successfully!")
        print(f"  Total images: {len(dataset)}")
        
        # Test loading a sample
        print("\nTesting sample loading...")
        batch = dataset[0]
        
        print(f"✓ Sample loaded successfully!")
        print(f"  Frames shape: {batch.sequence_frames.shape}")
        print(f"  Actions shape: {batch.sequence_actions.shape}")
        print(f"  Independent frames: {batch.independent_frames_mask}")
        print(f"  Actions mask: {batch.actions_mask}")
        print(f"  Frames valid mask: {batch.frames_valid_mask}")
        
        print("\n✓ All tests passed! OpenImages dataset is ready to use.")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
