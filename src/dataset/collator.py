from typing import List
import torch
from .batch import WorldModelBatch

# Re-export WorldModelBatch for compatibility if needed, 
# though consumers should ideally import from .batch
__all__ = ["StackCollator", "WorldModelBatch"]

class StackCollator:
    def __init__(self, shuffle: bool = True):
        self.shuffle = shuffle

    def __call__(self, batch: List[WorldModelBatch]) -> WorldModelBatch:
        # Stack all fields
        sequence_frames = torch.stack([b.sequence_frames for b in batch])
        sequence_actions = torch.stack([b.sequence_actions for b in batch])
        independent_frames_mask = torch.stack([b.independent_frames_mask for b in batch])
        actions_mask = torch.stack([b.actions_mask for b in batch])
        frames_valid_mask = torch.stack([b.frames_valid_mask for b in batch])
        dataset_indices = torch.stack([b.dataset_indices for b in batch])

        if self.shuffle:
            B = sequence_frames.shape[0]
            perm = torch.randperm(B)
            
            sequence_frames = sequence_frames[perm]
            sequence_actions = sequence_actions[perm]
            independent_frames_mask = independent_frames_mask[perm]
            actions_mask = actions_mask[perm]
            frames_valid_mask = frames_valid_mask[perm]
            dataset_indices = dataset_indices[perm]

        return WorldModelBatch(
            sequence_frames=sequence_frames,
            sequence_actions=sequence_actions,
            independent_frames_mask=independent_frames_mask,
            actions_mask=actions_mask,
            frames_valid_mask=frames_valid_mask,
            dataset_indices=dataset_indices,
        )
