from typing import List, Dict, Tuple
import random
import torch
from .common import WorldBatch

class StackCollator:
    """
    The collator is just here to manipulate the length of the sequence per step of the training. Following Dreamerv4, having training with mixed sequence length.
    """
    def __init__(
        self, 
        sequence_length_distribution: Dict[int, float],
        shuffle: bool = True,
    ):
        self.shuffle = shuffle
        
        self.sequence_length_choices: List[int] = []
        self.sequence_length_probs: List[float] = []
        
        for length, weight in sequence_length_distribution.items():
            self.sequence_length_choices.append(int(length))
            self.sequence_length_probs.append(float(weight))
        
        total = sum(self.sequence_length_probs)
        self.sequence_length_probs = [w / total for w in self.sequence_length_probs]
        
        self.max_sequence_length = max(self.sequence_length_choices)

    def _sample_sequence_length(self) -> int:
        """Sample a target sequence length from the distribution."""
        return random.choices(
            self.sequence_length_choices, 
            weights=self.sequence_length_probs, 
            k=1
        )[0]
    
    def _crop_sequence(
        self, 
        batch: List[WorldBatch], 
        target_length: int
    ) -> List[WorldBatch]:
        """
        Crop sequences to target_length, preferring the middle where there's
        less chance of having padded images.
        """
        cropped_batch = []
        
        for b in batch:
            T = b.sequence_frames.shape[0]
            
            if T == target_length:
                cropped_batch.append(b)
                continue
            
            valid_mask = b.frames_valid_mask  # [T]
            
            if valid_mask.any():
                valid_indices = torch.where(valid_mask)[0]
                center_of_valid = valid_indices.float().mean().long().item()
            else:
                # If no valid frames (shouldn't happen), use middle
                center_of_valid = T // 2
            
            start_idx = max(0, center_of_valid - target_length // 2)
            start_idx = min(start_idx, T - target_length)
            
            end_idx = start_idx + target_length
            
            cropped_batch.append(WorldBatch(
                sequence_frames=b.sequence_frames[start_idx:end_idx],
                sequence_actions=b.sequence_actions[start_idx:end_idx],
                independent_frames_mask=b.independent_frames_mask,
                actions_mask=b.actions_mask[start_idx:end_idx],
                frames_valid_mask=b.frames_valid_mask[start_idx:end_idx],
                dataset_indices=b.dataset_indices,
                dataset_names=b.dataset_names,
                episode_ids=b.episode_ids,
            ))
        
        return cropped_batch

    def __call__(self, batch: List[WorldBatch]) -> WorldBatch:
        target_length = self._sample_sequence_length()
        
        batch = self._crop_sequence(batch, target_length)
        
        sequence_frames = torch.stack([b.sequence_frames for b in batch])
        sequence_actions = torch.stack([b.sequence_actions for b in batch])
        independent_frames_mask = torch.stack([b.independent_frames_mask for b in batch])
        actions_mask = torch.stack([b.actions_mask for b in batch])
        frames_valid_mask = torch.stack([b.frames_valid_mask for b in batch])
        dataset_indices = torch.stack([b.dataset_indices for b in batch])
        episode_ids = torch.stack([b.episode_ids for b in batch])

        if self.shuffle:
            B = sequence_frames.shape[0]
            perm = torch.randperm(B)
            
            sequence_frames = sequence_frames[perm]
            sequence_actions = sequence_actions[perm]
            independent_frames_mask = independent_frames_mask[perm]
            actions_mask = actions_mask[perm]
            frames_valid_mask = frames_valid_mask[perm]
            dataset_indices = dataset_indices[perm]
            episode_ids = episode_ids[perm]

        return WorldBatch(
            sequence_frames=sequence_frames,
            sequence_actions=sequence_actions,
            independent_frames_mask=independent_frames_mask,
            actions_mask=actions_mask,
            frames_valid_mask=frames_valid_mask,
            dataset_indices=dataset_indices,
            dataset_names=batch[0].dataset_names,
            episode_ids=episode_ids,
        )

