from typing import List, Dict, Tuple
import random
import torch
from .common import WorldBatch

class StackCollator:
    """
    The collator is just here to manipulate the length of the sequence per step of the training. Following Dreamerv4, training with mixed sequence length.
    """
    def __init__(
        self, 
        sequence_length_distribution: Dict[int, float],
    ):
        self.sequence_length_choices: List[int] = []
        self.sequence_length_probs: List[float] = []
        
        for length, weight in sequence_length_distribution.items():
            self.sequence_length_choices.append(int(length))
            self.sequence_length_probs.append(float(weight))
        
        total = sum(self.sequence_length_probs)
        self.sequence_length_probs = [w / total for w in self.sequence_length_probs]
        
        self.max_sequence_length = max(self.sequence_length_choices)

    def _sample_sequence_length(self) -> int:
        return random.choices(
            self.sequence_length_choices, 
            weights=self.sequence_length_probs, 
            k=1
        )[0]

    def __call__(self, batch: List[WorldBatch]) -> WorldBatch:
        target_length = self._sample_sequence_length()
        
        sequence_frames = torch.stack([b.sequence_frames for b in batch])
        sequence_actions = torch.stack([b.sequence_actions for b in batch])
        independent_frames = torch.stack([b.independent_frames for b in batch])
        use_actions = torch.stack([b.use_actions for b in batch])
        padded_frames = torch.stack([b.padded_frames for b in batch])
        dataset_indices = torch.stack([b.dataset_indices for b in batch])
        episode_ids = torch.stack([b.episode_ids for b in batch])

        if target_length < sequence_frames.shape[1]:
            sequence_frames = sequence_frames[:, :target_length]
            sequence_actions = sequence_actions[:, :target_length]
            independent_frames = independent_frames[:, :target_length]
            use_actions = use_actions[:, :target_length]
            padded_frames = padded_frames[:, :target_length]
            if dataset_indices.ndim > 1 and dataset_indices.shape[1] > 1:
                dataset_indices = dataset_indices[:, :target_length]
            if episode_ids.ndim > 1 and episode_ids.shape[1] > 1:
                episode_ids = episode_ids[:, :target_length]

        return WorldBatch(
            sequence_frames=sequence_frames,
            sequence_actions=sequence_actions,
            independent_frames=independent_frames,
            use_actions=use_actions,
            padded_frames=padded_frames,
            dataset_indices=dataset_indices,
            dataset_names=batch[0].dataset_names, # dataset_names is usually metadata, just taking the first one is standard
            episode_ids=episode_ids,
        )