from dataclasses import dataclass
import torch

@dataclass
class WorldBatch:
    sequence_frames: torch.Tensor
    sequence_actions: torch.Tensor
    independent_frames_mask: torch.Tensor
    actions_mask: torch.Tensor
    frames_valid_mask: torch.Tensor
    dataset_indices: torch.Tensor
