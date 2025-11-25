from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from torchvision.datasets import Kinetics

from .common import WorldBatch
from .common import RESIZE_CROP_TRANSFORM_224

@dataclass
class KineticsDatasetConfig:
    root: str
    split: str
    fps: float = 3.0
    sequence_length: int = 15
    step_between_clips: int = 1

class KineticsDataset(Dataset):
    def __init__(
        self,
        cfg: KineticsDatasetConfig,
        action_dim: int,
    ):
        self.cfg = cfg
        self.action_dim = action_dim
        
        # Use values from config
        self.sequence_length = cfg.sequence_length
        self.fps = cfg.fps
        self.max_sequence_length = self.sequence_length
        
        # Initialize the raw Kinetics dataset
        self.dataset = Kinetics(
            root=cfg.root,
            frames_per_clip=self.sequence_length,
            step_between_clips=cfg.step_between_clips,
            frame_rate=self.fps,
            output_format="TCHW",
            num_workers=16,
            transform=RESIZE_CROP_TRANSFORM_224,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> WorldBatch:
        video, _, _ = self.dataset[index]
        video = video.to(torch.float32) / 255.0
        
        T = video.shape[0]
        
        actions = torch.zeros((T, self.action_dim), dtype=torch.float32)
        independent_frames_mask = torch.tensor(False, dtype=torch.bool)
        actions_mask = torch.zeros((T,), dtype=torch.bool)
        frames_valid_mask = torch.ones((T,), dtype=torch.bool)

        return WorldBatch(
            sequence_frames=video,
            sequence_actions=actions,
            independent_frames_mask=independent_frames_mask,
            actions_mask=actions_mask,
            frames_valid_mask=frames_valid_mask,
            dataset_indices=torch.tensor(-1, dtype=torch.long)
        )
