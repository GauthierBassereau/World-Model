import os
import random
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import torch
from torch.utils.data import Dataset

import decord
from decord import VideoReader
from decord import cpu
decord.bridge.set_bridge('torch')

from .common import WorldBatch
from .common import RESIZE_CROP_TRANSFORM_224

@dataclass
class VideoDatasetConfig:
    root: str
    fps: float = 3.0
    sequence_length: int = 15
    fps_error_threshold: float = 0.3

# TODO: manage the cuts in videos
class VideoDataset(Dataset):
    def __init__(self, cfg: VideoDatasetConfig, action_dim: int):
        self.cfg = cfg
        self.action_dim = action_dim
        self.sequence_length = cfg.sequence_length
        self.fps = cfg.fps
        self.fps_error_threshold = cfg.fps_error_threshold
        
        root_path = Path(cfg.root)
        cache_file = root_path / f"video_paths.txt"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                self.video_paths = [line.strip() for line in f if line.strip()]
        else:
            print(f"[VideoDataset] Scanning directory: {root_path}")
            paths = sorted(list(root_path.rglob("*.mp4")) + list(root_path.rglob("*.avi")))
            self.video_paths = [str(p) for p in paths]
            
            with open(cache_file, 'w') as f:
                for p in self.video_paths:
                    f.write(f"{p}\n")
            print(f"[VideoDataset] Cached {len(self.video_paths)} videos.")

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> WorldBatch:
        video_path = self.video_paths[index]
        
        vr = VideoReader(video_path, ctx=cpu(0))
        
        total_frames = len(vr)
        
        native_fps = vr.get_avg_fps()
        if np.isnan(native_fps) or native_fps <= 0:
            native_fps = 30.0
        
        step = int(round(native_fps / self.fps))
        exact_step = native_fps / self.fps
        step = max(1, int(round(exact_step)))
        effective_fps = native_fps / step
        fps_error = abs(effective_fps - self.fps)
        if fps_error > self.fps_error_threshold:
            raise ValueError(f"FPS Mismatch in {Path(video_path).name}: "
                    f"Native={native_fps:.2f}, Target={self.fps}, Step={step}. "
                    f"Resulting Effective FPS={effective_fps:.2f} (Error: {fps_error:.2f})")
        
        needed_frames = self.sequence_length
        span = needed_frames * step
        
        if total_frames > span:
            start_idx = random.randint(0, total_frames - span)
            indices = [start_idx + i * step for i in range(needed_frames)]
        else:
            indices = np.linspace(0, total_frames - 1, needed_frames).astype(int)

        vframes = vr.get_batch(indices) # Returns (T, H, W, C)
        
        vframes = vframes.permute(0, 3, 1, 2).to(torch.uint8)
        
        vframes = RESIZE_CROP_TRANSFORM_224(vframes)

        T = self.sequence_length
        actions = torch.zeros((T, self.action_dim), dtype=torch.float32)
        independent_frames_mask = torch.tensor(False, dtype=torch.bool)
        actions_mask = torch.zeros((T,), dtype=torch.bool)
        frames_valid_mask = torch.ones((T,), dtype=torch.bool)

        return WorldBatch(
            sequence_frames=vframes,
            sequence_actions=actions,
            independent_frames_mask=independent_frames_mask,
            actions_mask=actions_mask,
            frames_valid_mask=frames_valid_mask,
            dataset_indices=torch.tensor(-1, dtype=torch.long),
            dataset_names=torch.tensor(-1, dtype=torch.long),
            episode_ids=torch.tensor(-1, dtype=torch.long),
        )