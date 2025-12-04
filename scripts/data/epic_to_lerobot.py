#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Decord imports
from decord import VideoReader, cpu

from lerobot.datasets.lerobot_dataset import LeRobotDataset

EPIC_FPS = 15
TARGET_SIZE = 224

EPIC_FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": {
            "axes": ["accl_x", "accl_y", "accl_z", "gyro_x", "gyro_y", "gyro_z"],
        },
    },
    "observation.images.ego": {
        "dtype": "video",
        "shape": (TARGET_SIZE, TARGET_SIZE, 3),
        "names": ["height", "width", "channels"],
    },
}


def get_video_paths(root_dir: Path) -> list[Path]:
    """Find all MP4 video files and sort by size (largest first)."""
    # 1. Get all paths
    paths = list(root_dir.rglob("P*/videos/*.MP4"))
    
    # 2. Sort by file size using stat().st_size
    # reverse=True ensures the biggest files come first
    paths.sort(key=lambda p: p.stat().st_size, reverse=True)
    
    return paths


def load_sensor_data(participant_id: str, video_id: str, root_dir: Path) -> pd.DataFrame:
    """Load and merge Accelerometer and Gyroscope data."""
    meta_dir = root_dir / participant_id / "meta_data"
    accl_path = meta_dir / f"{video_id}-accl.csv"
    gyro_path = meta_dir / f"{video_id}-gyro.csv"

    if not accl_path.exists() or not gyro_path.exists():
        logging.warning(f"Missing sensor data for {video_id}")
        return None

    accl_df = pd.read_csv(accl_path)
    gyro_df = pd.read_csv(gyro_path)

    return accl_df, gyro_df


def interpolate_sensors(accl_df, gyro_df, target_timestamps_ms):
    """Interpolate sensor data to target timestamps."""
    # Accel
    accl_time = accl_df["Milliseconds"].values
    accl_data = accl_df[["AcclX", "AcclY", "AcclZ"]].values
    
    # Gyro
    gyro_time = gyro_df["Milliseconds"].values
    gyro_data = gyro_df[["GyroX", "GyroY", "GyroZ"]].values

    # Interpolate
    interp_accl = np.zeros((len(target_timestamps_ms), 3))
    for i in range(3):
        interp_accl[:, i] = np.interp(target_timestamps_ms, accl_time, accl_data[:, i])

    interp_gyro = np.zeros((len(target_timestamps_ms), 3))
    for i in range(3):
        interp_gyro[:, i] = np.interp(target_timestamps_ms, gyro_time, gyro_data[:, i])

    return np.hstack([interp_accl, interp_gyro])


def calculate_resize_dims(width, height, target_min):
    """
    Calculate new dims such that the smaller edge equals target_min.
    This preserves aspect ratio.
    """
    if width < height:
        # Portrait (unlikely for Epic Kitchens, but good for safety)
        new_w = target_min
        new_h = int(height * (target_min / width))
    else:
        # Landscape
        new_h = target_min
        new_w = int(width * (target_min / height))
    return new_w, new_h


def center_crop(frame, target_size):
    """
    Center crop a numpy image (H, W, C).
    """
    h, w, _ = frame.shape
    start_x = (w - target_size) // 2
    start_y = (h - target_size) // 2
    return frame[start_y : start_y + target_size, start_x : start_x + target_size, :]


def process_video(video_path: Path, root_dir: Path, lerobot_dataset: LeRobotDataset):
    """Process a single video using Decord with Aspect-Ratio preserving resize."""
    video_id = video_path.stem
    participant_id = video_path.parent.parent.name
    
    logging.info(f"Processing {video_id}...")

    # 1. Load sensor data
    sensors = load_sensor_data(participant_id, video_id, root_dir)
    if sensors is None:
        return
    accl_df, gyro_df = sensors

    # 2. Open Video to get ORIGINAL dimensions
    # We do not pass width/height yet because we need to calculate the ratio first
    try:
        vr_meta = VideoReader(str(video_path), ctx=cpu(0))
    except Exception as e:
        logging.error(f"Failed to open video {video_path}: {e}")
        return

    orig_h, orig_w, _ = vr_meta[0].shape
    
    # Calculate dimensions that preserve aspect ratio (Short-side Scale)
    # e.g. 1920x1080 -> 398x224
    resize_w, resize_h = calculate_resize_dims(orig_w, orig_h, TARGET_SIZE)
    
    # 3. Re-open Video with the optimized resize dimensions
    # This forces Decord to decode directly to 398x224 (very fast)
    del vr_meta # Free memory
    vr = VideoReader(str(video_path), ctx=cpu(0), width=resize_w, height=resize_h)

    # 4. Calculate Timings
    source_fps = vr.get_avg_fps()
    total_frames_source = len(vr)
    duration_sec = total_frames_source / source_fps
    
    num_target_frames = int(duration_sec * EPIC_FPS)
    target_indices = np.linspace(0, total_frames_source - 1, num_target_frames).astype(int)
    
    target_timestamps_sec = np.arange(num_target_frames) / EPIC_FPS
    target_timestamps_ms = target_timestamps_sec * 1000.0

    # 5. Interpolate actions
    actions = interpolate_sensors(accl_df, gyro_df, target_timestamps_ms)

    # 6. Batch Processing
    batch_size = 64
    
    for start_idx in tqdm(range(0, len(target_indices), batch_size), desc=f"Processing {video_id}", leave=False):
        end_idx = min(start_idx + batch_size, len(target_indices))
        batch_indices = target_indices[start_idx:end_idx]
        
        # Get Batch (returns e.g. [64, 224, 398, 3])
        video_data = vr.get_batch(batch_indices).asnumpy()
        
        for i, frame in enumerate(video_data):
            global_idx = start_idx + i
            
            # Center Crop to get exactly 224x224
            # e.g. [224, 398, 3] -> [224, 224, 3]
            frame_cropped = center_crop(frame, TARGET_SIZE)
            
            frame_dict = {
                "action": torch.from_numpy(actions[global_idx]).float(),
                "observation.images.ego": frame_cropped,
                "task": "kitchen activity",
            }
            
            lerobot_dataset.add_frame(frame_dict)

    lerobot_dataset.save_episode()


from lerobot.utils.constants import HF_LEROBOT_HOME

def port_epic_kitchens(
    raw_dir: Path,
    repo_id: str,
    push_to_hub: bool = False,
    debug: bool = False,
):
    logging.basicConfig(level=logging.INFO)
    
    root = HF_LEROBOT_HOME / repo_id
    if (root / "meta/info.json").exists():
        logging.info(f"Resuming from existing dataset at {root}")
        lerobot_dataset = LeRobotDataset(repo_id=repo_id)
    else:
        # Initialize dataset
        lerobot_dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=EPIC_FPS,
            features=EPIC_FEATURES,
            robot_type="epic_kitchens", 
        )
        lerobot_dataset.meta.update_chunk_settings(video_files_size_in_mb=500)
    
    # Find videos
    video_paths = get_video_paths(raw_dir)
    if debug:
        video_paths = video_paths[:1]

    logging.info(f"Found {len(video_paths)} videos.")

    # Resume logic: check for already processed videos using a simple text file
    processed_videos_path = lerobot_dataset.root / "processed_videos.txt"
    existing_video_ids = set()
    if processed_videos_path.exists():
        with open(processed_videos_path, "r") as f:
            existing_video_ids = set(line.strip() for line in f if line.strip())
    
    if len(existing_video_ids) > 0:
        logging.info(f"Resuming from {len(existing_video_ids)} existing episodes.")

    for video_path in tqdm(video_paths):
        video_id = video_path.stem
        if video_id in existing_video_ids:
            continue
        try:
            process_video(video_path, raw_dir, lerobot_dataset)
            
            # Log processed video
            with open(processed_videos_path, "a") as f:
                f.write(f"{video_id}\n")
                
        except Exception as e:
            logging.error(f"Error processing {video_path}: {e}")
            continue

    lerobot_dataset.finalize()

    if push_to_hub:
        lerobot_dataset.push_to_hub(tags=["epic-kitchens", "ego-centric"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Root directory of Epic Kitchens dataset",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repo ID",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload to Hub after creation",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run on a small subset for debugging",
    )
    args = parser.parse_args()

    port_epic_kitchens(**vars(args))


if __name__ == "__main__":
    main()