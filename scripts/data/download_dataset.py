# =================
# Download Droid Dataset with LeRobot
# =================

# from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
# import torch
# repo_id = "aractingi/droid_1.0.1"
# episodes = [0]
# delta_timestamps = {
#     "observation.images.wrist_left": [-1, -10/15, -5/15, 0, 5/15, 10/15, 1],
# }
# dataset = LeRobotDataset(repo_id, episodes=episodes, delta_timestamps=delta_timestamps)
# print(f"Dataset length: {len(dataset)}")

# =================
# Download EPIC KITCHENS
# =================

# from huggingface_hub import snapshot_download

# local_dir = snapshot_download(
#     repo_id="awsaf49/epic_kitchens_100",
#     repo_type="dataset",
#     local_dir="/gpfs/helios/home/gauthierbernarda/data/epic_kitchens_100",
#     local_dir_use_symlinks=False,  # safer on some HPC / GPFS setups
# )

# print("Files downloaded to:", local_dir)

import os
import subprocess
import uuid
from pathlib import Path
from tqdm import tqdm

INPUT_ROOT = "/gpfs/helios/home/gauthierbernarda/data/epic_kitchens_100"
OUTPUT_ROOT = "/gpfs/helios/home/gauthierbernarda/data/epic_kitchens_100_chunks"
SEGMENT_TIME = 120
input_path = Path(INPUT_ROOT)
output_path = Path(OUTPUT_ROOT)
output_path.mkdir(parents=True, exist_ok=True)

exts = ['*.mp4', '*.MP4', '*.avi', '*.AVI', '*.mov', '*.MOV', '*.mkv']
files = []
for ext in exts:
    files.extend(list(input_path.rglob(ext)))

print(f"Found {len(files)} videos to process.")

for video_file in tqdm(files, desc="Chunking videos"):
    unique_id = str(uuid.uuid4())[:8]
    save_pattern = output_path / f"{unique_id}_%03d.mp4"

    try:
        cmd = [
            "ffmpeg", "-y",
            "-fflags", "+genpts",
            "-i", str(video_file),
            "-c", "copy",
            "-map", "0:v",
            "-map_metadata", "-1",
            "-f", "segment",
            "-segment_time", str(SEGMENT_TIME),
            "-reset_timestamps", "1",
            "-loglevel", "error",
            str(save_pattern)
        ]

        subprocess.run(cmd, check=True)

    except subprocess.CalledProcessError as e:
        print(f"\nError processing {video_file.name}: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


# =================
# Split LeRobot Dataset, don't why it needs huge RAM for days, don't have the budget for it...
# =================

# from pathlib import Path
# from lerobot.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.datasets.dataset_tools import split_dataset

# base = LeRobotDataset("aractingi/droid_1.0.1")
# splits = {"train": 0.95, "val": 0.05}
# split_ds = split_dataset(base, splits, output_dir=Path("/gpfs/helios/home/gauthierbernarda/data/droid"))

# train_root = Path("/gpfs/helios/home/gauthierbernarda/data/droid_splits/train")
# val_root = Path("/gpfs/helios/home/gauthierbernarda/data/droid_splits/val")

# train_ds = LeRobotDataset("droid-train", root=train_root)
# val_ds   = LeRobotDataset("droid-val", root=val_root)