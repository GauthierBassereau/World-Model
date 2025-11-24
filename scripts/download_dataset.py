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
# Download Kinetics Dataset with Torchvision
# =================

# from torchvision.datasets import Kinetics

# ds = Kinetics(
#     root="/gpfs/helios/home/gauthierbernarda/data/kinetics",
#     split="train",
#     num_classes="700",
#     frame_rate=3,
#     frames_per_clip=9,
#     step_between_clips=1,
#     num_workers=16,
#     download=False,
# )
# print("Index built. Clips:", len(ds))


# =================
# Download ImageNet Dataset with Kaggle
# =================

import fiftyone.zoo as foz

# Defines the download. 
# Full training set is ~513 GB. 
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=[],
    dataset_dir="/gpfs/helios/home/gauthierbernarda/data/open_images_v7",
)

print("Download complete. Location: /gpfs/helios/home/gauthierbernarda/data/open_images_v7")

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