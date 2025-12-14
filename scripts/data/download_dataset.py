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

print("Done.")