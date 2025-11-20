# =================
# Download Droid Dataset with LeRobot
# =================

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch
repo_id = "aractingi/droid_1.0.1"
episodes = [2, 3, 4, 5]
delta_timestamps = {
    "observation.images.exterior_1_left": [-1, 10/15, 5/15, 0]
}

dataset = LeRobotDataset(repo_id, episodes=episodes, delta_timestamps=delta_timestamps)
print(dataset)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
for batch in dataloader:
    print(f"Received batch index {batch['index']}", end="\r")
# =================
# Split LeRobot Dataset
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