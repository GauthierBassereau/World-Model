# =================
# Download Droid Dataset with LeRobot
# =================

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch
repo_id = "aractingi/droid_1.0.1"
delta_timestamps = {
    "observation.images.exterior_1_left": [-1, -10/15, -5/15, 0],
    "observation.images.exterior_2_left": [-1, -10/15, -5/15, 0],
    "observation.images.wrist_left": [-1, -10/15, -5/15, 0],
    "observation.state": [-1, -10/15, -5/15, 0],
}
dataset = LeRobotDataset(repo_id, episodes=[2, 5], delta_timestamps=delta_timestamps)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
print(f"Length of dataloader: {len(dataloader)}")

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