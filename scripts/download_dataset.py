# =================
# Download Droid Dataset with LeRobot
# =================

from lerobot.datasets.lerobot_dataset import LeRobotDataset
repo_id = "aractingi/droid_1.0.1"
delta_timestamps = {
    "observation.images.exterior_1_left": [-45/15, -40/15, -35/15, -30/15, -25/15, -20/15, -15/15, -10/15, -5/15, 0],
    "observation.images.exterior_2_left": [-45/15, -40/15, -35/15, -30/15, -25/15, -20/15, -15/15, -10/15, -5/15, 0],
    "observation.images.wrist_left": [-45/15, -40/15, -35/15, -30/15, -25/15, -20/15, -15/15, -10/15, -5/15, 0],
    "observation.state": [-45/15, -40/15, -35/15, -30/15, -25/15, -20/15, -15/15, -10/15, -5/15, 0],
}
dataset = LeRobotDataset(repo_id, episodes=[50000], delta_timestamps=delta_timestamps)
print(f"Selected episodes: {dataset.episodes}")
print(f"Number of episodes selected: {dataset.num_episodes}")
print(f"Number of frames selected: {dataset.num_frames}")
import torch
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
)
print(f"Number of batches: {len(dataloader)}")

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