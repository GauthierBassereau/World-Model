# =================
# Download Droid Dataset with LeRobot
# =================

from lerobot.datasets.lerobot_dataset import LeRobotDataset
# repo_id = "aractingi/droid_1.0.1"
repo_id = "cadene/droid_1.0.1"
dataset = LeRobotDataset(repo_id)

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