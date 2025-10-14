import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

repo_id = "lerobot/droid_1.0.1"

ds_meta = LeRobotDatasetMetadata(repo_id)

camera_keys = ds_meta.camera_keys
print(f"{camera_keys=}") # camera_keys=['observation.images.wrist_left', 'observation.images.exterior_1_left', 'observation.images.exterior_2_left'] We will use the 2 exterior cameras
fps = ds_meta.fps
print(f"{fps=}")  # fps=15

delta_timestamps = {
    # loads 4 images separated by 5/15 seconds
    "observation.images.exterior_1_left": [-1, -10/15, -5/15, 0],
    "observation.images.exterior_2_left": [-1, -10/15, -5/15, 0],
    "observation.state.cartesian_position": [-1, -10/15, -5/15, 0],
}

dataset = LeRobotDataset(repo_id, episodes=[0, 1, 2, 3, 4, 5], delta_timestamps=delta_timestamps)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
)

for batch in dataloader:
    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            print(f"{key}: {batch[key].shape}")  # torch.Size([2, 4, 3, 180, 320])
        if isinstance(batch[key], list):
            print(f"{key}: list of length {len(batch[key])}")  # list of length 2
    camera_key = dataset.meta.camera_keys[1]
    print(f"{batch[camera_key].shape=}")  # torch.Size([2, 4, 3, 180, 320])
    print(f"observation state gripper cartesian position:", batch['observation.state.cartesian_position'].shape)  # torch.Size([2, 4, 6])
    break