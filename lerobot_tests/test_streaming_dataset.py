from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

delta_timestamps = {
    "observation.images.exterior_1_left": [-1, -10/15, -5/15, 0],
    "observation.images.exterior_2_left": [-1, -10/15, -5/15, 0],
    "observation.state.cartesian_position": [-1, -10/15, -5/15, 0],
}

dataset = StreamingLeRobotDataset(
    repo_id="lerobot/droid_1.0.1",
    delta_timestamps=delta_timestamps,
    streaming=True,
    buffer_size=1000,
)

# Iterate over the dataset
for i, item in enumerate(dataset):
    print(f"Sample {i}: Episode {item['episode_index']} Frame {item['frame_index']}")
    # item will contain stacked frames according to delta_timestamps
    if i >= 10:
        break