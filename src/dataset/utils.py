from typing import Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from .resilient_dataset import ResilientLeRobotDataset
if TYPE_CHECKING:
    from .configs import DatasetConfig
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as transforms_v2

_DROID_RESIZE_CROP_TRANSFORM = transforms_v2.Compose(
    [
        transforms_v2.Resize(
            size=(224, 320),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms_v2.CenterCrop((224, 224)),
    ]
)

def _compute_episode_midpoints(
    dataset: ResilientLeRobotDataset,
    episodes: Optional[List[int]],
) -> Tuple[List[int], List[int]]:
    """
    Return a single middle-frame index for each target episode (global index) and the episode ids used.

    When the dataset is initialized with a subset of episodes, the metadata still contains offsets for
    the full dataset. We therefore compute midpoints relative to the actually loaded subset ordering.
    """
    meta_eps = getattr(getattr(dataset, "meta", None), "episodes", None)
    if meta_eps is None:
        raise AttributeError("Dataset does not expose episode metadata required for midpoint sampling.")

    target_episode_ids = list(episodes) if episodes is not None else list(range(len(meta_eps)))

    loaded_episodes = getattr(dataset, "episodes", None)
    loaded_episodes = list(loaded_episodes) if loaded_episodes is not None else list(range(len(meta_eps)))

    midpoint_map: Dict[int, int] = {}
    current_idx = 0
    for ep_id in loaded_episodes:
        if ep_id < 0 or ep_id >= len(meta_eps):
            raise ValueError(
                f"Episode id {ep_id} requested but metadata only contains {len(meta_eps)} episodes."
            )
        ep = meta_eps[ep_id]
        length = ep.get("length")
        if length is None:
            raise ValueError(f"Episode {ep_id} is missing a length field required for midpoint calculation.")
        if ep_id in target_episode_ids:
            midpoint_map[ep_id] = current_idx + int(length) // 2
        current_idx += int(length)

    missing = [ep_id for ep_id in target_episode_ids if ep_id not in midpoint_map]
    if missing:
        raise ValueError(
            "Failed to compute midpoint indices for requested episodes (not present in loaded subset): "
            + ", ".join(str(ep) for ep in missing)
        )

    midpoint_indices = [midpoint_map[ep_id] for ep_id in target_episode_ids]
    return midpoint_indices, target_episode_ids


def _ensure_delta_timestamps(
    dataset_cfg: "DatasetConfig",
    metadata: LeRobotDatasetMetadata,
) -> Dict[str, Sequence[float]]:
    """Ensure delta timestamps are provided for all cameras and action key."""
    max_length = max(int(length) for length in dataset_cfg.sequence_length_distribution.keys())
    step = dataset_cfg.frame_delta_seconds
    if step <= 0:
        step = 1.0 / metadata.fps
    # Generate offsets symmetric around 0
    offsets = [step * (i - (max_length - 1) / 2) for i in range(max_length)]
    delta = {camera: list(offsets) for camera in dataset_cfg.cameras}

    for key in dataset_cfg.action_keys:
        delta[key] = list(offsets)
    return delta


def _coerce_frame_delta(value: Union[float, str]) -> float:
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            result = float(numerator) / float(denominator)
        else:
            result = float(text)
    else:
        raise TypeError(
            f"frame_delta_seconds must be a float or string, received {type(value).__name__}."
        )
    if result <= 0:
        raise ValueError("frame_delta_seconds must be strictly positive.")
    return result