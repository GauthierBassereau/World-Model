import torch
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as transforms_v2

@dataclass
class WorldBatch:
    sequence_frames: torch.Tensor
    sequence_actions: torch.Tensor
    independent_frames: torch.Tensor
    use_actions: torch.Tensor
    padded_frames: torch.Tensor
    dataset_indices: torch.Tensor
    dataset_names: Dict[int, str]
    episode_ids: torch.Tensor

RESIZE_CROP_TRANSFORM_224 = transforms_v2.Compose(
    [
        transforms_v2.Resize(
            size=224,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms_v2.CenterCrop(224),
        transforms_v2.ToDtype(torch.uint8, scale=True),
    ]
)

def get_delta_timestamps(
    action_mode: Optional[str], 
    fps: float,
    sequence_length: int,
    camera_keys: List[str]
) -> Dict[str, List[float]]:
    # We want to fetch T frames starting from t, so deltas are [0, 1/fps, ..., (T-1)/fps]
    # so that padded frames are at the end, -> doesn't affect learning since auto-regressive worldmodel and padded frames are not used for loss calculation
    delta_timestamps = {
        key: [i / fps for i in range(sequence_length)]
        for key in camera_keys
    }
    
    ACTION_KEYS = {
        "soar_relative_ee": ["action"],
        "soar_relative_ee_normalized": ["action"],
        "ur5_relative_local_ee_normalized": ["action.delta_local"],
    }
    
    if action_mode is not None:
        # action[i] is the action that takes us FROM frame[i-1] TO frame[i] (shifted by -1/fps)
        for key in ACTION_KEYS[action_mode]:
            delta_timestamps[key] = [(i - 1) / fps for i in range(sequence_length)]

    return delta_timestamps

def get_actions(
    action_mode: Optional[str], 
    item: Dict[str, Any], 
    sequence_length: int = 0,
    action_dim: int = 7,
    stats: Optional[Dict] = None,
) -> torch.Tensor:
    """Returns action tensor. For action_mode=None, returns zeros."""
    if action_mode is None:
        return torch.zeros(sequence_length, action_dim)
    if action_mode == "soar_relative_ee":
        return item["action"]
    if action_mode == "soar_relative_ee_normalized":
        return _normalize_actions(item["action"])
    if action_mode == "ur5_relative_local_ee_normalized":
        return _normalize_actions(item["action.delta_local"], stats=stats["action.delta_local"])
    raise ValueError(f"Unknown action mode: {action_mode}")

def _normalize_actions(
    actions: torch.Tensor,
    stats: Optional[Dict] = None,
    clamp_range: float = 5.0,
) -> torch.Tensor:
    """Normalize actions by mean/std and clamp."""
    if stats is not None:
        mean = torch.as_tensor(stats["mean"], device=actions.device)
        std = torch.as_tensor(stats["std"], device=actions.device)
    else:
        # Legacy fallback for soar
        mean = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], device=actions.device)
        std = torch.tensor([0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.5], device=actions.device)
    return torch.clamp((actions - mean) / (std + 1e-8), -clamp_range, clamp_range).float()