import torch
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
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
    action_mode: str, 
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
    # Shifted by -1/fps because action[i] is the action that leads TO frame[i] (from frame[i-1])
    delta_timestamps["action"] = [(i - 1) / fps for i in range(sequence_length)]

    return delta_timestamps

def get_actions(action_mode: str, item: Dict[str, Any], stats: Optional[Dict[str, Dict[str, float]]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if action_mode == "soar_relative_ee":
        d = item["action"].device
        return item["action"], torch.ones(item["action"].shape[0], dtype=torch.bool, device=d)
    elif action_mode == "soar_relative_ee_normalized":
        return _get_actions_soar_relative_normalized(item)
        
    raise ValueError(f"Unknown action mode for processing: {action_mode}")

def _get_actions_soar_relative_normalized(item: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    actions = item["action"]
    mean = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], device=actions.device)
    std = torch.tensor([0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.5], device=actions.device)
    
    # Clamp values to avoid outliers
    min_val = torch.tensor([-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -1.0], device=actions.device)
    max_val = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0], device=actions.device)
    
    normalized = (actions - mean) / std
    
    valid_mask = (normalized >= min_val) & (normalized <= max_val)
    valid_steps = valid_mask.all(dim=-1)
    
    return normalized, valid_steps