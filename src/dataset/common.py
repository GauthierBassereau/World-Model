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
        "ur5_relative_local_ee_normalized": ["action.5hz_delta_local"],
        "ur5_relative_base_ee_normalized": ["action.5hz_delta_base"],
        "ur5_ee_state_relative_base_ee_normalized": ["observation.state", "action.5hz_delta_base"],
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
        # This is horrible I know. I rewrite the stats manually because I need same norm for both train and eval. And this is the fastest way to do it. Hoping no one reads this.
        stats = {
            "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "std": [0.01841617003083229, 0.014865336008369923, 0.016895292326807976, 0.05121244490146637, 0.04898665100336075, 0.06159385293722153, 100]
        }
        return _normalize_actions(item["action.5hz_delta_local"], stats=stats)
    if action_mode == "ur5_relative_base_ee_normalized":
        stats = {
            "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "std": [0.018853237852454185, 0.014781574718654156, 0.016488293185830116, 0.04993394762277603, 0.05001137778162956, 0.06182064861059189, 100]
        }
        return _normalize_actions(item["action.5hz_delta_base"], stats=stats)
    if action_mode == "ur5_ee_state_relative_base_ee_normalized":
        stats = {
            "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "std": [0.018853237852454185, 0.014781574718654156, 0.016488293185830116, 0.04993394762277603, 0.05001137778162956, 0.06182064861059189, 100]
        }
        ee_state = item["observation.state"][..., 6:13].clone()
        # This is to recale the gripper vel from [0, 100] to [-pi, pi]
        ee_state[..., -1] = ee_state[..., -1] / 100.0 * (2 * 3.141592653589793) - 3.141592653589793
        base_action = _normalize_actions(item["action.5hz_delta_base"], stats=stats)
        return torch.cat([ee_state, base_action], dim=-1).float()
    raise ValueError(f"Unknown action mode: {action_mode}")

def _normalize_actions(
    actions: torch.Tensor,
    stats: Optional[Dict] = None,
    clamp_range: float = 10.0,
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
