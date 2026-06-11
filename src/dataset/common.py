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

UR5_DELTA_BASE_STATS = {
    "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "std": [
        0.018853237852454185,
        0.014781574718654156,
        0.016488293185830116,
        0.04993394762277603,
        0.05001137778162956,
        0.06182064861059189,
        100,
    ],
}

UR5_DELTA_LOCAL_STATS = {
    "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "std": [
        0.01841617003083229,
        0.014865336008369923,
        0.016895292326807976,
        0.05121244490146637,
        0.04898665100336075,
        0.06159385293722153,
        100,
    ],
}

LEGACY_SOAR_STATS = {
    "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
    "std": [0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.5],
}

SHARED_DELTA_ABSG_STATS = {
    "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
    "std": [0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.5],
}

BRIDGEV2_ACTION_STATS = {
    "mean": [
        0.0005735716009908971,
        0.00011228325259468299,
        0.0003373184047051246,
        -3.283241991818777e-05,
        -0.0018181045801051378,
        0.00010191781941100798,
        0.6233555981508744,
    ],
    "std": [
        0.010469300087834605,
        0.015241937360906514,
        0.012821477711892938,
        0.03112553826170278,
        0.032741803465084277,
        0.055619840206296726,
        0.48136343907227913,
    ],
}

BRIDGEV2_JOINT_POSITION_STATS = {
    "mean": [
        0.3078664089670387,
        0.03112775889086699,
        0.0652421230355093,
        0.010478423792782094,
        -0.11532839496184308,
        0.13292687768630143,
    ],
    "std": [
        0.06390384208181099,
        0.09462589615812915,
        0.05397352374040936,
        0.14631031040716821,
        0.17009176302991325,
        0.608639036668274,
    ],
}

BRIDGEV2_PROPRIO_ACTION_STATS = {
    "mean": BRIDGEV2_JOINT_POSITION_STATS["mean"] + BRIDGEV2_ACTION_STATS["mean"],
    "std": BRIDGEV2_JOINT_POSITION_STATS["std"] + BRIDGEV2_ACTION_STATS["std"],
}

ACTION_MODE_DELTA_STATS = {
    "bridgev2_action_normalized": BRIDGEV2_ACTION_STATS,
    "bridgev2_proprio_action_normalized": BRIDGEV2_PROPRIO_ACTION_STATS,
    "soar_matchur5": SHARED_DELTA_ABSG_STATS,
    "ur5_delta_absg": SHARED_DELTA_ABSG_STATS,
    "ur5_relative_local_ee_normalized": UR5_DELTA_LOCAL_STATS,
    "ur5_relative_base_ee_normalized": UR5_DELTA_BASE_STATS,
    "ur5_ee_state_relative_base_ee_normalized": UR5_DELTA_BASE_STATS,
}


def get_action_delta_stats(action_mode: Optional[str]) -> Optional[Dict[str, List[float]]]:
    return ACTION_MODE_DELTA_STATS.get(action_mode)


def denormalize_actions(
    actions: torch.Tensor,
    stats: Optional[Dict] = None,
) -> torch.Tensor:
    if stats is not None:
        mean = torch.as_tensor(stats["mean"], device=actions.device, dtype=actions.dtype)
        std = torch.as_tensor(stats["std"], device=actions.device, dtype=actions.dtype)
    else:
        mean = torch.tensor(LEGACY_SOAR_STATS["mean"], device=actions.device, dtype=actions.dtype)
        std = torch.tensor(LEGACY_SOAR_STATS["std"], device=actions.device, dtype=actions.dtype)
    return actions * (std + 1e-8) + mean

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
        "bridgev2_action_normalized": ["action"],
        "bridgev2_proprio_action_normalized": ["observation.state", "action"],
        "soar_relative_ee": ["action"],
        "soar_relative_ee_normalized": ["action"],
        "soar_matchur5": ["observation.state", "action"],
        "ur5_relative_local_ee_normalized": ["observation.state", "action.5hz_delta_local"],
        "ur5_relative_base_ee_normalized": ["observation.state", "action.5hz_delta_base"],
        "ur5_delta_absg": ["observation.state", "action", "action.5hz_delta_base"],
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
    if action_mode == "bridgev2_action_normalized":
        return _normalize_actions(item["action"], stats=BRIDGEV2_ACTION_STATS)
    if action_mode == "bridgev2_proprio_action_normalized":
        joint_positions = _normalize_actions(
            item["observation.state"][..., :6],
            stats=BRIDGEV2_JOINT_POSITION_STATS,
        )
        actions = _normalize_actions(item["action"], stats=BRIDGEV2_ACTION_STATS)
        return torch.cat([joint_positions, actions], dim=-1).float()
    if action_mode == "soar_relative_ee":
        return item["action"]
    if action_mode == "soar_relative_ee_normalized":
        return _normalize_actions(item["action"])
    if action_mode == "soar_matchur5":
        return _normalize_actions(_soar_matchur5_actions(item), stats=SHARED_DELTA_ABSG_STATS)
    if action_mode == "ur5_relative_local_ee_normalized":
        stats = UR5_DELTA_LOCAL_STATS
        return _normalize_actions(item["action.5hz_delta_local"], stats=stats)
    if action_mode == "ur5_relative_base_ee_normalized":
        stats = UR5_DELTA_BASE_STATS
        return _normalize_actions(item["action.5hz_delta_base"], stats=stats)
    if action_mode == "ur5_delta_absg":
        return _normalize_actions(_ur5_delta_absg_actions(item), stats=SHARED_DELTA_ABSG_STATS)
    if action_mode == "ur5_ee_state_relative_base_ee_normalized":
        stats = UR5_DELTA_BASE_STATS
        ee_state = item["observation.state"][..., 6:13].clone()
        # This is to recale the gripper vel from [0, 100] to [-pi, pi]
        ee_state[..., -1] = ee_state[..., -1] / 100.0 * (2 * 3.141592653589793) - 3.141592653589793
        base_action = _normalize_actions(item["action.5hz_delta_base"], stats=stats)
        return torch.cat([ee_state, base_action], dim=-1).float()
    raise ValueError(f"Unknown action mode: {action_mode}")

def _ur5_delta_absg_actions(item: Dict[str, Any]) -> torch.Tensor:
    actions = item["action.5hz_delta_base"].clone()
    actions[..., -1] = item["action"][..., -1] / 100.0
    return actions

def _soar_matchur5_actions(item: Dict[str, Any]) -> torch.Tensor:
    actions = item["action"].clone()
    current_rpy = item["observation.state"][..., 3:6]
    target_rpy = current_rpy + actions[..., 3:6]
    current_rotation = _rpy_to_matrix(current_rpy)
    target_rotation = _rpy_to_matrix(target_rpy)
    delta_rotation = target_rotation @ current_rotation.transpose(-1, -2)
    actions[..., 3:6] = _matrix_to_rotvec(delta_rotation)
    return actions

def _rpy_to_matrix(rpy: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = rpy.unbind(dim=-1)
    sr, cr = torch.sin(roll), torch.cos(roll)
    sp, cp = torch.sin(pitch), torch.cos(pitch)
    sy, cy = torch.sin(yaw), torch.cos(yaw)

    return torch.stack(
        [
            torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], dim=-1),
            torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], dim=-1),
            torch.stack([-sp, cp * sr, cp * cr], dim=-1),
        ],
        dim=-2,
    )

def _matrix_to_rotvec(matrix: torch.Tensor) -> torch.Tensor:
    trace = matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    theta = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))
    skew = torch.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        dim=-1,
    )
    sin_theta = torch.sin(theta)
    scale = torch.where(theta.abs() < 1e-4, 0.5 + theta.square() / 12.0, theta / (2.0 * sin_theta))
    return scale.unsqueeze(-1) * skew

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
        mean = torch.tensor(LEGACY_SOAR_STATS["mean"], device=actions.device)
        std = torch.tensor(LEGACY_SOAR_STATS["std"], device=actions.device)
    return torch.clamp((actions - mean) / (std + 1e-8), -clamp_range, clamp_range).float()
