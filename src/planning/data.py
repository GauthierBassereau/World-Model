from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pyrallis
import torch

from src.dataset.lerobot_dataset import LeRobotDataset, LeRobotDatasetConfig
from src.dataset.world_dataset import WorldDatasetConfig
from src.planning.action_tokens import ActionTokenBuilder
from src.planning.config import CEMConfig, PlanningConfig
from src.training.logger import WorldModelLogger


@dataclass
class PlanningSample:
    frames: torch.Tensor
    context_actions: torch.Tensor
    context_use_actions: torch.Tensor
    context_frames: int
    horizon: int
    target_step: int
    start_local_frame: int
    goal_local_frame: int
    initial_ee_state: Optional[torch.Tensor]
    reference_ee_positions: Optional[torch.Tensor]
    real_video_frames: torch.Tensor
    goal_image: torch.Tensor
    fps: float
    native_fps: float
    action_mode: str
    episode_index: int
    start_frame_index: int
    goal_frame_index: int


def _single_lerobot_dataset_config(dataset_cfg: WorldDatasetConfig) -> Tuple[str, Dict[str, Any]]:
    if len(dataset_cfg.datasets) != 1:
        raise ValueError("Planning expects eval_dataset to contain exactly one dataset.")

    dataset_name, ds_conf_dict = next(iter(dataset_cfg.datasets.items()))
    ds_type = ds_conf_dict.get("type")
    if ds_type != "lerobot":
        raise ValueError(f"Planning only supports lerobot datasets, got {ds_type!r}.")

    ds_conf_dict = ds_conf_dict.copy()
    del ds_conf_dict["type"]
    return dataset_name, ds_conf_dict


def load_planning_sample(
    *,
    planning_cfg: PlanningConfig,
    cem_cfg: CEMConfig,
    dataset_cfg: WorldDatasetConfig,
    action_builder: ActionTokenBuilder,
    logger: WorldModelLogger,
) -> PlanningSample:
    if planning_cfg.context_frames < 1:
        raise ValueError("planning.context_frames must be >= 1.")
    if planning_cfg.goal_time_s <= planning_cfg.start_time_s:
        raise ValueError("planning.goal_time_s must be greater than planning.start_time_s.")

    _, ds_conf_dict = _single_lerobot_dataset_config(dataset_cfg)
    action_mode = ds_conf_dict.get("action_mode")
    if action_mode is None:
        raise ValueError("Planning requires an action-conditioned dataset action_mode.")
    if action_mode != action_builder.action_mode:
        raise ValueError(
            f"Action builder was created for {action_builder.action_mode!r}, "
            f"but dataset uses {action_mode!r}."
        )

    target_step = max(1, int(round((planning_cfg.goal_time_s - planning_cfg.start_time_s) * dataset_cfg.fps)))
    horizon = cem_cfg.horizon if cem_cfg.horizon is not None else target_step
    if horizon < 1:
        raise ValueError("CEM horizon must be >= 1.")

    required_future_steps = max(horizon, target_step)
    sequence_length = planning_cfg.context_frames + required_future_steps + 1

    ds_conf_dict["episodes"] = [planning_cfg.episode_index]
    ds_conf_dict["episode_startpoint_only"] = False
    ds_conf_dict["sequence_length"] = sequence_length
    ds_conf_dict["fps"] = dataset_cfg.fps
    ds_conf_dict["action_dim"] = dataset_cfg.action_dim
    ds_cfg = pyrallis.decode(LeRobotDatasetConfig, ds_conf_dict)
    dataset = LeRobotDataset(ds_cfg, logger=logger)

    episode_meta = dataset.backend.meta.episodes[planning_cfg.episode_index]
    episode_start = int(episode_meta["dataset_from_index"])
    episode_end = int(episode_meta["dataset_to_index"])
    native_fps = float(dataset.native_fps)

    start_offset = int(round(planning_cfg.start_time_s * native_fps))
    goal_offset = int(round(planning_cfg.goal_time_s * native_fps))
    start_frame_index = episode_start + start_offset
    goal_frame_index = episode_start + goal_offset
    sequence_start = start_frame_index - planning_cfg.context_frames + 1

    if sequence_start < episode_start:
        raise ValueError(
            "Not enough frames before planning.start_time_s for the requested context. "
            f"Need sequence start {sequence_start}, episode starts at {episode_start}."
        )
    if goal_frame_index >= episode_end:
        raise ValueError(
            f"Goal time {planning_cfg.goal_time_s}s maps to frame {goal_frame_index}, "
            f"outside episode [{episode_start}, {episode_end})."
        )

    sequence_end = sequence_start + sequence_length
    if sequence_end > episode_end:
        logger.warning(
            "Planning sequence extends past the episode end (%d > %d). "
            "LeRobot padding may be present after the goal/reference horizon.",
            sequence_end,
            episode_end,
        )

    try:
        dataset_index = dataset.indices.index(sequence_start)
    except ValueError as exc:
        raise ValueError(f"Could not find sequence start frame {sequence_start} in selected episode.") from exc

    batch = dataset[dataset_index]
    frames = batch.sequence_frames
    actions = batch.sequence_actions
    use_actions = batch.use_actions
    padded_frames = batch.padded_frames

    start_local_frame = planning_cfg.context_frames - 1
    goal_local_frame = planning_cfg.context_frames + target_step - 1
    if goal_local_frame >= frames.shape[0]:
        raise ValueError(
            f"Goal local frame {goal_local_frame} outside loaded frame sequence length {frames.shape[0]}."
        )
    if bool(padded_frames[goal_local_frame].item()):
        raise ValueError("Goal frame is padded; choose an earlier goal time or a longer episode.")

    context_actions = actions[: planning_cfg.context_frames].unsqueeze(0)
    context_use_actions = use_actions[: planning_cfg.context_frames].unsqueeze(0)
    initial_ee_state = action_builder.state_from_action_tokens(actions, planning_cfg.context_frames)

    reference_ee_positions = None
    if action_builder.has_state_token:
        ref_start = planning_cfg.context_frames
        ref_end = min(ref_start + target_step + 1, actions.shape[0])
        if ref_end - ref_start >= 2:
            reference_ee_positions = actions[ref_start:ref_end, :3].detach().cpu()

    real_video_frames = frames[start_local_frame : goal_local_frame + 1].detach().cpu()
    goal_image = frames[goal_local_frame].detach().cpu()

    logger.info(
        "Loaded planning episode=%d start=%.3fs goal=%.3fs target_step=%d horizon=%d context_frames=%d.",
        planning_cfg.episode_index,
        planning_cfg.start_time_s,
        planning_cfg.goal_time_s,
        target_step,
        horizon,
        planning_cfg.context_frames,
    )

    return PlanningSample(
        frames=frames.unsqueeze(0),
        context_actions=context_actions,
        context_use_actions=context_use_actions,
        context_frames=planning_cfg.context_frames,
        horizon=horizon,
        target_step=target_step,
        start_local_frame=start_local_frame,
        goal_local_frame=goal_local_frame,
        initial_ee_state=initial_ee_state,
        reference_ee_positions=reference_ee_positions,
        real_video_frames=real_video_frames,
        goal_image=goal_image,
        fps=float(dataset_cfg.fps),
        native_fps=native_fps,
        action_mode=action_mode,
        episode_index=planning_cfg.episode_index,
        start_frame_index=start_frame_index,
        goal_frame_index=goal_frame_index,
    )

