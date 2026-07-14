import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import pyrallis
import torch
import torch.distributed as dist

from src.diffusion.euler_solver import EulerSolver
from src.planning.action_tokens import ActionTokenBuilder
from src.planning.cem import CEMWorldModelPlanner
from src.planning.config import PlanningScriptConfig
from src.planning.data import _single_lerobot_dataset_config, load_planning_sample
from src.rae_dino import (
    build_autoencoder,
    configured_autoencoder_resolution,
    validate_autoencoder_input_dim,
)
from src.training.logger import WorldModelLogger
from src.training.utils import set_seed
from src.world_model.backbone import WorldModelBackbone


def _initialize_distributed() -> tuple[torch.device, int, int, bool]:
    requested_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = requested_world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU planning requires CUDA devices.")
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device, rank, world_size, distributed


def _load_checkpoint(
    path: str,
    device: torch.device,
    logger: WorldModelLogger,
) -> tuple[Dict, Dict[str, torch.Tensor]]:
    logger.info("Loading checkpoint from %s...", path)
    checkpoint = torch.load(path, map_location=device)
    if "ema_model" in checkpoint:
        logger.info("Found EMA model in checkpoint; using EMA weights for planning.")
        state_dict = checkpoint["ema_model"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[10:] if key.startswith("_orig_mod.") else key] = value
    return checkpoint, cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Goal-conditioned CEM planning with a trained world model.")
    parser.add_argument("--config_path", type=str, default="configs/planning_ur5.yaml")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    config = pyrallis.parse(config_class=PlanningScriptConfig, config_path=args.config_path)
    device, rank, world_size, distributed = _initialize_distributed()
    # Keep sample selection and model initialization identical across ranks.
    seed = set_seed(config.planning.seed, world_size=1, rank=0)
    is_main_process = rank == 0

    logger = WorldModelLogger(config.logging, is_main_process=is_main_process)
    logger.info("Planning on device=%s seed=%d rank=%d/%d.", device, seed, rank, world_size)

    checkpoint, checkpoint_state = _load_checkpoint(
        config.checkpoint_path,
        device,
        logger,
    )
    saved_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if config.use_checkpoint_config and isinstance(saved_config, dict):
        if isinstance(saved_config.get("world_model"), dict):
            config.world_model = pyrallis.decode(
                type(config.world_model),
                saved_config["world_model"],
            )
            logger.info("Using world-model architecture saved in the checkpoint.")
        if isinstance(saved_config.get("autoencoder"), dict):
            config.autoencoder = pyrallis.decode(
                type(config.autoencoder),
                saved_config["autoencoder"],
            )
            logger.info("Using autoencoder profile saved in the checkpoint.")
    if config.planning.max_context_frames is None:
        config.planning.max_context_frames = config.world_model.temporal_context_length
    config.eval_dataset.image_size = configured_autoencoder_resolution(
        config.autoencoder
    )
    logger.init_wandb(pyrallis.encode(config))

    _, ds_conf_dict = _single_lerobot_dataset_config(config.eval_dataset)
    action_mode = ds_conf_dict.get("action_mode")
    if action_mode is None:
        raise ValueError("eval_dataset dataset action_mode must be set for planning.")
    if config.eval_dataset.action_dim != config.world_model.action_dim:
        raise ValueError(
            f"eval_dataset.action_dim={config.eval_dataset.action_dim} must match "
            f"world_model.action_dim={config.world_model.action_dim}."
        )

    action_builder = ActionTokenBuilder(
        action_mode=action_mode,
        action_dim=config.eval_dataset.action_dim,
        device=device,
    )

    sample = load_planning_sample(
        planning_cfg=config.planning,
        cem_cfg=config.cem,
        dataset_cfg=config.eval_dataset,
        action_builder=action_builder,
        logger=logger,
    )

    logger.info("Initializing world model...")
    model = WorldModelBackbone(config.world_model)
    model.load_state_dict(checkpoint_state)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    if config.planning.compile_model:
        model = torch.compile(model)

    logger.info("Initializing autoencoder...")
    autoencoder = build_autoencoder(config.autoencoder)
    validate_autoencoder_input_dim(autoencoder, config.world_model.input_dim)
    autoencoder.to(device)
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad_(False)

    solver = EulerSolver(config.euler_solver, config.signal_scheduler)
    planner = CEMWorldModelPlanner(
        planning_cfg=config.planning,
        cem_cfg=config.cem,
        visualization_cfg=config.visualization,
        model=model,
        autoencoder=autoencoder,
        solver=solver,
        action_builder=action_builder,
        logger=logger,
        device=device,
        rank=rank,
        world_size=world_size,
    )

    result = planner.run(sample)
    if is_main_process:
        result_path = Path(result.output_dir) / "planning_result.pt"
        torch.save(
            {
                "mode": result.mode,
                "deltas_denormalized": result.deltas,
                "planned_positions": result.planned_positions,
                "episode_index": sample.episode_index,
                "start_time_s": config.planning.start_time_s,
                "goal_time_s": config.planning.goal_time_s,
                "horizon": sample.horizon,
                "target_step": sample.target_step,
                "action_mode": sample.action_mode,
            },
            result_path,
        )
        logger.info("Saved planning result to %s.", result_path)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    logger.close()


if __name__ == "__main__":
    main()
