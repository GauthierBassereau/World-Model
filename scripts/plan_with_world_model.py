import argparse
import sys
from pathlib import Path
from typing import Dict

import pyrallis
import torch

from src.diffusion.euler_solver import EulerSolver
from src.planning.action_tokens import ActionTokenBuilder
from src.planning.cem import CEMWorldModelPlanner
from src.planning.config import PlanningScriptConfig
from src.planning.data import _single_lerobot_dataset_config, load_planning_sample
from src.rae_dino.rae import RAE
from src.training.logger import WorldModelLogger
from src.training.utils import set_seed
from src.world_model.backbone import WorldModelBackbone


def _load_checkpoint_state(path: str, device: torch.device, logger: WorldModelLogger) -> Dict[str, torch.Tensor]:
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
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Goal-conditioned CEM planning with a trained world model.")
    parser.add_argument("--config_path", type=str, default="configs/planning_ur5.yaml")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    config = pyrallis.parse(config_class=PlanningScriptConfig, config_path=args.config_path)
    if config.planning.max_context_frames is None:
        config.planning.max_context_frames = config.world_model.temporal_context_length

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = set_seed(config.planning.seed, world_size=1, rank=0)

    logger = WorldModelLogger(config.logging, is_main_process=True)
    logger.init_wandb(pyrallis.encode(config))
    logger.info("Planning on device=%s seed=%d.", device, seed)

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
    model.load_state_dict(_load_checkpoint_state(config.checkpoint_path, device, logger))
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    if config.planning.compile_model:
        model = torch.compile(model)

    logger.info("Initializing autoencoder...")
    autoencoder = RAE()
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
    )

    result = planner.run(sample)
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
    logger.close()


if __name__ == "__main__":
    main()
