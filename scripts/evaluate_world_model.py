import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from evaluation.world_model_evaluator import WorldModelEvaluator
from rae_dino.rae import RAE
from training.world_trainer import load_training_config, _dataclass_to_dict
from training.logger import WorldModelLogger
from world_model.flow_matching import EulerSolverConfig
from world_model.backbone import WorldModelBackbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained world model checkpoint.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretraining.yaml"),
        help="Path to the YAML config file used for training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the checkpoint to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run evaluation on (default: cuda).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Override the number of evaluation batches (default: config value).",
    )
    parser.add_argument(
        "--num-video-samples",
        type=int,
        default=None,
        help="Override how many sequences to log as videos (default: config value).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override evaluation batch size (default: config dataloader batch size).",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Evaluate the EMA weights if available in the checkpoint.",
    )
    parser.add_argument(
        "--rollout-start-frame",
        type=int,
        default=None,
        help="Override how many initial frames are kept clean before rolling out.",
    )
    parser.add_argument(
        "--rollout-signal-level",
        type=float,
        default=None,
        help="Override the signal level used for generated frames (default: config value).",
    )
    parser.add_argument(
        "--clean-signal-level",
        type=float,
        default=None,
        help="Override the signal level used for conditioned frames (default: config value).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("archive/evaluation"),
        help="Directory where evaluation artifacts (videos/plots) will be saved.",
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="Existing wandb run ID to log evaluation metrics/videos to.",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Global step of the evaluated checkpoint (used for wandb logging).",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, model: torch.nn.Module, use_ema: bool) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    key = "ema_model" if use_ema and "ema_model" in checkpoint else "model"
    state_dict = checkpoint.get(key)
    if state_dict is None:
        raise KeyError(f"Checkpoint {checkpoint_path} missing '{key}' weights.")
    model.load_state_dict(state_dict)

def _save_videos(videos: Dict[str, torch.Tensor], output_dir: Path, fps: int) -> List[Path]:
    saved_paths: List[Path] = []
    if not videos:
        return saved_paths
    for key, tensor in videos.items():
        name = key.replace("/", "_") + ".mp4"
        path = output_dir / name
        frames = tensor.detach().cpu().numpy()
        if frames.ndim != 4:
            continue
        frames = np.clip(frames, 0.0, 1.0)
        frames = (frames * 255.0).astype(np.uint8)
        frames = np.transpose(frames, (0, 2, 3, 1))
        imageio.mimwrite(path, frames, fps=fps, format="mp4")
        saved_paths.append(path)
    return saved_paths


def _collect_series(metrics: Dict[str, float], metric_type: str) -> Dict[str, List[Tuple[int, float]]]:
    series: Dict[str, List[Tuple[int, float]]] = {}
    prefix = f"{metric_type}_t+"
    for key, value in metrics.items():
        if not key.startswith("eval/"):
            continue
        parts = key.split("/")
        if len(parts) != 3:
            continue
        scenario, step_key = parts[1], parts[2]
        if not step_key.startswith(prefix):
            continue
        try:
            step = int(step_key[len(prefix):])
        except ValueError:
            continue
        series.setdefault(scenario, []).append((step, value))
    for values in series.values():
        values.sort(key=lambda item: item[0])
    return series


def _plot_rollout_series(metrics: Dict[str, float], metric_type: str, ylabel: str, output_path: Path) -> None:
    series = _collect_series(metrics, metric_type)
    if not series:
        return
    plt.figure(figsize=(6, 4))
    for scenario, values in series.items():
        steps = [step for step, _ in values]
        vals = [val for _, val in values]
        plt.plot(steps, vals, marker="o", label=scenario)
    plt.xlabel("Rollout step (frames after conditioning)")
    plt.ylabel(ylabel)
    plt.title(f"Rollout {metric_type} vs. horizon")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()

    config = load_training_config(args.config)
    eval_cfg = config.evaluation
    if args.max_batches is not None:
        eval_cfg.max_batches = max(1, args.max_batches)
    if args.num_video_samples is not None:
        eval_cfg.num_video_samples = max(1, args.num_video_samples)
    if args.batch_size is not None:
        eval_cfg.batch_size = max(1, args.batch_size)
    if args.rollout_start_frame is not None:
        eval_cfg.rollout_start_frame = max(1, args.rollout_start_frame)
    if args.rollout_signal_level is not None:
        eval_cfg.rollout_signal_level = float(args.rollout_signal_level)
    if args.clean_signal_level is not None:
        eval_cfg.clean_signal_level = float(args.clean_signal_level)

    if config.dataset.evaluation_episodes is None:
        raise ValueError(
            "dataset.evaluation_episodes must be specified in the config to run evaluation."
        )

    device = torch.device(args.device)
    model = WorldModelBackbone(config.world_model).to(device)
    load_model(args.checkpoint, model, args.use_ema)
    model.eval()

    autoencoder = RAE().to(device)
    autoencoder.eval()

    evaluator = WorldModelEvaluator(
        config=eval_cfg,
        dataset_cfg=config.dataset,
        dataloader_cfg=config.dataloader,
        diffusion_cfg=config.diffusion,
        autoencoder=autoencoder,
        device=device,
        seed=config.trainer.seed,
    )

    result = evaluator.evaluate(model)
    if result is None:
        raise RuntimeError(
            "Evaluation could not be run. Ensure dataset.evaluation_episodes is configured."
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fps = max(1, int(round(1.0 / config.dataset.frame_delta_seconds)))
    saved_videos = _save_videos(result.videos, output_dir, fps)
    mean_plot = output_dir / "rollout_mean.png"
    var_plot = output_dir / "rollout_variance.png"
    _plot_rollout_series(result.metrics, "mean", "Mean absolute error", mean_plot)
    _plot_rollout_series(result.metrics, "var", "Variance of absolute error", var_plot)

    if args.wandb_run_id:
        logger = WorldModelLogger(
            config.logging,
            euler_cfg=EulerSolverConfig(),
            sample_fps=fps,
            is_main_process=True,
        )
        logger.init_wandb(
            _dataclass_to_dict(config),
            run_id=args.wandb_run_id,
            resume="must",
        )
        step = args.checkpoint_step if args.checkpoint_step is not None else 0
        logger.start_step(step)
        logger.log_evaluation(result)
        logger.close()

    print("Evaluation metrics:")
    for key in sorted(result.metrics.keys()):
        print(f"  {key}: {result.metrics[key]:.6f}")
    if saved_videos:
        print("Saved videos:")
        for path in saved_videos:
            print(f"  {path}")
    if mean_plot.exists():
        print(f"Rollout mean plot saved to {mean_plot}")
    if var_plot.exists():
        print(f"Rollout variance plot saved to {var_plot}")


if __name__ == "__main__":
    main()
