"""
Generate an offline visualisation report for the world model pipeline.

The script simulates a single training step: it fetches a short sequence from
the dataset, encodes the frames with the DINO embedder, samples diffusion noise
using the configured scheduler, and exports the same plots previously produced
by the in-loop debuggers.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from training.world_model_trainer import WorldModelTrainingConfig, load_training_config
from training.datasets import (
    DatasetConfig,
    _ensure_delta_timestamps,
)
from training.diffusion import DimensionShiftedUniformScheduler, sample_base_noise
from visualization.diffusion import DiffusionVisualizer
from visualization.transformer import TransformerVisualizer
from vision.dino_v3 import DinoV3Embedder
from world_model.transformer import WorldModelBackbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline visualisation for the world model pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/world_model.yaml"),
        help="Path to the YAML config file used for training.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "visualization" / "visualization_reports",
        help="Directory where the visualisation artefacts will be written.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Dataset sample index to visualise.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional torch device override (e.g. cuda, mps, cpu).",
    )
    return parser.parse_args()


def resolve_device(preferred: Optional[str]) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_dataset_sequence(
    cfg: WorldModelTrainingConfig,
    sample_index: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    dataset_cfg: DatasetConfig = cfg.dataset
    metadata = LeRobotDatasetMetadata(dataset_cfg.repo_id)
    delta_timestamps = _ensure_delta_timestamps(dataset_cfg, metadata)
    dataset = LeRobotDataset(
        dataset_cfg.repo_id,
        episodes=list(dataset_cfg.episodes) if dataset_cfg.episodes else None,
        delta_timestamps=delta_timestamps,
    )
    if sample_index >= len(dataset):
        raise IndexError(f"Requested sample_index={sample_index} but dataset only has {len(dataset)} samples.")
    sample = dataset[sample_index]
    info = {
        "repo_id": dataset_cfg.repo_id,
        "sample_index": sample_index,
        "num_available_samples": len(dataset),
        "cameras": list(dataset_cfg.cameras),
        "action_key": dataset_cfg.action_key,
        "sequence_length_distribution": dataset_cfg.sequence_length_distribution,
    }
    return sample, info


def ensure_channel_first(frames: torch.Tensor) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError(f"Expected frames tensor with 4 dims, got shape {tuple(frames.shape)}.")
    if frames.shape[1] == 3:
        return frames
    if frames.shape[-1] == 3:
        return frames.permute(0, 3, 1, 2)
    raise ValueError(f"Unable to interpret frame layout with shape {tuple(frames.shape)}.")


def save_frame_sequence(frames: torch.Tensor, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_cpu = frames.detach().cpu()
    for idx, frame in enumerate(frames_cpu):
        img = frame
        if img.dtype.is_floating_point:
            data = img.clone()
            if data.max() <= 1.0:
                data = data * 255.0
            data = data.clamp(0.0, 255.0).round().to(torch.uint8)
        else:
            data = img.clamp(0, 255).to(torch.uint8)
        data = data.permute(1, 2, 0).numpy()
        image = Image.fromarray(data)
        image.save(output_dir / f"frame_{idx:02d}.png")


def build_sequence_tensors(
    sample: Dict[str, torch.Tensor],
    cfg: WorldModelTrainingConfig,
    device: torch.device,
    max_sequence_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    camera_key = cfg.dataset.cameras[0]
    if camera_key not in sample:
        available = ", ".join(sample.keys())
        raise KeyError(f"Camera key '{camera_key}' not found in sample. Available keys: {available}.")
    frames = torch.as_tensor(sample[camera_key])
    frames = ensure_channel_first(frames).to(torch.float32)
    if frames.shape[0] < max_sequence_length:
        raise ValueError(
            f"Camera sequence has {frames.shape[0]} frames but {max_sequence_length} are required "
            "per DatasetConfig.sequence_length_distribution."
        )
    frames = frames[-max_sequence_length:]

    action_key = cfg.dataset.action_key
    if not action_key or action_key not in sample:
        raise KeyError("Action key missing from dataset sample; cannot reconstruct training inputs.")
    actions = torch.as_tensor(sample[action_key]).to(torch.float32)
    if actions.ndim == 1:
        actions = actions.unsqueeze(-1)
    if actions.shape[0] < max_sequence_length:
        raise ValueError(
            f"Action sequence has {actions.shape[0]} steps but {max_sequence_length} are required."
        )
    actions = actions[-max_sequence_length:]

    actions_delta = torch.zeros((max_sequence_length, actions.shape[-1]), dtype=torch.float32)
    if max_sequence_length > 1:
        deltas = actions[1:] - actions[:-1]
        actions_delta[1:] = deltas

    frames = frames.to(device=device)
    actions_delta = actions_delta.to(device=device)
    return frames, actions_delta, camera_key


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    device = resolve_device(args.device or config.trainer.device)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    diffusion_dir = output_dir / "diffusion"
    transformer_dir = output_dir / "transformer"

    max_sequence_length = max(int(length) for length in config.dataset.sequence_length_distribution.keys())
    sample, dataset_info = load_dataset_sequence(config, args.sample_index)
    frames, actions_delta, camera_key = build_sequence_tensors(sample, config, device, max_sequence_length)

    embedder = DinoV3Embedder(config.vision)
    embedder.to(device)
    embedder.eval()

    with torch.no_grad():
        latents = embedder(frames)
    if latents.ndim != 3:
        raise ValueError(f"Expected latents tensor with 3 dims, got {tuple(latents.shape)}.")

    latents_batch = latents.unsqueeze(0)
    actions_batch = actions_delta.unsqueeze(0).to(latents_batch.dtype)

    scheduler = DimensionShiftedUniformScheduler(config.diffusion)
    tau = scheduler.sample(latents_batch)
    base_noise = sample_base_noise(latents_batch, config.diffusion)
    tau_factor = tau.unsqueeze(-1).unsqueeze(-1)
    noisy_latents = (1.0 - tau_factor) * base_noise + tau_factor * latents_batch
    target_velocity = latents_batch - base_noise

    diffusion_viz = DiffusionVisualizer(
        config.diffusion.debug,
        config.diffusion,
        output_dir=diffusion_dir,
    )
    diffusion_viz.generate(
        latents=latents_batch.cpu(),
        signal_levels=tau.cpu(),
        base_noise=base_noise.cpu(),
        noisy_latents=noisy_latents.cpu(),
        target_velocity=target_velocity.cpu(),
    )

    latent_tokens = latents_batch.shape[2]
    tokens_per_frame = latent_tokens + 2 + config.world_model.num_registers
    spatial_mask, _ = WorldModelBackbone._build_spatial_masks(
        latent_tokens,
        config.world_model.num_registers,
        device=torch.device("cpu"),
    )
    temporal_indices = WorldModelBackbone._build_temporal_indices(
        latent_tokens,
        device=torch.device("cpu"),
    )
    transformer_viz = TransformerVisualizer(
        config.world_model.debug,
        config.world_model,
        output_dir=transformer_dir,
    )
    transformer_viz.generate(
        latent_tokens=latent_tokens,
        tokens_per_frame=tokens_per_frame,
        temporal_indices=temporal_indices,
        temporal_context=config.world_model.temporal_context,
        spatial_mask=spatial_mask,
    )

    save_frame_sequence(frames, frames_dir)

    summary = {
        "device": str(device),
        "camera_key": camera_key,
        "sequence_length": max_sequence_length,
        "latent_shape": [int(dim) for dim in latents_batch.shape],
        "actions_shape": [int(dim) for dim in actions_batch.shape],
        "latent_stats": {
            "mean": float(latents_batch.mean().item()),
            "std": float(latents_batch.std(unbiased=False).item()),
        },
        "action_stats": {
            "mean": float(actions_batch.mean().item()),
            "std": float(actions_batch.std(unbiased=False).item()),
        },
        "diffusion": {
            "tau_mean": float(tau.mean().item()),
            "tau_std": float(tau.std(unbiased=False).item()),
            "base_noise_std": float(base_noise.std(unbiased=False).item()),
        },
        "latent_tokens": latent_tokens,
        "tokens_per_frame": tokens_per_frame,
        "dataset": dataset_info,
        "output_directories": {
            "base": str(output_dir),
            "frames": str(frames_dir),
            "diffusion": str(diffusion_dir),
            "transformer": str(transformer_dir),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Visualisation artefacts written to {output_dir}")


if __name__ == "__main__":
    main()
