import argparse
from fractions import Fraction
from pathlib import Path
from typing import Optional, Tuple

import pyrallis
import torch
from torchvision.io import write_video
from torchvision.utils import make_grid, save_image

from training.dataset import WorldModelBatch, build_world_model_dataloader
from training.world_trainer import WorldModelTrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the eval dataloader and export a sample sequence as a video."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretraining.yaml"),
        help="Path to the YAML config that defines eval_data (default: configs/pretraining.yaml).",
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        default=0,
        help="Which batch to pull from the eval dataloader (0-based).",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Which sample within the chosen batch to visualise (0-based).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("archive/eval_dataloader_sample.mp4"),
        help="Where to save the extracted video (default: archive/eval_dataloader_sample.mp4).",
    )
    parser.add_argument(
        "--grid-output",
        type=Path,
        default=Path("archive/eval_dataloader_sample_grid.png"),
        help="Optional contact sheet of the sequence frames (default: archive/eval_dataloader_sample_grid.png).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override frames-per-second for the video file; defaults to 1/frame_delta_seconds from the config.",
    )
    return parser.parse_args()


def load_eval_configs(config_path: Path) -> WorldModelTrainingConfig:
    return pyrallis.parse(
        config_class=WorldModelTrainingConfig,
        config_path=str(config_path),
        args=[],
    )


def fetch_batch(
    loader: torch.utils.data.DataLoader,
    target_batch_idx: int,
) -> Tuple[WorldModelBatch, int]:
    if target_batch_idx < 0:
        raise ValueError("batch-index must be >= 0.")
    for batch_idx, batch in enumerate(loader):
        if batch_idx == target_batch_idx:
            return batch, batch_idx
    raise IndexError(
        f"Dataloader ended before reaching batch {target_batch_idx}. "
        f"Length={len(loader)}; try a smaller --batch-index."
    )


def to_video_tensor(
    frames: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if valid_mask is not None:
        if valid_mask.ndim != 1 or valid_mask.shape[0] != frames.shape[0]:
            raise ValueError("frames_valid_mask must be 1D with length equal to sequence length.")
        if valid_mask.any():
            frames = frames[valid_mask]
        else:
            print("[!] frames_valid_mask has no valid entries; exporting all frames instead.")
    if frames.numel() == 0:
        raise ValueError("No frames available to export.")
    video = frames.clamp(0.0, 1.0) * 255.0
    video = video.to(dtype=torch.uint8).permute(0, 2, 3, 1).contiguous()
    return video


def save_contact_sheet(frames: torch.Tensor, output_path: Path) -> None:
    grid = make_grid(
        frames,
        nrow=min(8, frames.shape[0]),
        padding=2,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output_path)


def main() -> None:
    args = parse_args()
    training_cfg = load_eval_configs(args.config)
    dataset_cfg = training_cfg.eval_data.eval_dataset
    dataloader_cfg = training_cfg.eval_data.eval_dataloader

    loader = build_world_model_dataloader(
        dataset_cfg=dataset_cfg,
        dataloader_cfg=dataloader_cfg,
        device=torch.device("cpu"),
    )
    print(f"[ ] Eval dataset episodes: {dataset_cfg.episodes or 'all'} | length={len(loader.dataset)}")
    print(f"[ ] Eval dataloader batches: {len(loader)} | batch_size={loader.batch_size}")

    batch, batch_idx = fetch_batch(loader, args.batch_index)
    print(batch.frames_valid_mask[0])
    if args.sample_index < 0 or args.sample_index >= batch.sequence_frames.shape[0]:
        raise IndexError(
            f"sample-index {args.sample_index} is out of range for batch size {batch.sequence_frames.shape[0]}"
        )
    print(f"[ ] Fetched batch {batch_idx} from eval dataloader, with shape {batch.sequence_frames.shape}")
    dataset_index = batch_idx * loader.batch_size + args.sample_index

    frames = batch.sequence_frames[args.sample_index].cpu()
    valid_mask = batch.frames_valid_mask[args.sample_index].cpu() if batch.frames_valid_mask is not None else None
    display_frames = frames if valid_mask is None or not valid_mask.any() else frames[valid_mask]
    video = to_video_tensor(frames, valid_mask)

    # Ensure fps is a plain Python type; torchvision/pyav can't handle numpy scalar types here.
    fps = float(args.fps) if args.fps is not None else float(1.0 / float(dataset_cfg.frame_delta_seconds))
    # PyAV accepts a Fraction explicitly; this also guards against numpy scalars leaking through.
    fps_rational = Fraction(fps).limit_denominator()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(args.output), video, fps=fps_rational, video_codec="h264")
    print(
        f"[x] Wrote video for dataset idx {dataset_index} "
        f"({video.shape[0]} frames @ {fps:.2f} fps) to {args.output}"
    )

    if args.grid_output:
        save_contact_sheet(display_frames, args.grid_output)
        print(f"[x] Saved contact sheet to {args.grid_output}")

    episode_ids = getattr(loader.dataset, "episode_ids", None)
    if episode_ids is not None and dataset_index < len(episode_ids):
        print(f"[ ] Episode id for sample: {episode_ids[dataset_index]}")


if __name__ == "__main__":
    main()
