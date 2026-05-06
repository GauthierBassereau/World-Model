from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

from src.training.logger import WorldModelLogger


def frames_to_uint8(frames: torch.Tensor) -> np.ndarray:
    if frames.ndim != 4:
        raise ValueError(f"Expected frames with shape [T, C, H, W], got {tuple(frames.shape)}.")
    frames = frames.detach().cpu()
    if frames.dtype == torch.uint8:
        array = frames.numpy()
    else:
        array = (
            frames.to(dtype=torch.float32)
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(dtype=torch.uint8)
            .numpy()
        )
    return array


def save_image(frame: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.ndim != 3:
        raise ValueError(f"Expected frame with shape [C, H, W], got {tuple(frame.shape)}.")
    array = frames_to_uint8(frame.unsqueeze(0))[0]
    Image.fromarray(array.transpose(1, 2, 0)).save(path)


def save_video(frames: torch.Tensor, path: Path, fps: int, logger: WorldModelLogger) -> Optional[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    chw = frames_to_uint8(frames)
    thwc = chw.transpose(0, 2, 3, 1)

    try:
        import imageio.v3 as iio

        iio.imwrite(path, thwc, fps=fps)
        return path
    except Exception as imageio_v3_error:
        try:
            import imageio

            imageio.mimsave(path, list(thwc), fps=fps)
            return path
        except Exception:
            try:
                from torchvision.io import write_video

                write_video(str(path), torch.from_numpy(thwc), fps=fps)
                return path
            except Exception as torchvision_error:
                logger.warning(
                    "Could not save video to %s. imageio error=%r torchvision error=%r",
                    path,
                    imageio_v3_error,
                    torchvision_error,
                )
                return None


def log_wandb_video(logger: WorldModelLogger, key: str, frames: torch.Tensor, fps: int, step: int = 0) -> None:
    if logger.wandb_run is None or logger._wandb is None:
        return
    array = frames_to_uint8(frames)
    logger.wandb_run.log({key: logger._wandb.Video(array, fps=fps, format="mp4")}, step=step)


def log_wandb_image(logger: WorldModelLogger, key: str, image_path: Path, step: int = 0) -> None:
    if logger.wandb_run is None or logger._wandb is None:
        return
    logger.wandb_run.log({key: logger._wandb.Image(str(image_path))}, step=step)


def log_wandb_metrics(logger: WorldModelLogger, metrics: Dict[str, Any], step: int = 0) -> None:
    if logger.wandb_run is None or not metrics:
        return
    logger.wandb_run.log(metrics, step=step)


def save_trajectory_plot(
    *,
    planned_positions: torch.Tensor,
    reference_positions: Optional[torch.Tensor],
    path: Path,
    title: str,
) -> Path:
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    planned = planned_positions.detach().cpu().to(dtype=torch.float32).numpy()
    reference = None
    if reference_positions is not None:
        reference = reference_positions.detach().cpu().to(dtype=torch.float32).numpy()

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    if reference is not None and reference.shape[0] > 0:
        ax.plot(
            reference[:, 0],
            reference[:, 1],
            reference[:, 2],
            color="#202020",
            linewidth=2.5,
            label="reference trajectory",
        )
        ax.scatter(reference[0, 0], reference[0, 1], reference[0, 2], color="#202020", s=45, marker="o")
        ax.scatter(reference[-1, 0], reference[-1, 1], reference[-1, 2], color="#202020", s=60, marker="x")

    if planned.shape[0] > 0:
        ax.plot(
            planned[:, 0],
            planned[:, 1],
            planned[:, 2],
            color="#2f6fed",
            linewidth=2.5,
            label="planned trajectory",
        )
        ax.scatter(planned[0, 0], planned[0, 1], planned[0, 2], color="#2f6fed", s=45, marker="o")
        ax.scatter(planned[-1, 0], planned[-1, 1], planned[-1, 2], color="#2f6fed", s=60, marker="x")

    all_points = planned if reference is None else np.concatenate([planned, reference], axis=0)
    if all_points.size > 0:
        mins = all_points.min(axis=0)
        maxs = all_points.max(axis=0)
        centers = (mins + maxs) / 2.0
        radius = max(float((maxs - mins).max()) / 2.0, 1e-3)
        ax.set_xlim(centers[0] - radius, centers[0] + radius)
        ax.set_ylim(centers[1] - radius, centers[1] + radius)
        ax.set_zlim(centers[2] - radius, centers[2] + radius)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
