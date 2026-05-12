from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

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


def _wandb_step_kwargs(step: Optional[int]) -> Dict[str, int]:
    return {} if step is None else {"step": step}


def log_wandb_video(
    logger: WorldModelLogger,
    key: str,
    frames: torch.Tensor,
    fps: int,
    step: Optional[int] = None,
) -> None:
    if logger.wandb_run is None or logger._wandb is None:
        return
    array = frames_to_uint8(frames)
    logger.wandb_run.log(
        {key: logger._wandb.Video(array, fps=fps, format="mp4")},
        **_wandb_step_kwargs(step),
    )


def log_wandb_video_file(
    logger: WorldModelLogger,
    key: str,
    path: Path,
    *,
    fmt: str,
    step: Optional[int] = None,
) -> None:
    if logger.wandb_run is None or logger._wandb is None:
        return
    try:
        media = logger._wandb.Video(str(path), format=fmt)
    except Exception:
        media = logger._wandb.Image(str(path))
    logger.wandb_run.log({key: media}, **_wandb_step_kwargs(step))


def log_wandb_image(
    logger: WorldModelLogger,
    key: str,
    image_path: Path,
    step: Optional[int] = None,
) -> None:
    if logger.wandb_run is None or logger._wandb is None:
        return
    logger.wandb_run.log(
        {key: logger._wandb.Image(str(image_path))},
        **_wandb_step_kwargs(step),
    )


def log_wandb_metrics(
    logger: WorldModelLogger,
    metrics: Dict[str, Any],
    step: Optional[int] = None,
) -> None:
    if logger.wandb_run is None or not metrics:
        return
    logger.wandb_run.log(metrics, **_wandb_step_kwargs(step))


def log_wandb_line_series(
    logger: WorldModelLogger,
    key: str,
    *,
    xs: Sequence[int],
    ys: Sequence[Sequence[float]],
    labels: Sequence[str],
    title: str,
    xname: str = "iteration",
    step: Optional[int] = None,
) -> None:
    if logger.wandb_run is None or logger._wandb is None:
        return
    logger.wandb_run.log(
        {
            key: logger._wandb.plot.line_series(
                xs=list(xs),
                ys=[list(values) for values in ys],
                keys=list(labels),
                title=title,
                xname=xname,
            )
        },
        **_wandb_step_kwargs(step),
    )


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


def save_trajectory_evolution_gif(
    *,
    snapshots: Sequence[Tuple[int, float, torch.Tensor]],
    reference_positions: Optional[torch.Tensor],
    path: Path,
    title: str,
    fps: int,
) -> Path:
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not snapshots:
        raise ValueError("Cannot save trajectory evolution GIF without snapshots.")

    path.parent.mkdir(parents=True, exist_ok=True)
    planned_arrays = [
        positions.detach().cpu().to(dtype=torch.float32).numpy()
        for _, _, positions in snapshots
    ]
    reference = None
    if reference_positions is not None:
        reference = reference_positions.detach().cpu().to(dtype=torch.float32).numpy()

    axis_points = planned_arrays if reference is None else [*planned_arrays, reference]
    all_points = np.concatenate([points for points in axis_points if points.size > 0], axis=0)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 1e-3)

    frames = []
    for idx, (iteration, best_cost, _) in enumerate(snapshots):
        fig = plt.figure(figsize=(8, 7), dpi=130)
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
            ax.scatter(reference[0, 0], reference[0, 1], reference[0, 2], color="#202020", s=35, marker="o")
            ax.scatter(reference[-1, 0], reference[-1, 1], reference[-1, 2], color="#202020", s=50, marker="x")

        for previous in planned_arrays[:idx]:
            if previous.shape[0] > 0:
                ax.plot(previous[:, 0], previous[:, 1], previous[:, 2], color="#8fb3ff", alpha=0.20, linewidth=1.5)

        current = planned_arrays[idx]
        if current.shape[0] > 0:
            ax.plot(
                current[:, 0],
                current[:, 1],
                current[:, 2],
                color="#2f6fed",
                linewidth=3.0,
                label="best planned trajectory",
            )
            ax.scatter(current[0, 0], current[0, 1], current[0, 2], color="#2f6fed", s=45, marker="o")
            ax.scatter(current[-1, 0], current[-1, 1], current[-1, 2], color="#2f6fed", s=60, marker="x")

        ax.set_xlim(centers[0] - radius, centers[0] + radius)
        ax.set_ylim(centers[1] - radius, centers[1] + radius)
        ax.set_zlim(centers[2] - radius, centers[2] + radius)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"{title}\niteration {iteration} | best cost {best_cost:.5f}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)

    duration_ms = max(1, int(round(1000.0 / max(1, fps))))
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return path
