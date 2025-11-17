"""
Structured logging utilities for the world model trainer.

The ``WorldModelLogger`` centralises local logging, WandB integration, and
auxiliary visualisations so that the trainer can remain focused on the
optimization loop.
"""

import logging
from typing import Any, Dict, Iterable, Optional, TYPE_CHECKING, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from world_model.flow_matching import EulerSolverConfig

if TYPE_CHECKING:
    from training.world_trainer import LoggingConfig
    import wandb
    from evaluation.world_model_evaluator import EvaluationSummary
    

def _create_local_logger() -> logging.Logger:
    logger = logging.getLogger("world_model_trainer")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("\x1b[90m%(asctime)s\x1b[0m | %(levelname)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class WorldModelLogger:
    def __init__(
        self,
        logging_cfg: "LoggingConfig",
        euler_cfg: EulerSolverConfig,
        sample_fps: Optional[float] = None,
        is_main_process: bool = True,
    ) -> None:
        self.cfg = logging_cfg
        self.local = _create_local_logger()
        self.is_main_process = is_main_process
        self.sample_interval = logging_cfg.sample_interval
        self.noise_log_limit = logging_cfg.tau_log_limit if logging_cfg.tau_log_limit > 0 else None
        self.noise_log_count = 0
        self.current_step = 0
        self.current_micro_step = 0
        self.wandb_run: Optional["wandb.sdk.wandb_run.Run"] = None
        self._wandb = None
        if not (euler_cfg.min_signal == 0.0 and euler_cfg.max_signal == 1.0):
            raise ValueError("Euler solver bounds must be [0.0, 1.0].")
        self._sample_fps = sample_fps
        self._wandb_eval_defined = False

    # ------------------------------------------------------------------ helpers
    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        if self.is_main_process:
            self.local.info(message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        if self.is_main_process:
            self.local.warning(message, *args, **kwargs)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        if self.is_main_process:
            self.local.debug(message, *args, **kwargs)

    # ------------------------------------------------------------------ lifecycle
    def init_wandb(
        self,
        config: Dict[str, Any],
        *,
        run_id: Optional[str] = None,
        resume: Optional[str] = None,
    ) -> Optional["wandb.sdk.wandb_run.Run"]:
        if not self.is_main_process:
            return None
        if self.wandb_run is not None or self._wandb is not None:
            return self.wandb_run
        try:
            import wandb  # type: ignore[attr-defined]
        except ImportError:
            self.info("Weights & Biases not available; skipping logging.")
        self._wandb = wandb
        self.wandb_run = wandb.init(
            project=self.cfg.project,
            entity=self.cfg.entity,
            name=self.cfg.run_name,
            config=config,
            id=run_id,
            resume=resume,
        )
        return self.wandb_run

    def close(self) -> None:
        if self.is_main_process and self.wandb_run:
            self.wandb_run.finish()
        self.wandb_run = None
        self._wandb = None

    # ------------------------------------------------------------------ state
    def start_step(self, step: int) -> None:
        self.current_step = step
        self.current_micro_step = 0

    def start_micro_step(self, micro_step: int) -> None:
        self.current_micro_step = micro_step

    # ------------------------------------------------------------------ logging hooks
    def log_training_metrics(self, metrics: Dict[str, float]) -> None:
        if not self.is_main_process:
            return
        if self.cfg.log_interval and self.current_step % self.cfg.log_interval == 0:
            message = " ".join(f"{key}={value:.5f}" for key, value in metrics.items())
            self.info("step=%d %s", self.current_step, message)
            if self.wandb_run is not None:
                self.wandb_run.log(
                    {f"train/{key}": value for key, value in metrics.items()},
                    step=self.current_step,
                )

    def process_gradients(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scaler: Optional[torch.amp.GradScaler] = None,
        clip_norm: Optional[float] = None,
    ) -> Dict[str, float]:
        if scaler is not None:
            scaler.unscale_(optimizer)

        params_with_grad = [param for param in model.parameters() if param.grad is not None]
        if not params_with_grad:
            return {}

        if clip_norm is not None:
            total_norm = torch.nn.utils.clip_grad_norm_(params_with_grad, clip_norm)
        else:
            total_norm = self._compute_grad_norm(params_with_grad)
            if total_norm is None:
                return {}

        if isinstance(total_norm, torch.Tensor):
            grad_norm = float(total_norm.detach().cpu().item())
        else:
            grad_norm = float(total_norm)
        return {"grad_norm": grad_norm}

    def log_distr_noise(
        self,
        tau: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        bins: int = 50,
    ) -> None:
        if not self.is_main_process:
            return
        if self.noise_log_limit is not None and self.noise_log_count >= self.noise_log_limit:
            return

        values = tau.detach().to(dtype=torch.float32).flatten()
        if values.numel() == 0:
            return

        weight_values: Optional[torch.Tensor] = None
        if weights is not None:
            if weights.shape != tau.shape:
                raise ValueError("Weights tensor must match tau shape for logging.")
            weight_values = weights.detach().to(dtype=torch.float32).flatten()

        if self.wandb_run is not None and self._wandb is not None:
            logs: Dict[str, Any] = {}
            edges = torch.linspace(0.0, 1.0, bins + 1, device=values.device)
            centers = 0.5 * (edges[:-1] + edges[1:])
            if weight_values is not None:
                indices = torch.bucketize(values, edges, right=True) - 1
                indices = indices.clamp(min=0, max=bins - 1)
                weight_hist = torch.zeros(bins, device=values.device, dtype=torch.float32)
                weight_hist.scatter_add_(0, indices, weight_values)
                histogram = torch.stack([centers, weight_hist], dim=1)
                logs["noise/tau_weight_histogram"] = self._wandb.Table(
                    data=histogram.detach().cpu().tolist(),
                    columns=["tau", "weight_sum"],
                )
                self.debug(
                    "Logged tau-weight histogram for %d samples (bins=%d).",
                    int(weight_values.numel()),
                    bins,
                )
            else:
                indices = torch.bucketize(values, edges, right=True) - 1
                indices = indices.clamp(min=0, max=bins - 1)
                counts = torch.zeros(bins, device=values.device, dtype=torch.float32)
                ones = torch.ones_like(values)
                counts.scatter_add_(0, indices, ones)
                histogram = torch.stack([centers, counts], dim=1)
                logs["noise/tau_histogram"] = self._wandb.Table(
                    data=histogram.detach().cpu().tolist(),
                    columns=["tau", "count"],
                )
                self.debug(
                    "Logged tau histogram for %d samples (bins=%d).",
                    int(values.numel()),
                    bins,
                )
            if logs:
                self.wandb_run.log(logs, step=self.current_step, commit=False)

        self.noise_log_count += 1

    # ------------------------------------------------------------------ visual logging
    def log_evaluation(self, result: "EvaluationSummary") -> None:
        if not self.is_main_process or result is None:
            return
        if result.metrics:
            summary_parts = [
                f"{key}={value:.5f}"
                for key, value in result.metrics.items()
                if key.endswith("/mean") or key.endswith("/var")
            ]
            if summary_parts:
                self.info("step=%d eval %s", self.current_step, " ".join(summary_parts))
        if self.wandb_run is not None and result.metrics:
            if not self._wandb_eval_defined:
                try:
                    self.wandb_run.define_metric("eval_step")
                    self.wandb_run.define_metric("eval/*", step_metric="eval_step")
                except Exception:
                    pass
                self._wandb_eval_defined = True
            payload: Dict[str, Any] = {"eval_step": self.current_step}
            payload.update(result.metrics)
            plot_images = self._build_rollout_plot_images(result.metrics)
            if plot_images:
                payload.update(plot_images)
            self.wandb_run.log(payload, step=self.current_step)
            video_payload = self._build_video_table(result.videos)
            if video_payload:
                self.wandb_run.log(video_payload, step=self.current_step)
        if self.wandb_run is None or not self._wandb or not result.videos:
            return

    def _video_from_frames(self, frames: torch.Tensor) -> np.ndarray:
        if frames.ndim != 4:
            raise ValueError("Frames must have shape [T, C, H, W].")
        array = (
            frames.detach()
            .to(dtype=torch.float32)
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(dtype=torch.uint8)
            .cpu()
            .numpy()
        )
        return array

    def _resolve_fps(self) -> int:
        if self._sample_fps is None:
            return 10
        return max(1, int(round(self._sample_fps)))

    def _build_rollout_plot_images(self, metrics: Dict[str, float]) -> Dict[str, "wandb.Image"]:
        if self._wandb is None:
            return {}
        plots: Dict[str, "wandb.Image"] = {}
        entries = (
            ("mean", "Mean absolute error"),
            ("var", "Variance of absolute error"),
        )
        for metric_type, ylabel in entries:
            figure = self._build_rollout_plot(metrics, metric_type, ylabel)
            if figure is None:
                continue
            image = self._wandb.Image(figure)
            plt.close(figure)
            plots[f"eval/rollout_{metric_type}_plot"] = image
        return plots

    @staticmethod
    def _collect_rollout_series(metrics: Dict[str, float], metric_type: str) -> Dict[str, List[Tuple[int, float]]]:
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

    def _build_rollout_plot(self, metrics: Dict[str, float], metric_type: str, ylabel: str):
        series = self._collect_rollout_series(metrics, metric_type)
        if not series:
            return None
        fig, ax = plt.subplots(figsize=(6, 4))
        for scenario, values in series.items():
            steps = [step for step, _ in values]
            vals = [val for _, val in values]
            ax.plot(steps, vals, marker="o", label=scenario)
        ax.set_xlabel("Rollout step (frames after conditioning)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Rollout {metric_type} vs. horizon")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        return fig

    def _build_video_table(self, videos: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        if self._wandb is None or not videos:
            return {}
        table = self._wandb.Table(columns=["name", "video"])
        fps_value = self._resolve_fps()
        for key, frames in videos.items():
            array = self._video_from_frames(frames)
            table.add_data(key, self._wandb.Video(array, fps=fps_value, format="mp4"))
        return {"eval/videos": table}

    @staticmethod
    def _compute_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> Optional[float]:
        norm_sq: Optional[torch.Tensor] = None
        for param in parameters:
            grad = param.grad
            if grad is None:
                continue
            value = grad.detach().float().pow(2).sum()
            norm_sq = value if norm_sq is None else norm_sq + value
        if norm_sq is None:
            return None
        return float(norm_sq.sqrt().item())
