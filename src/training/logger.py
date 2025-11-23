"""
Structured logging utilities for the world model trainer.

The ``WorldModelLogger`` centralises local logging, WandB integration, and
auxiliary visualisations so that the trainer can remain focused on the
optimization loop.
"""

import time
import logging
import math
from typing import Any, Dict, Iterable, Optional, TYPE_CHECKING, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
import yaml

from src.world_model.diffusion import EulerSolverConfig

if TYPE_CHECKING:
    from src.training.world_trainer import LoggingConfig
    import wandb
    from src.training.world_evaluator import EvaluationSummary
    

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
        self._tau_values: List[torch.Tensor] = []
        self._tau_bins: Optional[int] = None
        self._tau_hist_logged = False
        self._last_step_time: Optional[float] = None
        self._step_duration: Optional[float] = None

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

    def log_config(self, config: Dict[str, Any]) -> None:
        if self.is_main_process:
            self.info("Configuration:\n%s", yaml.dump(config, sort_keys=False))

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
        if self.is_main_process:
            if self.noise_log_limit is not None:
                self._flush_tau_histograms(force=True)
            if self.wandb_run:
                self.wandb_run.finish()
        self.wandb_run = None
        self._wandb = None

    # ------------------------------------------------------------------ state
    def start_step(self, step: int) -> None:
        self.current_step = step
        self.current_micro_step = 0
        now = time.time()
        if self._last_step_time is not None:
            self._step_duration = now - self._last_step_time
        self._last_step_time = now

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
                payload: Dict[str, float] = {}
                for key, value in metrics.items():
                    if key == "learning_rate" or key.startswith("grad_norm"):
                        payload[f"debug/{key}"] = value
                    else:
                        payload[f"train/{key}"] = value
                if self._step_duration is not None:
                    payload["debug/train_seconds_per_step"] = self._step_duration
                if payload:
                    self.wandb_run.log(payload, step=self.current_step)

    def log_grad_norm(
        self,
        *,
        model: nn.Module,
        key: str = "grad_norm",
    ) -> Dict[str, float]:
        params_with_grad = [param for param in model.parameters() if param.grad is not None]
        if not params_with_grad:
            return {}

        total_norm = self._compute_grad_norm(params_with_grad)
        if total_norm is None:
            return {}

        if isinstance(total_norm, torch.Tensor):
            grad_norm = float(total_norm.detach().cpu().item())
        else:
            grad_norm = float(total_norm)
        return {key: grad_norm}

    def log_distr_noise(
        self,
        tau: torch.Tensor,
        bins: int = 25,
    ) -> None:
        if not self.is_main_process:
            return

        values = tau.detach().to(dtype=torch.float32).flatten()
        if values.numel() == 0:
            return

        if self.wandb_run is None or self._wandb is None:
            return

        if self.noise_log_limit is None:
            self._log_tau_histogram(
                values=values,
                bins=bins,
            )
            self.noise_log_count += 1
            return

        if self._tau_hist_logged:
            return

        self._tau_values.append(values.cpu())
        self._tau_bins = bins
        self.noise_log_count += 1
        if self.noise_log_limit is not None and self.noise_log_count >= self.noise_log_limit:
            self._flush_tau_histograms()

    def _flush_tau_histograms(self, force: bool = False) -> None:
        if self._tau_hist_logged:
            return
        if not force and (self.noise_log_limit is None or self.noise_log_count < self.noise_log_limit):
            return
        if not self._tau_values:
            return
        if self.wandb_run is None or self._wandb is None:
            return

        values = torch.cat(self._tau_values, dim=0)
        bins = self._tau_bins or 25
        self._log_tau_histogram(values=values, bins=bins)
        self._tau_hist_logged = True
        self._tau_values.clear()
        self._tau_bins = None

    def _log_tau_histogram(
        self,
        *,
        values: torch.Tensor,
        bins: int,
    ) -> None:
        if self.wandb_run is None or self._wandb is None:
            return
        logs: Dict[str, Any] = {}
        device = values.device
        edges = torch.linspace(0.0, 1.0, bins + 1, device=device)
        centers = 0.5 * (edges[:-1] + edges[1:])
        sample_count = int(values.numel())
        indices = torch.bucketize(values, edges, right=True) - 1
        indices = indices.clamp(min=0, max=bins - 1)
        counts = torch.zeros(bins, device=device, dtype=torch.float32)
        ones = torch.ones_like(values)
        counts.scatter_add_(0, indices, ones)
        table = self._wandb.Table(columns=["tau", "count"])
        flat = torch.stack([centers, counts], dim=1).detach().cpu().tolist()
        for tau_value, count in flat:
            truncated_tau = math.floor(float(tau_value) * 1000.0) / 1000.0
            table.add_data(truncated_tau, float(count))
        logs["debug/tau_histogram"] = self._wandb.plot.bar(
            table,
            "tau",
            "count",
            title="Tau histogram",
        )
        self.debug(
            "Logged tau histogram for %d samples (bins=%d).",
            sample_count,
            bins,
        )
        if logs:
            self.wandb_run.log(logs, step=self.current_step, commit=False)

    # ------------------------------------------------------------------ visual logging
    def log_evaluation(self, result: "EvaluationSummary") -> None:
        if not self.is_main_process or result is None:
            return
        if result.metrics:
            summary_keys = sorted(
                key
                for key in result.metrics.keys()
                if key.startswith("evaluation/l1_loss/")
                or key.startswith("evaluation/l2_loss/")
                or key.startswith("evaluation/var/")
            )
            summary_parts = [f"{key}={result.metrics[key]:.5f}" for key in summary_keys]
            if summary_parts:
                self.info("step=%d eval %s", self.current_step, " ".join(summary_parts))
        if self.wandb_run is not None and result.metrics:
            if not self._wandb_eval_defined:
                try:
                    self.wandb_run.define_metric("evaluation_step")
                    self.wandb_run.define_metric("evaluation/*", step_metric="evaluation_step")
                except Exception:
                    pass
                self._wandb_eval_defined = True
            payload: Dict[str, Any] = {"evaluation_step": self.current_step}
            payload.update(result.metrics)
            self.wandb_run.log(payload, step=self.current_step)
        if self.wandb_run is not None and result.videos:
            video_payload = self._build_video_payload(result.videos)
            if video_payload:
                self.wandb_run.log(video_payload, step=self.current_step)

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

    def _build_video_payload(self, videos: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        if self._wandb is None or not videos:
            return {}
        payload: Dict[str, Any] = {}
        fps_value = self._resolve_fps()
        for key, frames in videos.items():
            array = self._video_from_frames(frames)
            payload[key] = self._wandb.Video(array, fps=fps_value, format="mp4")
        return payload

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
