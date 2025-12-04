import time
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, TYPE_CHECKING, List

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from torch.optim import Optimizer
import yaml

if TYPE_CHECKING:
    import wandb
    from src.training.world_evaluator import EvaluationSummary
    
@dataclass
class LoggingConfig:
    project: str = "world_model"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    log_interval: int = 10
    checkpoint_interval: int = 1_000
    output_dir: str = "checkpoints"
    signal_level_log_limit: int = 200


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

# This is dirty code, need to clean this TODO
class WorldModelLogger:
    def __init__(
        self,
        logging_cfg: "LoggingConfig",
        is_main_process: bool = True,
    ) -> None:
        self.cfg = logging_cfg
        self.local = _create_local_logger()
        self.is_main_process = is_main_process
        self.signal_log_limit = logging_cfg.signal_level_log_limit if logging_cfg.signal_level_log_limit > 0 else None
        self.signal_log_count = 0
        self.current_step = 0
        self.current_micro_step = 0
        self.wandb_run: Optional["wandb.sdk.wandb_run.Run"] = None
        self._wandb = None
        self._wandb_eval_defined = False
        self._signal_values: List[torch.Tensor] = []
        self._signal_bins: Optional[int] = None
        self._signal_hist_logged = False
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
            if self.signal_log_limit is not None:
                self._flush_signal_histograms(force=True)
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

    def log_distr_signal(
        self,
        signal_levels: torch.Tensor,
        bins: int = 25,
    ) -> None:
        if self._signal_hist_logged:
            return

        # Gather signals from all processes if distributed training is active
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            if world_size > 1:
                signal_levels = signal_levels.contiguous()
                gathered_signals = [torch.zeros_like(signal_levels) for _ in range(world_size)]
                torch.distributed.all_gather(gathered_signals, signal_levels)
                signal_levels = torch.cat(gathered_signals)

        values = signal_levels.detach().to(dtype=torch.float32).flatten()
        if values.numel() == 0:
            return

        if self.signal_log_limit is None:
            if self.is_main_process:
                self._log_signal_histogram(
                    values=values,
                    bins=bins,
                )
            self.signal_log_count += 1
            return

        self._signal_values.append(values.cpu())
        self._signal_bins = bins
        self.signal_log_count += 1
        if self.signal_log_limit is not None and self.signal_log_count >= self.signal_log_limit:
            self._flush_signal_histograms()

    def _flush_signal_histograms(self, force: bool = False) -> None:
        if self._signal_hist_logged:
            return
        if not force and (self.signal_log_limit is None or self.signal_log_count < self.signal_log_limit):
            return
        
        # Mark as logged and clear buffers on all processes to prevent memory leaks and unnecessary gathering
        self._signal_hist_logged = True
        
        if not self._signal_values:
            return

        if not self.is_main_process:
            self._signal_values.clear()
            self._signal_bins = None
            return

        if self.wandb_run is None or self._wandb is None:
            self._signal_values.clear()
            self._signal_bins = None
            return

        values = torch.cat(self._signal_values, dim=0)
        bins = self._signal_bins or 25
        self._log_signal_histogram(values=values, bins=bins)
        
        self._signal_values.clear()
        self._signal_bins = None

    def _log_signal_histogram(
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
        table = self._wandb.Table(columns=["signal", "count"])
        flat = torch.stack([centers, counts], dim=1).detach().cpu().tolist()
        for signal_value, count in flat:
            truncated_signal = math.floor(float(signal_value) * 1000.0) / 1000.0
            table.add_data(truncated_signal, float(count))
        logs["debug/signal_histogram"] = self._wandb.plot.bar(
            table,
            "signal",
            "count",
            title="Signal histogram",
        )
        self.debug(
            "Logged signal histogram for %d samples (bins=%d).",
            sample_count,
            bins,
        )
        if logs:
            self.wandb_run.log(logs, step=self.current_step, commit=False)

    # ------------------------------------------------------------------ visual logging
    def log_evaluation(self, result: "EvaluationSummary", fps: int = 5) -> None:
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
            video_payload = self._build_video_payload(result.videos, fps=fps)
            if video_payload:
                self.wandb_run.log(video_payload, step=self.current_step)
        
        if self.wandb_run is not None and hasattr(result, "plots") and result.plots:
            plot_payload = {}
            for key, plot_data in result.plots.items():
                if self._wandb is None:
                    continue
                # plot_data has xs, ys, keys, title
                plot_payload[key] = self._wandb.plot.line_series(
                    xs=plot_data["xs"],
                    ys=plot_data["ys"],
                    keys=plot_data["keys"],
                    title=plot_data["title"],
                    xname=plot_data.get("xname", "diffusion_step")
                )
            if plot_payload:
                self.wandb_run.log(plot_payload, step=self.current_step)

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

    def _build_video_payload(self, videos: Dict[str, torch.Tensor], fps: int = 5) -> Dict[str, Any]:
        if self._wandb is None or not videos:
            return {}
        payload: Dict[str, Any] = {}
        for key, frames in videos.items():
            array = self._video_from_frames(frames)
            payload[key] = self._wandb.Video(array, fps=fps, format="mp4")
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
