"""
Structured logging utilities for the world model trainer.

The ``WorldModelLogger`` centralises local logging, WandB integration, and
auxiliary visualisations so that the trainer can remain focused on the
optimization loop.
"""

import logging
from typing import Any, Callable, Dict, Iterable, Optional, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from world_model.flow_matching import EulerSolver, EulerSolverConfig

if TYPE_CHECKING:
    from training.world_trainer import LoggingConfig
    import wandb
    

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
        decode_fn: Callable[[torch.Tensor], torch.Tensor],
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
        self._solver = EulerSolver(euler_cfg)
        self._decode_latents = decode_fn
        self._sample_fps = sample_fps
        self._sample_logged_this_step = False

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
    def init_wandb(self, config: Dict[str, Any]) -> Optional["wandb.sdk.wandb_run.Run"]:
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
        self._sample_logged_this_step = False

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
    def maybe_log_micro_step_video(
        self,
        model: nn.Module,
        *,
        latents: torch.Tensor,
        noisy_latents: torch.Tensor,
        noise_levels: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        actions_mask: Optional[torch.Tensor] = None,
        independant_frames_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if not self.is_main_process:
            return
        if (
            self.sample_interval is None
            or self.wandb_run is None
            or self._wandb is None
            or self.current_step % self.sample_interval != 0
            or self._sample_logged_this_step
        ):
            return

        batch_size = latents.shape[0]
        candidate_idx: Optional[int] = None
        for idx in range(batch_size):
            independent = False
            if independant_frames_mask is not None:
                mask = independant_frames_mask[idx]
                independent = bool(mask.detach().bool().item())
            if independent:
                continue
            has_actions = True
            if actions_mask is not None:
                mask_row = actions_mask[idx]
                has_actions = bool(torch.all(mask_row.detach().bool()).item())
            if not has_actions:
                continue
            candidate_idx = idx
            break

        if candidate_idx is None:
            return

        seq_latents = latents[candidate_idx : candidate_idx + 1].detach()
        seq_noisy_latents = noisy_latents[candidate_idx : candidate_idx + 1].detach()
        seq_noise_levels = noise_levels[candidate_idx : candidate_idx + 1].detach()

        seq_actions = actions[candidate_idx : candidate_idx + 1].detach() if actions is not None else None
        seq_action_mask = (
            actions_mask[candidate_idx : candidate_idx + 1].detach() if actions_mask is not None else None
        )
        seq_indep = (
            independant_frames_mask[candidate_idx : candidate_idx + 1].detach()
            if independant_frames_mask is not None
            else None
        )

        model_kwargs: Dict[str, torch.Tensor] = {}
        if seq_actions is not None:
            model_kwargs["actions"] = seq_actions
        if seq_action_mask is not None:
            model_kwargs["actions_mask"] = seq_action_mask
        if seq_indep is not None:
            model_kwargs["independant_frames_mask"] = seq_indep

        was_training = model.training
        if was_training:
            model.eval()

        with torch.no_grad():
            denoised_latents = self._solver.sample(
                model,
                seq_noisy_latents,
                initial_signal=seq_noise_levels,
                **model_kwargs,
            )

        if was_training:
            model.train()

        noisy_decoded = self._decode_sequence(seq_noisy_latents.squeeze(0))
        clean_decoded = self._decode_sequence(seq_latents.squeeze(0))
        denoised_decoded = self._decode_sequence(denoised_latents.squeeze(0))

        noisy_clean_mse = float(F.mse_loss(seq_noisy_latents, seq_latents).item())
        denoised_clean_mse = float(F.mse_loss(denoised_latents, seq_latents).item())
        signal_level = float(seq_noise_levels.mean().item())

        videos = {
            "samples/clean_reconstruction": self._video_from_frames(clean_decoded),
            "samples/noisy_reconstruction": self._video_from_frames(noisy_decoded),
            "samples/denoised_reconstruction": self._video_from_frames(denoised_decoded),
        }
        fps_value = self._resolve_fps()
        wandb_videos = {
            key: self._wandb.Video(value, fps=fps_value, format="mp4") for key, value in videos.items()
        }
        if self.wandb_run is not None:
            scalar_logs = {
                "samples/selected_signal_level": signal_level,
                "samples/noisy_clean_mse": noisy_clean_mse,
                "samples/denoised_clean_mse": denoised_clean_mse,
            }
            self.wandb_run.log(scalar_logs, step=self.current_step, commit=False)
            self.wandb_run.log(wandb_videos, step=self.current_step, commit=False)
        self.info(
            "step=%d candidate_idx=%d signal=%.4f mse(noisy,clean)=%.6f mse(denoised,clean)=%.6f",
            self.current_step,
            candidate_idx,
            signal_level,
            noisy_clean_mse,
            denoised_clean_mse,
        )
        self._sample_logged_this_step = True
    def _decode_sequence(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 3:
            raise ValueError("Expected latents with shape [T, tokens, dim] for decoding.")
        with torch.no_grad():
            decoded = self._decode_latents(latents.contiguous())
        if decoded.ndim != 4:
            raise ValueError("Decoder returned tensor with unexpected shape.")
        return decoded

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
