"""
Structured logging utilities for the world model trainer.

The ``WorldModelLogger`` centralises local logging, WandB integration, and
auxiliary visualisations so that the trainer can remain focused on the
optimization loop.
"""

import logging
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from training.world_trainer import LoggingConfig


def _create_local_logger() -> logging.Logger:
    logger = logging.getLogger("world_model_trainer")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("\x1b[90m%(asctime)s | %(levelname)s | %(message)s\x1b[0m")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class WorldModelLogger:
    def __init__(
        self,
        logging_cfg: "LoggingConfig",
    ) -> None:
        self.cfg = logging_cfg
        self.local = _create_local_logger()
        self.sample_interval = logging_cfg.sample_interval
        self.noise_log_limit = logging_cfg.tau_log_limit if logging_cfg.tau_log_limit > 0 else None
        self.noise_log_count = 0
        self.current_step = 0
        self.current_micro_step = 0
        self._tau_running_mean = 0.0
        self._tau_running_std = 0.0
        self._tau_stat_count = 0
        self.wandb_run: Optional["wandb.sdk.wandb_run.Run"] = None
        self._wandb = None

    # ------------------------------------------------------------------ helpers
    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.local.info(message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.local.warning(message, *args, **kwargs)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.local.debug(message, *args, **kwargs)

    # ------------------------------------------------------------------ lifecycle
    def init_wandb(self, config: Dict[str, Any]) -> Optional["wandb.sdk.wandb_run.Run"]:
        if self.wandb_run is not None or self._wandb is not None:
            return self.wandb_run
        try:
            import wandb  # type: ignore[attr-defined]
        except ImportError:
            self.info("Weights & Biases not available; skipping logging.")
            self._wandb = None
            self.wandb_run = None
            return None
        self._wandb = wandb
        self.wandb_run = wandb.init(
            project=self.cfg.project,
            entity=self.cfg.entity,
            name=self.cfg.run_name,
            config=config,
        )
        return self.wandb_run

    def close(self) -> None:
        if self.wandb_run:
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
    def log_dataloader(self, dataloader: DataLoader) -> None:
        try:
            steps = len(dataloader)
        except TypeError:
            steps = None

        if steps is None:
            self.info("Dataloader length unavailable (object does not implement __len__).")
        else:
            self.info("Dataloader contains %d steps.", steps)

        if self.wandb_run is not None:
            self.wandb_run.log(
                {"debug/dataloader_length": steps if steps is not None else "unknown"},
                step=0,
                commit=False,
            )

    def log_training_metrics(self, metrics: Dict[str, float]) -> None:
        if self.cfg.log_interval and self.current_step % self.cfg.log_interval == 0:
            message = " ".join(f"{key}={value:.5f}" for key, value in metrics.items())
            self.info("step=%d %s", self.current_step, message)
            if self.wandb_run is not None:
                self.wandb_run.log(
                    {f"train/{key}": value for key, value in metrics.items()},
                    step=self.current_step,
                )

    def log_distr_noise(self, tau: torch.Tensor, bins: int = 50) -> None:
        if self.noise_log_limit is not None and self.noise_log_count >= self.noise_log_limit:
            return

        values = tau.detach().to(dtype=torch.float32).flatten()
        if values.numel() == 0:
            return

        min_val = float(values.min().item())
        max_val = float(values.max().item())
        mean_val = float(values.mean().item())
        std_val = float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0
        hist = torch.histc(values, bins=bins, min=min_val, max=max_val) if values.numel() > 1 else values

        self._tau_stat_count += 1
        self._tau_running_mean += (mean_val - self._tau_running_mean) / self._tau_stat_count
        self._tau_running_std += (std_val - self._tau_running_std) / self._tau_stat_count

        self.info(
            "step=%d micro_step=%d | tau mean=%.4f std=%.4f min=%.4f max=%.4f bins=%d | running mean=%.4f std=%.4f",
            self.current_step,
            self.current_micro_step,
            mean_val,
            std_val,
            min_val,
            max_val,
            bins if values.numel() > 1 else 1,
            self._tau_running_mean,
            self._tau_running_std,
        )
        self.debug(
            "Tau histogram counts: %s",
            hist.tolist() if values.numel() > 1 else [float(values.item())],
        )

        if self.wandb_run is not None and self._wandb is not None:
            logs: Dict[str, Any] = {
                "noise/tau_mean": mean_val,
                "noise/tau_std": std_val,
                "noise/tau_running_mean": self._tau_running_mean,
                "noise/tau_running_std": self._tau_running_std,
            }
            if values.numel() > 1:
                logs["noise/tau_histogram"] = self._wandb.Histogram(values.cpu().numpy())
            self.wandb_run.log(logs, step=self.current_step, commit=False)

        self.noise_log_count += 1

    def log_sample_sequence(
        self,
        frame: torch.Tensor,
        noisy_latent: torch.Tensor,
        denoised_latent: torch.Tensor,
        decode_latent: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        if self.sample_interval is None or self.sample_interval <= 0:
            return
        if self.current_step % self.sample_interval != 0 or self.current_micro_step != 0:
            return

        frame = frame.detach().to(dtype=torch.float32)
        if frame.ndim != 3:
            self.warning(
                "log_sample_sequence expected frame with shape [C, H, W], got %s",
                tuple(frame.shape),
            )
            return

        if frame.max() > 1.5:
            frame = frame / 255.0
        frame = frame.clamp(0.0, 1.0)

        self.info(
            "Logging sample reconstruction at step=%d micro_step=%d.",
            self.current_step,
            self.current_micro_step,
        )

        if self.wandb_run is None or self._wandb is None:
            return

        decode_owner = getattr(decode_latent, "__self__", None)
        decode_device = noisy_latent.device
        if decode_owner is not None:
            try:
                decode_device = next(decode_owner.parameters()).device
            except (StopIteration, AttributeError):  # pragma: no cover - defensive
                decode_device = noisy_latent.device

        def _decode_single(latent: torch.Tensor) -> torch.Tensor:
            latent = latent.detach().unsqueeze(0).to(decode_device)
            with torch.no_grad():
                decoded = decode_latent(latent)
            if decoded.ndim != 4 or decoded.shape[0] != 1:
                raise ValueError(
                    "decode_latent expected to return tensor with shape [1, C, H, W], "
                    f"got {tuple(decoded.shape)}."
                )
            return decoded.squeeze(0).to(torch.float32)

        try:
            noisy_image = _decode_single(noisy_latent)
            denoised_image = _decode_single(denoised_latent)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.warning("Failed to decode latents for sample logging: %s", exc)
            return

        def _to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
            tensor = tensor.detach().clamp(0.0, 1.0)
            array = tensor.mul(255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            return array

        target_image = _to_uint8_image(frame)
        noisy_image_np = _to_uint8_image(noisy_image)
        denoised_image_np = _to_uint8_image(denoised_image)

        self.wandb_run.log(
            {
                "sample/target": self._wandb.Image(target_image, caption="target"),
                "sample/noisy": self._wandb.Image(noisy_image_np, caption="noisy"),
                "sample/denoised": self._wandb.Image(denoised_image_np, caption="denoised"),
            },
            step=self.current_step,
            commit=False,
        )
