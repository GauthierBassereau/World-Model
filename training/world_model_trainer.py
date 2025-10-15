"""
Minimal training loop for the Dreamer-style world model.

The trainer wires together the DINO tokenizer, LeRobot dataloaders, optimizer,
and logging utilities (including Weights & Biases).

Config keys consumed here:
    trainer.max_steps
    trainer.grad_accum_steps
    trainer.precision
    trainer.seed
    trainer.device
    trainer.single_batch_overfit
    optimizer.lr / betas / weight_decay / eps / grad_clip_norm
    logging.project / entity / run_name / log_interval / checkpoint_interval / output_dir
    diffusion.min_signal / max_signal / base_dimension / noise_mean / noise_std
    diffusion.debug.enabled / output_dir / histogram_bins / max_updates
    ema.enabled / decay / device
"""

import copy
import logging
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from training.datasets import (
    DataloaderConfig,
    DatasetConfig,
    WorldModelBatch,
    build_world_model_dataloader,
)
from training.diffusion import (
    DiffusionConfig,
    DiffusionDebugConfig,
    DimensionShiftedUniformScheduler,
    sample_base_noise,
)
from vision.dino_v3 import DinoV3Embedder, DinoVisionConfig
from world_model.transformer import TransformerDebugConfig, WorldModelConfig


@dataclass
class OptimizerConfig:
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    eps: float = 1e-8
    grad_clip_norm: Optional[float] = 1.0


@dataclass
class TrainerLoopConfig:
    max_steps: int = 10_000
    grad_accum_steps: int = 1
    precision: str = "bf16"
    seed: int = 1234
    device: Optional[str] = None
    resume_checkpoint: Optional[str] = None
    single_batch_overfit: bool = False


@dataclass
class LoggingConfig:
    project: str = "dinowm"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    log_interval: int = 10
    checkpoint_interval: int = 1_000
    output_dir: str = "checkpoints"


@dataclass
class EMAConfig:
    enabled: bool = False
    decay: float = 0.999
    device: Optional[str] = None

    def validate(self) -> None:
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("ema.decay must satisfy 0.0 <= decay < 1.0.")


@dataclass
class WorldModelTrainingConfig:
    dataset: DatasetConfig = DatasetConfig()
    dataloader: DataloaderConfig = DataloaderConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    trainer: TrainerLoopConfig = TrainerLoopConfig()
    logging: LoggingConfig = LoggingConfig()
    vision: DinoVisionConfig = DinoVisionConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    world_model: WorldModelConfig = WorldModelConfig()
    ema: EMAConfig = EMAConfig()


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("world_model_trainer")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def load_training_config(path: str | Path) -> WorldModelTrainingConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw: Dict[str, Any] = yaml.safe_load(handle)

    diffusion_kwargs = dict(raw.get("diffusion", {}))
    if "tau_min" in diffusion_kwargs and "min_signal" not in diffusion_kwargs:
        diffusion_kwargs["min_signal"] = diffusion_kwargs.pop("tau_min")
    if "tau_max" in diffusion_kwargs and "max_signal" not in diffusion_kwargs:
        diffusion_kwargs["max_signal"] = diffusion_kwargs.pop("tau_max")
    if "tau_distribution" in diffusion_kwargs:
        distribution = diffusion_kwargs.pop("tau_distribution")
        if distribution != "uniform":
            raise ValueError("Only uniform diffusion schedules are supported.")
    diffusion_kwargs.pop("tau_power", None)
    diffusion_kwargs.pop("ramp_slope", None)
    diffusion_kwargs.pop("ramp_intercept", None)
    if isinstance(diffusion_kwargs.get("debug"), dict):
        debug_kwargs = dict(diffusion_kwargs["debug"])
        if "histogram_bins" in debug_kwargs and "num_hist_bins" not in debug_kwargs:
            debug_kwargs["num_hist_bins"] = debug_kwargs.pop("histogram_bins")
        debug_kwargs.pop("max_updates", None)
        diffusion_kwargs["debug"] = DiffusionDebugConfig(**debug_kwargs)

    if "debug" in raw:
        logging.getLogger("world_model_trainer").warning(
            "Top-level 'debug' config is deprecated; use diffusion.debug instead."
        )

    world_model_kwargs = dict(raw.get("world_model", {}))
    debug_section_raw = world_model_kwargs.get("debug", {})
    if isinstance(debug_section_raw, TransformerDebugConfig):
        debug_section = asdict(debug_section_raw)
    elif isinstance(debug_section_raw, dict):
        debug_section = dict(debug_section_raw)
    else:
        debug_section = {}
    if "num_tokens" in world_model_kwargs:
        debug_section.setdefault("num_tokens", world_model_kwargs.pop("num_tokens"))
    world_model_kwargs["debug"] = TransformerDebugConfig(**debug_section)

    logging_kwargs = dict(raw.get("logging", {}))
    logging_kwargs.pop("use_wandb", None)

    cfg = WorldModelTrainingConfig(
        dataset=DatasetConfig(**raw.get("dataset", {})),
        dataloader=DataloaderConfig(**raw.get("dataloader", {})),
        optimizer=OptimizerConfig(**raw.get("optimizer", {})),
        trainer=TrainerLoopConfig(**raw.get("trainer", {})),
        logging=LoggingConfig(**logging_kwargs),
        vision=DinoVisionConfig(**raw.get("vision", {})),
        diffusion=DiffusionConfig(**diffusion_kwargs),
        world_model=WorldModelConfig(**world_model_kwargs),
        ema=EMAConfig(**raw.get("ema", {})),
    )
    cfg.diffusion.validate()
    cfg.ema.validate()
    cfg.world_model.debug.validate()
    return cfg


def _dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {key: _dataclass_to_dict(value) for key, value in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(item) for item in obj]
    return obj


class WorldModelTrainer:
    def __init__(
        self,
        config: WorldModelTrainingConfig,
        model: nn.Module,
        embedder: Optional[DinoV3Embedder] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.embedder = embedder or DinoV3Embedder(config.vision)
        self.logger = _setup_logger()
        self.device = self._resolve_device()
        self.flow_cfg = config.diffusion
        self.scheduler = DimensionShiftedUniformScheduler(self.flow_cfg)
        self.ema_cfg = config.ema
        self.ema_model: Optional[nn.Module] = None
        self._ema_param_pairs: List[Tuple[torch.nn.Parameter, torch.nn.Parameter]] = []
        self._ema_buffer_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._ema_checkpoint_state: Optional[Dict[str, torch.Tensor]] = None
        self._cached_overfit_batch: Optional[WorldModelBatch] = None

        self._seed_everything(config.trainer.seed)
        self.model.to(self.device)
        self.embedder.to(self.device)

        self.dataloader = build_world_model_dataloader(
            dataset_cfg=config.dataset,
            dataloader_cfg=config.dataloader,
            embedder=self.embedder,
            device=self.device,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.optimizer.lr,
            betas=config.optimizer.betas,
            weight_decay=config.optimizer.weight_decay,
            eps=config.optimizer.eps,
        )
        self.use_scaler = config.trainer.precision == "fp16" and self.device.type == "cuda"
        self.use_autocast = config.trainer.precision in {"bf16", "fp16"} and self.device.type == "cuda"
        self.autocast_dtype = (
            torch.bfloat16 if config.trainer.precision == "bf16" else torch.float16
        )
        if self.use_scaler:
            self.scaler = torch.amp.GradScaler(device_type="cuda", enabled=True)
        else:
            self.scaler = None
        self.global_step = 0
        self._maybe_load_checkpoint()
        if self.ema_cfg.enabled:
            self._init_ema_model()
        self.wandb_run = self._init_wandb()
        if self.config.trainer.single_batch_overfit:
            self.logger.info("Single-batch overfit enabled; caching the first batch for repeated updates.")

    def _resolve_device(self) -> torch.device:
        if self.config.trainer.device:
            return torch.device(self.config.trainer.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _seed_everything(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _init_ema_model(self) -> None:
        ema_device = torch.device(self.ema_cfg.device) if self.ema_cfg.device else self.device
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.to(ema_device)
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
        self._ema_param_pairs = [
            (ema_param, param)
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters())
        ]
        self._ema_buffer_pairs = [
            (ema_buffer, buffer)
            for ema_buffer, buffer in zip(self.ema_model.buffers(), self.model.buffers())
        ]
        if self._ema_checkpoint_state is not None:
            self.ema_model.load_state_dict(self._ema_checkpoint_state)
            self._ema_checkpoint_state = None
            origin = "loaded from checkpoint"
        else:
            origin = "initialized from current model weights"
        self.logger.info(
            "EMA enabled with decay %.6f on %s (%s)",
            self.ema_cfg.decay,
            ema_device,
            origin,
        )

    def _update_ema(self) -> None:
        if self.ema_model is None:
            return
        decay = self.ema_cfg.decay
        if decay >= 1.0:
            return
        with torch.no_grad():
            for ema_param, param in self._ema_param_pairs:
                source = param.data
                if source.device != ema_param.device:
                    source = source.to(ema_param.device)
                ema_param.data.mul_(decay).add_(source, alpha=1.0 - decay)
            for ema_buffer, buffer in self._ema_buffer_pairs:
                source = buffer.data
                if source.device != ema_buffer.device:
                    source = source.to(ema_buffer.device)
                ema_buffer.data.copy_(source)

    def _init_wandb(self):
        import wandb

        wandb_run = wandb.init(
            project=self.config.logging.project,
            entity=self.config.logging.entity,
            name=self.config.logging.run_name,
            config=_dataclass_to_dict(self.config),
        )
        return wandb_run

    def train(self) -> None:
        if self.global_step >= self.config.trainer.max_steps:
            self.logger.info(
                "Checkpoint already at or beyond max_steps (%d >= %d); nothing to do.",
                self.global_step,
                self.config.trainer.max_steps,
            )
            self.close()
            return

        self.logger.info(
            "Starting world model training for %d steps on %s (resuming at step %d)",
            self.config.trainer.max_steps,
            self.device,
            self.global_step,
        )
        data_iter = iter(self.dataloader)

        for step in range(self.global_step + 1, self.config.trainer.max_steps + 1):
            self.optimizer.zero_grad(set_to_none=True)
            accum_metrics: Dict[str, float] = {}
            for _ in range(self.config.trainer.grad_accum_steps):
                batch_start_time = time.perf_counter()
                batch, data_iter = self._next_batch(data_iter)
                metrics = self._train_micro_step(batch)
                metrics["batch_time_sec"] = time.perf_counter() - batch_start_time
                for key, value in metrics.items():
                    accum_metrics[key] = accum_metrics.get(key, 0.0) + value
            if self.config.optimizer.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optimizer.grad_clip_norm)

            if self.use_scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self._update_ema()

            self.global_step = step
            mean_metrics = {
                key: value / self.config.trainer.grad_accum_steps for key, value in accum_metrics.items()
            }
            self._log_step(mean_metrics)

            if (
                self.config.logging.checkpoint_interval
                and step % self.config.logging.checkpoint_interval == 0
            ):
                self._save_checkpoint(step)

        self.logger.info("Finished training")

    def _train_micro_step(self, batch: WorldModelBatch) -> Dict[str, float]:
        sequence_latents = batch.sequence_latents.to(self.device)
        sequence_actions = batch.sequence_actions.to(self.device)

        actions_tensor: Optional[torch.Tensor]
        action_mask: Optional[torch.Tensor] = None
        if getattr(self.model, "action_proj", None) is not None:
            actions_tensor = sequence_actions
            if batch.actions_dropped_mask is not None:
                mask = (~batch.actions_dropped_mask).to(sequence_latents.device)
                if mask.ndim == 1:
                    mask = mask.unsqueeze(1).expand(-1, sequence_actions.shape[1])
                elif mask.shape != sequence_actions.shape[:2]:
                    raise ValueError(
                        "actions_dropped_mask must broadcast to [B, T] when actions are provided."
                    )
                action_mask = mask
        else:
            actions_tensor = None

        tau = self.scheduler.sample(sequence_latents)
        base_noise = sample_base_noise(sequence_latents, self.flow_cfg)
        tau_factor = tau.unsqueeze(-1).unsqueeze(-1)
        noisy_latents = (1.0 - tau_factor) * base_noise + tau_factor * sequence_latents
        target_velocity = sequence_latents - base_noise

        if self.use_autocast:
            autocast_ctx = torch.autocast(
                device_type=self.device.type,
                dtype=self.autocast_dtype,
                enabled=True,
            )
        else:
            autocast_ctx = nullcontext()
        with autocast_ctx:
            outputs = self.model(
                noisy_latents,
                noise_levels=tau,
                actions=actions_tensor,
                action_mask=action_mask,
            )
            pred_velocity = outputs.get("pred_velocity")

            if pred_velocity.dtype != target_velocity.dtype:
                target_velocity = target_velocity.to(pred_velocity.dtype)
            loss = F.mse_loss(pred_velocity, target_velocity)
            scaled_loss = loss / self.config.trainer.grad_accum_steps

        if self.use_scaler:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return {
            "loss": float(loss.detach().cpu()),
            "signal_level": float(tau.mean().detach().cpu()),
        }

    def _next_batch(self, data_iter):
        if self.config.trainer.single_batch_overfit:
            if self._cached_overfit_batch is None:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)
                self._cached_overfit_batch = batch
            return self._cached_overfit_batch, data_iter
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(self.dataloader)
            batch = next(data_iter)
        return batch, data_iter

    def _log_step(self, metrics: Dict[str, float]) -> None:
        if self.global_step % self.config.logging.log_interval == 0:
            message = " ".join(f"{key}={value:.5f}" for key, value in metrics.items())
            self.logger.info("step=%d %s", self.global_step, message)
            if self.wandb_run:
                self.wandb_run.log(
                    {f"train/{key}": value for key, value in metrics.items()},
                    step=self.global_step,
                )

    def _save_checkpoint(self, step: int) -> None:
        output_dir = Path(self.config.logging.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / f"world_model_step_{step:06d}.pt"
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step,
        }
        if self.use_scaler:
            payload["scaler"] = self.scaler.state_dict()
        payload["config"] = _dataclass_to_dict(self.config)
        if self.ema_model is not None:
            payload["ema_model"] = self.ema_model.state_dict()
        torch.save(payload, checkpoint_path)
        self.logger.info("Saved checkpoint to %s", checkpoint_path)

    def close(self) -> None:
        if self.wandb_run:
            self.wandb_run.finish()
            self.wandb_run = None

    def _maybe_load_checkpoint(self) -> None:
        checkpoint_path = self.config.trainer.resume_checkpoint
        if not checkpoint_path:
            return
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.use_scaler and "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.global_step = int(ckpt.get("step", 0))
        if "ema_model" in ckpt:
            self._ema_checkpoint_state = ckpt["ema_model"]
        elif self.ema_cfg.enabled:
            self._ema_checkpoint_state = None
        self.logger.info("Resumed training state from %s at step %d", checkpoint_path, self.global_step)


def train_from_config(
    config_path: Union[str, Path],
    model: nn.Module,
    embedder: Optional[DinoV3Embedder] = None,
) -> None:
    config = load_training_config(config_path)
    trainer = WorldModelTrainer(config, model, embedder)
    trainer.train()
