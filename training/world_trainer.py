import copy
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, is_dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import yaml

from rae_dino.rae import RAE
from training.dataset import (
    DataloaderConfig,
    DatasetConfig,
    WorldModelBatch,
    build_world_model_dataloader,
)
from world_model.flow_matching import (
    DiffusionConfig,
    DimensionShiftedUniformScheduler,
    EulerSolverConfig,
    sample_base_noise,
)
from world_model.backbone import WorldModelConfig

from training.logger import WorldModelLogger

@dataclass
class OptimizerConfig:
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    eps: float = 1e-8
    grad_clip_norm: Optional[float] = 1.0


@dataclass
class TrainerLoopConfig:
    max_steps: Optional[int] = 10_000
    grad_accum_steps: int = 1
    precision: str = "bf16"
    seed: int = 1234
    device: Optional[str] = None
    resume_checkpoint: Optional[str] = None
    single_batch_overfit: bool = False
    distributed: "DistributedConfig" = field(default_factory=lambda: DistributedConfig())


@dataclass
class DistributedConfig:
    enabled: bool = False
    backend: str = "nccl"
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    timeout_minutes: float = 30.0


@dataclass
class LoggingConfig:
    project: str = "world_model"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    log_interval: int = 10
    checkpoint_interval: int = 1_000
    output_dir: str = "checkpoints"
    sample_interval: Optional[int] = 1_000
    tau_log_limit: int = 200


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
    diffusion: DiffusionConfig = DiffusionConfig()
    world_model: WorldModelConfig = WorldModelConfig()
    ema: EMAConfig = EMAConfig()
    
    
def load_training_config(path: str | Path) -> WorldModelTrainingConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw: Dict[str, Any] = yaml.safe_load(handle)

    diffusion_kwargs = dict(raw.get("diffusion", {}))
    world_model_kwargs = dict(raw.get("world_model", {}))
    trainer_kwargs = dict(raw.get("trainer", {}))
    distributed_kwargs = trainer_kwargs.pop("distributed", {})
    trainer_cfg = TrainerLoopConfig(**trainer_kwargs)
    if distributed_kwargs:
        trainer_cfg.distributed = DistributedConfig(**distributed_kwargs)

    cfg = WorldModelTrainingConfig(
        dataset=DatasetConfig(**raw.get("dataset", {})),
        dataloader=DataloaderConfig(**raw.get("dataloader", {})),
        optimizer=OptimizerConfig(**raw.get("optimizer", {})),
        trainer=trainer_cfg,
        logging=LoggingConfig(**raw.get("logging", {})),
        diffusion=DiffusionConfig(**diffusion_kwargs),
        world_model=WorldModelConfig(**world_model_kwargs),
        ema=EMAConfig(**raw.get("ema", {})),
    )
    cfg.diffusion.validate()
    cfg.ema.validate()
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
        autoencoder: Optional[RAE] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.autoencoder = autoencoder or self._build_autoencoder(config.vision)
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0

        if self.config.trainer.distributed.enabled:
            self._init_distributed()

        self.device = self._resolve_device()
        self.is_main_process = not self.is_distributed or self.rank == 0

        self.model.to(self.device)
        self.autoencoder.to(self.device)
        self.autoencoder.eval()
        for param in self.autoencoder.parameters():
            param.requires_grad_(False)

        self.flow_cfg = config.diffusion
        self.scheduler = DimensionShiftedUniformScheduler(self.flow_cfg)
        self.ema_cfg = config.ema
        self.ema_model: Optional[nn.Module] = None
        self._ema_param_pairs: List[Tuple[torch.nn.Parameter, torch.nn.Parameter]] = []
        self._ema_buffer_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._ema_checkpoint_state: Optional[Dict[str, torch.Tensor]] = None
        self._cached_overfit_batch: Optional[WorldModelBatch] = None
        self._sampler_epoch = 0
        self._dataloader_iter = None

        self._seed_everything(config.trainer.seed + self.rank)

        euler_cfg = EulerSolverConfig(
            step_size=EulerSolverConfig().step_size,
            min_signal=0.0,
            max_signal=1.0,
        )
        frame_delta = self.config.dataset.frame_delta_seconds
        sample_fps = 1.0 / frame_delta if frame_delta > 0 else None
        self.logger = WorldModelLogger(
            config.logging,
            diffusion_cfg=self.flow_cfg,
            euler_cfg=euler_cfg,
            decode_fn=self.autoencoder.decode,
            sample_fps=sample_fps,
            is_main_process=self.is_main_process,
        )

        self.dataloader = build_world_model_dataloader(
            dataset_cfg=config.dataset,
            dataloader_cfg=config.dataloader,
            device=self.device,
            rank=self.rank if self.is_distributed else None,
            world_size=self.world_size if self.is_distributed else None,
            seed=config.trainer.seed + self.rank,
        )
        self._train_module: nn.Module
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.optimizer.lr,
            betas=config.optimizer.betas,
            weight_decay=config.optimizer.weight_decay,
            eps=config.optimizer.eps,
        )
        if self.is_distributed:
            if self.device.type != "cuda":
                raise RuntimeError("Distributed training currently expects CUDA devices.")
            self._train_module = DistributedDataParallel(
                self.model,
                device_ids=[self.device.index],
                output_device=self.device.index,
                broadcast_buffers=self.config.trainer.distributed.broadcast_buffers,
                find_unused_parameters=self.config.trainer.distributed.find_unused_parameters,
            )
        else:
            self._train_module = self.model
        autocast_enabled = config.trainer.precision in {"bf16", "fp16"} and self.device.type == "cuda"
        autocast_dtype = torch.bfloat16 if config.trainer.precision == "bf16" else torch.float16
        if autocast_enabled:
            self._autocast_scope = partial(
                torch.autocast,
                device_type=self.device.type,
                dtype=autocast_dtype,
                enabled=True,
            )
        else:
            self._autocast_scope = nullcontext
        self.use_scaler = config.trainer.precision == "fp16" and self.device.type == "cuda" # don't need to use scaler with bf16 since it rarely overflows in practice (from very safe source)
        self.scaler = torch.amp.GradScaler(enabled=self.use_scaler)
        self.logger.info(
            "AMP settings: autocast=%s (dtype=%s), grad_scaler=%s",
            autocast_enabled,
            autocast_dtype if autocast_enabled else None,
            self.use_scaler,
        )

        self.global_step = 0
        self._maybe_load_checkpoint()
        if self.ema_cfg.enabled:
            self._init_ema_model()
        self.logger.init_wandb(_dataclass_to_dict(self.config))
        if self.config.trainer.single_batch_overfit:
            self.logger.info("Single-batch overfit enabled; reusing the first batch for all updates.")

        self.logger.log_dataloader(self.dataloader)

    def _build_autoencoder(self) -> RAE:
        return RAE()

    def _init_distributed(self) -> None:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is required for distributed training.")
        if dist.is_initialized():
            self.is_distributed = True
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
            return
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA availability.")
        torch.cuda.set_device(local_rank)
        backend = self.config.trainer.distributed.backend
        timeout = timedelta(minutes=self.config.trainer.distributed.timeout_minutes)
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timeout,
        )
        self.is_distributed = True
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = local_rank

    def _resolve_device(self) -> torch.device:
        if self.is_distributed:
            return torch.device("cuda", self.local_rank)
        if self.config.trainer.device:
            return torch.device(self.config.trainer.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _init_ema_model(self) -> None:
        ema_device = torch.device(self.ema_cfg.device) if self.ema_cfg.device else self.device
        self.ema_model = copy.deepcopy(self.model).to(ema_device)
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

    def _update_ema(self) -> None:
        if not self.ema_model:
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

    def train(self) -> None:
        total_steps = self._resolve_total_steps()
        if self.global_step >= total_steps:
            self.logger.info(
                "Checkpoint already at or beyond max_steps (%d >= %d); nothing to do.",
                self.global_step,
                total_steps,
            )
            self.close()
            return

        self.logger.info(
            "Starting world model training for %d steps on %s (resuming at step %d)",
            total_steps,
            self.device,
            self.global_step,
        )
        self._dataloader_iter = None

        for step in range(self.global_step + 1, total_steps + 1):
            self.logger.start_step(step)
            self._train_module.train()
            self.optimizer.zero_grad(set_to_none=True) # set_to_none for potential memory savings hehe
            accum_metrics: Dict[str, float] = {}

            for accum_idx in range(self.config.trainer.grad_accum_steps):
                self.logger.start_micro_step(accum_idx)
                batch = self._next_batch()
                metrics = self._train_micro_step(batch)
                for key, value in metrics.items():
                    accum_metrics[key] = accum_metrics.get(key, 0.0) + value

            grad_metrics = self.logger.process_gradients(
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler if self.use_scaler else None,
                clip_norm=self.config.optimizer.grad_clip_norm,
            )

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
            if grad_metrics:
                mean_metrics.update(grad_metrics)
            mean_metrics = self._sync_metrics(mean_metrics)
            self.logger.log_training_metrics(mean_metrics)

            if (
                self.config.logging.checkpoint_interval
                and step % self.config.logging.checkpoint_interval == 0
            ):
                self._save_checkpoint(step)

        self.logger.info("Finished training")

    def _train_micro_step(
        self,
        batch: WorldModelBatch,
    ) -> Dict[str, float]:
        frames_cpu = batch.sequence_frames

        frames = frames_cpu.to(self.device, non_blocking=True)
        actions = batch.sequence_actions.to(self.device, non_blocking=True)
        independant_frames_mask = (
            batch.independant_frames_mask.to(self.device, non_blocking=True)
            if batch.independant_frames_mask is not None
            else None
        )
        actions_mask = (
            batch.actions_mask.to(self.device, non_blocking=True)
            if batch.actions_mask is not None
            else None
        )

        latents = self._encode_frames(frames)

        tau, tau_weights = self.scheduler.sample(latents) # High signal level -> tau around 1
        base_noise = sample_base_noise(latents, self.flow_cfg)
        tau_factor = tau.unsqueeze(-1).unsqueeze(-1)
        noisy_latents = (1.0 - tau_factor) * base_noise + tau_factor * latents
        self.logger.log_distr_noise(tau, weights=tau_weights)

        with self._autocast_scope():
            outputs = self._train_module(
                noisy_latents,
                noise_levels=tau,
                actions=actions,
                independant_frames_mask=independant_frames_mask,
                actions_mask=actions_mask,
            )
            pred_clean_latents = outputs.get("pred_clean_latents")
            if pred_clean_latents.dtype != latents.dtype:
                latents = latents.to(pred_clean_latents.dtype)
                tau_weights = tau_weights.to(pred_clean_latents.dtype)
            mse = F.mse_loss(pred_clean_latents, latents, reduction="none")
            mse = mse.mean(dim=(-1, -2))
            loss = (mse * tau_weights.to(mse.dtype)).mean()
            scaled_loss = loss / self.config.trainer.grad_accum_steps

        if self.use_scaler:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if self.is_main_process:
            self.logger.maybe_log_micro_step_video(
                self.model,
                latents=latents.detach(),
                noisy_latents=noisy_latents.detach(),
                noise_levels=tau.detach(),
                actions=actions.detach() if actions is not None else None,
                actions_mask=actions_mask.detach() if actions_mask is not None else None,
                independant_frames_mask=independant_frames_mask.detach()
                if independant_frames_mask is not None
                else None,
            )

        return {
            "loss": float(loss.detach().cpu()),
            "signal_level": float(tau.mean().detach().cpu()),
        }

    @torch.no_grad()
    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = frames.shape
        flat = frames.view(batch * steps, channels, height, width)
        latents = self.autoencoder.encode(flat)
        tokens, dim = latents.shape[1], latents.shape[2]
        return latents.view(batch, steps, tokens, dim)

    def _next_batch(self) -> WorldModelBatch:
        if self.config.trainer.single_batch_overfit:
            if self._cached_overfit_batch is None:
                self._cached_overfit_batch = self._fetch_batch()
            return self._cached_overfit_batch
        return self._fetch_batch()

    def _fetch_batch(self) -> WorldModelBatch:
        if self._dataloader_iter is None:
            self._dataloader_iter = self._create_data_iter()
        try:
            return next(self._dataloader_iter)
        except StopIteration:
            if self.is_main_process:
                self.logger.warning("Data iterator exhausted (StopIteration); reinitializing dataloader iterator.")
            self._dataloader_iter = self._create_data_iter()
            return next(self._dataloader_iter)

    def _create_data_iter(self):
        sampler = getattr(self.dataloader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self._sampler_epoch)
        self._sampler_epoch += 1
        return iter(self.dataloader)

    def _sync_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        if not self.is_distributed or not metrics:
            return metrics
        keys = sorted(metrics.keys())
        values = torch.tensor(
            [float(metrics[key]) for key in keys],
            device=self.device,
            dtype=torch.float32,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= self.world_size
        return {key: float(value) for key, value in zip(keys, values.tolist())}

    def _resolve_total_steps(self) -> int:
        configured = self.config.trainer.max_steps
        if configured is not None:
            return configured
        try:
            total = len(self.dataloader)
        except Exception: # for iter dataset (i.e. LeRobotStreaming), there is no length...
            total = 99999999999 # FIX ME
        if total <= 0:
            raise ValueError(
                "TrainerLoopConfig.max_steps resolved to zero steps; check dataset configuration."
            )
        return total

    def _save_checkpoint(self, step: int) -> None:
        if not self.is_main_process:
            return
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
        self.logger.close()
        if self.is_distributed and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

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


def train_from_config(
    config_path: Union[str, Path],
    model: nn.Module,
    autoencoder: Optional[RAE] = None,
) -> None:
    config = load_training_config(config_path)
    trainer = WorldModelTrainer(config, model, autoencoder)
    trainer.train()
