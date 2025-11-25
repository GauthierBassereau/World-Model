import copy
import random
import datetime
import time
import yaml
from contextlib import nullcontext
from dataclasses import asdict, dataclass, is_dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.rae_dino.rae import RAE
from src.dataset.loader import build_world_dataloader, DataloaderConfig
from src.dataset.world_dataset import WorldDatasetConfig, WorldBatch
from src.dataset.droid_dataset import DroidDatasetConfig
from src.world_model.backbone import WorldModelConfig
from src.training.world_evaluator import WorldModelEvaluator, EvaluationConfig
from src.world_model.diffusion import (
    DiffusionConfig,
    EulerSolverConfig,
    NoiseScheduler,
    latents_to_velocity,
    sample_base_noise,
)
from src.training.logger import WorldModelLogger
from src.training.utils import set_seed, sync_metrics


@dataclass
class OptimizerConfig:
    lr: Union[float, Dict[str, float]] = 1e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    eps: float = 1e-8
    grad_clip_norm: Optional[float] = None


@dataclass
class TrainerLoopConfig:
    max_steps: Optional[int] = None
    grad_accum_steps: int = 1
    precision: str = "bf16"
    seed: int = 1234
    device: Optional[str] = None
    evaluation_interval: int = 1000
    load_checkpoint: Optional[str] = None
    resume: bool = False


@dataclass
class LoggingConfig:
    project: str = "world_model"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    log_interval: int = 10
    checkpoint_interval: int = 1_000
    output_dir: str = "checkpoints"
    tau_log_limit: int = 200


@dataclass
class EMAConfig:
    enabled: bool = False
    decay: float = 0.999
    start_step: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("ema.decay must satisfy 0.0 <= decay < 1.0.")


@dataclass
class WorldModelTrainingConfig:
    train_dataset: WorldDatasetConfig = field(default_factory=lambda: WorldDatasetConfig(datasets={}, weights={}))
    train_dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    eval_dataset: WorldDatasetConfig = field(default_factory=lambda: WorldDatasetConfig(datasets={}, weights={}))
    eval_dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    evaluator: EvaluationConfig = field(default_factory=EvaluationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer: TrainerLoopConfig = field(default_factory=TrainerLoopConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    ode_solver: EulerSolverConfig = field(default_factory=EulerSolverConfig)


class WorldModelTrainer:
    def __init__(
        self,
        config: WorldModelTrainingConfig,
        model: nn.Module
    ) -> None:
        self.config = config
        self.flow_cfg = config.diffusion
        self.scheduler = NoiseScheduler(self.flow_cfg)
        self.ema_cfg = config.ema
        self.ema_model: Optional[nn.Module] = None
        self._ema_param_pairs: List[Tuple[torch.nn.Parameter, torch.nn.Parameter]] = []
        self._ema_buffer_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._ema_checkpoint_state: Optional[Dict[str, torch.Tensor]] = None

        self._sampler_epoch = 0
        self._dataloader_iter = None
        self._checkpoint_dir: Optional[Path] = None

        # Managing devices and distributed setup
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.device_idx = self.rank % torch.cuda.device_count()
        torch.cuda.set_device(self.device_idx)
        self.device = torch.device("cuda", self.device_idx)
        self.is_main_process = self.rank == 0

        self.model = model
        self.model.to(self.device).train()
        self.autoencoder = RAE() # For now hardcoded RAE, but later could be any AE        
        self.autoencoder.to(self.device).eval()
        for param in self.autoencoder.parameters():
            param.requires_grad_(False)
        
        self.model = torch.compile(self.model)
        self.autoencoder = torch.compile(self.autoencoder)

        seed = set_seed(config.trainer.seed, self.world_size, self.rank)

        sample_fps = self.config.train_dataset.fps
        self.logger = WorldModelLogger(
            config.logging,
            euler_cfg=self.config.ode_solver,
            sample_fps=sample_fps,
            is_main_process=self.is_main_process,
        )

        self.logger.log_config(asdict(config))

        self.dataloader = build_world_dataloader(
            dataset_cfg=config.train_dataset,
            dataloader_cfg=config.train_dataloader,
            grad_accum_steps=config.trainer.grad_accum_steps,
            seed=seed,
            rank=self.rank,
            world_size=self.world_size,
        )

        self.evaluator: Optional[WorldModelEvaluator] = None
        try:
            evaluator = WorldModelEvaluator(
                config=config.evaluator,
                dataset_cfg=config.eval_dataset,
                dataloader_cfg=config.eval_dataloader,
                diffusion_cfg=self.flow_cfg,
                autoencoder=self.autoencoder,
                device=self.device,
                seed=seed,
                solver_cfg=self.config.ode_solver,
                rank=self.rank,
                world_size=self.world_size,
                is_main_process=self.is_main_process,
            )
            if evaluator.dataloader is None:
                if self.is_main_process:
                    self.logger.warning(
                        "Evaluation dataset split unavailable; metrics will be skipped."
                    )
            else:
                self.evaluator = evaluator
        except ValueError as exc:
            if self.is_main_process:
                self.logger.warning(
                    "Evaluation disabled due to configuration issue: %s",
                    exc,
                )
            
        initial_lr = config.optimizer.lr
        if isinstance(initial_lr, dict):
            # If dict, use the value at the smallest step, or 0.0 if empty (unlikely)
            if initial_lr:
                # Keys are strings from config, convert to int
                schedule = {int(k): v for k, v in initial_lr.items()}
                min_step = min(schedule.keys())
                initial_lr = schedule[min_step]
            else:
                initial_lr = 0.0

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=initial_lr,
            betas=config.optimizer.betas,
            weight_decay=config.optimizer.weight_decay,
            eps=config.optimizer.eps,
        )

        self._init_lr_schedule()
        
        self._train_module: nn.Module
        if self.world_size > 1:
            self._train_module = DistributedDataParallel(
                self.model,
                device_ids=[self.device_idx],
                gradient_as_bucket_view=False,
            )
            print(f"[rank {self.rank}] Backbone wrapped in DDP.", flush=True)
        else:
            self._train_module = self.model

        autocast_enabled = config.trainer.precision in {"bf16", "bfloat16"} and self.device.type == "cuda"
        if autocast_enabled:
            self._autocast_scope = partial(
                torch.autocast,
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=True,
            )
        else:
            self._autocast_scope = nullcontext
        self.logger.info(
            "AMP settings: autocast=%s (dtype=%s)",
            autocast_enabled,
            torch.bfloat16 if autocast_enabled else None,
        )

        self.global_step = 0
        self._maybe_load_checkpoint()
        if self.ema_cfg.enabled:
            self._init_ema_model()
        self.logger.init_wandb(asdict(self.config))
        self._checkpoint_dir = self._resolve_checkpoint_dir()



    def _init_lr_schedule(self) -> None:
        lr_config = self.config.optimizer.lr
        self._lr_schedule: Optional[List[Tuple[int, float]]] = None
        
        if isinstance(lr_config, dict):
            # Convert keys to int and sort
            try:
                schedule = {int(k): v for k, v in lr_config.items()}
            except ValueError as e:
                raise ValueError(f"LR schedule keys must be convertible to integers. Got error: {e}")
            self._lr_schedule = sorted(schedule.items(), key=lambda x: x[0])
        else:
            # Constant LR, no schedule needed effectively, but we can just set it as None
            self._lr_schedule = None

    def _init_ema_model(self) -> None:
        ema_device = torch.device(self.device)
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
        
        step = self.global_step
        if step < self.ema_cfg.start_step:
            return

        cur_decay = self.ema_cfg.decay

        with torch.no_grad():
            for ema_param, param in self._ema_param_pairs:
                source = param.data
                if source.device != ema_param.device:
                    source = source.to(ema_param.device)
                ema_param.data.mul_(cur_decay).add_(source, alpha=1.0 - cur_decay)
            for ema_buffer, buffer in self._ema_buffer_pairs:
                source = buffer.data
                if source.device != ema_buffer.device:
                    source = source.to(ema_buffer.device)
                ema_buffer.data.copy_(source)
        
    def _apply_lr_schedule(self, step: int) -> None:
        if self._lr_schedule is None:
            return

        # Linear interpolation
        # Schedule is a sorted list of (step, lr)
        # Find the segment [start_step, end_step] that contains 'step'
        
        target_lr = self._lr_schedule[-1][1] # Default to last LR if beyond
        
        if step <= self._lr_schedule[0][0]:
            target_lr = self._lr_schedule[0][1]
        else:
            for i in range(len(self._lr_schedule) - 1):
                start_step, start_lr = self._lr_schedule[i]
                end_step, end_lr = self._lr_schedule[i+1]
                
                if start_step < step <= end_step:
                    progress = (step - start_step) / (end_step - start_step)
                    target_lr = start_lr + (end_lr - start_lr) * progress
                    break
        
        for group in self.optimizer.param_groups:
            group["lr"] = target_lr

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


        for step in range(self.global_step + 1, total_steps + 1):
            self._apply_lr_schedule(step)
            if self.device.type == "cuda" and step == 1:
                torch.cuda.reset_peak_memory_stats(self.device)
            self.logger.start_step(step)
            self._train_module.train()
            self.optimizer.zero_grad()
            accum_metrics: Dict[str, float] = {}

            for accum_idx in range(self.config.trainer.grad_accum_steps):
                self.logger.start_micro_step(accum_idx)
                batch = self._next_batch()
                metrics = self._train_micro_step(batch)
                for key, value in metrics.items():
                    accum_metrics[key] = accum_metrics.get(key, 0.0) + value

            grad_metrics = self.logger.log_grad_norm(model=self.model, key="grad_norm_before_clip")

            if self.config.optimizer.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.optimizer.grad_clip_norm,
                )

            grad_metrics.update(self.logger.log_grad_norm(model=self.model, key="grad_norm_after_clip"))
            
            self.optimizer.step()

            self._update_ema()

            if self.device.type == "cuda" and step == 1:
                peak_mem_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
                print(f"[GPU {self.device.index}] step {step} peak memory: {peak_mem_gb:.2f} GB", flush=True)

            self.global_step = step
            mean_metrics = {
                key: value / self.config.trainer.grad_accum_steps for key, value in accum_metrics.items()
            }
            lr_value = float(self.optimizer.param_groups[0]["lr"])
            mean_metrics["learning_rate"] = lr_value
            if grad_metrics:
                mean_metrics.update(grad_metrics)
            mean_metrics = self._sync_metrics(mean_metrics)
            self.logger.log_training_metrics(mean_metrics)
            self._maybe_run_evaluation(step)

            if (
                self.config.logging.checkpoint_interval
                and step % self.config.logging.checkpoint_interval == 0
            ):
                self._save_checkpoint(step)

        self.logger.info("Finished training")

    def _train_micro_step(
        self,
        batch: WorldBatch,
    ) -> Dict[str, float]:
        frames_cpu = batch.sequence_frames

        frames = frames_cpu.to(self.device, non_blocking=True)
        actions = batch.sequence_actions.to(self.device, non_blocking=True)
        independent_frames_mask = (
            batch.independent_frames_mask.to(self.device, non_blocking=True)
            if batch.independent_frames_mask is not None
            else None
        )
        actions_mask = (
            batch.actions_mask.to(self.device, non_blocking=True)
            if batch.actions_mask is not None
            else None
        )
        frames_valid_mask = batch.frames_valid_mask.to(self.device, non_blocking=True)

        latents = self._encode_frames(frames)

        tau = self.scheduler.sample(latents) # High signal level -> tau around 1
        base_noise = sample_base_noise(latents, self.flow_cfg)
        tau_factor = tau.unsqueeze(-1).unsqueeze(-1)
        noisy_latents = (1.0 - tau_factor) * base_noise + tau_factor * latents
        self.logger.log_distr_noise(tau)

        with self._autocast_scope():
            outputs = self._train_module(
                noisy_latents,
                noise_levels=tau,
                actions=actions,
                independent_frames_mask=independent_frames_mask,
                actions_mask=actions_mask,
            )
            pred_clean_latents = outputs.get("pred_clean_latents")

            # Loss computation -> Basically following Dreamerv4 and JiT(https://arxiv.org/pdf/2511.13720v1) papers
            # It is better to predict in x space because manifold is lower dim there than in noise space
            # But keep the v prediction formula, because it reweights the loss depending on tau
            v_true = latents - base_noise
            v_pred = latents_to_velocity(
                pred_clean_latents=pred_clean_latents,
                noisy_latents=noisy_latents,
                noise_levels=tau,
            )
            loss_unreduced = F.mse_loss(v_pred, v_true, reduction="none")

            # Apply valid frame mask, some sequences may have padding frames which should not contribute to loss
            frame_mask = frames_valid_mask.to(device=loss_unreduced.device, dtype=loss_unreduced.dtype)
            frame_loss = loss_unreduced.mean(dim=(-1, -2))
            masked_loss = frame_loss * frame_mask
            denom = frame_mask.sum().clamp_min(1.0)
            loss = masked_loss.sum() / denom
            scaled_loss = loss / self.config.trainer.grad_accum_steps

        scaled_loss.backward()

        return {
            "loss": float(loss) # because it is later divided by grad_accum_steps for logging
        }

    @torch.no_grad()
    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = frames.shape
        flat = frames.view(batch * steps, channels, height, width)
        latents = self.autoencoder.encode(flat).detach().clone() # Detach and clone needed because compile mess with autograd graph
        tokens, dim = latents.shape[1], latents.shape[2]
        return latents.view(batch, steps, tokens, dim)

    def _next_batch(self) -> WorldBatch:

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
        return sync_metrics(metrics, self.world_size, self.device)

    def _maybe_run_evaluation(self, step: int) -> None:
        if self.evaluator is None:
            return
        interval = self.config.trainer.evaluation_interval
        if interval is None or interval <= 0 or step % interval != 0:
            return
        eval_model = self.ema_model if self.ema_model is not None else self.model
        model_device = next(eval_model.parameters()).device
        if model_device != self.device:
            self.logger.warning(
                "Evaluation model device %s does not match trainer device %s; skipping evaluation.",
                model_device,
                self.device,
            )
            return
        if self.is_main_process:
            self.logger.info("Running evaluation...")
        result = self.evaluator.evaluate(eval_model)
        if result is not None and self.is_main_process:
            self.logger.log_evaluation(result)

    def _resolve_total_steps(self) -> int:
        configured = self.config.trainer.max_steps
        if configured is not None:
            return configured
        total = len(self.dataloader)
        if total <= 0:
            raise ValueError(
                "TrainerLoopConfig.max_steps resolved to zero steps; check dataset configuration."
            )
        return total

    def _save_checkpoint(self, step: int) -> None:
        if not self.is_main_process:
            return
        checkpoint_dir = self._checkpoint_dir or self._resolve_checkpoint_dir()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"world_model_step_{step:06d}.pt"
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step,
        }
        payload["config"] = asdict(self.config)
        if self.ema_model is not None:
            payload["ema_model"] = self.ema_model.state_dict()
        torch.save(payload, checkpoint_path)
        self.logger.info("Saved checkpoint to %s", checkpoint_path)
        self._sync_ema_across_ranks()

    def _sync_ema_across_ranks(self):
        if self.world_size <= 1 or self.ema_model is None:
            return
        
        # Broadcast EMA parameters from rank 0 to ensure consistency (there can be some divergence due to non-deterministic floating-point addition)
        for param in self.ema_model.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in self.ema_model.buffers():
            dist.broadcast(buffer.data, src=0)

    def close(self) -> None:
        self.logger.close()
        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    def _resolve_checkpoint_dir(self) -> Path:
        base_dir = Path(self.config.logging.output_dir)
        run_name = self._get_active_run_name()
        if run_name:
            return base_dir / run_name
        return base_dir / "no_name"

    def _get_active_run_name(self) -> Optional[str]:
        wandb_run = getattr(self.logger, "wandb_run", None)
        if wandb_run is not None:
            name = getattr(wandb_run, "name", None)
            if name:
                return name
        return None

    def _maybe_load_checkpoint(self) -> None:
        checkpoint_path = self.config.trainer.load_checkpoint
        if not checkpoint_path:
            return
        
        self.logger.info("Loading checkpoint from %s...", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])

        if self.config.trainer.resume:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.global_step = int(ckpt.get("step", 0))
            self.logger.info("Resuming training from step %d", self.global_step)
            for param_group in self.optimizer.param_groups:
                self.logger.info("Learning rate: %f", param_group["lr"])

            self.logger.info("Be sure to use a different seed to avoid seeing same data")

            if "ema_model" in ckpt:
                self._ema_checkpoint_state = ckpt["ema_model"]
            elif self.ema_cfg.enabled:
                self._ema_checkpoint_state = None
        else:
            self.logger.info("Loaded model weights for finetuning/initialization. Starting from step 0.")
            self._ema_checkpoint_state = None
