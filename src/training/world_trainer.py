import copy
import datetime
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from src.dataset.loader import build_world_dataloader, DataloaderConfig
from src.dataset.world_dataset import WorldDataset, WorldBatch, WorldDatasetConfig

from src.world_model.backbone import WorldModelConfig
from src.world_model.components import RMSNorm

from src.diffusion.common import calculate_velocity_1_to_2
from src.diffusion.euler_solver import EulerSolverConfig
from src.diffusion.patch_forcing import sample_ltg_patch_signal_levels
from src.diffusion.signal_scheduler import SignalScheduler, SignalSchedulerConfig

from src.training.logger import WorldModelLogger, LoggingConfig
from src.training.utils import set_seed, sync_metric_stats, sync_metrics
from src.training.world_evaluator import WorldModelEvaluator, EvaluationConfig
from src.rae_dino import AutoencoderConfig


@dataclass
class OptimizerConfig:
    lr: Union[float, Dict[str, float]] = 1e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    eps: float = 1e-8
    grad_clip_norm: Optional[float] = None


@dataclass
class TrainerLoopConfig:
    loss_type: str = "position"
    grad_accum_steps: int = 1
    precision: str = "bf16"
    seed: int = 1234
    evaluation_interval: int = 1000
    load_checkpoint: Optional[str] = None
    resume: bool = False
    loss_weighting: Optional[str] = None
    loss_weighting_intercept: float = 0.1
    loss_weighting_slope: float = 0.9


@dataclass
class PatchForcingConfig:
    enabled: bool = False
    sampler: str = "ltg"
    ltg_std: float = 0.6
    difficulty_loss_weight: float = 0.01

    def __post_init__(self) -> None:
        self.sampler = self.sampler.lower()
        if self.sampler != "ltg":
            raise ValueError(f"Unsupported patch forcing sampler: {self.sampler}")
        if self.ltg_std <= 0.0:
            raise ValueError("patch_forcing.ltg_std must be positive.")
        if self.difficulty_loss_weight < 0.0:
            raise ValueError("patch_forcing.difficulty_loss_weight must be non-negative.")


@dataclass
class EMAConfig:
    enabled: bool = False
    decay: float = 0.999
    start_step: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("ema.decay must be in [0, 1).")
        if self.start_step < 0:
            raise ValueError("ema.start_step must be non-negative.")


@dataclass
class WorldModelTrainingConfig:
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    train_dataset: WorldDatasetConfig = field(default_factory=lambda: WorldDatasetConfig(datasets={}, weights={}))
    train_dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    eval_dataset: WorldDatasetConfig = field(default_factory=lambda: WorldDatasetConfig(datasets={}, weights={}))
    eval_dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    trainer: TrainerLoopConfig = field(default_factory=TrainerLoopConfig)
    evaluator: EvaluationConfig = field(default_factory=EvaluationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    signal_scheduler: SignalSchedulerConfig = field(default_factory=SignalSchedulerConfig)
    patch_forcing: PatchForcingConfig = field(default_factory=PatchForcingConfig)
    euler_solver: EulerSolverConfig = field(default_factory=EulerSolverConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


MetricStats = Dict[str, Tuple[float, float]]


@dataclass
class MicroStepLoss:
    loss_sum: torch.Tensor
    valid_frame_count: torch.Tensor
    total_frame_count: torch.Tensor
    metric_stats: MetricStats


class WorldModelTrainer:
    def __init__(
        self,
        config: WorldModelTrainingConfig,
        model: torch.nn.Module,
        autoencoder: torch.nn.Module,
    ) -> None:
        self.config = config
        self._dataloader_iter = None
        self._sampler_epoch = 0
        self.epoch_step = 0
        self._epoch_batches_consumed = 0
        self._legacy_sampler_seed_epoch: Optional[int] = None
        self.global_step = 0
        self._pending_rng_state: Optional[Dict[str, Any]] = None
        self._resume_wandb_run_id: Optional[str] = None
        self._train_module: torch.nn.Module
        autocast_enabled = config.trainer.precision in {"bf16", "bfloat16"}
        if isinstance(config.optimizer.lr, dict):
            schedule = {int(k): v for k, v in config.optimizer.lr.items()}
            min_step = min(schedule.keys())
            initial_lr = schedule[min_step]
        else:
            initial_lr = config.optimizer.lr

        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.device_idx = self.rank % torch.cuda.device_count()
        torch.cuda.set_device(self.device_idx)
        self.device = torch.device("cuda", self.device_idx)
        self.is_main_process = self.rank == 0

        if config.world_model.predict_patch_difficulty:
            if config.trainer.loss_type != "velocity":
                raise ValueError("Patch difficulty training currently requires trainer.loss_type=velocity.")
        
        process_seed = set_seed(config.trainer.seed, self.world_size, self.rank)
        self._process_seed = process_seed

        self.autoencoder = autoencoder
        self.autoencoder.to(self.device).eval()
        for param in self.autoencoder.parameters():
            param.requires_grad_(False)
        self.model = model
        self.model.to(self.device).train()
        self.model = torch.compile(self.model)

        if self.world_size > 1:
            self._train_module = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device_idx],
                gradient_as_bucket_view=False,
            )
        else:
            self._train_module = self.model

        self.logger = WorldModelLogger(
            config.logging,
            is_main_process=self.is_main_process,
        )
        if self.is_main_process:
            self.logger.info("Model: %s", self.model)
            self.logger.info("Parameters: %s", sum(p.numel() for p in self.model.parameters()))
        
        if autocast_enabled:
            self._autocast_scope = partial(
                torch.autocast,
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=True,
            )
        else:
            self._autocast_scope = nullcontext
        
        if self.is_main_process:
            self.logger.info("AMP settings: autocast=%s (dtype=%s)", autocast_enabled, torch.bfloat16 if autocast_enabled else None)

        dataset = WorldDataset(config.train_dataset, logger=self.logger, seed=config.trainer.seed)
        if self.is_main_process:
            self.logger.info(dataset)
        
        self.dataloader = build_world_dataloader(
            dataset=dataset,
            dataloader_cfg=config.train_dataloader,
            grad_accum_steps=config.trainer.grad_accum_steps,
            # Sampler seeds must be identical on every rank. Process-local
            # randomness remains rank-specific through process_seed above.
            seed=config.trainer.seed,
            rank=self.rank,
            world_size=self.world_size,
        )
        if self.is_main_process:
            self.logger.info(f"Training dataloader has length: {len(self.dataloader)}")

        self.evaluator = WorldModelEvaluator(
            config=config.evaluator,
            dataset_cfg=config.eval_dataset,
            dataloader_cfg=config.eval_dataloader,
            signal_scheduler_cfg=config.signal_scheduler,
            euler_solver_cfg=config.euler_solver,
            autoencoder=self.autoencoder,
            logger=self.logger,
            device=self.device,
            seed=config.trainer.seed,
            rank=self.rank,
            world_size=self.world_size,
            is_main_process=self.is_main_process,
        )
            
        self.optimizer = self._create_optimizer(initial_lr)

        self._init_lr_schedule()

        self.signal_scheduler = SignalScheduler(config.signal_scheduler)

        self.ema_model: Optional[torch.nn.Module] = None
        self._ema_param_pairs: List[Tuple[torch.nn.Parameter, torch.nn.Parameter]] = []
        self._ema_buffer_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._ema_checkpoint_state: Optional[Dict[str, torch.Tensor]] = None
        self._ema_checkpoint_started: Optional[bool] = None
        self._ema_started = False

        self._maybe_load_checkpoint()
        if config.ema.enabled:
            self._init_ema_model()

        self.logger.init_wandb(
            asdict(self.config),
            run_id=self._resume_wandb_run_id,
            resume="allow" if self._resume_wandb_run_id is not None else None,
        )
        # self.logger.log_config(asdict(config))

        base_dir = Path(config.logging.output_dir)
        wandb_run = getattr(self.logger, "wandb_run", None)
        checkpoint_run_name = (
            wandb_run.name
            if wandb_run is not None
            else (config.logging.run_name or "unnamed_run")
        )
        if self.is_main_process:
            self._checkpoint_dir = base_dir / checkpoint_run_name
        else:
            self._checkpoint_dir = None

    def _create_optimizer(self, initial_lr: float):
        decay = set()
        no_decay = set()
        
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.Embedding, RMSNorm)
        
        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif pn.endswith('weight'):
                    decay.add(fpn)

        no_decay.add('base_action_embed')
        no_decay.add('register_tokens')
        no_decay.add('signal_token_offsets')
        # torch.compile() prefixes parameter names with '_orig_mod.'
        no_decay.add('_orig_mod.base_action_embed')
        no_decay.add('_orig_mod.register_tokens')
        no_decay.add('_orig_mod.signal_token_offsets')
        
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        
        # Filter to only include params that actually exist (handles compiled vs non-compiled models)
        decay = decay & param_dict.keys()
        no_decay = no_decay & param_dict.keys()
        
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.config.optimizer.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        optimizer = torch.optim.AdamW(optim_groups, lr=initial_lr, betas=self.config.optimizer.betas, eps=self.config.optimizer.eps)
        return optimizer

    def _init_lr_schedule(self) -> None:
        lr_config = self.config.optimizer.lr
        self._lr_schedule: Optional[List[Tuple[int, float]]] = None
        
        if isinstance(lr_config, dict):
            schedule = {int(k): v for k, v in lr_config.items()}
            self._lr_schedule = sorted(schedule.items(), key=lambda x: x[0])
        else:
            self._lr_schedule = None

    def _init_ema_model(self) -> None:
        ema_device = torch.device(self.device)
        self.ema_model = copy.deepcopy(self.model).to(ema_device)
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
        self.ema_model.eval()

        ema_named_params = list(self.ema_model.named_parameters())
        online_named_params = list(self.model.named_parameters())
        if [name for name, _ in ema_named_params] != [name for name, _ in online_named_params]:
            raise RuntimeError("EMA and online model parameter names do not match.")
        self._ema_param_pairs = [
            (ema_param, param)
            for (_, ema_param), (_, param) in zip(ema_named_params, online_named_params)
        ]

        ema_named_buffers = list(self.ema_model.named_buffers())
        online_named_buffers = list(self.model.named_buffers())
        if [name for name, _ in ema_named_buffers] != [name for name, _ in online_named_buffers]:
            raise RuntimeError("EMA and online model buffer names do not match.")
        self._ema_buffer_pairs = [
            (ema_buffer, buffer)
            for (_, ema_buffer), (_, buffer) in zip(ema_named_buffers, online_named_buffers)
        ]

        loaded_ema = self._ema_checkpoint_state is not None
        if self._ema_checkpoint_state is not None:
            self.ema_model.load_state_dict(self._ema_checkpoint_state)
            self._ema_checkpoint_state = None
        if self._ema_checkpoint_started is not None:
            self._ema_started = self._ema_checkpoint_started
        else:
            # A legacy EMA checkpoint has no explicit lifecycle flag. An EMA
            # newly enabled while resuming after start_step begins as an exact
            # copy of the online checkpoint and is valid immediately.
            self._ema_started = self.global_step >= self.config.ema.start_step

        if loaded_ema:
            self.logger.info(
                "Restored EMA weights from step %d (active=%s).",
                self.global_step,
                self._ema_started,
            )
        elif self.global_step > 0 and self._ema_started:
            self.logger.warning(
                "Checkpoint has no EMA weights; initializing EMA from the online model at step %d.",
                self.global_step,
            )

    def _update_ema(self) -> None:
        if self.ema_model is None:
            return
        
        step = self.global_step
        if step < self.config.ema.start_step:
            return

        with torch.no_grad():
            if not self._ema_started:
                # Delayed EMA must start from the current trained weights. If
                # it remained at initialization and decayed from there, it
                # would take thousands of updates to become usable.
                self._copy_online_to_ema()
                self._ema_started = True
                self.logger.info("EMA activated from online weights at step %d.", step)
                return

            cur_decay = self.config.ema.decay
            for ema_param, param in self._ema_param_pairs:
                source = param.detach()
                if source.device != ema_param.device:
                    source = source.to(ema_param.device)
                ema_param.mul_(cur_decay).add_(source, alpha=1.0 - cur_decay)
            for ema_buffer, buffer in self._ema_buffer_pairs:
                source = buffer.detach()
                if source.device != ema_buffer.device:
                    source = source.to(ema_buffer.device)
                ema_buffer.copy_(source)

    def _copy_online_to_ema(self) -> None:
        for ema_param, param in self._ema_param_pairs:
            ema_param.copy_(param.detach().to(device=ema_param.device))
        for ema_buffer, buffer in self._ema_buffer_pairs:
            ema_buffer.copy_(buffer.detach().to(device=ema_buffer.device))
        
    def _apply_lr_schedule(self, step: int) -> None:
        if self._lr_schedule is None:
            return
        
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
        self._restore_pending_rng_state()
        self.logger.info(
            "Starting world model training on %s (epoch %d, epoch step %d, "
            "epoch batches consumed %d, global step %d)",
            self.device,
            self._sampler_epoch,
            self.epoch_step,
            self._epoch_batches_consumed,
            self.global_step,
        )

        while True:
            self.epoch_step += 1
            self.global_step += 1
            self._apply_lr_schedule(self.global_step)
            if self.device.type == "cuda" and self.global_step == 1:
                torch.cuda.reset_peak_memory_stats(self.device)
            self.logger.start_step(self.global_step)
            self._train_module.train()
            self.optimizer.zero_grad()
            accum_metric_stats: MetricStats = {}
            local_valid_frame_count = torch.zeros((), device=self.device, dtype=torch.float32)
            local_total_frame_count = torch.zeros((), device=self.device, dtype=torch.float32)

            for accum_idx in range(self.config.trainer.grad_accum_steps):
                self.logger.start_micro_step(accum_idx)
                batch = self._next_batch()
                with self._backward_sync_context(accum_idx):
                    micro_step_loss = self._compute_micro_step_loss(batch)
                    micro_step_loss.loss_sum.backward()
                local_valid_frame_count += micro_step_loss.valid_frame_count
                local_total_frame_count += micro_step_loss.total_frame_count
                self._accumulate_metric_stats(accum_metric_stats, micro_step_loss.metric_stats)

            global_valid_frame_count = self._sync_scalar_sum(local_valid_frame_count)
            global_total_frame_count = self._sync_scalar_sum(local_total_frame_count)
            # DDP averages gradients across ranks. We backpropagate summed frame losses on
            # each micro-step, then normalize once using the global number of valid frames
            # seen across all ranks and accumulation steps.
            self._scale_gradients(self.world_size / max(global_valid_frame_count, 1.0))

            grad_metrics = self.logger.log_grad_norm(model=self.model, key="grad_norm_before_clip")

            if self.config.optimizer.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.optimizer.grad_clip_norm,
                )

            grad_metrics.update(self.logger.log_grad_norm(model=self.model, key="grad_norm_after_clip"))
            
            self.optimizer.step()

            self._update_ema()

            if self.device.type == "cuda" and self.global_step == 1:
                peak_mem_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
                self.logger.info(f"[GPU {self.device.index}] global step {self.global_step} peak memory: {peak_mem_gb:.2f} GB")
            mean_metrics = self._finalize_metric_stats(
                sync_metric_stats(accum_metric_stats, self.world_size, self.device)
            )
            mean_metrics["valid_frames"] = global_valid_frame_count
            mean_metrics["total_frames"] = global_total_frame_count
            lr_value = float(self.optimizer.param_groups[0]["lr"])
            scalar_metrics = {"learning_rate": lr_value}
            if grad_metrics:
                scalar_metrics.update(grad_metrics)
            mean_metrics.update(sync_metrics(scalar_metrics, self.world_size, self.device))
            self.logger.log_training_metrics(mean_metrics)

            self._maybe_run_evaluation(self.global_step)

            if (
                self.config.logging.checkpoint_interval
                and self.global_step % self.config.logging.checkpoint_interval == 0
            ):
                # Save after evaluation. Evaluation uses an isolated RNG scope,
                # but this ordering also keeps checkpoint state aligned with
                # all work associated with the logged step.
                self._save_checkpoint(self.global_step)

    def _compute_micro_step_loss(
        self,
        batch: WorldBatch,
    ) -> MicroStepLoss:
        frames = batch.sequence_frames.to(self.device, non_blocking=True)
        actions = batch.sequence_actions.to(self.device, non_blocking=True)
        independent_frames = batch.independent_frames.to(self.device, non_blocking=True)
        use_actions = batch.use_actions.to(self.device, non_blocking=True)
        padded_frames = batch.padded_frames.to(self.device, non_blocking=True)
        frames_valid_mask = ~padded_frames
        dataset_indices = batch.dataset_indices.to(self.device, non_blocking=True)
        
        frames = frames.float() / 255.0
        with torch.no_grad():
            batch_size, steps, channels, height, width = frames.shape
            flat = frames.view(batch_size * steps, channels, height, width)
            latents = self.autoencoder.encode(flat).detach().clone() # Detach and clone needed because compile messes up autograd graph
            tokens, dim = latents.shape[1], latents.shape[2]
            latents = latents.view(batch_size, steps, tokens, dim)
        
        base_signal_levels, scheduler_steps = self.signal_scheduler.sample_with_base(latents)
        signal_levels = sample_ltg_patch_signal_levels(
            base_signal_levels=base_signal_levels,
            num_tokens=tokens,
            ltg_std=self.config.patch_forcing.ltg_std,
        ) if self.config.patch_forcing.enabled else base_signal_levels
        self.logger.log_distr_signal(signal_levels)
        base_noise = torch.randn_like(latents)
        signal_levels_expanded = self._expand_signal_to_latents(signal_levels, latents)
        
        noisy_latents = (1.0 - signal_levels_expanded) * base_noise + signal_levels_expanded * latents
        with self._autocast_scope():
            outputs = self._train_module(
                noisy_latents,
                signal_levels=base_signal_levels,
                actions=actions,
                independent_frames=independent_frames,
                use_actions=use_actions,
            )
            micro_step_loss = self._compute_loss(
                latents=latents,
                noisy_latents=noisy_latents,
                outputs=outputs,
                signal_levels=signal_levels,
                scheduler_steps=scheduler_steps,
                base_noise=base_noise,
                frames_valid_mask=frames_valid_mask,
                independent_frames=independent_frames,
                dataset_indices=dataset_indices,
                dataset_names=batch.dataset_names,
                use_actions=use_actions,
            )

        return micro_step_loss

    def _compute_loss(
        self,
        latents: torch.Tensor,
        noisy_latents: torch.Tensor,
        outputs: torch.Tensor,
        signal_levels: torch.Tensor,
        scheduler_steps: torch.Tensor,
        base_noise: torch.Tensor,
        frames_valid_mask: torch.Tensor,
        independent_frames: torch.Tensor,
        dataset_indices: torch.Tensor,
        dataset_names: Dict[int, str],
        use_actions: torch.Tensor,
    ) -> MicroStepLoss:
        v_true: Optional[torch.Tensor] = None
        v_pred: Optional[torch.Tensor] = None
        if self.config.trainer.loss_type == "position":
            loss_unreduced = torch.nn.functional.mse_loss(latents, outputs.latents, reduction="none")
        else:
            target_signal = torch.ones_like(signal_levels)
            min_denom = self.config.euler_solver.min_denom
            v_true = calculate_velocity_1_to_2(
                latent_1=noisy_latents,
                latent_2=latents,
                signal_levels_1=signal_levels,
                signal_levels_2=target_signal,
                min_denom=min_denom,
            )
            v_pred = calculate_velocity_1_to_2(
                latent_1=noisy_latents,
                latent_2=outputs.latents,
                signal_levels_1=signal_levels,
                signal_levels_2=target_signal,
                min_denom=min_denom,
            )
            loss_unreduced = torch.nn.functional.mse_loss(v_pred, v_true, reduction="none")

        if self.config.trainer.loss_weighting in ("linear", "linear_signal") and self.config.trainer.loss_type == "position":
            # Weight scales linearly with signal level (after scheduler transformation)
            weights = self.config.trainer.loss_weighting_intercept + self.config.trainer.loss_weighting_slope * signal_levels
            weights = self._expand_signal_to_latents(weights, loss_unreduced)
            loss_unreduced = loss_unreduced * weights
        elif self.config.trainer.loss_weighting == "linear_scheduler" and self.config.trainer.loss_type == "position":
            # Weight scales linearly with scheduler step (before transformation)
            # This is useful when using dimension shift, so weights scale uniformly with the original uniform distribution
            weights = self.config.trainer.loss_weighting_intercept + self.config.trainer.loss_weighting_slope * scheduler_steps
            weights = self._expand_signal_to_latents(weights, loss_unreduced)
            loss_unreduced = loss_unreduced * weights

        # Apply valid frame mask, some sequences may have padding frames which should not contribute to loss
        frames_valid_mask = frames_valid_mask.to(dtype=loss_unreduced.dtype)
        frame_loss = loss_unreduced.mean(dim=(-1, -2))
        valid_frame_loss = frame_loss * frames_valid_mask
        loss_sum = valid_frame_loss.sum()
        valid_frame_count = frames_valid_mask.sum().detach()
        total_frame_count = torch.tensor(
            float(frames_valid_mask.numel()),
            device=frames_valid_mask.device,
            dtype=torch.float32,
        )

        metrics = {
            "raw_l2_loss": self._to_metric_stat(loss_sum, valid_frame_count),
        }
        if outputs.patch_logvar is not None and self.config.patch_forcing.difficulty_loss_weight > 0.0:
            if v_true is None or v_pred is None:
                raise RuntimeError("Patch difficulty loss requires velocity targets.")

            logvar = outputs.patch_logvar.float()
            logvar_expanded = self._expand_signal_to_latents(logvar, v_true)
            variance = torch.exp(logvar_expanded).clamp_min(1e-8)
            squared_error = (v_true.float() - v_pred.detach().float()).pow(2)
            difficulty_loss_unreduced = 0.5 * (
                math.log(2.0 * math.pi)
                + logvar_expanded
                + squared_error / variance
            )
            difficulty_frame_loss = difficulty_loss_unreduced.mean(dim=(-1, -2))
            valid_difficulty_frame_loss = difficulty_frame_loss * frames_valid_mask
            difficulty_loss_sum = valid_difficulty_frame_loss.sum()
            loss_sum = loss_sum + self.config.patch_forcing.difficulty_loss_weight * difficulty_loss_sum
            metrics["patch_difficulty_nll_loss"] = self._to_metric_stat(
                difficulty_loss_sum,
                valid_frame_count,
            )
            metrics["total_loss"] = self._to_metric_stat(loss_sum, valid_frame_count)

        if self.global_step % 10 == 0:
            breakdown = self._compute_loss_breakdown(
                valid_frame_loss,
                frames_valid_mask,
                independent_frames,
                dataset_indices,
                dataset_names,
                use_actions,
            )
            metrics.update(breakdown)

        return MicroStepLoss(
            loss_sum=loss_sum,
            valid_frame_count=valid_frame_count,
            total_frame_count=total_frame_count,
            metric_stats=metrics,
        )

    @staticmethod
    def _expand_signal_to_latents(signal_levels: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        while signal_levels.ndim < latents.ndim:
            signal_levels = signal_levels.unsqueeze(-1)
        return signal_levels

    def _compute_loss_breakdown(
        self,
        valid_frame_loss: torch.Tensor,
        frames_valid_mask: torch.Tensor,
        independent_frames: torch.Tensor,
        dataset_indices: torch.Tensor,
        dataset_names: Dict[int, str],
        use_actions: torch.Tensor,
    ) -> MetricStats:
        metrics: MetricStats = {}

        frames_valid_mask_bool = frames_valid_mask.to(dtype=torch.bool)
        independent_frames_bool = independent_frames.to(dtype=torch.bool)
        use_actions_bool = use_actions.to(dtype=torch.bool)

        global_masks = {
            "l2_loss/independent_frames": frames_valid_mask_bool & independent_frames_bool,
            "l2_loss/dependent_frames": frames_valid_mask_bool & (~independent_frames_bool),
            "l2_loss/with_actions": frames_valid_mask_bool & use_actions_bool,
            "l2_loss/without_actions": frames_valid_mask_bool & (~use_actions_bool),
        }
        for key, mask in global_masks.items():
            metrics[key] = self._masked_metric_stat(valid_frame_loss, mask)

        for idx, name in dataset_names.items():
            dataset_mask = dataset_indices == idx
            dataset_loss = valid_frame_loss[dataset_mask]
            dataset_valid_mask = frames_valid_mask_bool[dataset_mask]
            dataset_independent = independent_frames_bool[dataset_mask]
            dataset_use_actions = use_actions_bool[dataset_mask]

            metrics[f"l2_loss/{name}"] = self._masked_metric_stat(dataset_loss, dataset_valid_mask)
            metrics[f"l2_loss/{name}/independent"] = self._masked_metric_stat(
                dataset_loss,
                dataset_valid_mask & dataset_independent,
            )
            metrics[f"l2_loss/{name}/dependent"] = self._masked_metric_stat(
                dataset_loss,
                dataset_valid_mask & (~dataset_independent),
            )
            metrics[f"l2_loss/{name}/with_actions"] = self._masked_metric_stat(
                dataset_loss,
                dataset_valid_mask & dataset_use_actions,
            )
            metrics[f"l2_loss/{name}/without_actions"] = self._masked_metric_stat(
                dataset_loss,
                dataset_valid_mask & (~dataset_use_actions),
            )

        return metrics

    @staticmethod
    def _accumulate_metric_stats(
        accum_metric_stats: MetricStats,
        metric_stats: MetricStats,
    ) -> None:
        for key, (loss_sum, denom) in metric_stats.items():
            prev_loss_sum, prev_denom = accum_metric_stats.get(key, (0.0, 0.0))
            accum_metric_stats[key] = (prev_loss_sum + loss_sum, prev_denom + denom)

    @staticmethod
    def _finalize_metric_stats(metric_stats: MetricStats) -> Dict[str, float]:
        finalized: Dict[str, float] = {}
        for key, (loss_sum, denom) in metric_stats.items():
            finalized[key] = loss_sum / denom if denom > 0.0 else 0.0
        return finalized

    @staticmethod
    def _to_metric_stat(loss_sum: torch.Tensor, count: torch.Tensor) -> Tuple[float, float]:
        return float(loss_sum.detach()), float(count.detach())

    @classmethod
    def _masked_metric_stat(
        cls,
        valid_frame_loss: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[float, float]:
        mask = mask.to(dtype=valid_frame_loss.dtype)
        return cls._to_metric_stat((valid_frame_loss * mask).sum(), mask.sum())

    def _backward_sync_context(self, accum_idx: int):
        if self.world_size > 1 and accum_idx < self.config.trainer.grad_accum_steps - 1:
            return self._train_module.no_sync()
        return nullcontext()

    def _sync_scalar_sum(self, value: torch.Tensor) -> float:
        total = value.detach().to(device=self.device, dtype=torch.float32)
        if self.world_size > 1:
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        return float(total.item())

    def _scale_gradients(self, scale: float) -> None:
        if scale == 1.0:
            return
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.mul_(scale)

    def _next_batch(self) -> WorldBatch:
        if self._dataloader_iter is None:
            self._dataloader_iter = self._create_data_iter()
        try:
            batch = next(self._dataloader_iter)
        except StopIteration:
            if self.is_main_process:
                self.logger.info(
                    "Epoch %d finished after %d dataloader batches. Starting epoch %d.",
                    self._sampler_epoch,
                    self._epoch_batches_consumed,
                    self._sampler_epoch + 1,
                )
            self._sampler_epoch += 1
            self.epoch_step = 0
            self._epoch_batches_consumed = 0
            self._dataloader_iter = self._create_data_iter()
            batch = next(self._dataloader_iter)
        self._epoch_batches_consumed += 1
        return batch

    def _create_data_iter(self):
        sampler = getattr(self.dataloader, "sampler", None)
        legacy_seed_epoch = getattr(self, "_legacy_sampler_seed_epoch", None)
        if sampler is not None and hasattr(sampler, "seed"):
            sampler.seed = (
                self._process_seed
                if self._sampler_epoch == legacy_seed_epoch
                else self.config.trainer.seed
            )
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self._sampler_epoch)
        if sampler is not None and hasattr(sampler, "set_start_index"):
            sample_offset = self._epoch_batches_consumed * self.dataloader.batch_size
            sampler.set_start_index(sample_offset)
        self.dataloader.dataset.reshuffle_virtual_map(self.config.trainer.seed + self._sampler_epoch)
        worker_generator = getattr(self.dataloader, "generator", None)
        if worker_generator is not None:
            worker_generator.manual_seed(self._process_seed + self._sampler_epoch)
        return iter(self.dataloader)

    def _maybe_run_evaluation(self, step: int) -> None:
        interval = self.config.trainer.evaluation_interval
        if interval is None or interval <= 0 or step % interval != 0:
            return
        use_ema = self.ema_model is not None and self._ema_started
        eval_model = self.ema_model if use_ema else self.model
        self.logger.info(
            "Evaluating %s model at step %d.",
            "EMA" if use_ema else "online",
            step,
        )

        # Evaluation samples diffusion noise. Keep it reproducible and prevent
        # it from changing the training RNG stream.
        python_rng_state = random.getstate()
        cuda_devices = [self.device_idx] if self.device.type == "cuda" else []
        eval_seed = self.config.trainer.seed * self.world_size + self.rank
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                random.seed(eval_seed)
                torch.set_rng_state(torch.Generator().manual_seed(eval_seed).get_state())
                if self.device.type == "cuda":
                    torch.cuda.manual_seed(eval_seed)
                self.evaluator.evaluate(eval_model)
        finally:
            random.setstate(python_rng_state)

    def _save_checkpoint(self, step: int) -> None:
        # This collective must be entered by every rank. Keeping it after the
        # rank-zero early return deadlocks as soon as EMA checkpointing is used.
        self._sync_ema_across_ranks()
        rng_states = self._gather_rng_states()
        if not self.is_main_process:
            return
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self._checkpoint_dir / f"world_model_step_{step:06d}.pt"
        payload = {
            "checkpoint_version": 2,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self._sampler_epoch,
            "epoch_step": self.epoch_step,
            "global_step": self.global_step,
            "world_size": self.world_size,
            "data_state": {
                "sampler_epoch": self._sampler_epoch,
                "batches_consumed": self._epoch_batches_consumed,
            },
            "rng_states": rng_states,
            "ema_started": self._ema_started,
        }
        wandb_run = getattr(self.logger, "wandb_run", None)
        if wandb_run is not None and getattr(wandb_run, "id", None):
            payload["wandb_run_id"] = wandb_run.id
        payload["config"] = asdict(self.config)
        if self.ema_model is not None and self._ema_started:
            payload["ema_model"] = self.ema_model.state_dict()
        torch.save(payload, checkpoint_path)
        self.logger.info("Saved checkpoint to %s", checkpoint_path)

    def _sync_ema_across_ranks(self):
        if self.world_size <= 1 or self.ema_model is None or not self._ema_started:
            return
        # Broadcast EMA parameters from rank 0 to ensure consistency (there can be some divergence due to non-deterministic floating-point addition)
        for param in self.ema_model.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in self.ema_model.buffers():
            dist.broadcast(buffer.data, src=0)

    def _local_rng_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
        }
        if self.device.type == "cuda":
            state["cuda"] = torch.cuda.get_rng_state(self.device)
        return state

    def _gather_rng_states(self) -> List[Dict[str, Any]]:
        local_state = self._local_rng_state()
        if self.world_size <= 1:
            return [local_state]
        gathered: List[Optional[Dict[str, Any]]] = [None] * self.world_size
        dist.all_gather_object(gathered, local_state)
        return [state for state in gathered if state is not None]

    def _restore_pending_rng_state(self) -> None:
        if self._pending_rng_state is None:
            return
        state = self._pending_rng_state
        random.setstate(state["python"])
        torch.set_rng_state(state["torch"])
        if self.device.type == "cuda" and "cuda" in state:
            torch.cuda.set_rng_state(state["cuda"], self.device)
        self._pending_rng_state = None
        self.logger.info("Restored process RNG state from checkpoint.")

    def _maybe_load_checkpoint(self) -> None:
        checkpoint_path = self.config.trainer.load_checkpoint
        if not checkpoint_path:
            return
        
        self.logger.info("Loading checkpoint from %s...", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        
        # When resuming, require exact match. When finetuning, allow missing/extra keys.
        strict = self.config.trainer.resume
        incompatible = self.model.load_state_dict(ckpt["model"], strict=strict)
        
        if not strict:
            if incompatible.missing_keys:
                self.logger.warning(
                    "Missing keys in checkpoint (will be randomly initialized): %s",
                    incompatible.missing_keys
                )
            if incompatible.unexpected_keys:
                self.logger.warning(
                    "Unexpected keys in checkpoint (will be ignored): %s",
                    incompatible.unexpected_keys
                )

        if self.config.trainer.resume:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.epoch_step = int(ckpt.get("epoch_step", 0))
            self.global_step = int(ckpt.get("global_step", 0))
            self._resume_wandb_run_id = ckpt.get("wandb_run_id")

            data_state = ckpt.get("data_state")
            if isinstance(data_state, dict):
                self._sampler_epoch = int(data_state.get("sampler_epoch", 0))
                self._epoch_batches_consumed = int(data_state.get("batches_consumed", 0))
            else:
                # Legacy checkpoints incremented `epoch` as soon as an iterator
                # was created and did not save a dataloader cursor.
                saved_next_epoch = int(ckpt.get("epoch", 0))
                self._sampler_epoch = max(saved_next_epoch - 1, 0)
                self._epoch_batches_consumed = (
                    self.epoch_step * self.config.trainer.grad_accum_steps
                )
                if self.world_size > 1:
                    # Old distributed runs seeded each rank's sampler
                    # differently. Preserve that ordering for the unfinished
                    # legacy epoch, then use the corrected shared seed from the
                    # next epoch onward.
                    self._legacy_sampler_seed_epoch = self._sampler_epoch
                self.logger.warning(
                    "Legacy checkpoint: inferred current epoch %d and %d consumed "
                    "dataloader batches from epoch_step.",
                    self._sampler_epoch,
                    self._epoch_batches_consumed,
                )

            self.logger.info(
                "Resuming training from epoch %d, epoch step %d, epoch batch %d, global step %d",
                self._sampler_epoch,
                self.epoch_step,
                self._epoch_batches_consumed,
                self.global_step,
            )
            for param_group in self.optimizer.param_groups:
                self.logger.info("Learning rate: %f", param_group["lr"])

            saved_config = ckpt.get("config")
            saved_seed = None
            if isinstance(saved_config, dict) and isinstance(saved_config.get("trainer"), dict):
                saved_seed = saved_config["trainer"].get("seed")
                saved_grad_accum = saved_config["trainer"].get("grad_accum_steps")
                if (
                    saved_grad_accum is not None
                    and int(saved_grad_accum) != self.config.trainer.grad_accum_steps
                ):
                    self.logger.warning(
                        "Resume grad_accum_steps changed from %s to %d. The saved "
                        "legacy cursor and optimizer-step batch semantics may no "
                        "longer match.",
                        saved_grad_accum,
                        self.config.trainer.grad_accum_steps,
                    )
            if (
                isinstance(saved_config, dict)
                and isinstance(saved_config.get("train_dataloader"), dict)
            ):
                saved_batch_size = saved_config["train_dataloader"].get("batch_size")
                if (
                    saved_batch_size is not None
                    and int(saved_batch_size) != self.config.train_dataloader.batch_size
                ):
                    self.logger.warning(
                        "Resume global batch size changed from %s to %d; continuation "
                        "will not preserve the original optimization or data stream.",
                        saved_batch_size,
                        self.config.train_dataloader.batch_size,
                    )
            if saved_seed is not None and int(saved_seed) != self.config.trainer.seed:
                self.logger.warning(
                    "Resume seed changed from %s to %d. Keep the original seed to "
                    "continue the saved data order; changing it can repeat seen "
                    "samples and skip unseen ones.",
                    saved_seed,
                    self.config.trainer.seed,
                )

            saved_world_size = ckpt.get("world_size")
            rng_states = ckpt.get("rng_states")
            if (
                isinstance(rng_states, list)
                and len(rng_states) == self.world_size
                and (saved_world_size is None or int(saved_world_size) == self.world_size)
            ):
                self._pending_rng_state = rng_states[self.rank]
            elif rng_states is not None:
                self.logger.warning(
                    "Checkpoint RNG state is incompatible with world size %d; "
                    "resume will preserve the data cursor but is not bitwise exact.",
                    self.world_size,
                )
            else:
                self.logger.warning(
                    "Legacy checkpoint has no RNG state; resume will preserve the "
                    "data cursor but is not bitwise exact."
                )

            if self.config.ema.enabled and "ema_model" in ckpt:
                self._ema_checkpoint_state = ckpt["ema_model"]
                self._ema_checkpoint_started = bool(
                    ckpt.get(
                        "ema_started",
                        self.global_step >= self.config.ema.start_step,
                    )
                )
            elif self.config.ema.enabled:
                self._ema_checkpoint_state = None
                self._ema_checkpoint_started = (
                    self.global_step >= self.config.ema.start_step
                )
        else:
            self.logger.info("Loaded model weights for finetuning/initialization. Starting from step 0.")
            self._ema_checkpoint_state = None
