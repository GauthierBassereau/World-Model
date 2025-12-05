import copy
import datetime
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from src.training.logger import LoggingConfig

from src.dataset.loader import build_world_dataloader, DataloaderConfig
from src.dataset.world_dataset import WorldDataset, WorldBatch, WorldDatasetConfig

from src.world_model.backbone import WorldModelConfig

from src.diffusion.common import calculate_velocity_1_to_2
from src.diffusion.euler_solver import EulerSolverConfig
from src.diffusion.signal_scheduler import SignalScheduler, SignalSchedulerConfig

from src.training.logger import WorldModelLogger
from src.training.utils import set_seed, sync_metrics
from src.training.world_evaluator import WorldModelEvaluator, EvaluationConfig


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
class EMAConfig:
    enabled: bool = False
    decay: float = 0.999
    start_step: int = 0


@dataclass
class WorldModelTrainingConfig:
    train_dataset: WorldDatasetConfig = field(default_factory=lambda: WorldDatasetConfig(datasets={}, weights={}))
    train_dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    eval_dataset: WorldDatasetConfig = field(default_factory=lambda: WorldDatasetConfig(datasets={}, weights={}))
    eval_dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    trainer: TrainerLoopConfig = field(default_factory=TrainerLoopConfig)
    evaluator: EvaluationConfig = field(default_factory=EvaluationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    signal_scheduler: SignalSchedulerConfig = field(default_factory=SignalSchedulerConfig)
    euler_solver: EulerSolverConfig = field(default_factory=EulerSolverConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


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
        self.global_step = 0
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
        
        seed = set_seed(config.trainer.seed, self.world_size, self.rank)

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
            seed=seed,
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
            seed=seed,
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

        self._maybe_load_checkpoint()
        if config.ema.enabled:
            self._init_ema_model()

        self.logger.init_wandb(asdict(self.config))
        # self.logger.log_config(asdict(config))

        base_dir = Path(config.logging.output_dir)
        wandb_run = getattr(self.logger, "wandb_run", "no_name")
        if self.is_main_process:
            self._checkpoint_dir = base_dir / wandb_run.name
        else:
            self._checkpoint_dir = None


    def _create_optimizer(self, initial_lr: float) -> torch.optim.Optimizer:
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias"):
                no_decay_params.append(param)
                continue
            if param.ndim <= 1: # for RMSNorm
                no_decay_params.append(param)
                continue
            decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.optimizer.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(
            optim_groups,
            lr=initial_lr,
            betas=self.config.optimizer.betas,
            eps=self.config.optimizer.eps,
        )

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
        if step < self.config.ema.start_step:
            return

        cur_decay = self.config.ema.decay

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
        self.logger.info(
            "Starting world model training on %s (resuming at epoch %d, epoch step %d, global step %d)",
            self.device,
            self._sampler_epoch,
            self.epoch_step,
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

            if self.device.type == "cuda" and self.global_step == 1:
                peak_mem_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
                print(f"[GPU {self.device.index}] global step {self.global_step} peak memory: {peak_mem_gb:.2f} GB", flush=True)
            mean_metrics = {
                key: value / self.config.trainer.grad_accum_steps for key, value in accum_metrics.items()
            }
            lr_value = float(self.optimizer.param_groups[0]["lr"])
            mean_metrics["learning_rate"] = lr_value
            if grad_metrics:
                mean_metrics.update(grad_metrics)
            mean_metrics = sync_metrics(mean_metrics, self.world_size, self.device)
            self.logger.log_training_metrics(mean_metrics)
            self._maybe_run_evaluation(self.global_step)

            if (
                self.config.logging.checkpoint_interval
                and self.global_step % self.config.logging.checkpoint_interval == 0
            ):
                self._save_checkpoint(self.global_step)

    def _train_micro_step(
        self,
        batch: WorldBatch,
    ) -> Dict[str, float]:
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
        
        signal_levels, scheduler_steps = self.signal_scheduler.sample_with_base(latents)
        self.logger.log_distr_signal(signal_levels)
        base_noise = torch.randn_like(latents)
        signal_levels_expanded = signal_levels.unsqueeze(-1).unsqueeze(-1)
        
        noisy_latents = (1.0 - signal_levels_expanded) * base_noise + signal_levels_expanded * latents

        with self._autocast_scope():
            outputs = self._train_module(
                noisy_latents,
                signal_levels=signal_levels,
                actions=actions,
                independent_frames=independent_frames,
                use_actions=use_actions,
            )

            loss, metrics = self._compute_loss(
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

        loss.backward()

        return metrics

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
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self.config.trainer.loss_type == "position":
            loss_unreduced = torch.nn.functional.mse_loss(latents, outputs.latents, reduction="none")
        else:
            v_true = latents - base_noise
            v_pred = calculate_velocity_1_to_2(
                latent_1=noisy_latents,
                latent_2=outputs.latents,
                signal_levels_1=signal_levels,
                signal_levels_2=torch.ones_like(signal_levels),
                min_denom=0.05,
            )
            loss_unreduced = torch.nn.functional.mse_loss(v_pred, v_true, reduction="none")

        if self.config.trainer.loss_weighting in ("linear", "linear_signal"):
            # Weight scales linearly with signal level (after scheduler transformation)
            weights = self.config.trainer.loss_weighting_intercept + self.config.trainer.loss_weighting_slope * signal_levels
            weights = weights.unsqueeze(-1).unsqueeze(-1)
            loss_unreduced = loss_unreduced * weights
        elif self.config.trainer.loss_weighting == "linear_scheduler":
            # Weight scales linearly with scheduler step (before transformation)
            # This is useful when using dimension shift, so weights scale uniformly with the original uniform distribution
            weights = self.config.trainer.loss_weighting_intercept + self.config.trainer.loss_weighting_slope * scheduler_steps
            weights = weights.unsqueeze(-1).unsqueeze(-1)
            loss_unreduced = loss_unreduced * weights

        # Apply valid frame mask, some sequences may have padding frames which should not contribute to loss
        frames_valid_mask = frames_valid_mask.to(dtype=loss_unreduced.dtype)
        frame_loss = loss_unreduced.mean(dim=(-1, -2))
        valid_frame_loss = frame_loss * frames_valid_mask
        denom = frames_valid_mask.sum().clamp_min(1.0)
        loss = valid_frame_loss.sum() / denom
        scaled_loss = loss / self.config.trainer.grad_accum_steps
        
        metrics = {
            "l2_loss": float(loss),
        }

        if self.global_step % 10 == 0:
            metrics.update(self._compute_loss_breakdown(
                valid_frame_loss,
                frames_valid_mask,
                independent_frames,
                dataset_indices,
                dataset_names,
                use_actions,
            ))
            
        return scaled_loss, metrics

    def _compute_loss_breakdown(
        self,
        valid_frame_loss: torch.Tensor,
        frames_valid_mask: torch.Tensor,
        independent_frames: torch.Tensor,
        dataset_indices: torch.Tensor,
        dataset_names: Dict[int, str],
        use_actions: torch.Tensor,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        
        indep_mask_bool = independent_frames.to(dtype=torch.bool)
        mask_indep = frames_valid_mask * indep_mask_bool.to(dtype=frames_valid_mask.dtype)
        mask_dept = frames_valid_mask * (~indep_mask_bool).to(dtype=frames_valid_mask.dtype)

        denom_indep = mask_indep.sum()
        if denom_indep > 0:
            loss_indep = (valid_frame_loss * mask_indep).sum() / denom_indep
            metrics["l2_loss/independent_frames"] = float(loss_indep)
        
        denom_dept = mask_dept.sum()
        if denom_dept > 0:
            loss_dept = (valid_frame_loss * mask_dept).sum() / denom_dept
            metrics["l2_loss/dependent_frames"] = float(loss_dept)
            
        action_mask_bool = use_actions.to(dtype=torch.bool)
        mask_action = frames_valid_mask * action_mask_bool.to(dtype=frames_valid_mask.dtype)
        mask_no_action = frames_valid_mask * (~action_mask_bool).to(dtype=frames_valid_mask.dtype)
        
        denom_action = mask_action.sum()
        if denom_action > 0:
            loss_action = (valid_frame_loss * mask_action).sum() / denom_action
            metrics["l2_loss/with_actions"] = float(loss_action)
            
        denom_no_action = mask_no_action.sum()
        if denom_no_action > 0:
            loss_no_action = (valid_frame_loss * mask_no_action).sum() / denom_no_action
            metrics["l2_loss/without_actions"] = float(loss_no_action)
        
        num_datasets = max(dataset_names.keys()) + 1
        
        loss_sum_b = valid_frame_loss.sum(dim=1)
        denom_sum_b = frames_valid_mask.sum(dim=1)
        
        loss_indep_b = (valid_frame_loss * mask_indep).sum(dim=1)
        denom_indep_b = mask_indep.sum(dim=1)
        
        loss_dept_b = (valid_frame_loss * mask_dept).sum(dim=1)
        denom_dept_b = mask_dept.sum(dim=1)
        
        loss_action_b = (valid_frame_loss * mask_action).sum(dim=1)
        denom_action_b = mask_action.sum(dim=1)
        
        loss_no_action_b = (valid_frame_loss * mask_no_action).sum(dim=1)
        denom_no_action_b = mask_no_action.sum(dim=1)
        
        ds_loss_total = torch.zeros(num_datasets, device=valid_frame_loss.device, dtype=valid_frame_loss.dtype)
        ds_denom_total = torch.zeros(num_datasets, device=valid_frame_loss.device, dtype=valid_frame_loss.dtype)
        
        ds_loss_indep = torch.zeros(num_datasets, device=valid_frame_loss.device, dtype=valid_frame_loss.dtype)
        ds_denom_indep = torch.zeros(num_datasets, device=valid_frame_loss.device, dtype=valid_frame_loss.dtype)
        
        ds_loss_dept = torch.zeros(num_datasets, device=valid_frame_loss.device, dtype=valid_frame_loss.dtype)
        ds_denom_dept = torch.zeros(num_datasets, device=valid_frame_loss.device, dtype=valid_frame_loss.dtype)
        
        ds_loss_action = torch.zeros(num_datasets, device=valid_frame_loss.device, dtype=valid_frame_loss.dtype)
        ds_denom_action = torch.zeros(num_datasets, device=valid_frame_loss.device, dtype=valid_frame_loss.dtype)
        
        ds_loss_no_action = torch.zeros(num_datasets, device=valid_frame_loss.device, dtype=valid_frame_loss.dtype)
        ds_denom_no_action = torch.zeros(num_datasets, device=valid_frame_loss.device, dtype=valid_frame_loss.dtype)
        
        ds_loss_total.scatter_add_(0, dataset_indices, loss_sum_b)
        ds_denom_total.scatter_add_(0, dataset_indices, denom_sum_b)
        
        ds_loss_indep.scatter_add_(0, dataset_indices, loss_indep_b)
        ds_denom_indep.scatter_add_(0, dataset_indices, denom_indep_b)
        
        ds_loss_dept.scatter_add_(0, dataset_indices, loss_dept_b)
        ds_denom_dept.scatter_add_(0, dataset_indices, denom_dept_b)
        
        ds_loss_action.scatter_add_(0, dataset_indices, loss_action_b)
        ds_denom_action.scatter_add_(0, dataset_indices, denom_action_b)
        
        ds_loss_no_action.scatter_add_(0, dataset_indices, loss_no_action_b)
        ds_denom_no_action.scatter_add_(0, dataset_indices, denom_no_action_b)
        
        for idx, name in dataset_names.items():
            d_total = ds_denom_total[idx]
            if d_total > 0:
                metrics[f"l2_loss/{name}"] = float(ds_loss_total[idx] / d_total)
                
            d_indep = ds_denom_indep[idx]
            if d_indep > 0:
                metrics[f"l2_loss/{name}/independent"] = float(ds_loss_indep[idx] / d_indep)
                
            d_dept = ds_denom_dept[idx]
            if d_dept > 0:
                metrics[f"l2_loss/{name}/dependent"] = float(ds_loss_dept[idx] / d_dept)
            
            d_action = ds_denom_action[idx]
            if d_action > 0:
                metrics[f"l2_loss/{name}/with_actions"] = float(ds_loss_action[idx] / d_action)
                
            d_no_action = ds_denom_no_action[idx]
            if d_no_action > 0:
                metrics[f"l2_loss/{name}/without_actions"] = float(ds_loss_no_action[idx] / d_no_action)
                    
        return metrics

    def _next_batch(self) -> WorldBatch:
        if self._dataloader_iter is None:
            self._dataloader_iter = self._create_data_iter()
        try:
            return next(self._dataloader_iter)
        except StopIteration:
            if self.is_main_process:
                self.logger.info(f"Epoch {self._sampler_epoch} finished at step {self.epoch_step}. Starting epoch {self._sampler_epoch + 1}.")
            self.epoch_step = 0
            self._dataloader_iter = self._create_data_iter()
            return next(self._dataloader_iter)

    def _create_data_iter(self):
        sampler = getattr(self.dataloader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self._sampler_epoch)
        self.dataloader.dataset.reshuffle_virtual_map(self.config.trainer.seed + self._sampler_epoch)
        self._sampler_epoch += 1
        return iter(self.dataloader)

    def _maybe_run_evaluation(self, step: int) -> None:
        interval = self.config.trainer.evaluation_interval
        if interval is None or interval <= 0 or step % interval != 0:
            return
        eval_model = self.ema_model if self.config.ema.enabled else self.model
        self.evaluator.evaluate(eval_model)

    def _save_checkpoint(self, step: int) -> None:
        if not self.is_main_process:
            return
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self._checkpoint_dir / f"world_model_step_{step:06d}.pt"
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self._sampler_epoch,
            "epoch_step": self.epoch_step,
            "global_step": self.global_step,
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

    def _maybe_load_checkpoint(self) -> None:
        checkpoint_path = self.config.trainer.load_checkpoint
        if not checkpoint_path:
            return
        
        self.logger.info("Loading checkpoint from %s...", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])

        if self.config.trainer.resume:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self._sampler_epoch = int(ckpt.get("epoch", 0))
            self.epoch_step = int(ckpt.get("epoch_step", 0))
            self.global_step = int(ckpt.get("global_step", 0))
            self.logger.info("Resuming training from epoch %d, epoch step %d, global step %d", 
                           self._sampler_epoch, self.epoch_step, self.global_step)
            for param_group in self.optimizer.param_groups:
                self.logger.info("Learning rate: %f", param_group["lr"])

            self.logger.warning("Be sure to use a different seed to avoid seeing same data")

            if "ema_model" in ckpt:
                self._ema_checkpoint_state = ckpt["ema_model"]
            elif self.config.ema.enabled:
                self._ema_checkpoint_state = None
        else:
            self.logger.info("Loaded model weights for finetuning/initialization. Starting from step 0.")
            self._ema_checkpoint_state = None
