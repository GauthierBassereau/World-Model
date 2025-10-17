"""
Minimal training loop for the Stage-1 decoder used in the DiT-RAE pipeline.

The trainer keeps the DINOv3 encoder frozen, streams ImageNet samples directly
from Hugging Face, and optimises a ViT-L decoder with reconstruction and
perceptual losses. The implementation deliberately omits adversarial losses to
stay lightweight while preserving the core training recipe.
"""

from dataclasses import asdict, dataclass
import copy
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset

from PIL import Image
from torchvision import transforms
from torchvision.models import VGG16_Weights, vgg16

import yaml

from vision.dino_v3 import DinoV3Embedder, DinoVisionConfig
from vision.image_decoder import ImageDecoderConfig, VisionDecoder


def _logger() -> logging.Logger:
    logger = logging.getLogger("decoder_trainer")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _to_tuple2(value: Sequence[int] | int) -> Tuple[int, int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError(f"Expected sequence of length 2, got {value!r}.")
        return int(value[0]), int(value[1])
    value_int = int(value)
    return value_int, value_int


@dataclass
class DecoderDatasetConfig:
    dataset_name: str = "imagenet-1k"
    dataset_config: Optional[str] = None
    split: str = "train"
    image_key: str = "image"
    buffer_size: int = 2048
    min_scale: float = 0.75
    max_scale: float = 1.0
    random_flip: bool = True
    seed: int = 0

    def validate(self) -> None:
        if not 0.0 < self.min_scale <= self.max_scale <= 1.0:
            raise ValueError("Dataset scale range must satisfy 0 < min_scale <= max_scale <= 1.")
        if self.buffer_size < 0:
            raise ValueError("buffer_size must be non-negative.")


@dataclass
class DecoderDataloaderConfig:
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.num_workers != 0:
            raise ValueError("Streaming dataloader requires num_workers=0.")


@dataclass
class OptimizerConfig:
    lr: float = 2e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    eps: float = 1e-8
    grad_clip: Optional[float] = 1.0


@dataclass
class SchedulerConfig:
    enabled: bool = True
    warmup_steps: int = 2000
    total_steps: int = 200_000
    min_lr: float = 2e-5

    def validate(self) -> None:
        if self.enabled and self.total_steps <= 0:
            raise ValueError("scheduler.total_steps must be positive when enabled.")
        if self.warmup_steps < 0:
            raise ValueError("scheduler.warmup_steps must be non-negative.")


@dataclass
class TrainerLoopConfig:
    max_steps: int = 200_000
    grad_accum_steps: int = 1
    precision: str = "bf16"
    seed: int = 42
    device: Optional[str] = None
    resume_checkpoint: Optional[str] = None

    def validate(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("trainer.max_steps must be positive.")
        if self.grad_accum_steps <= 0:
            raise ValueError("trainer.grad_accum_steps must be positive.")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("trainer.precision must be one of {'fp32', 'fp16', 'bf16'}.")


@dataclass
class LoggingConfig:
    log_interval: int = 100
    checkpoint_interval: int = 5_000
    output_dir: str = "checkpoints/decoder"
    run_name: Optional[str] = None
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    sample_interval: int = 0


@dataclass
class LossConfig:
    recon_weight: float = 1.0
    perceptual_weight: float = 1.0


@dataclass
class EMAConfig:
    enabled: bool = True
    decay: float = 0.999
    device: Optional[str] = None

    def validate(self) -> None:
        if self.enabled and not 0.0 < self.decay < 1.0:
            raise ValueError("ema.decay must lie in (0, 1).")


@dataclass
class DecoderTrainingConfig:
    decoder: ImageDecoderConfig = ImageDecoderConfig()
    vision: DinoVisionConfig = DinoVisionConfig()
    dataset: DecoderDatasetConfig = DecoderDatasetConfig()
    dataloader: DecoderDataloaderConfig = DecoderDataloaderConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    trainer: TrainerLoopConfig = TrainerLoopConfig()
    logging: LoggingConfig = LoggingConfig()
    losses: LossConfig = LossConfig()
    ema: EMAConfig = EMAConfig()

    def validate(self) -> None:
        self.decoder.validate()
        self.dataset.validate()
        self.dataloader.validate()
        self.scheduler.validate()
        self.trainer.validate()
        self.ema.validate()
        if self.logging.sample_interval < 0:
            raise ValueError("logging.sample_interval must be >= 0.")


class StreamingImageDataset(IterableDataset):
    """Streams images from Hugging Face datasets with on-the-fly augmentation."""

    def __init__(
        self,
        cfg: DecoderDatasetConfig,
        image_size: Tuple[int, int],
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_size = image_size
        interpolation = transforms.InterpolationMode.BICUBIC
        target_h, target_w = image_size
        aspect_ratio = target_w / target_h
        augmentations = [
            transforms.RandomResizedCrop(
                image_size,
                scale=(cfg.min_scale, cfg.max_scale),
                ratio=(aspect_ratio, aspect_ratio),
                interpolation=interpolation,
            )
        ]
        if cfg.random_flip:
            augmentations.append(transforms.RandomHorizontalFlip())
        augmentations.append(transforms.ToTensor())
        self.transform = transforms.Compose(augmentations)
        self.epoch: int = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _iter_dataset(self):
        from datasets import load_dataset

        dataset = load_dataset(
            self.cfg.dataset_name,
            self.cfg.dataset_config,
            split=self.cfg.split,
            streaming=True,
        )
        if self.cfg.buffer_size > 0:
            dataset = dataset.shuffle(
                buffer_size=self.cfg.buffer_size,
                seed=self.cfg.seed + self.epoch,
            )
        return dataset

    def __iter__(self) -> Iterator[torch.Tensor]:
        dataset = self._iter_dataset()
        for example in dataset:
            image = example[self.cfg.image_key]
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            if image.mode != "RGB":
                image = image.convert("RGB")
            yield self.transform(image)


class PerceptualLoss(nn.Module):
    """LPIPS-style perceptual metric built from VGG16 feature maps."""

    def __init__(self) -> None:
        super().__init__()
        features = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features.eval()
        slices = [4, 9, 16, 23]
        prev = 0
        blocks = []
        for idx in slices:
            block = nn.Sequential(*[layer for layer in features[prev:idx]])
            block.eval()
            blocks.append(block)
            prev = idx
        self.blocks = nn.ModuleList(blocks)
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.eval()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.mean) / self.std
        y_norm = (y - self.mean) / self.std
        loss = x.new_zeros(())
        feat_x, feat_y = x_norm, y_norm
        for block in self.blocks:
            feat_x = block(feat_x)
            feat_y = block(feat_y)
            loss = loss + F.l1_loss(feat_x, feat_y)
        return loss


class DecoderTrainer:
    def __init__(
        self,
        config: DecoderTrainingConfig,
        decoder: Optional[VisionDecoder] = None,
        embedder: Optional[DinoV3Embedder] = None,
    ) -> None:
        self.config = config
        self.logger = _logger()
        self.device = self._resolve_device(config.trainer.device)

        self.decoder = decoder or VisionDecoder(config.decoder)
        self.decoder.to(self.device)

        self.embedder = embedder or DinoV3Embedder(config.vision)
        self.embedder.model.to(self.device)
        self.embedder.eval()
        for param in self.embedder.parameters():
            param.requires_grad_(False)

        self.optimizer = AdamW(
            self.decoder.parameters(),
            lr=config.optimizer.lr,
            betas=config.optimizer.betas,
            weight_decay=config.optimizer.weight_decay,
            eps=config.optimizer.eps,
        )
        self.scheduler = self._build_scheduler()

        self.grad_clip = config.optimizer.grad_clip
        self.loss_weights = config.losses
        self.perceptual_loss = None
        if self.loss_weights.perceptual_weight > 0.0:
            self.perceptual_loss = PerceptualLoss().to(self.device)

        self.autocast_dtype, self.scaler = self._setup_precision()

        self.ema_model: Optional[VisionDecoder]
        if config.ema.enabled:
            ema_device = torch.device(config.ema.device) if config.ema.device else self.device
            self.ema_model = copy.deepcopy(self.decoder).to(ema_device)
            self.ema_decay = config.ema.decay
        else:
            self.ema_model = None
            self.ema_decay = 0.0

        self._wandb = None
        self._wandb_run = None

        torch.manual_seed(config.trainer.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.trainer.seed)

        self._configure_wandb()

    def _resolve_device(self, requested: Optional[str]) -> torch.device:
        if requested:
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")

    def _setup_precision(self) -> Tuple[Optional[torch.dtype], Optional[GradScaler]]:
        precision = self.config.trainer.precision
        device_type = self.device.type
        if precision == "fp16":
            if device_type != "cuda":
                raise ValueError("fp16 precision is only supported on CUDA devices.")
            return torch.float16, GradScaler()
        if precision == "bf16":
            if device_type == "cuda":
                is_supported = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
                return (torch.bfloat16, None) if is_supported else (None, None)
            return torch.bfloat16, None
        return None, None

    def _build_scheduler(self) -> Optional[LambdaLR]:
        cfg = self.config.scheduler
        if not cfg.enabled:
            return None
        total_steps = max(cfg.total_steps, self.config.trainer.max_steps)

        def schedule(step: int) -> float:
            if step < cfg.warmup_steps:
                return float(step + 1) / float(max(1, cfg.warmup_steps))
            progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            min_ratio = cfg.min_lr / self.config.optimizer.lr
            return min_ratio + (1.0 - min_ratio) * cosine

        return LambdaLR(self.optimizer, schedule)

    def _update_ema(self) -> None:
        if self.ema_model is None:
            return
        ema_params = dict(self.ema_model.named_parameters())
        model_params = dict(self.decoder.named_parameters())
        for name, param in model_params.items():
            if name not in ema_params:
                continue
            ema_param = ema_params[name]
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def _save_checkpoint(self, step: int) -> None:
        output_dir = Path(self.config.logging.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "step": step,
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "config": asdict(self.config),
        }
        if self.ema_model is not None:
            state["ema"] = self.ema_model.state_dict()
        path = output_dir / f"decoder_step_{step:07d}.pt"
        torch.save(state, path)
        self.logger.info("Saved checkpoint to %s", path)

    def _load_checkpoint(self, path: str | Path) -> int:
        checkpoint = torch.load(path, map_location="cpu")
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)
        if self.scheduler is not None and checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if self.ema_model is not None and "ema" in checkpoint:
            self.ema_model.load_state_dict(checkpoint["ema"])
        step = int(checkpoint.get("step", 0))
        self.logger.info("Resumed from %s at step %d", path, step)
        return step

    def _configure_wandb(self) -> None:
        if not self.config.logging.use_wandb:
            self._wandb = None
            self._wandb_run = None
            return
        try:
            import wandb  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "logging.use_wandb=True but wandb is not installed. "
                "Install wandb or disable logging.use_wandb."
            ) from exc

        run_kwargs: Dict[str, Any] = {"config": self._wandb_config_dict()}
        if self.config.logging.wandb_project:
            run_kwargs["project"] = self.config.logging.wandb_project
        if self.config.logging.wandb_entity:
            run_kwargs["entity"] = self.config.logging.wandb_entity
        if self.config.logging.run_name:
            run_kwargs["name"] = self.config.logging.run_name
        run_kwargs["resume"] = "allow"

        self._wandb = wandb
        self._wandb_run = wandb.init(**run_kwargs)

    def _wandb_config_dict(self) -> Dict[str, Any]:
        raw = asdict(self.config)

        def convert(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(v) for v in value]
            if isinstance(value, Path):
                return str(value)
            return value

        return convert(raw)

    def _wandb_log(self, metrics: Dict[str, float], step: int) -> None:
        if self._wandb_run is None:
            return
        self._wandb_run.log(metrics, step=step)

    def _wandb_log_reconstruction(
        self,
        reference: torch.Tensor,
        reconstruction: torch.Tensor,
        step: int,
    ) -> None:
        if self._wandb_run is None:
            return
        ref = reference[0].clamp(0.0, 1.0).permute(1, 2, 0).numpy()
        recon = reconstruction[0].clamp(0.0, 1.0).permute(1, 2, 0).numpy()
        ref_uint8 = (ref * 255.0).round().astype(np.uint8)
        recon_uint8 = (recon * 255.0).round().astype(np.uint8)
        panel = np.concatenate([ref_uint8, recon_uint8], axis=1)
        caption = "Left: input | Right: reconstruction"
        self._wandb_run.log(
            {"samples/reconstruction": self._wandb.Image(panel, caption=caption)},
            step=step,
        )

    def train(self) -> None:
        self.config.validate()
        dataset = StreamingImageDataset(
            self.config.dataset,
            self.config.decoder.image_size,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.dataloader.batch_size,
            num_workers=0,
            pin_memory=self.config.dataloader.pin_memory,
            drop_last=True,
        )
        data_iter = iter(loader)
        global_step = 0
        if self.config.trainer.resume_checkpoint:
            global_step = self._load_checkpoint(self.config.trainer.resume_checkpoint)
        self.decoder.train()
        if self.ema_model is not None:
            self.ema_model.eval()

        log_interval = max(1, self.config.logging.log_interval)
        ckpt_interval = max(1, self.config.logging.checkpoint_interval)
        grad_accum = self.config.trainer.grad_accum_steps

        while global_step < self.config.trainer.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            step_metrics: Dict[str, float] = {}
            sample_images: Optional[torch.Tensor] = None
            sample_recons: Optional[torch.Tensor] = None
            for accumulate_idx in range(grad_accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    dataset.set_epoch(dataset.epoch + 1)
                    data_iter = iter(loader)
                    batch = next(data_iter)

                images = batch.to(self.device, non_blocking=True)
                with torch.no_grad():
                    tokens = self.embedder(images)
                tokens = tokens.to(self.device)

                if self.autocast_dtype is not None:
                    autocast_cm = torch.autocast(
                        device_type=self.device.type,
                        dtype=self.autocast_dtype,
                    )
                else:
                    autocast_cm = torch.autocast(device_type=self.device.type, enabled=False)

                with autocast_cm:
                    recon = self.decoder(tokens)
                    recon = recon.clamp(0.0, 1.0)
                    recon_loss = F.l1_loss(recon, images)
                    loss = self.loss_weights.recon_weight * recon_loss
                    perc_value = torch.zeros_like(loss)
                    if self.perceptual_loss is not None and self.loss_weights.perceptual_weight > 0.0:
                        perc_value = self.perceptual_loss(recon, images)
                        loss = loss + self.loss_weights.perceptual_weight * perc_value

                loss = loss / grad_accum
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if sample_images is None:
                    sample_images = images[:1].detach().cpu()
                    sample_recons = recon[:1].detach().cpu()

                step_metrics.setdefault("recon", 0.0)
                step_metrics["recon"] += recon_loss.detach().item() / grad_accum
                if self.perceptual_loss is not None:
                    step_metrics.setdefault("perceptual", 0.0)
                    step_metrics["perceptual"] += perc_value.detach().item() / grad_accum

            if self.grad_clip is not None and self.grad_clip > 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.grad_clip)

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            self._update_ema()

            total_loss = step_metrics["recon"]
            if "perceptual" in step_metrics:
                total_loss += self.loss_weights.perceptual_weight * step_metrics["perceptual"]
            step_metrics["total"] = total_loss

            if global_step % log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                metrics = {
                    "loss/total": step_metrics["total"],
                    "loss/recon": step_metrics["recon"],
                    "lr": lr,
                }
                msg_parts = [
                    f"step={global_step}",
                    f"loss={step_metrics['total']:.4f}",
                    f"recon={step_metrics['recon']:.4f}",
                    f"lr={lr:.2e}",
                ]
                if "perceptual" in step_metrics:
                    metrics["loss/perceptual"] = step_metrics["perceptual"]
                    msg_parts.append(f"perc={step_metrics['perceptual']:.4f}")
                self.logger.info(" | ".join(msg_parts))
                self._wandb_log(metrics, global_step)

            if (
                self.config.logging.sample_interval > 0
                and global_step % self.config.logging.sample_interval == 0
                and sample_images is not None
                and sample_recons is not None
            ):
                self._wandb_log_reconstruction(sample_images, sample_recons, global_step)

            if global_step > 0 and global_step % ckpt_interval == 0:
                self._save_checkpoint(global_step)

            global_step += 1

        if self._wandb_run is not None:
            self._wandb_run.finish()


def load_decoder_config(path: str | Path) -> DecoderTrainingConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    decoder_cfg = dict(raw.get("decoder", {}))
    if "image_size" in decoder_cfg:
        decoder_cfg["image_size"] = tuple(decoder_cfg["image_size"])
    if "patch_size" in decoder_cfg:
        patch_val = decoder_cfg["patch_size"]
        if isinstance(patch_val, list):
            decoder_cfg["patch_size"] = tuple(patch_val)
    if "latent_grid" in decoder_cfg and decoder_cfg["latent_grid"] is not None:
        decoder_cfg["latent_grid"] = tuple(decoder_cfg["latent_grid"])

    dataset_cfg = dict(raw.get("dataset", {}))
    dataloader_cfg = dict(raw.get("dataloader", {}))
    optimizer_cfg = dict(raw.get("optimizer", {}))
    scheduler_cfg = dict(raw.get("scheduler", {}))
    trainer_cfg = dict(raw.get("trainer", {}))
    logging_cfg = dict(raw.get("logging", {}))
    losses_cfg = dict(raw.get("losses", {}))
    ema_cfg = dict(raw.get("ema", {}))
    vision_cfg = dict(raw.get("vision", {}))

    cfg = DecoderTrainingConfig(
        decoder=ImageDecoderConfig(**decoder_cfg),
        vision=DinoVisionConfig(**vision_cfg),
        dataset=DecoderDatasetConfig(**dataset_cfg),
        dataloader=DecoderDataloaderConfig(**dataloader_cfg),
        optimizer=OptimizerConfig(**optimizer_cfg),
        scheduler=SchedulerConfig(**scheduler_cfg),
        trainer=TrainerLoopConfig(**trainer_cfg),
        logging=LoggingConfig(**logging_cfg),
        losses=LossConfig(**losses_cfg),
        ema=EMAConfig(**ema_cfg),
    )
    cfg.validate()
    return cfg
