from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Dict

from src.dataset.configs import DataloaderConfig, DatasetConfig
from src.training.world_evaluator import EvaluationConfig
from src.world_model.diffusion import DiffusionConfig, EulerSolverConfig
from src.world_model.backbone import WorldModelConfig


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
class TrainDataConfig:
    train_dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train_dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)


@dataclass
class EvalDataConfig:
    eval_dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(episode_midpoint_only=True)
    )
    eval_dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)


@dataclass
class WorldModelTrainingConfig:
    train_data: TrainDataConfig = field(default_factory=TrainDataConfig)
    eval_data: EvalDataConfig = field(default_factory=EvalDataConfig)
    evaluator: EvaluationConfig = field(default_factory=EvaluationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer: TrainerLoopConfig = field(default_factory=TrainerLoopConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    ode_solver: EulerSolverConfig = field(default_factory=EulerSolverConfig)
