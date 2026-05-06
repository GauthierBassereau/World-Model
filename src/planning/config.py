from dataclasses import dataclass, field
from typing import List, Optional

from src.dataset.world_dataset import WorldDatasetConfig
from src.diffusion.euler_solver import EulerSolverConfig
from src.diffusion.signal_scheduler import SignalSchedulerConfig
from src.training.logger import LoggingConfig
from src.world_model.backbone import WorldModelConfig


@dataclass
class CEMConfig:
    iterations: int = 8
    population_size: int = 128
    elite_fraction: float = 0.1
    horizon: Optional[int] = None
    rollout_batch_size: int = 32
    initial_delta_std: List[float] = field(
        default_factory=lambda: [0.025, 0.025, 0.025, 0.07, 0.07, 0.07, 10.0]
    )
    min_delta_std: List[float] = field(
        default_factory=lambda: [0.0025, 0.0025, 0.0025, 0.007, 0.007, 0.007, 1.0]
    )
    max_delta_abs: List[float] = field(
        default_factory=lambda: [0.08, 0.08, 0.08, 0.25, 0.25, 0.25, 50.0]
    )
    mean_momentum: float = 0.1
    std_momentum: float = 0.1
    action_l2_weight: float = 0.01
    include_zero_sequence: bool = True


@dataclass
class MPCConfig:
    steps: Optional[int] = None
    execute_steps: int = 1
    shrink_horizon_to_goal: bool = True


@dataclass
class PlanningConfig:
    mode: str = "classic_cem"
    episode_index: int = 9
    start_time_s: float = 0.0
    goal_time_s: float = 9.0
    context_frames: int = 1
    rollout_signal_level: float = 0.6
    precision: str = "bf16"
    compile_model: bool = False
    max_context_frames: Optional[int] = None
    output_dir: str = "planning_outputs"
    seed: int = 1234
    mpc: MPCConfig = field(default_factory=MPCConfig)


@dataclass
class VisualizationConfig:
    log_iterations: List[int] = field(default_factory=lambda: [0, 2, 5, 7])
    decode_best_samples: bool = True
    save_real_video: bool = True
    save_goal_image: bool = True
    save_trajectory_plots: bool = True
    save_planned_video: bool = True
    fps: int = 5


@dataclass
class PlanningScriptConfig:
    checkpoint_path: str = "checkpoints/training_ur5_400M_res/world_model_step_013000.pt"
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    cem: CEMConfig = field(default_factory=CEMConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    eval_dataset: WorldDatasetConfig = field(default_factory=lambda: WorldDatasetConfig(datasets={}, weights={}))
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    signal_scheduler: SignalSchedulerConfig = field(default_factory=SignalSchedulerConfig)
    euler_solver: EulerSolverConfig = field(default_factory=EulerSolverConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

