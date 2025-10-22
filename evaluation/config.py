from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from world_model.flow_matching import EulerSolverConfig


@dataclass
class EvaluatorConfig:
    """Configuration bundle used to build a :class:`WorldModelEvaluator`."""

    config_path: Path
    checkpoint_path: Path
    device: Optional[str] = None
    use_ema_weights: bool = True
    solver: EulerSolverConfig = field(default_factory=EulerSolverConfig)


@dataclass
class RolloutConfig:
    """
    Parameters that control a rollout call.

    Attributes:
        context: Number of frames supplied as conditioning context. These frames
            are encoded and kept fixed when sampling future latents.
        horizon: Number of autoregressive steps to sample.
        decode: Decode predicted latents back to RGB frames when True.
    """

    context: int
    horizon: int
    decode: bool = True
