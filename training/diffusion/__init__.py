"""
Utilities for diffusion schedules and samplers used by the world model trainer.

The helpers in this package keep the training loop lightweight while leaving room for future
inference solvers (e.g., Euler) and alternative schedules.
"""

from .config import DiffusionConfig, DiffusionDebugConfig
from .noise import sample_base_noise
from .scheduler import DimensionShiftedUniformScheduler

__all__ = [
    "DiffusionConfig",
    "DiffusionDebugConfig",
    "DimensionShiftedUniformScheduler",
    "sample_base_noise",
]
