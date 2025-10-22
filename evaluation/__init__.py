"""
Evaluation utilities for Dreamer-style world model inference.

This package provides tools to load a trained checkpoint, run diffusion-based
rollouts, evaluate sequences against datasets, and drive interactive demos.
"""

from .config import (
    EvaluatorConfig,
    RolloutConfig,
)
from .dataset_runner import DatasetEvaluator
from .evaluator import WorldModelEvaluator

__all__ = [
    "EvaluatorConfig",
    "RolloutConfig",
    "DatasetEvaluator",
    "WorldModelEvaluator",
]
