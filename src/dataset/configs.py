from dataclasses import dataclass
from typing import Dict, Union

from .droid_dataset import DroidDatasetConfig
from .kinetics_dataset import KineticsDatasetConfig
from .openimages_dataset import OpenImagesDatasetConfig


@dataclass
class DataloaderConfig:
    batch_size: int = 2
    shuffle: bool = False
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class WorldDatasetConfig:
    datasets: Dict[str, Union[DroidDatasetConfig, KineticsDatasetConfig, OpenImagesDatasetConfig]]
    weights: Dict[str, float]
    action_dim: int = 8

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError("WorldDatasetConfig must contain at least one dataset.")
        if set(self.datasets.keys()) != set(self.weights.keys()):
            raise ValueError("Keys in datasets and weights must match.")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("Weights must be non-negative.")