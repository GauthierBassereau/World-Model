from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union
from .utils import _coerce_frame_delta

@dataclass
class DataloaderConfig:
    batch_size: int = 2
    shuffle: bool = False
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class LeRobotDatasetConfig:
    repo_id: str = "aractingi/droid_1.0.1"
    decoder_retry_attempts: int = 5
    cameras: Tuple[str, ...] = (
        "observation.images.exterior_1_left",
        "observation.images.exterior_2_left",
        "observation.images.wrist_left",
    )
    camera_probabilities: Optional[Dict[str, float]] = None
    action_keys: Tuple[str, ...] = (
        "observation.state",
    )
    action_representation: str = "delta"
    action_normalization: Optional[str] = None
    action_normalization_params: Optional[Dict[str, List[float]]] = None

    episodes: Optional[List[int]] = None
    episode_midpoint_only: bool = False
    sequence_length_distribution: Dict[int, float] = field(default_factory=lambda: {4: 1.0})
    frame_delta_seconds: Union[float, str] = 5.0 / 15.0
    independent_frames_probability: float = 0.0
    drop_action_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.decoder_retry_attempts < 1:
            raise ValueError("decoder_retry_attempts must be >= 1.")
        self.frame_delta_seconds = _coerce_frame_delta(self.frame_delta_seconds)
        if not self.sequence_length_distribution:
            raise ValueError(
                "LeRobotDatasetConfig.sequence_length_distribution must contain at least one entry."
            )
        self.sequence_length_distribution = {
            int(length): float(weight)
            for length, weight in self.sequence_length_distribution.items()
        }
        keys: Sequence[str]
        if isinstance(self.action_keys, str):
            keys = (self.action_keys,)
        else:
            keys = tuple(self.action_keys)
        if not keys:
            raise ValueError("LeRobotDatasetConfig.action_keys must contain at least one key.")
        self.action_keys = keys
        if self.action_representation not in {"delta", "position"}:
            raise ValueError(
                "LeRobotDatasetConfig.action_representation must be either 'delta' or 'position'."
            )

        if self.action_normalization not in {None, "min_max", "mean_std"}:
            raise ValueError(
                "LeRobotDatasetConfig.action_normalization must be one of None, 'min_max', or 'mean_std'."
            )
        if self.action_normalization is None:
            if self.action_normalization_params is not None:
                raise ValueError(
                    "action_normalization_params provided without a corresponding action_normalization."
                )
        else:
            if self.action_normalization_params is None:
                raise ValueError(
                    "action_normalization_params must be provided when action_normalization is set."
                )
            expected_keys = {"min", "max"} if self.action_normalization == "min_max" else {"mean", "std"}
            provided_keys = set(self.action_normalization_params.keys())
            missing = expected_keys - provided_keys
            extra = provided_keys - expected_keys
            if missing:
                raise ValueError(
                    "action_normalization_params missing required entries: " + ", ".join(sorted(missing))
                )
            if extra:
                raise ValueError(
                    "action_normalization_params contains unexpected entries: " + ", ".join(sorted(extra))
                )
            first_key = next(iter(expected_keys))
            first_value = self.action_normalization_params[first_key]
            if not isinstance(first_value, Sequence):
                raise TypeError(
                    f"action_normalization_params['{first_key}'] must be a sequence of floats."
                )
            reference_length = len(first_value)
            for key in expected_keys:
                value = self.action_normalization_params[key]
                if not isinstance(value, Sequence):
                    raise TypeError(
                        f"action_normalization_params['{key}'] must be a sequence of floats."
                    )
                if len(value) != reference_length:
                    raise ValueError(
                        "All action_normalization_params sequences must have the same length."
                    )
        self.episodes = self._normalize_episode_indices(self.episodes)

    @staticmethod
    def _normalize_episode_indices(indices: Optional[Sequence[int]]) -> Optional[List[int]]:
        if indices is None:
            return None
        normalized = [int(idx) for idx in indices]
        return normalized or None


@dataclass
class KineticsDatasetConfig:
    root: str
    frames_per_clip: int = 16
    step_between_clips: int = 1
    transform: Optional[str] = None  # Placeholder for now, could be more complex


@dataclass
class ImageNetDatasetConfig:
    root: str


@dataclass
class WorldModelDatasetConfig:
    datasets: Dict[str, Union[LeRobotDatasetConfig, KineticsDatasetConfig, ImageNetDatasetConfig]]
    weights: Dict[str, float]

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError("WorldModelDatasetConfig must contain at least one dataset.")
        if set(self.datasets.keys()) != set(self.weights.keys()):
            raise ValueError("Keys in datasets and weights must match.")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("Weights must be non-negative.")

