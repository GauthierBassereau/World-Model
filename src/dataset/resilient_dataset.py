import logging
import random
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)

class ResilientLeRobotDataset(LeRobotDataset):
    """
    A wrapper around LeRobotDataset that handles decoding errors by 
    resampling a random index from the dataset.
    """
    def __init__(
        self,
        *args,
        max_decode_failures: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_decode_failures = max_decode_failures

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        attempts = 0
        current_index = index
        
        while attempts < self.max_decode_failures:
            try:
                return super().__getitem__(current_index)
            
            except (RuntimeError, OSError, ValueError) as e:
                # We catch RuntimeError (torchcodec), OSError (file corruption), 
                # and ValueError (sometimes generic decoding issues)
                attempts += 1
                
                # Pick a new random index from the total available frames
                new_index = random.randint(0, len(self) - 1)
                
                # Log a warning so we aren't training on garbage silently
                logger.warning(
                    f"Data loading failed at index {current_index}. "
                    f"Error: {e}. "
                    f"Retry {attempts}/{self.max_decode_failures} with new index {new_index}."
                )
                
                current_index = new_index

        # If we exhaust retries, raise the last error
        raise RuntimeError(
            f"Failed to fetch a valid sample after {self.max_retries} retries. "
            "The dataset might be heavily corrupted or the decoder is incompatible."
        )
