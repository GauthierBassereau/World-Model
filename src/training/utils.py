import random
import torch
import torch.distributed as dist
from typing import Dict

def set_seed(seed: int, world_size: int, rank: int) -> int:
    """
    Sets the random seed for reproducibility, adjusting for distributed training.
    Returns the effective seed used.
    """
    effective_seed = seed * world_size + rank
    random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    return effective_seed

def sync_metrics(metrics: Dict[str, float], world_size: int, device: torch.device) -> Dict[str, float]:
    """
    Synchronizes metrics across all processes by averaging them.
    """
    if world_size == 1 or not metrics:
        return metrics
    
    keys = sorted(metrics.keys())
    values = torch.tensor(
        [float(metrics[key]) for key in keys],
        device=device,
        dtype=torch.float32,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= world_size
    return {key: float(value) for key, value in zip(keys, values.tolist())}
