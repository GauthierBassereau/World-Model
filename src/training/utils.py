import random
from typing import Dict, Tuple

import torch
import torch.distributed as dist

def set_seed(seed: int, world_size: int, rank: int) -> int:
    effective_seed = seed * world_size + rank
    random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    return effective_seed

def sync_metrics(metrics: Dict[str, float], world_size: int, device: torch.device) -> Dict[str, float]:
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


def sync_metric_stats(
    metric_stats: Dict[str, Tuple[float, float]],
    world_size: int,
    device: torch.device,
) -> Dict[str, Tuple[float, float]]:
    if world_size == 1 or not metric_stats:
        return metric_stats

    keys = sorted(metric_stats.keys())
    values = torch.tensor(
        [[float(metric_stats[key][0]), float(metric_stats[key][1])] for key in keys],
        device=device,
        dtype=torch.float32,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {
        key: (float(values[idx, 0]), float(values[idx, 1]))
        for idx, key in enumerate(keys)
    }
