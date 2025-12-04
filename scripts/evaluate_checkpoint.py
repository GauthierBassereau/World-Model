import datetime
import torch
import torch.distributed as dist
import pyrallis
from dataclasses import dataclass, field
from src.training.world_evaluator import WorldModelEvaluator, EvaluationConfig
from src.training.logger import WorldModelLogger, LoggingConfig
from src.world_model.backbone import WorldModelBackbone, WorldModelConfig
from src.rae_dino.rae import RAE
from src.dataset.world_dataset import WorldDatasetConfig
from src.dataset.loader import DataloaderConfig
from src.diffusion.signal_scheduler import SignalSchedulerConfig
from src.diffusion.euler_solver import EulerSolverConfig
from src.training.utils import set_seed

@dataclass
class EvaluationScriptConfig:
    checkpoint_path: str
    evaluator: EvaluationConfig = field(default_factory=EvaluationConfig)
    eval_dataset: WorldDatasetConfig = field(default_factory=lambda: WorldDatasetConfig(datasets={}, weights={}))
    eval_dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    signal_scheduler: SignalSchedulerConfig = field(default_factory=SignalSchedulerConfig)
    euler_solver: EulerSolverConfig = field(default_factory=EulerSolverConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 1234

def main() -> None:
    config = pyrallis.parse(config_class=EvaluationScriptConfig)

    if torch.cuda.is_available():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device_idx = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_idx)
        device = torch.device("cuda", device_idx)
    else:
        dist.init_process_group(backend="gloo", timeout=datetime.timedelta(minutes=30))
        world_size = 1
        rank = 0
        device = torch.device("cpu")

    is_main_process = rank == 0
    seed = set_seed(config.seed, world_size, rank)

    logger = WorldModelLogger(config.logging, is_main_process=is_main_process)
    logger.init_wandb(pyrallis.encode(config))

    logger.info(f"Initialized process {rank}/{world_size} on device {device}")

    logger.info("Initializing World Model...")
    model = WorldModelBackbone(config.world_model)
    model.to(device)
    
    logger.info(f"Loading checkpoint from {config.checkpoint_path}...")
    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
        
    if "ema_model" in checkpoint:
        logger.info("Found EMA model in checkpoint, loading it for evaluation.")
        state_dict = checkpoint["ema_model"]

    # Remove _orig_mod prefix from keys, from torch compile
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.eval()
    
    logger.info("Initializing Autoencoder...")
    autoencoder = RAE()
    autoencoder.to(device)
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad_(False)

    logger.info("Initializing Evaluator...")
    evaluator = WorldModelEvaluator(
        config=config.evaluator,
        dataset_cfg=config.eval_dataset,
        dataloader_cfg=config.eval_dataloader,
        signal_scheduler_cfg=config.signal_scheduler,
        euler_solver_cfg=config.euler_solver,
        autoencoder=autoencoder,
        logger=logger,
        device=device,
        seed=seed,
        rank=rank,
        world_size=world_size,
        is_main_process=is_main_process,
    )

    logger.info("Starting Evaluation...")
    evaluator.evaluate(model)
    logger.info("Evaluation Completed.")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
