import argparse
from pathlib import Path

from training.world_trainer import (
    WorldModelTrainer,
    load_training_config,
)
from world_model.transformer import WorldModelBackbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Dreamer-style world model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/world_model.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional wandb run name override.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint to resume from.",
    )
    parser.add_argument(
        "--single-batch-overfit",
        action="store_true",
        help="Reuse the very first loader batch for every step to sanity-check convergence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_training_config(args.config)
    if args.run_name:
        config.logging.run_name = args.run_name
    if args.resume:
        config.trainer.resume_checkpoint = str(args.resume)
    if args.single_batch_overfit:
        config.trainer.single_batch_overfit = True

    model = WorldModelBackbone(config.world_model)
    trainer = WorldModelTrainer(config, model)
    trainer.train()


if __name__ == "__main__":
    main()
