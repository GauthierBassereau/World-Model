"""
CLI entry point for training the Stage-1 image decoder.

Usage:
    python scripts/train_decoder.py --config configs/decoder.yaml
"""

import argparse
import logging
from pathlib import Path

from training.decoder_trainer import DecoderTrainer, load_decoder_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ViT decoder on streamed ImageNet samples.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/decoder.yaml"),
        help="Path to the YAML config describing the decoder training run.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint to resume from.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override passed through to the trainer (e.g., 'cuda', 'mps').",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["fp32", "fp16", "bf16"],
        help="Optional precision override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config = load_decoder_config(args.config)
    if args.resume:
        config.trainer.resume_checkpoint = str(args.resume)
    if args.device:
        config.trainer.device = args.device
    if args.precision:
        config.trainer.precision = args.precision

    trainer = DecoderTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
