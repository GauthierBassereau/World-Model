#!/usr/bin/env python3
import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from evaluation import EvaluatorConfig, WorldModelEvaluator
from evaluation.interactive import InteractiveWorldModelUI
from training.dataset import build_world_model_dataloader


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    ui_path = Path(path)
    if not ui_path.exists():
        raise FileNotFoundError(f"UI config file not found: {ui_path}")
    with open(ui_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {ui_path}, found {type(data).__name__}.")
    return data


def _resolve_arg(value, config_dict: Dict[str, Any], key: str, default=None):
    if value is not None:
        return value
    if key in config_dict and config_dict[key] is not None:
        return config_dict[key]
    return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch an interactive Dreamer world model UI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ui-config",
        type=str,
        default=None,
        help="Optional YAML file with defaults (see configs/world_model_ui.yaml).",
    )
    parser.add_argument(
        "--training-config",
        type=str,
        default=None,
        help="Path to the training YAML used to build the model and dataloader.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path containing model weights.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (e.g. cpu, cuda, cuda:1).",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Disable loading EMA weights even if available.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Zero-based index of the batch to draw from the dataset.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Number of frames to keep as conditioning context (default: full sequence).",
    )
    parser.add_argument(
        "--disable-actions",
        action="store_true",
        help="Ignore dataset actions when seeding the UI.",
    )
    parser.add_argument(
        "--translation-step",
        type=float,
        default=None,
        help="Initial translation step applied to key bindings.",
    )
    parser.add_argument(
        "--rotation-step",
        type=float,
        default=None,
        help="Initial rotation step applied to key bindings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_dict = _load_yaml(args.ui_config)
    evaluator_cfg_dict = cfg_dict.get("evaluator", {})
    ui_cfg_dict = cfg_dict.get("ui", {})

    training_config_path = _resolve_arg(
        args.training_config,
        evaluator_cfg_dict,
        "config_path",
        default="configs/pretraining.yaml",
    )
    checkpoint_path = _resolve_arg(args.checkpoint, evaluator_cfg_dict, "checkpoint_path")
    if checkpoint_path is None:
        raise ValueError("A checkpoint path must be specified via --checkpoint or the UI config.")
    device = _resolve_arg(args.device, evaluator_cfg_dict, "device")

    use_ema = evaluator_cfg_dict.get("use_ema_weights", True)
    if args.no_ema:
        use_ema = False

    evaluator = WorldModelEvaluator.from_config(
        EvaluatorConfig(
            config_path=training_config_path,
            checkpoint_path=checkpoint_path,
            device=device,
            use_ema_weights=use_ema,
        )
    )

    # Prepare dataloader with batch_size=1 for deterministic sampling.
    dataloader_cfg = replace(evaluator.training_cfg.dataloader)
    dataloader_cfg.batch_size = 1
    dataloader_cfg.shuffle = False
    dataloader_cfg.num_workers = max(0, dataloader_cfg.num_workers)

    dataloader = build_world_model_dataloader(
        dataset_cfg=evaluator.training_cfg.dataset,
        dataloader_cfg=dataloader_cfg,
        device=evaluator.device,
    )

    target_index = _resolve_arg(args.sample_index, ui_cfg_dict, "sample_index", default=0)
    if target_index < 0:
        raise ValueError("sample_index must be non-negative.")

    iterator = iter(dataloader)
    batch = None
    for _ in range(target_index + 1):
        try:
            batch = next(iterator)
        except StopIteration as exc:
            raise IndexError(
                f"Requested sample_index {target_index} but the dataloader ran out of batches."
            ) from exc
    if batch is None:
        raise RuntimeError("Failed to fetch a batch from the dataloader.")

    context_frames = batch.sequence_frames.cpu()
    context_actions = batch.sequence_actions.cpu() if batch.sequence_actions is not None else None
    context_mask = batch.actions_mask.cpu() if batch.actions_mask is not None else None

    context_length = _resolve_arg(args.context_length, ui_cfg_dict, "context_length")
    if context_length is not None:
        if context_length <= 0:
            raise ValueError("context_length must be positive when provided.")
        if context_length > context_frames.shape[1]:
            raise ValueError(
                f"context_length {context_length} exceeds available frames {context_frames.shape[1]}."
            )
        context_frames = context_frames[:, :context_length, ...]
        if context_actions is not None:
            context_actions = context_actions[:, :context_length, ...]
        if context_mask is not None:
            context_mask = context_mask[:, :context_length]

    disable_actions = args.disable_actions or bool(ui_cfg_dict.get("disable_actions", False))
    if disable_actions:
        context_actions = None
        context_mask = None

    translation_step = _resolve_arg(
        args.translation_step, ui_cfg_dict, "translation_step", default=0.02
    )
    rotation_step = _resolve_arg(args.rotation_step, ui_cfg_dict, "rotation_step", default=0.05)

    ui = InteractiveWorldModelUI(evaluator)
    ui.translation_step = float(translation_step)
    ui.rotation_step = float(rotation_step)
    ui._build_default_bindings()

    ui.load_context(
        frames=context_frames,
        actions=context_actions,
        actions_mask=context_mask,
    )

    print("Loaded context tensor with shape:", tuple(context_frames.shape))
    if context_actions is not None:
        print("Actions included with shape:", tuple(context_actions.shape))
    else:
        print("Actions disabled for this session.")

    ui.run()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
