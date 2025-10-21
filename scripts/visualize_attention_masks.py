"""Standalone helper to visualise the spatial and temporal attention masks.

This mirrors the mask-building logic in `world_model.backbone.WorldModelBackbone`
and the way masks are consumed inside the attention modules. It never touches
the model code, letting you experiment safely.
"""

import argparse
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import torch


def build_spatial_mask(latent_tokens: int, num_registers: int, device: torch.device) -> torch.Tensor:
    tokens_per_frame = latent_tokens + 2 + num_registers
    mask = torch.zeros(tokens_per_frame, tokens_per_frame, dtype=torch.bool, device=device)
    reg_end = num_registers
    noise_idx = num_registers
    action_idx = num_registers + 1
    latent_start = num_registers + 2

    mask[latent_start:, latent_start:] = True
    mask[latent_start:, noise_idx] = True
    mask[latent_start:, action_idx] = True
    mask[latent_start:, :reg_end] = True

    mask[noise_idx, noise_idx] = True
    mask[action_idx, action_idx] = True

    mask[:reg_end, latent_start:] = True
    mask[:reg_end, noise_idx] = True
    mask[:reg_end, action_idx] = True
    mask[:reg_end, :reg_end] = True

    return mask


def build_temporal_mask(
    time_steps: int,
    context_length: Optional[int],
    independent_frame_mask: Optional[torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.tril(
        torch.ones(time_steps, time_steps, dtype=torch.bool, device=device),
        diagonal=0,
    )
    if context_length is not None and context_length < time_steps:
        context_band = torch.triu(mask, diagonal=-(context_length - 1))
        mask = mask & context_band
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, time_steps, time_steps).clone()

    if independent_frame_mask is not None:
        independent = independent_frame_mask.to(device=device, dtype=torch.bool).view(batch_size, 1, 1, 1)
        identity_mask = torch.eye(time_steps, dtype=torch.bool, device=device).view(1, 1, time_steps, time_steps)
        mask = torch.where(independent, identity_mask, mask)

    return mask


def attention_mask_after_application_spatial(mask_2d: torch.Tensor) -> torch.Tensor:
    # Mirrors the broadcasting in SpatialAttention.forward.
    mask_bool = mask_2d.to(dtype=torch.bool).view(1, 1, mask_2d.shape[0], mask_2d.shape[1])
    return mask_bool[0, 0]


def attention_mask_after_application_temporal(mask_4d: torch.Tensor, batch_index: int) -> torch.Tensor:
    # Mirrors the broadcasting in TemporalAttention.forward for one batch element.
    mask_bool = mask_4d.to(dtype=torch.bool).unsqueeze(2)
    return mask_bool[batch_index, 0, 0]


def parse_independent_indices(raw: Iterable[str], batch_size: int) -> torch.Tensor:
    mask = torch.zeros(batch_size, 1)
    for item in raw:
        idx = int(item)
        if idx < 0 or idx >= batch_size:
            raise ValueError(f"Independent index {idx} outside [0, {batch_size - 1}]")
        mask[idx, 0] = 1.0
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise spatial and temporal attention masks.")
    parser.add_argument("--latent-tokens", type=int, default=8, help="Number of latent tokens per frame.")
    parser.add_argument("--num-registers", type=int, default=4, help="Register token count.")
    parser.add_argument("--time-steps", type=int, default=5, help="Sequence length for the temporal mask.")
    parser.add_argument("--context-length", type=int, default=3, help="Temporal context window (set <=0 for full).")
    parser.add_argument("--batch-size", type=int, default=3, help="Batch size for temporal mask visualisation.")
    parser.add_argument(
        "--independent-indices",
        nargs="*",
        default=(1,2),
        help="Batch indices flagged as independent frames (no temporal attention).",
    )
    parser.add_argument("--device", default="cpu", help="Device used for torch tensors.")
    args = parser.parse_args()

    device = torch.device(args.device)
    context_length = args.context_length if args.context_length > 0 else None
    independent_mask = None
    if args.independent_indices:
        independent_mask = parse_independent_indices(args.independent_indices, args.batch_size)

    spatial_mask_2d = build_spatial_mask(args.latent_tokens, args.num_registers, device=device)
    temporal_mask_4d = build_temporal_mask(
        time_steps=args.time_steps,
        context_length=context_length,
        independent_frame_mask=independent_mask,
        batch_size=args.batch_size,
        device=device,
    )

    spatial_applied = attention_mask_after_application_spatial(spatial_mask_2d)
    temporal_applied = [
        attention_mask_after_application_temporal(temporal_mask_4d, batch_index=i) for i in range(args.batch_size)
    ]

    fig_spatial, ax_spatial = plt.subplots(1, 1, figsize=(5, 5))
    ax_spatial.imshow(spatial_applied.cpu().int(), cmap="Greys", vmin=0, vmax=1)
    ax_spatial.set_title("Spatial Attention Mask (post-broadcast)")
    ax_spatial.set_xlabel("Key token index")
    ax_spatial.set_ylabel("Query token index")
    fig_spatial.tight_layout()

    fig_temporal, axes_temporal = plt.subplots(1, args.batch_size, figsize=(5 * args.batch_size, 5))
    if args.batch_size == 1:
        axes_temporal = [axes_temporal]

    for idx, axis in enumerate(axes_temporal):
        axis.imshow(temporal_applied[idx].cpu().int(), cmap="Greys", vmin=0, vmax=1)
        axis.set_title(f"Temporal Attention Mask (batch {idx})")
        axis.set_xlabel("Key time index")
        axis.set_ylabel("Query time index")

    fig_temporal.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
