"""Lightweight decoder used to map latent tokens back into pixel space."""
import math
from pathlib import Path
from typing import Any, Mapping, Tuple, Type

import torch
import torch.nn as nn

__all__ = ["ImageDecoderTranspose", "build_image_decoder"]


def _initialize_weights(module: nn.Module) -> None:
    """Apply Kaiming initialization to convolutional modules."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(module.weight.data, nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight.data)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)


class ImageDecoderTranspose(nn.Module):
    """
    Simple decoder composed of transposed convolutions.

    The input is expected to be shaped [batch, num_tokens, feature_dim],
    where ``num_tokens`` forms a square grid (e.g. 32x32). Tokens are first
    reshaped into a spatial grid and projected to a high-dimensional feature
    map before being progressively upsampled to the target resolution.
    """

    def __init__(
        self,
        observation_shape: Tuple[int, int, int] = (3, 512, 512),
        feature_dim: int = 1024,
        activation: Type[nn.Module] = nn.ReLU,
        depth: int = 64,
        kernel_size: int = 5,
        stride: int = 3,
    ) -> None:
        super().__init__()
        channels, height, width = observation_shape
        act = activation()
        base_channels = depth

        self.observation_shape = observation_shape
        self.feature_dim = feature_dim

        self.proj = nn.Conv2d(feature_dim, base_channels * 32, kernel_size=1)
        blocks = []
        in_channels = base_channels * 32
        for out_channels in (base_channels * 8, base_channels * 4, base_channels * 2, base_channels):
            blocks.extend(
                [
                    act if len(blocks) == 0 else activation(),
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding=1,
                    ),
                ]
            )
            in_channels = out_channels
        blocks.append(activation())
        blocks.append(
            nn.ConvTranspose2d(
                in_channels, channels, kernel_size, stride, padding=1
            )
        )
        blocks.append(
            nn.Upsample(size=(height, width), mode="bilinear", align_corners=False)
        )

        self.network = nn.Sequential(*blocks)

        self.apply(_initialize_weights)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode a batch of latent tokens into images.

        Args:
            latents: Tensor shaped [batch, num_tokens, feature_dim].

        Returns:
            Reconstructed images shaped [batch, C, H, W].
        """
        if latents.dim() != 3:
            raise ValueError(
                f"Expected latents with shape [B, N, E], received {tuple(latents.shape)}"
            )

        batch, num_tokens, embed_dim = latents.shape
        if embed_dim != self.feature_dim:
            raise ValueError(
                f"Decoder expects feature_dim={self.feature_dim}, "
                f"received {embed_dim}"
            )

        grid_size = int(math.isqrt(num_tokens))
        if grid_size * grid_size != num_tokens:
            raise ValueError("Number of tokens must form a square grid.")

        # (B, N, E) -> (B, E, H', W')
        latents = latents.view(batch, grid_size, grid_size, embed_dim)
        latents = latents.permute(0, 3, 1, 2).contiguous()

        x = self.proj(latents)
        x = self.network(x)
        return x


def build_image_decoder(config: Mapping[str, Any]) -> ImageDecoderTranspose:
    """
    Construct an :class:`ImageDecoderTranspose` from a configuration mapping.

    Expected keys:
        observation_shape: Sequence of three ints, defaults to (3, 512, 512).
        feature_dim: Integer latent dimensionality, defaults to 1024.
        activation: Fully-qualified name of an nn.Module (optional).
        depth, kernel_size, stride: Integers matching the decoder signature.
    """

    observation_shape = tuple(config.get("observation_shape", (3, 512, 512)))
    feature_dim = int(config.get("feature_dim", 1024))
    depth = int(config.get("depth", 64))
    kernel_size = int(config.get("kernel_size", 5))
    stride = int(config.get("stride", 3))

    activation_cls: Type[nn.Module]
    activation_name = config.get("activation")
    if activation_name is None:
        activation_cls = nn.ReLU
    elif isinstance(activation_name, str):
        activation_cls = _resolve_activation(activation_name)
    else:
        raise TypeError("activation must be a string with a module path.")

    return ImageDecoderTranspose(
        observation_shape=observation_shape,
        feature_dim=feature_dim,
        activation=activation_cls,
        depth=depth,
        kernel_size=kernel_size,
        stride=stride,
    )


def _resolve_activation(name: str) -> Type[nn.Module]:
    module_path, _, attr = name.rpartition(".")
    if not module_path:
        module = getattr(nn, attr, None)
        if module is None:
            raise AttributeError(f"Activation {name!r} not found in torch.nn.")
        return module  # type: ignore[return-value]
    module = __import__(module_path, fromlist=[attr])
    activation_cls = getattr(module, attr, None)
    if activation_cls is None:
        raise AttributeError(f"Activation {name!r} not found in module {module_path!r}.")
    return activation_cls


if __name__ == "__main__":
    import yaml
    path = Path("configs/decoder.yaml")
    config = yaml.safe_load(Path(path).read_text())
    model = build_image_decoder(config).to("mps")
    feature_dim = int(config.get("feature_dim", 1024))
    num_tokens = int(config.get("num_tokens", 1024))
    dummy = torch.randn(2, num_tokens, feature_dim).to("mps")
    output = model(dummy)
    print("Decoded image shape:", tuple(output.shape))
