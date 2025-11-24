from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .configs import DatasetConfig
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as transforms_v2

# Standard transform for all datasets
IMAGE_RESIZE_CROP_TRANSFORM_224 = transforms_v2.Compose(
    [
        transforms_v2.Resize(
            size=(224, 320),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms_v2.CenterCrop((224, 224)),
    ]
)