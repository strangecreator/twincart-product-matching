from __future__ import annotations

import typing as tp

# albumentations & related imports
import albumentations as A
from albumentations.pytorch import ToTensorV2


def _build_one(operation: dict[str, tp.Any]) -> A.BasicTransform:
    name = operation["name"]
    kwargs = {k: v for k, v in operation.items() if k != "name"}

    if not hasattr(A, name):
        raise ValueError(f"Unknown albumentations transform: {name}.")

    return getattr(A, name)(**kwargs)


def build_image_transforms(config_transforms, mode: str) -> A.Compose:
    input_size = int(config_transforms.image.input_size)
    operations: list[dict[str, tp.Any]] = list(getattr(config_transforms.image, mode))

    return A.Compose(
        [
            A.Resize(input_size, input_size, p=1.0)
        ] + [
            _build_one(operation)
            for operation in operations
        ] + [
            ToTensorV2(p=1.0)
        ]
    )
