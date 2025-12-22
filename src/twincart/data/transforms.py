from __future__ import annotations

from typing import Any, Dict, List

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _build_one(op: Dict[str, Any]) -> A.BasicTransform:
    name = op["name"]
    kwargs = {k: v for k, v in op.items() if k != "name"}

    if not hasattr(A, name):
        raise ValueError(f"Unknown albumentations transform: {name}")

    cls = getattr(A, name)
    return cls(**kwargs)


def build_image_transforms(cfg_transforms, mode: str) -> A.Compose:
    # cfg_transforms.image.{train|infer} list + input_size
    input_size = int(cfg_transforms.image.input_size)
    ops: List[Dict[str, Any]] = list(getattr(cfg_transforms.image, mode))

    # Enforce resize first (explicit, stable behavior)
    augs = [A.Resize(input_size, input_size, p=1.0)]
    augs += [_build_one(op) for op in ops]
    augs += [ToTensorV2(p=1.0)]

    return A.Compose(augs)
