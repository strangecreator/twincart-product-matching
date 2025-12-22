from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ImageSample:
    image: torch.Tensor
    label: int
    posting_id: str


class ShopeeImageDataset(Dataset[ImageSample]):
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Path,
        transform,
        label_to_id: dict[str, int],
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.df)

    def _read_rgb(self, path: Path) -> torch.Tensor:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = self.transform(image=img)["image"]
        return out

    def __getitem__(self, idx: int) -> ImageSample:
        row = self.df.iloc[idx]
        image_id = row["image"]
        posting_id = row["posting_id"]
        label_group = str(row["label_group"])

        path = self.images_dir / f"{image_id}"
        if not path.exists():
            # Kaggle dataset typically uses .jpg
            alt = self.images_dir / f"{image_id}.jpg"
            path = alt if alt.exists() else path

        image = self._read_rgb(path)
        label = self.label_to_id[label_group]

        return ImageSample(image=image, label=label, posting_id=str(posting_id))
