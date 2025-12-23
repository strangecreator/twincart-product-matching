from __future__ import annotations

import pathlib
from dataclasses import dataclass

# utility imports
import cv2
import pandas as pd

# torch & related imports
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
        images_dir: pathlib.Path,
        transform,
        label_to_id: dict[str, int],
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.df)

    def _read_rgb(self, path: pathlib.Path) -> torch.Tensor:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if img is None:
            raise FileNotFoundError(f"Cannot read image: `{path}`.")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(image=img)["image"]

    def __getitem__(self, index: int) -> ImageSample:
        row = self.df.iloc[index]
        image_id = row["image"]
        posting_id = row["posting_id"]
        label_group = str(row["label_group"])

        path = self.images_dir / f"{image_id}"

        if not path.exists():
            alt = self.images_dir / f"{image_id}.jpg"
            path = alt if alt.exists() else path

        image = self._read_rgb(path)
        label = self.label_to_id[label_group]

        return ImageSample(image=image, label=label, posting_id=str(posting_id))
