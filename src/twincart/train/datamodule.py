from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from twincart.data.dataset_image import ShopeeImageDataset
from twincart.data.dataset_text import ShopeeTextDataset
from twincart.train.sampler_pairs import GroupPairBatchSampler


def _make_label_map(df: pd.DataFrame) -> dict[str, int]:
    groups = sorted({str(x) for x in df["label_group"].tolist()})
    return {g: i for i, g in enumerate(groups)}


class TwinCartDataModule(pl.LightningDataModule):
    def __init__(
        self,
        mode: Literal["image", "text"],
        df: pd.DataFrame,
        fold_index: int,
        images_dir: Path,
        transforms_cfg: Any,
        image_model_cfg: Any,
        text_model_cfg: Any,
        batch_size: int,
        num_workers: int,
        seed: int,
        use_pair_sampler: bool,
        text_aug_p: float,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.df = df
        self.fold_index = int(fold_index)
        self.images_dir = images_dir
        self.transforms_cfg = transforms_cfg
        self.image_model_cfg = image_model_cfg
        self.text_model_cfg = text_model_cfg
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self.use_pair_sampler = bool(use_pair_sampler)
        self.text_aug_p = float(text_aug_p)

        self.label_to_id: dict[str, int] = {}

        self.train_ds = None
        self.val_ds = None

    def setup(self, stage: str | None = None) -> None:
        if "fold" not in self.df.columns:
            raise ValueError("Expected 'fold' column in dataframe. Run prepare_folds first.")

        train_df = self.df[self.df["fold"] != self.fold_index].reset_index(drop=True)
        val_df = self.df[self.df["fold"] == self.fold_index].reset_index(drop=True)

        self.label_to_id = _make_label_map(self.df)

        if self.mode == "image":
            from twincart.data.transforms import build_image_transforms

            tfm_train = build_image_transforms(self.transforms_cfg, mode="train")
            tfm_val = build_image_transforms(self.transforms_cfg, mode="infer")

            self.train_ds = ShopeeImageDataset(train_df, self.images_dir, tfm_train, self.label_to_id)
            self.val_ds = ShopeeImageDataset(val_df, self.images_dir, tfm_val, self.label_to_id)
        else:
            tok_name = str(self.text_model_cfg.backbone)
            max_len = int(self.transforms_cfg.text.max_length)
            use_fast = bool(self.text_model_cfg.use_fast_tokenizer)

            self.train_ds = ShopeeTextDataset(
                train_df,
                tokenizer_name=tok_name,
                max_length=max_len,
                label_to_id=self.label_to_id,
                aug_p=self.text_aug_p,
                use_fast=use_fast,
            )
            self.val_ds = ShopeeTextDataset(
                val_df,
                tokenizer_name=tok_name,
                max_length=max_len,
                label_to_id=self.label_to_id,
                aug_p=0.0,
                use_fast=use_fast,
            )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        if self.use_pair_sampler:
            train_df = self.df[self.df["fold"] != self.fold_index].reset_index(drop=True)
            labels = [self.label_to_id[str(x)] for x in train_df["label_group"].astype(str).tolist()]
            sampler = GroupPairBatchSampler(labels, batch_size=self.batch_size, seed=self.seed)
            return DataLoader(
                self.train_ds,
                batch_sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
                collate_fn=_collate(self.mode),
            )

        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=_collate(self.mode),
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=_collate(self.mode),
        )


def _collate(mode: Literal["image", "text"]):
    def collate(batch):
        if mode == "image":
            images = torch.stack([b.image for b in batch], dim=0)
            labels = torch.tensor([b.label for b in batch], dtype=torch.long)
            posting_ids = [b.posting_id for b in batch]
            return {"image": images, "label": labels, "posting_id": posting_ids}

        input_ids = torch.stack([b.input_ids for b in batch], dim=0)
        attention_mask = torch.stack([b.attention_mask for b in batch], dim=0)
        labels = torch.tensor([b.label for b in batch], dtype=torch.long)
        posting_ids = [b.posting_id for b in batch]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": labels,
            "posting_id": posting_ids,
        }

    return collate
