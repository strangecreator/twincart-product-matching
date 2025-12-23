from __future__ import annotations

import random
from dataclasses import dataclass

# utility imports
import pandas as pd

# torch & related imports
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def _random_delete(words: list[str]) -> list[str]:
    if len(words) <= 2:
        return words

    i = random.randrange(len(words))
    return [w for j, w in enumerate(words) if j != i]


def _random_swap(words: list[str]) -> list[str]:
    if len(words) <= 2:
        return words

    i, j = random.sample(range(len(words)), 2)
    out = words[:]

    out[i], out[j] = out[j], out[i]

    return out


def augment_title(title: str, p: float) -> str:
    if random.random() >= p:
        return title

    words = title.split()

    if len(words) <= 2:
        return title

    ops = [
        lambda w: _random_delete(w),
        lambda w: _random_swap(w),
        lambda w: _random_delete(_random_swap(w)),
        lambda w: _random_swap(_random_swap(w)),
    ]
    return " ".join(random.choice(ops)(words))


@dataclass(frozen=True)
class TextSample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    label: int
    posting_id: str


class ShopeeTextDataset(Dataset[TextSample]):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer_name: str,
        max_length: int,
        label_to_id: dict[str, int],
        aug_p: float = 0.0,
        use_fast: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.max_length = int(max_length)
        self.label_to_id = label_to_id
        self.aug_p = float(aug_p)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=use_fast)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> TextSample:
        row = self.df.iloc[index]
        posting_id = str(row["posting_id"])
        label_group = str(row["label_group"])
        title = str(row["title"])

        title = augment_title(title, p=self.aug_p)

        enc = self.tokenizer(
            title,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        label = self.label_to_id[label_group]

        return TextSample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            label=label,
            posting_id=posting_id,
        )
