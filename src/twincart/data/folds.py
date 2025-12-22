from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class FoldConfig:
    num_folds: int
    seed: int


def _size_bin(group_size: int) -> int:
    # Simple monotonic binning; tweak later if you want.
    if group_size <= 1:
        return 0
    if group_size <= 3:
        return 1
    if group_size <= 7:
        return 2
    if group_size <= 15:
        return 3
    return 4


def make_folds(df: pd.DataFrame, cfg: FoldConfig) -> pd.DataFrame:
    required = {"posting_id", "label_group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train.csv missing columns: {missing}")

    group_sizes = df.groupby("label_group")["posting_id"].transform("count")
    y = group_sizes.map(_size_bin).astype(int)

    splitter = StratifiedGroupKFold(
        n_splits=cfg.num_folds, shuffle=True, random_state=cfg.seed
    )

    folds = df.copy()
    folds["fold"] = -1

    groups = df["label_group"].astype(str)
    for fold_idx, (_, val_idx) in enumerate(splitter.split(df, y=y, groups=groups)):
        folds.iloc[val_idx, folds.columns.get_loc("fold")] = fold_idx

    if (folds["fold"] < 0).any():
        raise RuntimeError("Some rows did not get a fold assignment.")

    return folds
