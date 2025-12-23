from __future__ import annotations

from dataclasses import dataclass

# utility imports
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class FoldConfig:
    num_folds: int
    seed: int


def _size_bin(group_size: int) -> int:
    if group_size <= 1:
        return 0
    elif group_size <= 3:
        return 1
    elif group_size <= 7:
        return 2
    elif group_size <= 15:
        return 3
    else:
        return 4


def make_folds(df: pd.DataFrame, config: FoldConfig) -> pd.DataFrame:
    missing = {"posting_id", "label_group"} - set(df.columns)

    if missing:
        raise ValueError(f"train.csv missing columns: {missing}")

    group_sizes = df.groupby("label_group")["posting_id"].transform("count")
    y = group_sizes.map(_size_bin).astype(int)

    splitter = StratifiedGroupKFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed)

    folds = df.copy()
    folds["fold"] = -1

    groups = df["label_group"].astype(str)
    for fold_index, (_, val_index) in enumerate(splitter.split(df, y=y, groups=groups)):
        folds.iloc[val_index, folds.columns.get_loc("fold")] = fold_index

    if (folds["fold"] < 0).any():
        raise RuntimeError("Some rows did not get a fold assignment.")

    return folds
