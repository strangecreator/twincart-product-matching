from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator

from torch.utils.data import Sampler


class GroupPairBatchSampler(Sampler[list[int]]):
    """
    Batch = pairs sampled from the same label_group.
    For batch_size=32 -> sample 16 groups, 2 items per group.
    """

    def __init__(
        self,
        labels: list[int],
        batch_size: int,
        seed: int,
        drop_last: bool = True,
    ) -> None:
        if batch_size % 2 != 0:
            raise ValueError("batch_size must be even for pair sampling.")
        self.labels = labels
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        group_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, y in enumerate(labels):
            group_to_indices[y].append(idx)
        self.group_to_indices = dict(group_to_indices)
        self.groups = list(self.group_to_indices.keys())

        self.pairs_per_batch = self.batch_size // 2

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        groups = self.groups[:]
        rng.shuffle(groups)

        # cycle through groups; for each batch pick pairs_per_batch groups
        for start in range(0, len(groups), self.pairs_per_batch):
            chosen = groups[start : start + self.pairs_per_batch]
            if len(chosen) < self.pairs_per_batch and self.drop_last:
                break

            batch: list[int] = []
            for g in chosen:
                idxs = self.group_to_indices[g]
                if len(idxs) >= 2:
                    i1, i2 = rng.sample(idxs, 2)
                else:
                    # singleton group: duplicate (not ideal but avoids crash)
                    i1 = i2 = idxs[0]
                batch.extend([i1, i2])

            if len(batch) == self.batch_size:
                yield batch

    def __len__(self) -> int:
        # approximate
        full = len(self.groups) // self.pairs_per_batch
        return full
