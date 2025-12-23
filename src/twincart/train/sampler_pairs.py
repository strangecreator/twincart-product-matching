from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator

# torch & related imports
from torch.utils.data import Sampler


class GroupPairBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        labels: list[int],
        batch_size: int,
        seed: int,
        drop_last: bool = True,
    ) -> None:
        if batch_size % 2 != 0:
            raise ValueError("Batch size must be even for pair sampling.")

        self.labels = labels
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        group_to_indices: dict[int, list[int]] = defaultdict(list)

        for index, y in enumerate(labels):
            group_to_indices[y].append(index)

        self.group_to_indices = dict(group_to_indices)
        self.groups = list(self.group_to_indices.keys())

        self.pairs_per_batch = self.batch_size // 2

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        groups = self.groups[:]
        rng.shuffle(groups)

        for start in range(0, len(groups), self.pairs_per_batch):
            chosen_groups = groups[start : (start + self.pairs_per_batch)]

            if len(chosen_groups) < self.pairs_per_batch and self.drop_last:
                break

            batch: list[int] = []

            for group in chosen_groups:
                indices = self.group_to_indices[group]

                if len(indices) >= 2:
                    i, j = rng.sample(indices, 2)
                else:
                    i = j = indices[0]  # singleton group

                batch.extend([i, j])

            if len(batch) == self.batch_size:
                yield batch

    def __len__(self) -> int:
        return len(self.groups) // self.pairs_per_batch
