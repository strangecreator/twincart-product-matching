from __future__ import annotations

from dataclasses import dataclass

import torch
from pytorch_metric_learning import losses, miners


@dataclass(frozen=True)
class MultiSimilarityCfg:
    alpha: float
    beta: float
    base: float
    miner_enabled: bool
    miner_epsilon: float


@dataclass(frozen=True)
class XbmCfg:
    enabled: bool
    memory_size: int


class MetricLearningLoss(torch.nn.Module):
    def __init__(self, ms: MultiSimilarityCfg, xbm: XbmCfg, embedding_size: int) -> None:
        super().__init__()
        base_loss = losses.MultiSimilarityLoss(alpha=ms.alpha, beta=ms.beta, base=ms.base)

        use_miner = ms.miner_enabled and (not xbm.enabled)
        self.miner = miners.MultiSimilarityMiner(epsilon=ms.miner_epsilon) if use_miner else None

        if xbm.enabled:
            self.loss = losses.CrossBatchMemory(
                loss=base_loss,
                embedding_size=embedding_size,
                memory_size=xbm.memory_size,
            )
        else:
            self.loss = base_loss

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.miner is not None:
            hard_pairs = self.miner(embeddings, labels)
            return self.loss(embeddings, labels, hard_pairs)
        return self.loss(embeddings, labels)
