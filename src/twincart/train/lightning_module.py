from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import faiss
import numpy as np
import pytorch_lightning as pl
import torch
from madgrad import MADGRAD
from torch.optim.lr_scheduler import CosineAnnealingLR

from twincart.models.image_encoder import ImageEncoder
from twincart.models.text_encoder import TextEncoder
from twincart.train.losses import MetricLearningLoss, MultiSimilarityCfg, XbmCfg


@dataclass
class TrainCfg:
    mode: Literal["image", "text"]
    lr_start: float
    lr_end: float
    max_epochs: int


class TwinCartModule(pl.LightningModule):
    def __init__(
        self,
        mode: Literal["image", "text"],
        image_model_cfg: Any,
        text_model_cfg: Any,
        loss_cfg: Any,
        xbm_cfg: Any,
        optim_cfg: Any,
        sched_cfg: Any,
        train_cfg: TrainCfg,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["image_model_cfg", "text_model_cfg", "train_cfg"])

        self.mode = mode
        self.train_cfg = train_cfg

        if mode == "image":
            pooling = image_model_cfg.pooling
            self.encoder = ImageEncoder(
                backbone=str(image_model_cfg.backbone),
                pretrained=bool(image_model_cfg.pretrained),
                embedding_dim=int(image_model_cfg.embedding_dim),
                gem_p=float(pooling.p_train),
                gem_eps=float(pooling.eps),
            )
            xbm = XbmCfg(enabled=bool(xbm_cfg.enabled), memory_size=int(xbm_cfg.memory_size))
            miner_enabled = False
            miner_eps = 0.1

            emb_dim = int(image_model_cfg.embedding_dim)
        else:
            self.encoder = TextEncoder(
                backbone=str(text_model_cfg.backbone),
                embedding_dim=int(text_model_cfg.embedding_dim),
            )
            xbm = XbmCfg(enabled=False, memory_size=0)
            miner_enabled = True  # 7th place used miner for text
            miner_eps = float(loss_cfg.miner.epsilon)

            emb_dim = int(text_model_cfg.embedding_dim)

        ms = MultiSimilarityCfg(
            alpha=float(loss_cfg.alpha),
            beta=float(loss_cfg.beta),
            base=float(loss_cfg.base),
            miner_enabled=miner_enabled,
            miner_epsilon=float(miner_eps),
        )
        self.criterion = MetricLearningLoss(ms=ms, xbm=xbm, embedding_size=emb_dim)

        self.optim_cfg = optim_cfg
        self.sched_cfg = sched_cfg

        self._val_embs: list[np.ndarray] = []
        self._val_labels: list[np.ndarray] = []

    def forward(self, batch: Any) -> torch.Tensor:
        if self.mode == "image":
            return self.encoder(batch["image"])
        return self.encoder(batch["input_ids"], batch["attention_mask"])

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        emb = self.forward(batch)
        labels = batch["label"]
        loss = self.criterion(emb, labels)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=labels.size(0))
        return loss

    @torch.no_grad()
    def validation_step(self, batch: Any, batch_idx: int) -> None:
        emb = self.forward(batch).detach().float().cpu().numpy()
        labels = batch["label"].detach().cpu().numpy().astype(np.int64)
        self._val_embs.append(emb)
        self._val_labels.append(labels)

    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        if not self._val_embs:
            return

        embs = np.concatenate(self._val_embs, axis=0).astype(np.float32)
        labels = np.concatenate(self._val_labels, axis=0).astype(np.int64)
        self._val_embs.clear()
        self._val_labels.clear()

        # cosine similarity on normalized vectors = inner product
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)

        # search top-2 because top-1 is itself
        sims, idxs = index.search(embs, 2)
        nn = idxs[:, 1]
        correct = (labels[nn] == labels).mean().item()
        self.log("val_recall1", float(correct), prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        lr = float(self.train_cfg.lr_start)
        wd = float(self.optim_cfg.weight_decay)
        momentum = float(self.optim_cfg.momentum)

        opt = MADGRAD(self.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

        # cosine annealing; T_max in epochs, with eta_min = lr_end
        scheduler = CosineAnnealingLR(opt, T_max=int(self.train_cfg.max_epochs), eta_min=float(self.train_cfg.lr_end))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
