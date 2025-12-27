from __future__ import annotations

import typing as tp
from dataclasses import dataclass

# utility imports
import faiss
import numpy as np

# torch & related imports
import torch
from madgrad import MADGRAD
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR

# local imports
from twincart.models.text_encoder import TextEncoder
from twincart.models.image_encoder import ImageEncoder
from twincart.train.losses import MetricLearningLoss, MultiSimilarityCfg, XbmCfg


@dataclass
class TrainCfg:
    mode: tp.Literal["image", "text"]
    lr_start: float
    lr_end: float
    max_epochs: int


class TwinCartModule(pl.LightningModule):
    def __init__(
        self,
        mode: tp.Literal["image", "text"],
        image_model_config: tp.Any,
        text_model_config: tp.Any,
        loss_config: tp.Any,
        xbm_config: tp.Any,
        optim_config: tp.Any,
        sched_config: tp.Any,
        train_config: TrainCfg,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["image_model_config", "text_model_config", "train_config"])

        self.mode = mode
        self.train_config = train_config

        if mode == "image":
            pooling = image_model_config.pooling
            self.encoder = ImageEncoder(
                backbone=str(image_model_config.backbone),
                pretrained=bool(image_model_config.pretrained),
                embedding_dim=int(image_model_config.embedding_dim),
                gem_p=float(pooling.p_train),
                gem_eps=float(pooling.eps),
            )
            xbm = XbmCfg(enabled=bool(xbm_config.enabled), memory_size=int(xbm_config.memory_size))
            miner_enabled = False
            miner_eps = 0.1

            emb_dim = int(image_model_config.embedding_dim)
        else:
            self.encoder = TextEncoder(
                backbone=str(text_model_config.backbone),
                embedding_dim=int(text_model_config.embedding_dim),
            )
            xbm = XbmCfg(enabled=False, memory_size=0)
            miner_enabled = True
            miner_eps = float(loss_config.miner.epsilon)

            emb_dim = int(text_model_config.embedding_dim)

        multi_similarity_config = MultiSimilarityCfg(
            alpha=float(loss_config.alpha),
            beta=float(loss_config.beta),
            base=float(loss_config.base),
            miner_enabled=miner_enabled,
            miner_epsilon=float(miner_eps),
        )
        self.criterion = MetricLearningLoss(ms=multi_similarity_config, xbm=xbm, embedding_size=emb_dim)

        self.optim_config = optim_config
        self.sched_config = sched_config

        self._val_embs: list[np.ndarray] = []
        self._val_labels: list[np.ndarray] = []

    def forward(self, batch: tp.Any) -> torch.Tensor:
        if self.mode == "image":
            return self.encoder(batch["image"])

        return self.encoder(batch["input_ids"], batch["attention_mask"])

    def training_step(self, batch: tp.Any, batch_index: int) -> torch.Tensor:
        embeddings = self.forward(batch)
        labels = batch["label"]

        loss = self.criterion(embeddings, labels)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=labels.size(0))
        return loss

    @torch.no_grad()
    def validation_step(self, batch: tp.Any, batch_index: int) -> None:
        embeddings = self.forward(batch).detach().float().cpu().numpy()
        labels = batch["label"].detach().cpu().numpy().astype(np.int64)

        self._val_embs.append(embeddings)
        self._val_labels.append(labels)

    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        if not self._val_embs:
            return

        embeddings = np.concatenate(self._val_embs, axis=0).astype(np.float32)
        labels = np.concatenate(self._val_labels, axis=0).astype(np.int64)
        self._val_embs.clear()
        self._val_labels.clear()

        # cosine similarity is inner product
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        # searching for top-2 (cause top-1 is itself)
        similarities, indices = index.search(embeddings, 2)
        correct = (labels[indices[:, 1]] == labels).mean().item()
        self.log("val_recall1", float(correct), prog_bar=True, on_epoch=True)

    def configure_optimizers(self) -> dict:
        lr = float(self.train_config.lr_start)
        wd = float(self.optim_config.weight_decay)
        momentum = float(self.optim_config.momentum)

        opt = MADGRAD(self.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

        scheduler = CosineAnnealingLR(opt, T_max=int(self.train_config.max_epochs), eta_min=float(self.train_config.lr_end))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
