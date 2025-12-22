from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from twincart.models.pooling import GeMPooling


class ImageEncoder(nn.Module):
    def __init__(
        self,
        backbone: str,
        pretrained: bool,
        embedding_dim: int,
        gem_p: float,
        gem_eps: float,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features

        self.pool = GeMPooling(p=gem_p, eps=gem_eps)
        self.fc = nn.Linear(feat_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)  # typically [B,C,H,W]
        if feats.dim() == 2:
            # some backbones output [B,C] already
            pooled = feats
        else:
            pooled = self.pool(feats)
        emb = self.fc(pooled)
        emb = F.normalize(emb, p=2, dim=1)
        return emb
