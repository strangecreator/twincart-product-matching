from __future__ import annotations

# torch & related imports
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# local imports
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
        feats = self.backbone.forward_features(x)  # (B, c) or (B, c, h, w)

        if feats.dim() == 2:
            pooled = feats
        else:
            pooled = self.pool(feats)

        embedding = self.fc(pooled)
        return F.normalize(embedding, p=2, dim=1)
