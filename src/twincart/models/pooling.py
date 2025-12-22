from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeMPooling(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = float(p)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, kernel_size=(x.size(-2), x.size(-1)))
        x = x.pow(1.0 / self.p)
        return x.flatten(1)
