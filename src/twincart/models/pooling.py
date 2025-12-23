from __future__ import annotations

# torch & related imports
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeMPooling(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()

        self.p = float(p)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, c, h, w)
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        x = x.pow(1.0 / self.p)
        return x.flatten(1)
