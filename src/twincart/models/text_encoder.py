from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(self, backbone: str, embedding_dim: int) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(backbone)
        hidden = self.model.config.hidden_size
        self.fc = nn.Linear(hidden, embedding_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT: out.last_hidden_state exists; take CLS token at position 0
        cls = out.last_hidden_state[:, 0, :]
        emb = self.fc(cls)
        emb = F.normalize(emb, p=2, dim=1)
        return emb
