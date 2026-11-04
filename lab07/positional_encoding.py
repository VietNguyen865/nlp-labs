"""Sinusoidal positions for batch-first embeddings, including odd dimensions."""
import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        if d_model <= 0 or max_len <= 0:
            raise ValueError("d_model and max_len must be positive")
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        if x.ndim != 3 or x.size(-1) != self.pe.size(-1):
            raise ValueError("Expected (batch, sequence, d_model)")
        if x.size(1) > self.pe.size(1):
            raise ValueError("Sequence exceeds max_len")
        return x + self.pe[:, :x.size(1)].to(dtype=x.dtype)
