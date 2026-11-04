"""Minimal post-norm Transformer encoder; True in padding_mask means PAD."""
import math
import torch
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        if n_heads <= 0 or d_model % n_heads:
            raise ValueError("d_model must be divisible by positive n_heads")
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None, return_attention=False):
        B, T, D = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.d_k).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if padding_mask is not None:
            if padding_mask.shape != (B, T) or padding_mask.dtype != torch.bool:
                raise ValueError("padding_mask must be bool with shape (B, T)")
            if padding_mask.all(dim=1).any():
                raise ValueError("Each sequence needs at least one non-padding token")
            scores = scores.masked_fill(padding_mask[:, None, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attention = (self.dropout(weights) @ v).transpose(1, 2).reshape(B, T, D)
        x = self.norm1(x + self.dropout(self.out_proj(attention)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return (x, weights) if return_attention else x
