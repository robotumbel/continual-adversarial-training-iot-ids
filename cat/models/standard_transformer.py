"""
standard_transformer.py — Encoder-only Standard Transformer IDS (baseline).
Fixed sinusoidal positional encoding, vanilla multi-head attention.
No adaptive gating, no incremental learning, single classification head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class StandardMultiHeadAttention(nn.Module):
    """Vanilla multi-head self-attention — no adaptive gating."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k    = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        H, dk   = self.n_heads, self.d_k

        Q = self.W_q(x).view(B, L, H, dk).transpose(1, 2)   # [B,H,L,dk]
        K = self.W_k(x).view(B, L, H, dk).transpose(1, 2)
        V = self.W_v(x).view(B, L, H, dk).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        ctx = torch.matmul(attn, V)                           # [B,H,L,dk]
        ctx = ctx.transpose(1, 2).contiguous().view(B, L, D)  # [B,L,D]
        return self.W_o(ctx), attn


class StandardTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn  = StandardMultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, attn_w = self.attn(x, mask)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x, attn_w


class StandardTransformerIDS(nn.Module):
    """
    Standard encoder-only Transformer for IDS (baseline).
    Fixed sinusoidal PE, vanilla attention, single classification head.
    """
    def __init__(self, input_dim: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 3, d_ff: int = 256, n_classes: int = 2,
                 max_seq_len: int = 50, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pe        = _SinusoidalPE(d_model, max_len=max_seq_len + 10)
        self.blocks    = nn.ModuleList([
            StandardTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: [B, seq_len, input_dim]
        x = self.drop(self.pe(self.embedding(x)))

        attn_weights = []
        for block in self.blocks:
            x, w = block(x, mask)
            attn_weights.append(w)

        pooled  = x.mean(dim=1)                          # global avg pool
        logits  = self.classification_head(pooled)
        anomaly = self.anomaly_scorer(pooled)
        return {
            "logits":          logits,
            "anomaly_score":   anomaly,
            "embeddings":      pooled,
            "attention_weights": attn_weights,
        }
