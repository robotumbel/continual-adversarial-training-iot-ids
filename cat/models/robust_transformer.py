"""
robust_transformer.py — Robust Denoising Transformer (RDT) for IoT IDS.
======================================================================
A Transformer encoder whose architectural elements are chosen for a
PRINCIPLED adversarial-robustness reason (unlike capacity-oriented gates):

  1. Feature-denoising blocks (non-local means; Xie et al., CVPR 2019,
     "Feature Denoising for Improving Adversarial Robustness").
     After each attention sublayer a non-local-means operation smooths
     the token-feature map, explicitly attenuating the perturbation that
     an adversary injects into the features, then a 1x1 projection with a
     learnable residual scale re-injects the denoised signal.

  2. Spectral-normalized linear layers (Miyato et al., 2018).
     Q/K/V/output projections and the FFN are spectrally normalized so
     the layer's Lipschitz constant is bounded; a bounded Lipschitz
     constant limits how much the output can move for an
     epsilon-bounded input perturbation, which is exactly the quantity
     an L-infinity adversary controls.

Both mechanisms act ON TOP of adversarial training, targeting the
feature-space perturbation that AT alone does not structurally constrain.

The forward signature and output dict
({logits, anomaly_score, embeddings}) match the other backbones so the
Paper-3 runners can use it interchangeably.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:                                  # newer API (torch >= 2.1)
    from torch.nn.utils.parametrizations import spectral_norm as _sn
except Exception:                     # fallback
    from torch.nn.utils import spectral_norm as _sn


def sn_linear(in_f, out_f):
    """Spectrally-normalized linear layer (bounded Lipschitz constant)."""
    return _sn(nn.Linear(in_f, out_f))


class _SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class FeatureDenoiseBlock(nn.Module):
    """Non-local-means feature denoising (Xie et al., 2019).

    For token features x [B, L, D]:
      affinity  f = softmax(x x^T / sqrt(D))          [B, L, L]
      denoised  z = f x                                [B, L, D]
      out       = x + gamma * W_proj(z)                (residual)
    The learnable scalar gamma is initialised small so the block starts
    close to identity and the denoising strength is learned.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.proj  = sn_linear(d_model, d_model)      # 1x1 projection
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, x):                              # x: [B, L, D]
        f = torch.softmax(torch.matmul(x, x.transpose(-2, -1)) * self.scale,
                          dim=-1)                       # [B, L, L]
        z = torch.matmul(f, x)                          # [B, L, D] denoised
        return x + self.gamma * self.proj(z)


class SNMultiHeadAttention(nn.Module):
    """Multi-head self-attention with spectrally-normalized projections."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = sn_linear(d_model, d_model)
        self.W_k = sn_linear(d_model, d_model)
        self.W_v = sn_linear(d_model, d_model)
        self.W_o = sn_linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        H, dk = self.n_heads, self.d_k
        Q = self.W_q(x).view(B, L, H, dk).transpose(1, 2)
        K = self.W_k(x).view(B, L, H, dk).transpose(1, 2)
        V = self.W_v(x).view(B, L, H, dk).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        ctx = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(ctx), attn


class RobustBlock(nn.Module):
    """Attention -> feature denoising -> FFN, all with residual+LayerNorm."""
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn    = SNMultiHeadAttention(d_model, n_heads, dropout)
        self.denoise = FeatureDenoiseBlock(d_model)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.norm3   = nn.LayerNorm(d_model)
        self.ff      = nn.Sequential(
            sn_linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            sn_linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        a, w = self.attn(x, mask)
        x = self.norm1(x + self.drop(a))
        x = self.norm2(x + self.denoise(x))        # feature denoising
        x = self.norm3(x + self.drop(self.ff(x)))
        return x, w


class RobustDenoiseTransformerIDS(nn.Module):
    """Robust Denoising Transformer for IoT IDS.

    Same constructor signature as StandardTransformerIDS so the Paper-3
    runners can build it interchangeably.
    """
    def __init__(self, input_dim: int, d_model: int = 96, n_heads: int = 4,
                 n_layers: int = 3, d_ff: int = 384, n_classes: int = 2,
                 max_seq_len: int = 50, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pe        = _SinusoidalPE(d_model, max_len=max_seq_len + 10)
        self.blocks    = nn.ModuleList([
            RobustBlock(d_model, n_heads, d_ff, dropout)
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
        x = self.drop(self.pe(self.embedding(x)))
        attn_weights = []
        for block in self.blocks:
            x, w = block(x, mask)
            attn_weights.append(w)
        pooled  = x.mean(dim=1)
        logits  = self.classification_head(pooled)
        anomaly = self.anomaly_scorer(pooled)
        return {
            "logits":            logits,
            "anomaly_score":     anomaly,
            "embeddings":        pooled,
            "attention_weights": attn_weights,
        }
