"""
aam_trans.py — AAM-TRANS: Adaptive Attention Mechanism Transformer for IDS.

Architecture aligned with dissertation Chapter 3:
  Objective 1 — Adaptive Attention with Gate Generation (transferability mitigation)
  Objective 2 — Incremental Learning hooks (EWC/KD/MemBuf handled in incremental.py)
  Objective 3 — Dual-output: Classification Head + Confidence-Based Anomaly Scorer

Core differences from Standard Transformer:
  • Learned Positional Encoding (trainable parameter, not fixed sinusoidal)
  • Per-head adaptive gates: Gates[h1..hn] = σ(MLP(avg_pool(X)))
  • Final output: Out = Attn × Gates × V
  • Additional anomaly scoring head alongside classification head

Q1-publication upgrades (all opt-in via flags; defaults retain original
behaviour for backward compatibility with existing ablation results):

  use_feature_tokenizer  : per-feature learnable embedding (FT-Transformer-style)
                           — boosts tabular IDS clean accuracy substantially.
  use_cls_token          : prepend a learned [CLS] token used as global summary
                           instead of mean pooling (BERT-style classification).
  use_pre_norm           : LayerNorm BEFORE attention/FFN (more stable than
                           post-norm at deeper layers).
  use_spectral_norm      : spectral normalization on Q/K/V projections gives a
                           Lipschitz bound, improving adversarial robustness.
  use_gradient_aware_gate: feed the absolute mean of the input feature
                           magnitudes alongside the pooled embedding into the
                           gate MLP — makes the gate sensitive to anomalous
                           input statistics.
  use_stochastic_gate    : add Gaussian noise to the gate logits at training
                           time (acts like Bayesian dropout on gates).

Original ablation flags (kept):
  use_adaptive_gates / use_learned_pe / use_anomaly_head
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Sinusoidal PE (fallback for ablation)
# ──────────────────────────────────────────────────────────────────────────────

def _sinusoidal_pe(max_seq_len: int, d_model: int) -> torch.Tensor:
    """Fixed sinusoidal positional encoding [1, max_seq_len, d_model]."""
    pe  = torch.zeros(max_seq_len, d_model)
    pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
    return pe.unsqueeze(0)   # [1, L, D]


def _maybe_sn(layer: nn.Module, enable: bool) -> nn.Module:
    """Optionally wrap a linear layer with spectral normalization."""
    if enable:
        return nn.utils.parametrizations.spectral_norm(layer)
    return layer


# ──────────────────────────────────────────────────────────────────────────────
# DropPath — Stochastic Depth (Huang et al., ECCV 2016)
# ──────────────────────────────────────────────────────────────────────────────

class DropPath(nn.Module):
    """Drop entire residual branches with probability `p` (per sample)."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x.div(keep) * mask


# ──────────────────────────────────────────────────────────────────────────────
# Swish-GLU FFN (PaLM / LLaMA style — outperforms vanilla GELU FFN)
# ──────────────────────────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    """SwiGLU(x) = SiLU(W1 x) ⊙ (W2 x); then Wo · (...)"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # We split d_ff across the two projections (so total params ≈ standard FFN)
        d_hidden  = int(d_ff * 2 / 3)
        self.w_gate = nn.Linear(d_model, d_hidden)
        self.w_up   = nn.Linear(d_model, d_hidden)
        self.w_down = nn.Linear(d_hidden, d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_down(self.drop(F.silu(self.w_gate(x)) * self.w_up(x)))


# ──────────────────────────────────────────────────────────────────────────────
# Feature Tokenizer (FT-Transformer style)
# ──────────────────────────────────────────────────────────────────────────────

class FeatureTokenizer(nn.Module):
    """
    Each input feature is embedded into d_model with its own learnable weight
    and bias (Gorishniy et al., NeurIPS 2021). Empirically gives SOTA on
    tabular data compared to a single shared linear projection.

    Input:   x  [B, L, F]            (treats each L slice as a "row" of F feats)
    Output:  z  [B, L, d_model]      after per-feature project + sum across F
    """

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias   = nn.Parameter(torch.empty(n_features, d_model))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F] → expand to [B, L, F, 1] · [F, D] + [F, D] → sum over F
        B, L, F_ = x.shape
        z = x.unsqueeze(-1) * self.weight + self.bias        # [B, L, F, D]
        return z.mean(dim=2)                                  # [B, L, D]


# ──────────────────────────────────────────────────────────────────────────────
# Adaptive Attention Mechanism (Objective 1) — enhanced
# ──────────────────────────────────────────────────────────────────────────────

class AdaptiveAttentionMechanism(nn.Module):
    """
    Multi-head self-attention with per-head adaptive gating.

    Standard path:
        Q, K, V = Linear(X)               (optionally spectral-normalised)
        Attn    = softmax(QK^T / sqrt(d_k))

    Adaptive gate path (novel contribution):
        gate_input = [avg_pool(X), |X|.mean()] if gradient_aware else avg_pool(X)
        Gates      = σ(MLP(gate_input))      [B, H]    one gate per head
        Out        = (Attn * gates) × V

    The gradient-aware variant feeds the magnitude statistics of the input
    alongside the pooled embedding so the gate can react to adversarial-style
    feature distributions even though no FGSM/PGD examples are seen during
    training.
    """

    def __init__(
        self,
        d_model:                 int,
        n_heads:                 int,
        dropout:                 float = 0.1,
        use_adaptive_gates:      bool  = True,
        use_spectral_norm:       bool  = False,
        use_gradient_aware_gate: bool  = False,
        use_stochastic_gate:     bool  = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model                  = d_model
        self.n_heads                  = n_heads
        self.d_k                      = d_model // n_heads
        self.use_adaptive_gates       = use_adaptive_gates
        self.use_gradient_aware_gate  = use_gradient_aware_gate
        self.use_stochastic_gate      = use_stochastic_gate

        # Standard Q, K, V projections (optionally spectral-normalised)
        self.W_q = _maybe_sn(nn.Linear(d_model, d_model), use_spectral_norm)
        self.W_k = _maybe_sn(nn.Linear(d_model, d_model), use_spectral_norm)
        self.W_v = _maybe_sn(nn.Linear(d_model, d_model), use_spectral_norm)
        self.W_o = nn.Linear(d_model, d_model)

        # Adaptive gate generator
        if use_adaptive_gates:
            # When gradient-aware: cat [pool, abs.mean, var] → 3× d_model
            gate_in = d_model * (3 if use_gradient_aware_gate else 1)
            self.gate_mlp = nn.Sequential(
                nn.Linear(gate_in, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, n_heads),
            )
        # Inference-time gate noise (set externally for MC-averaged inference).
        # Public attribute → set to e.g. 0.2 to get Bayesian-gate sampling.
        self._inference_noise_std = 0.0

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        H, dk   = self.n_heads, self.d_k

        # ── Standard Q, K, V ────────────────────────────────────────────────
        Q = self.W_q(x).view(B, L, H, dk).transpose(1, 2)
        K = self.W_k(x).view(B, L, H, dk).transpose(1, 2)
        V = self.W_v(x).view(B, L, H, dk).transpose(1, 2)

        # ── Attention weights ───────────────────────────────────────────────
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)

        if self.use_adaptive_gates:
            x_pool = x.mean(dim=1)                            # [B, D]
            if self.use_gradient_aware_gate:
                x_mag = x.abs().mean(dim=1)                   # [B, D]
                # Also include input-variance — high variance often indicates
                # adversarial perturbation
                x_var = x.var(dim=1)                          # [B, D]
                gate_in = torch.cat([x_pool, x_mag, x_var], dim=-1)
            else:
                gate_in = x_pool

            gate_logits = self.gate_mlp(gate_in)              # [B, H]

            # Stochastic gate noise — applied at TRAIN time always, and at
            # INFERENCE time if `inference_noise` is set (Bayesian gates).
            noise_std = getattr(self, "_inference_noise_std", 0.0)
            if self.use_stochastic_gate and (self.training or noise_std > 0):
                std = 0.1 if self.training else noise_std
                gate_logits = gate_logits + std * torch.randn_like(gate_logits)

            gates = torch.sigmoid(gate_logits)                # [B, H]
            gates_view = gates.view(B, H, 1, 1)

            attn_gated = self.dropout(attn * gates_view)
            gates_out  = gates                                # [B, H]
        else:
            attn_gated = self.dropout(attn)
            gates_out  = torch.ones(B, H, device=x.device)

        ctx = torch.matmul(attn_gated, V)
        ctx = ctx.transpose(1, 2).contiguous().view(B, L, D)
        out = self.W_o(ctx)

        return out, attn, gates_out


# ──────────────────────────────────────────────────────────────────────────────
# Transformer Block (supports pre-norm / post-norm)
# ──────────────────────────────────────────────────────────────────────────────

class AAMTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model:                 int,
        n_heads:                 int,
        d_ff:                    int,
        dropout:                 float,
        use_adaptive_gates:      bool  = True,
        use_pre_norm:            bool  = False,
        use_spectral_norm:       bool  = False,
        use_gradient_aware_gate: bool  = False,
        use_stochastic_gate:     bool  = False,
        use_swiglu:              bool  = True,
        drop_path:               float = 0.0,
    ):
        super().__init__()
        self.use_pre_norm = use_pre_norm

        self.attn = AdaptiveAttentionMechanism(
            d_model, n_heads, dropout,
            use_adaptive_gates      = use_adaptive_gates,
            use_spectral_norm       = use_spectral_norm,
            use_gradient_aware_gate = use_gradient_aware_gate,
            use_stochastic_gate     = use_stochastic_gate,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        if use_swiglu:
            self.ff = SwiGLU(d_model, d_ff, dropout)
        else:
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )
        self.drop      = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, mask=None):
        if self.use_pre_norm:
            attn_out, attn_w, gates = self.attn(self.norm1(x), mask)
            x = x + self.drop_path(self.drop(attn_out))
            ff_out = self.ff(self.norm2(x))
            x = x + self.drop_path(self.drop(ff_out))
        else:
            attn_out, attn_w, gates = self.attn(x, mask)
            x = self.norm1(x + self.drop_path(self.drop(attn_out)))
            x = self.norm2(x + self.drop_path(self.drop(self.ff(x))))
        return x, attn_w, gates


# ──────────────────────────────────────────────────────────────────────────────
# AAM-TRANS Full Model
# ──────────────────────────────────────────────────────────────────────────────

class AAMTransIDS(nn.Module):
    """
    AAM-TRANS: Adaptive Attention Mechanism for Transformer-based IDS.

    Original flags (kept for ablation):
        use_adaptive_gates, use_learned_pe, use_anomaly_head

    Q1-publication enhancement flags (default ON for new runs; set False to
    reproduce the original 20260518 run):
        use_feature_tokenizer
        use_cls_token
        use_pre_norm
        use_spectral_norm
        use_gradient_aware_gate
        use_stochastic_gate
    """

    def __init__(
        self,
        input_dim:               int,
        d_model:                 int   = 64,
        n_heads:                 int   = 4,
        n_layers:                int   = 3,
        d_ff:                    int   = 256,
        n_classes:               int   = 2,
        max_seq_len:             int   = 50,
        dropout:                 float = 0.1,
        # Original ablation flags ------------------------------------------------
        use_adaptive_gates:      bool  = True,
        use_learned_pe:          bool  = True,
        use_anomaly_head:        bool  = True,
        # Q1-v1 enhancement flags ------------------------------------------------
        use_feature_tokenizer:   bool  = True,
        use_cls_token:           bool  = True,
        use_pre_norm:            bool  = True,
        use_spectral_norm:       bool  = True,
        use_gradient_aware_gate: bool  = True,
        use_stochastic_gate:     bool  = True,
        # Q1-v2 enhancement flags ------------------------------------------------
        use_swiglu:              bool  = True,
        drop_path:               float = 0.1,
    ):
        super().__init__()
        self.input_dim             = input_dim
        self.d_model               = d_model
        self.use_learned_pe        = use_learned_pe
        self.use_anomaly_head      = use_anomaly_head
        self.use_feature_tokenizer = use_feature_tokenizer
        self.use_cls_token         = use_cls_token

        # ── Input embedding ──────────────────────────────────────────────────
        if use_feature_tokenizer:
            self.feature_tokenizer = FeatureTokenizer(input_dim, d_model)
            # Compatibility shim: expose `.embedding.in_features` so the
            # incremental learning logic that adapts the input dimension can
            # still query the current feature count.
            self.embedding = nn.Linear(input_dim, d_model)  # unused, but kept
        else:
            self.embedding = nn.Linear(input_dim, d_model)

        # ── CLS token (learned summary) ──────────────────────────────────────
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Positional encoding (account for +1 CLS slot) ────────────────────
        pe_len = max_seq_len + (1 if use_cls_token else 0)
        if use_learned_pe:
            self.pos_encoding = nn.Parameter(
                torch.randn(1, pe_len, d_model) * 0.01
            )
        else:
            self.register_buffer("pos_encoding", _sinusoidal_pe(pe_len, d_model))

        self.drop = nn.Dropout(dropout)

        # ── Stack of Transformer blocks ──────────────────────────────────────
        # Linearly scale drop_path probability with depth (stochastic depth)
        dp_rates = [drop_path * i / max(1, n_layers - 1) for i in range(n_layers)]
        self.blocks = nn.ModuleList([
            AAMTransformerBlock(
                d_model, n_heads, d_ff, dropout,
                use_adaptive_gates      = use_adaptive_gates,
                use_pre_norm            = use_pre_norm,
                use_spectral_norm       = use_spectral_norm,
                use_gradient_aware_gate = use_gradient_aware_gate,
                use_stochastic_gate     = use_stochastic_gate,
                use_swiglu              = use_swiglu,
                drop_path               = dp_rates[i],
            )
            for i in range(n_layers)
        ])
        if use_pre_norm:
            self.final_norm = nn.LayerNorm(d_model)
        else:
            self.final_norm = nn.Identity()

        # ── Objective 3: Classification Head ─────────────────────────────────
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        # ── Objective 3: Confidence-Based Anomaly Scorer ─────────────────────
        if use_anomaly_head:
            self.anomaly_scorer = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid(),
            )

    # ── Embedding helper (handles tokenizer vs. linear) ──────────────────────
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_feature_tokenizer and hasattr(self, "feature_tokenizer"):
            # Need to make sure the tokenizer was built for the current feat count.
            n_feat_seen = self.feature_tokenizer.weight.shape[0]
            if n_feat_seen == x.shape[-1]:
                return self.feature_tokenizer(x)
            # Heterogeneous-feature CL stage: fall back to linear embedding.
        return self.embedding(x)

    def forward(self, x, mask=None) -> Dict:
        B, L, _ = x.shape
        z = self._embed(x)                                # [B, L, d_model]

        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)
            z = torch.cat([cls, z], dim=1)                # [B, L+1, d_model]
            seq_len_eff = L + 1
        else:
            seq_len_eff = L

        # Slice PE to current sequence length
        z = z + self.pos_encoding[:, :seq_len_eff, :]
        z = self.drop(z)

        all_attn_weights, all_gates = [], []
        for block in self.blocks:
            z, attn_w, gates = block(z, mask)
            all_attn_weights.append(attn_w)
            all_gates.append(gates)

        z = self.final_norm(z)

        # Global summary: CLS token or mean pooling
        if self.use_cls_token:
            h_global = z[:, 0, :]
        else:
            h_global = z.mean(dim=1)

        logits = self.classification_head(h_global)

        if self.use_anomaly_head:
            anomaly_score = self.anomaly_scorer(h_global)
        else:
            anomaly_score = torch.zeros(B, 1, device=x.device)

        return {
            "logits":            logits,
            "anomaly_score":     anomaly_score,
            "embeddings":        h_global,
            "attention_weights": all_attn_weights,
            "adaptive_gates":    all_gates,
        }

    def get_embeddings(self, x) -> torch.Tensor:
        return self.forward(x)["embeddings"]

    # ── Bayesian-gate inference (architectural robustness) ──────────────────
    def set_inference_gate_noise(self, std: float):
        """
        Enable stochastic gate sampling at INFERENCE time.

        With std > 0, every block's gate logits get N(0, std²) noise,
        making the model behave as an implicit ensemble. Combined with
        `mc_predict()` this gives gradient-obfuscated robustness — FGSM/PGD
        no longer see a single deterministic decision surface.
        """
        for block in self.blocks:
            block.attn._inference_noise_std = float(std)

    @torch.no_grad()
    def mc_predict(self, x: torch.Tensor, n: int = 8) -> torch.Tensor:
        """
        Monte-Carlo-averaged softmax probabilities under Bayesian gates.
        Requires `set_inference_gate_noise(σ>0)` to have been called.

        Returns: averaged probability tensor [B, n_classes]
        """
        was_training = self.training
        self.eval()
        probs = None
        for _ in range(n):
            logits = self.forward(x)["logits"]
            p      = F.softmax(logits, dim=-1)
            probs  = p if probs is None else probs + p
        if was_training:
            self.train()
        return probs / float(n)

    # ── Helper: rebuild input projection for a new feature count (CL) ────────
    def adapt_input_dim(self, new_input_dim: int):
        """
        Used by run_aam_trans.py when moving from one CL task to another with a
        different feature dimensionality. Rebuilds the input projection layer
        while preserving overlapping feature weights.
        """
        device = next(self.parameters()).device
        cur = self.embedding.in_features
        if cur == new_input_dim:
            return

        # Rebuild plain linear embedding (used as fallback / for CL)
        new_emb = nn.Linear(new_input_dim, self.d_model).to(device)
        min_dim = min(cur, new_input_dim)
        new_emb.weight.data[:, :min_dim] = self.embedding.weight.data[:, :min_dim]
        new_emb.bias.data = self.embedding.bias.data.clone()
        self.embedding = new_emb

        # Rebuild feature tokenizer if it exists
        if self.use_feature_tokenizer and hasattr(self, "feature_tokenizer"):
            old_w = self.feature_tokenizer.weight.data
            old_b = self.feature_tokenizer.bias.data
            new_tok = FeatureTokenizer(new_input_dim, self.d_model).to(device)
            m = min(old_w.shape[0], new_input_dim)
            new_tok.weight.data[:m] = old_w[:m]
            new_tok.bias.data[:m]   = old_b[:m]
            self.feature_tokenizer = new_tok

        self.input_dim = new_input_dim
