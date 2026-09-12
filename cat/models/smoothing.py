"""
smoothing.py — Randomized Smoothing inference for certified robustness.

Cohen et al. (ICML 2019): if f is any classifier and g(x) = argmax_c P[f(x+η)=c]
for η ~ N(0, σ² I), then g is provably robust within an L2 radius
    R = σ · Φ⁻¹(p_A)
where p_A is the (lower-bounded) prediction probability of the top class.

We expose two utilities:

    smooth_predict       — Monte-Carlo majority vote (n samples, σ)
    certified_radius     — returns (predicted_class, certified_radius)
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from typing import Tuple


@torch.no_grad()
def smooth_predict(
    model,
    x:      torch.Tensor,
    sigma:  float = 0.05,
    n:      int   = 16,
    batch:  int   = 4,
) -> torch.Tensor:
    """
    Smoothed prediction by Monte-Carlo averaging over n Gaussian-noise samples.

    Args:
        model : returns {"logits": [B, C], ...}
        x     : [B, L, F]
        sigma : noise std
        n     : total MC samples
        batch : how many noisy copies to forward at once (memory tradeoff)

    Returns:
        averaged softmax probabilities [B, C]
    """
    model.eval()
    B = x.size(0)
    sum_probs = None
    remaining = n
    while remaining > 0:
        k = min(batch, remaining)
        x_rep = x.unsqueeze(0).expand(k, *x.shape).reshape(k * B, *x.shape[1:])
        noise = torch.randn_like(x_rep) * sigma
        logits = model(x_rep + noise)["logits"]
        probs  = F.softmax(logits, dim=-1).view(k, B, -1).mean(dim=0)
        sum_probs = probs if sum_probs is None else sum_probs + probs
        remaining -= k
    return sum_probs / math.ceil(n / batch)


@torch.no_grad()
def certified_radius(
    model,
    x:     torch.Tensor,
    sigma: float = 0.05,
    n:     int   = 100,
    batch: int   = 8,
    alpha: float = 0.001,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (preds, radius) where radius is a lower bound on the L2-perturbation
    that the smoothed classifier provably resists.

    Uses Hoeffding lower-bound on p_A with confidence (1-α).
    """
    probs = smooth_predict(model, x, sigma=sigma, n=n, batch=batch)
    preds = probs.argmax(dim=-1)
    pA    = probs.gather(1, preds.unsqueeze(1)).squeeze(1)
    # Hoeffding: pA_lower = pA - sqrt(log(1/alpha) / (2n))
    pA_low = (pA - math.sqrt(math.log(1.0 / alpha) / (2 * n))).clamp(1e-6, 1 - 1e-6)
    # R = σ Φ⁻¹(pA_low) — use torch.distributions.Normal icdf
    normal = torch.distributions.Normal(0.0, 1.0)
    radius = sigma * normal.icdf(pA_low)
    radius = torch.where(pA_low > 0.5, radius, torch.zeros_like(radius))
    return preds, radius
