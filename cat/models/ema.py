"""
ema.py — Exponential Moving Average of model weights.

Typical recipe:
    ema = ModelEMA(model, decay=0.999)
    for x, y in loader:
        loss = train_step(model, x, y); optimizer.step()
        ema.update(model)
    # at eval time:
    val_acc = evaluate(ema.module, val_loader)

EMA weights are usually 0.1–1.0 % better than the raw weights at convergence
and they smooth out the noisy late-epoch optimizer trajectory.
"""

from __future__ import annotations
import copy
from typing import Optional
import torch
import torch.nn as nn


class ModelEMA:
    """Standard EMA of parameters and buffers (BN stats included)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay  = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd)
