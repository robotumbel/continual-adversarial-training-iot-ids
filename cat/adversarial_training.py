"""
adversarial_training.py — PGD-based Adversarial Training (AT) for AAM-TRANS.

This is the defense that the SLR corpus and our own four failed non-AT
attempts both point to as the only reliable path. It is used in two roles:

  * Paper 2 (negative-result study) — as the POSITIVE CONTROL (V7): the
    one variant that actually works, against which all non-AT defenses
    are shown to fail.
  * Paper 3 (AAM-TRANS + AT) — as the MAIN METHOD.

Two training objectives are provided:
  standard PGD-AT  (Madry et al., ICLR 2018):  min_w  E[ CE(f(x_adv), y) ]
  TRADES           (Zhang et al., ICML 2019):  CE(f(x),y) + beta * KL(f(x)||f(x_adv))

x_adv is produced by PGD; perturbations are L-inf, epsilon-bounded.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# PGD perturbation for TRAINING (lean, in-graph, on-device)
# ──────────────────────────────────────────────────────────────────────────────

def pgd_perturb(
    model:   nn.Module,
    x:       torch.Tensor,
    y:       torch.Tensor,
    epsilon: float = 0.10,
    alpha:   float = 0.02,
    steps:   int   = 7,
    random_start: bool = True,
) -> torch.Tensor:
    """
    L-infinity PGD adversarial example generation for adversarial training.

    Returns x_adv (detached, same device as x). The model is left in
    whatever mode the caller set; gradients flow only w.r.t. the input.
    """
    was_training = model.training
    model.eval()                       # freeze BN/dropout during PGD search

    x_orig = x.detach()
    if random_start:
        delta = torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
        x_adv = (x_orig + delta).detach()
    else:
        x_adv = x_orig.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)["logits"]
        loss   = F.cross_entropy(logits, y)
        grad   = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x_orig - epsilon),
                              x_orig + epsilon)
        x_adv = x_adv.detach()

    if was_training:
        model.train()
    return x_adv


# ──────────────────────────────────────────────────────────────────────────────
# Loss objectives
# ──────────────────────────────────────────────────────────────────────────────

def standard_at_loss(model, x, y, epsilon, alpha, steps,
                     cls_loss_fn=None) -> torch.Tensor:
    """Madry-style PGD-AT: train purely on the adversarial example."""
    x_adv  = pgd_perturb(model, x, y, epsilon, alpha, steps)
    logits = model(x_adv)["logits"]
    if cls_loss_fn is not None:
        return cls_loss_fn(logits, y)
    return F.cross_entropy(logits, y)


def trades_loss(model, x, y, epsilon, alpha, steps, beta=6.0,
                cls_loss_fn=None) -> torch.Tensor:
    """
    TRADES objective (Zhang et al., 2019):

        L = CE(f(x), y) + beta * KL( f(x) || f(x_adv) )

    where x_adv is found by PGD maximising the KL term. beta trades clean
    accuracy against robustness (paper default beta = 6).
    """
    # ── find x_adv that maximises KL( f(x) || f(x_adv) ) ────────────────────
    was_training = model.training
    model.eval()
    with torch.no_grad():
        p_clean = F.softmax(model(x)["logits"], dim=-1)

    x_orig = x.detach()
    x_adv  = (x_orig + 0.001 * torch.randn_like(x_orig)).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logp_adv = F.log_softmax(model(x_adv)["logits"], dim=-1)
        kl = F.kl_div(logp_adv, p_clean, reduction="batchmean")
        grad = torch.autograd.grad(kl, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x_orig - epsilon),
                              x_orig + epsilon)
        x_adv = x_adv.detach()
    if was_training:
        model.train()

    # ── combined loss ───────────────────────────────────────────────────────
    logits_clean = model(x)["logits"]
    ce = cls_loss_fn(logits_clean, y) if cls_loss_fn is not None \
        else F.cross_entropy(logits_clean, y)
    logp_adv = F.log_softmax(model(x_adv)["logits"], dim=-1)
    p_clean2 = F.softmax(logits_clean, dim=-1)
    kl = F.kl_div(logp_adv, p_clean2, reduction="batchmean")
    return ce + beta * kl


# ──────────────────────────────────────────────────────────────────────────────
# Adversarial training epoch + full loop
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch_adversarial(
    model, loader, optimizer, device,
    epsilon: float = 0.10, alpha: float = 0.02, steps: int = 7,
    method:  str   = "trades", beta: float = 6.0,
    anomaly_weight: float = 0.3, grad_clip: float = 1.0,
    cls_loss_fn = None,
) -> Dict[str, float]:
    """One epoch of adversarial training. method = 'standard' | 'trades'."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if method == "standard":
            loss = standard_at_loss(model, x, y, epsilon, alpha, steps,
                                    cls_loss_fn)
        else:
            loss = trades_loss(model, x, y, epsilon, alpha, steps, beta,
                               cls_loss_fn)

        # auxiliary anomaly-head loss on clean inputs
        out = model(x)
        anm = F.binary_cross_entropy(out["anomaly_score"].squeeze(-1),
                                     (y > 0).float())
        loss = loss + anomaly_weight * anm

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        with torch.no_grad():
            preds = out["logits"].argmax(1)
            correct += int((preds == y).sum())
            total   += int(y.size(0))

    return {"loss": total_loss / max(1, len(loader)),
            "accuracy": correct / max(1, total)}


def train_model_adversarial(
    model, train_loader: DataLoader, val_loader: DataLoader,
    device: torch.device,
    epochs: int = 20, learning_rate: float = 1e-3, weight_decay: float = 1e-2,
    epsilon: float = 0.10, alpha: float = 0.02, steps: int = 7,
    method:  str = "trades", beta: float = 6.0,
    anomaly_weight: float = 0.3, grad_clip: float = 1.0,
    patience: int = 7, save_path: Optional[str] = None,
    model_name: str = "AAM-TRANS-AT", cls_loss_fn = None,
) -> Dict:
    """
    Full adversarial-training loop. Validation uses *robust* accuracy
    (accuracy on PGD-perturbed validation data) for early stopping, since
    that is the quantity AT is meant to optimise.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_robust_acc": []}
    best, no_improve = -1.0, 0

    for epoch in range(1, epochs + 1):
        st = train_epoch_adversarial(
            model, train_loader, optimizer, device,
            epsilon=epsilon, alpha=alpha, steps=steps,
            method=method, beta=beta, anomaly_weight=anomaly_weight,
            grad_clip=grad_clip, cls_loss_fn=cls_loss_fn,
        )
        scheduler.step()

        # ── robust validation accuracy ──
        model.eval()
        rc, rt = 0, 0
        for xv, yv in val_loader:
            xv = xv.to(device); yv = yv.to(device)
            xv_adv = pgd_perturb(model, xv, yv, epsilon, alpha, steps)
            with torch.no_grad():
                pv = model(xv_adv)["logits"].argmax(1)
            rc += int((pv == yv).sum()); rt += int(yv.size(0))
        rob = rc / max(1, rt)

        history["train_loss"].append(st["loss"])
        history["train_acc"].append(st["accuracy"])
        history["val_robust_acc"].append(rob)
        logger.info(f"[{model_name}] Epoch {epoch:3d}/{epochs} | "
                    f"Loss={st['loss']:.4f} | TrainAcc={st['accuracy']*100:.2f}% "
                    f"| ValRobustAcc={rob*100:.2f}%")

        if rob > best:
            best, no_improve = rob, 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

    if save_path:
        import os
        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path, map_location=device))
    history["best_val_robust_acc"] = best
    return history
