"""
training.py — Training and evaluation loops for all models.

Provides:
  train_epoch()             — one training epoch (standard + Q1 enhancements)
  train_epoch_incremental() — one training epoch with EWC+KD+replay
  evaluate_epoch()          — validation accuracy
  train_model()             — full training loop with early stopping + checkpoint
  train_model_two_stage()   — decoupled representation+classifier training (BBN/cRT)

Q1-publication enhancements (all opt-in via TrainingConfig flags, default ON):
  * AdamW + cosine schedule with linear warm-up
  * Class-Balanced Focal Loss (replaces CE when samples_per_class is provided)
  * Label smoothing
  * Manifold-style input mixup
  * TRADES KL regularization (smoothness w.r.t. small Gaussian noise)
  * EMA of model weights for evaluation
  * Stochastic weight averaging via EMA `.module`
"""

from __future__ import annotations

import os
import math
import logging
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from losses import (
    ClassBalancedFocalLoss,
    LabelSmoothingCrossEntropy,
    TRADESRegularizer,
    JacobianRegularizer,
    ConfidencePenalty,
    mixup_data,
    mixup_criterion,
    gaussian_noise_augment,
    cutmix_features,
    compute_samples_per_class,
)
from imbalance import LDAMLoss, LogitAdjustedCrossEntropy
from models.ema import ModelEMA
from models.sam import SAM

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Loss factory
# ──────────────────────────────────────────────────────────────────────────────

def build_classification_loss(
    samples_per_class: Optional[np.ndarray] = None,
    n_classes:         int    = 2,
    # V3.5 imbalance-loss selector — order of priority:
    #   ldam → logit_adjust → focal → label-smooth CE → vanilla CE
    use_ldam:          bool   = False,
    ldam_max_m:        float  = 0.5,
    ldam_s:            float  = 30.0,
    use_logit_adjust:  bool   = False,
    logit_adjust_tau:  float  = 1.0,
    use_focal:         bool   = True,
    focal_gamma:       float  = 2.0,
    focal_beta:        float  = 0.9999,
    label_smoothing:   float  = 0.1,
) -> nn.Module:
    """
    Pick the appropriate classification loss given config.

    Priority order:
      1. LDAM Loss          (severe long-tailed imbalance)
      2. Logit Adjustment   (mild-to-moderate imbalance)
      3. Class-Balanced Focal Loss
      4. Label-Smoothed CE
      5. Vanilla CE
    """
    if use_ldam and samples_per_class is not None:
        return LDAMLoss(
            samples_per_class=samples_per_class,
            max_m=ldam_max_m, s=ldam_s,
        )
    if use_logit_adjust and samples_per_class is not None:
        return LogitAdjustedCrossEntropy(
            samples_per_class=samples_per_class,
            tau=logit_adjust_tau,
            label_smoothing=label_smoothing,
        )
    if use_focal and samples_per_class is not None:
        return ClassBalancedFocalLoss(
            samples_per_class=samples_per_class,
            beta=focal_beta,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )
    if label_smoothing > 0:
        return LabelSmoothingCrossEntropy(epsilon=label_smoothing)
    return nn.CrossEntropyLoss()


# ──────────────────────────────────────────────────────────────────────────────
# Cosine schedule with linear warm-up
# ──────────────────────────────────────────────────────────────────────────────

def cosine_warmup_lambda(
    current_step:  int,
    warmup_steps:  int,
    total_steps:   int,
    min_lr_ratio:  float = 0.01,
) -> float:
    if current_step < warmup_steps:
        return float(current_step + 1) / float(max(1, warmup_steps))
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_cosine_scheduler(
    optimizer,
    warmup_steps:  int,
    total_steps:   int,
    min_lr_ratio:  float = 0.01,
):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_warmup_lambda(
            s, warmup_steps, total_steps, min_lr_ratio
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Standard training epoch (Q1-enhanced)
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model:            nn.Module,
    loader:           DataLoader,
    optimizer:        torch.optim.Optimizer,
    device:           torch.device,
    cls_loss_fn:      nn.Module,
    anomaly_weight:   float                  = 0.3,
    mixup_alpha:      float                  = 0.0,
    cutmix_alpha:     float                  = 0.0,
    cutmix_p:         float                  = 0.5,
    noise_aug_eps:    float                  = 0.0,
    noise_aug_p:      float                  = 0.0,
    trades_reg:       Optional[nn.Module]    = None,
    jacobian_reg:     Optional[nn.Module]    = None,
    confidence_pen:   Optional[nn.Module]    = None,
    scheduler                                = None,
    ema:              Optional[ModelEMA]     = None,
    grad_clip:        float                  = 1.0,
    use_sam:          bool                   = False,
) -> Dict[str, float]:
    """
    One training epoch. Supports a full regularization toolkit:
      * mixup / cutmix
      * Gaussian noise augmentation (input-space)
      * TRADES smoothness regularizer
      * Jacobian (input-gradient norm) regularizer
      * Confidence penalty
      * SAM 2-step optimizer
    All optional via flags.
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    def _compute_loss(_x, _y):
        """Single forward; returns (loss, logits, ce, anm)."""
        # Augmentations applied INSIDE so SAM second step recomputes the same.
        x_aug = _x

        # Random Gaussian noise augmentation (NOT adversarial)
        if noise_aug_eps > 0 and noise_aug_p > 0:
            x_aug = gaussian_noise_augment(x_aug, eps_max=noise_aug_eps, p=noise_aug_p)

        # Mixup / CutMix — mutually exclusive per batch
        use_cutmix = cutmix_alpha > 0 and (torch.rand(1).item() < cutmix_p)
        if use_cutmix:
            x_aug, ya, yb, lam = cutmix_features(x_aug, _y, alpha=cutmix_alpha, p=1.0)
            do_mix = True
        elif mixup_alpha > 0:
            x_aug, ya, yb, lam = mixup_data(x_aug, _y, alpha=mixup_alpha)
            do_mix = True
        else:
            ya, yb, lam, do_mix = _y, _y, 1.0, False

        out     = model(x_aug)
        logits  = out["logits"]
        anomaly = out["anomaly_score"]

        if do_mix:
            ce = mixup_criterion(cls_loss_fn, logits, ya, yb, lam)
        else:
            ce = cls_loss_fn(logits, _y)

        y_bin = (_y > 0).float()
        anm   = F.binary_cross_entropy(anomaly.squeeze(-1), y_bin)

        total = ce + anomaly_weight * anm
        if confidence_pen is not None:
            total = total + confidence_pen(logits)
        if trades_reg is not None:
            total = total + trades_reg(model, _x)
        if jacobian_reg is not None:
            total = total + jacobian_reg(model, _x, _y)
        return total, logits, ce, anm

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_sam and isinstance(optimizer, SAM):
            # ── SAM 2-step ──
            loss, logits, _, _ = _compute_loss(x, y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.first_step(zero_grad=True)

            loss2, logits, _, _ = _compute_loss(x, y)
            loss2.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.second_step(zero_grad=True)
        else:
            loss, logits, _, _ = _compute_loss(x, y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        with torch.no_grad():
            preds    = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)

    return {
        "loss":     total_loss / max(1, len(loader)),
        "accuracy": correct / max(1, total),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Incremental training epoch (Objective 2)  — kept signature-compatible
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch_incremental(
    model,
    loader:              DataLoader,
    optimizer:           torch.optim.Optimizer,
    device:              torch.device,
    incremental_trainer,
    memory_loader:       Optional[DataLoader],
    anomaly_weight:      float = 0.3,
    scheduler                  = None,
    ema:                 Optional[ModelEMA] = None,
    grad_clip:           float = 1.0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    mem_iter = iter(memory_loader) if memory_loader else None

    for x_new, y_new in loader:
        x_new = x_new.to(device, non_blocking=True)
        y_new = y_new.to(device, non_blocking=True)

        if mem_iter is not None:
            try:
                x_mem, y_mem = next(mem_iter)
            except StopIteration:
                mem_iter = iter(memory_loader)
                x_mem, y_mem = next(mem_iter)
            x_mem = x_mem.to(device, non_blocking=True)
            y_mem = y_mem.to(device, non_blocking=True)

            f_new, f_mem = x_new.shape[-1], x_mem.shape[-1]
            if f_mem < f_new:
                pad = torch.zeros(*x_mem.shape[:-1], f_new - f_mem, device=device)
                x_mem = torch.cat([x_mem, pad], dim=-1)
            elif f_mem > f_new:
                x_mem = x_mem[..., :f_new]

            x_batch = torch.cat([x_new, x_mem], dim=0)
            y_batch = torch.cat([y_new, y_mem], dim=0)
        else:
            x_batch, y_batch = x_new, y_new

        optimizer.zero_grad(set_to_none=True)
        loss, _ = incremental_trainer.compute_total_loss(
            model, x_batch, y_batch, anomaly_weight
        )
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        with torch.no_grad():
            preds    = model(x_new)["logits"].argmax(dim=1)
            correct += (preds == y_new).sum().item()
            total   += y_new.size(0)

    return {
        "loss":     total_loss / max(1, len(loader)),
        "accuracy": correct / max(1, total),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation epoch
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_epoch(
    model,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out  = model(x)
            pred = out["logits"].argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy   = float((all_preds == all_labels).mean())
    return accuracy, all_preds, all_labels


# ──────────────────────────────────────────────────────────────────────────────
# Full training loop with early stopping + Q1 enhancements
# ──────────────────────────────────────────────────────────────────────────────

def train_model(
    model,
    train_loader:        DataLoader,
    val_loader:          DataLoader,
    device:              torch.device,
    epochs:              int   = 15,
    learning_rate:       float = 1e-3,
    weight_decay:        float = 1e-2,
    patience:            int   = 5,
    anomaly_weight:      float = 0.3,
    save_path:           Optional[str] = None,
    model_name:          str   = "model",
    # ── Loss / sampling ──
    # NOTE: All v2 regularizers default to OFF so that legacy callers
    # (e.g. baseline RNN/LSTM training, which cannot double-backward through
    # cuDNN) get a safe vanilla recipe. AAM-TRANS opts in explicitly via
    # train_kwargs in run_aam_trans.py.
    samples_per_class:   Optional[np.ndarray] = None,
    n_classes:           int   = 2,
    # ── V3.5 imbalance-loss selectors ──
    use_ldam:            bool  = False,
    ldam_max_m:          float = 0.5,
    ldam_s:              float = 30.0,
    use_logit_adjust:    bool  = False,
    logit_adjust_tau:    float = 1.0,
    use_focal:           bool  = False,
    focal_gamma:         float = 1.0,
    focal_beta:          float = 0.9999,
    label_smoothing:     float = 0.0,
    # ── Augmentation ──
    mixup_alpha:         float = 0.0,
    cutmix_alpha:        float = 0.0,
    cutmix_p:            float = 0.5,
    noise_aug_eps:       float = 0.0,
    noise_aug_p:         float = 0.0,
    # ── Smoothness / robustness ──
    use_trades:          bool  = False,
    trades_sigma:        float = 0.10,
    trades_beta:         float = 6.0,
    use_jacobian_reg:    bool  = False,
    jacobian_weight:     float = 0.05,
    use_confidence_pen:  bool  = False,
    confidence_weight:   float = 0.05,
    # ── Optimizer ──
    use_adamw:           bool  = True,
    use_sam:             bool  = False,
    sam_rho:             float = 0.05,
    use_cosine_schedule: bool  = True,
    warmup_frac:         float = 0.10,
    grad_clip:           float = 1.0,
    # ── EMA ──
    use_ema:             bool  = True,
    ema_decay:           float = 0.999,
) -> Dict:
    # ── Optimizer ────────────────────────────────────────────────────────────
    if use_adamw:
        base_opt = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        base_opt = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    if use_sam:
        optimizer = SAM(model.parameters(), base_opt, rho=sam_rho, adaptive=False)
    else:
        optimizer = base_opt

    # ── Scheduler (attach to BASE optimizer; SAM proxies to it) ──────────────
    sched_target = base_opt
    if use_cosine_schedule:
        steps_per_epoch = max(1, len(train_loader))
        total_steps     = steps_per_epoch * epochs
        warmup_steps    = int(total_steps * warmup_frac)
        scheduler = make_cosine_scheduler(
            sched_target, warmup_steps=warmup_steps, total_steps=total_steps
        )
        plateau_sched = None
    else:
        scheduler     = None
        plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            sched_target, mode="max", factor=0.5, patience=3
        )

    # ── Loss + regularizers ──────────────────────────────────────────────────
    cls_loss_fn = build_classification_loss(
        samples_per_class=samples_per_class, n_classes=n_classes,
        use_ldam=use_ldam, ldam_max_m=ldam_max_m, ldam_s=ldam_s,
        use_logit_adjust=use_logit_adjust, logit_adjust_tau=logit_adjust_tau,
        use_focal=use_focal, focal_gamma=focal_gamma, focal_beta=focal_beta,
        label_smoothing=label_smoothing,
    )
    trades_reg   = TRADESRegularizer(sigma=trades_sigma, beta=trades_beta) \
                       if use_trades         else None

    # ── Safety: Jacobian reg needs double-backward, but cuDNN RNN/LSTM/GRU
    # do not support it (NotImplementedError on _cudnn_rnn_backward).
    # Force-disable for any model containing such modules.
    has_rnn = any(
        isinstance(m, (nn.RNN, nn.LSTM, nn.GRU,
                       nn.RNNCell, nn.LSTMCell, nn.GRUCell))
        for m in model.modules()
    )
    if use_jacobian_reg and has_rnn:
        logger.warning(
            f"[{model_name}] Jacobian reg disabled — model contains cuDNN "
            f"RNN/LSTM/GRU which does not support double backward."
        )
        use_jacobian_reg = False

    jacobian_reg = JacobianRegularizer(weight=jacobian_weight) \
                       if use_jacobian_reg   else None
    conf_pen     = ConfidencePenalty(weight=confidence_weight) \
                       if use_confidence_pen else None

    # ── EMA ──────────────────────────────────────────────────────────────────
    ema = ModelEMA(model, decay=ema_decay) if use_ema else None

    history    = {"train_loss": [], "train_acc": [], "val_acc": []}
    best_val   = 0.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        train_stats = train_epoch(
            model=model, loader=train_loader, optimizer=optimizer, device=device,
            cls_loss_fn=cls_loss_fn, anomaly_weight=anomaly_weight,
            mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, cutmix_p=cutmix_p,
            noise_aug_eps=noise_aug_eps, noise_aug_p=noise_aug_p,
            trades_reg=trades_reg, jacobian_reg=jacobian_reg,
            confidence_pen=conf_pen,
            scheduler=scheduler, ema=ema, grad_clip=grad_clip,
            use_sam=use_sam,
        )

        # Validate using EMA weights if available (slightly cleaner signal)
        eval_model = ema.module if ema is not None else model
        val_acc, _, _ = evaluate_epoch(eval_model, val_loader, device)

        if plateau_sched is not None:
            plateau_sched.step(val_acc)

        history["train_loss"].append(train_stats["loss"])
        history["train_acc"].append(train_stats["accuracy"])
        history["val_acc"].append(val_acc)

        logger.info(
            f"[{model_name}] Epoch {epoch:3d}/{epochs} | "
            f"Loss={train_stats['loss']:.4f} | "
            f"TrainAcc={train_stats['accuracy']*100:.2f}% | "
            f"ValAcc(EMA)={val_acc*100:.2f}%"
        )

        if val_acc > best_val:
            best_val   = val_acc
            no_improve = 0
            if save_path:
                state = eval_model.state_dict()
                torch.save(state, save_path)
                logger.info(f"  -> Saved best model ({val_acc*100:.2f}%)")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

    if save_path and os.path.exists(save_path):
        # Load best (EMA-weight) checkpoint back into the live model so all
        # downstream code (attacks, adversarial eval, EWC) operates on the same
        # weights that achieved best_val.
        model.load_state_dict(torch.load(save_path, map_location=device))

    history["best_val_acc"] = best_val
    return history


# ──────────────────────────────────────────────────────────────────────────────
# Two-Stage (Decoupled) Training — BBN / cRT
# ──────────────────────────────────────────────────────────────────────────────

def train_model_two_stage(
    model,
    train_loader_uniform: DataLoader,
    train_loader_balanced: DataLoader,
    val_loader:            DataLoader,
    device:                torch.device,
    epochs_stage1:         int = 12,
    epochs_stage2:         int = 4,
    learning_rate:         float = 1e-3,
    weight_decay:          float = 1e-2,
    save_path:             Optional[str] = None,
    model_name:            str   = "model",
    **kwargs,
) -> Dict:
    """
    BBN/cRT-style two-stage training for severe imbalance (CICIoMT multi-class).

    Stage 1: train the whole network on the natural (instance-balanced) loader.
    Stage 2: freeze the encoder; re-initialise the classification head and
             train ONLY the head on a class-balanced loader. This typically
             lifts minority-class recall dramatically without hurting majority.
    """
    logger.info(f"[{model_name}] Two-stage training: Stage 1 (full network, "
                f"{epochs_stage1} epochs)…")
    hist1 = train_model(
        model=model,
        train_loader=train_loader_uniform, val_loader=val_loader, device=device,
        epochs=epochs_stage1, learning_rate=learning_rate,
        weight_decay=weight_decay, save_path=save_path,
        model_name=f"{model_name}_s1", **kwargs,
    )

    logger.info(f"[{model_name}] Two-stage training: Stage 2 "
                f"(classifier-only, {epochs_stage2} epochs)…")
    # Freeze everything except the classification head
    for name, p in model.named_parameters():
        p.requires_grad_(False)
    for p in model.classification_head.parameters():
        p.requires_grad_(True)

    # Re-init final linear of the head for a clean retrain
    last_linear = model.classification_head[-1]
    if isinstance(last_linear, nn.Linear):
        nn.init.normal_(last_linear.weight, std=0.01)
        nn.init.zeros_(last_linear.bias)

    s2_kwargs = dict(kwargs)
    # Disable expensive regularizers in stage 2 — head-only training.
    s2_kwargs["mixup_alpha"]        = 0.0
    s2_kwargs["cutmix_alpha"]       = 0.0
    s2_kwargs["use_trades"]         = False
    s2_kwargs["use_jacobian_reg"]   = False
    s2_kwargs["use_confidence_pen"] = False
    s2_kwargs["use_sam"]            = False
    s2_kwargs["noise_aug_eps"]      = 0.0
    hist2 = train_model(
        model=model,
        train_loader=train_loader_balanced, val_loader=val_loader, device=device,
        epochs=epochs_stage2, learning_rate=learning_rate * 0.1,
        weight_decay=weight_decay, save_path=save_path,
        model_name=f"{model_name}_s2", **s2_kwargs,
    )

    # Un-freeze for downstream tasks
    for p in model.parameters():
        p.requires_grad_(True)

    history = {
        "stage1": hist1,
        "stage2": hist2,
        "best_val_acc": max(hist1["best_val_acc"], hist2["best_val_acc"]),
        "train_loss": hist1["train_loss"] + hist2["train_loss"],
        "train_acc":  hist1["train_acc"]  + hist2["train_acc"],
        "val_acc":    hist1["val_acc"]    + hist2["val_acc"],
    }
    return history


# ──────────────────────────────────────────────────────────────────────────────
# Incremental training loop (full, multi-task)
# ──────────────────────────────────────────────────────────────────────────────

def train_incremental_task(
    model,
    new_train_loader: DataLoader,
    new_val_loader:   DataLoader,
    device:           torch.device,
    incremental_trainer,
    memory_loader:    Optional[DataLoader],
    epochs:           int   = 10,
    learning_rate:    float = 5e-4,
    weight_decay:     float = 1e-2,
    patience:         int   = 5,
    anomaly_weight:   float = 0.3,
    save_path:        Optional[str] = None,
    model_name:       str  = "aam_trans_incremental",
    use_ema:          bool = True,
    ema_decay:        float = 0.999,
    grad_clip:        float = 1.0,
) -> Dict:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    ema = ModelEMA(model, decay=ema_decay) if use_ema else None

    history    = {"train_loss": [], "train_acc": [], "val_acc": []}
    best_val   = 0.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        train_stats = train_epoch_incremental(
            model=model, loader=new_train_loader, optimizer=optimizer,
            device=device, incremental_trainer=incremental_trainer,
            memory_loader=memory_loader, anomaly_weight=anomaly_weight,
            scheduler=None, ema=ema, grad_clip=grad_clip,
        )

        eval_model = ema.module if ema is not None else model
        val_acc, _, _ = evaluate_epoch(eval_model, new_val_loader, device)
        plateau_sched.step(val_acc)

        history["train_loss"].append(train_stats["loss"])
        history["train_acc"].append(train_stats["accuracy"])
        history["val_acc"].append(val_acc)

        logger.info(
            f"[{model_name}] Incremental Epoch {epoch:3d}/{epochs} | "
            f"Loss={train_stats['loss']:.4f} | "
            f"TrainAcc={train_stats['accuracy']*100:.2f}% | "
            f"ValAcc(EMA)={val_acc*100:.2f}%"
        )

        if val_acc > best_val:
            best_val   = val_acc
            no_improve = 0
            if save_path:
                torch.save(eval_model.state_dict(), save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    history["best_val_acc"] = best_val
    return history
