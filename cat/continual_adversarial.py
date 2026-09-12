"""
continual_adversarial.py — Paper 3 core integration.
=====================================================
Combines CONTINUAL LEARNING (EWC + Knowledge Distillation + class-balanced
replay) with ADVERSARIAL TRAINING (PGD-TRADES) in a single training loop.

This is the missing piece that neither run_advtrain.py (AT only, single
task) nor run_aam_trans.py (CL only, adversarial *evaluation* but clean
training) provided.

Per-batch objective:

    L_total =  CE(f(x), y)                          # clean classification
             + beta_trades * KL( f(x) || f(x_adv) )  # TRADES robustness
             + anomaly_w   * BCE(anomaly(x), y>0)    # auxiliary anomaly head
             + kd_w        * L_KD                     # knowledge distillation
             + ewc_w       * L_EWC                    # elastic weight consolidation

where x_adv is produced by a PGD inner loop that maximises the TRADES KL
term (reused from adversarial_training.pgd-style search).

The continual-learning machinery (memory buffer, EWC Fisher, KD teacher)
is delegated to an existing ``IncrementalTrainer`` instance so that the
behaviour matches the non-AT pipeline exactly, with the adversarial term
added on top.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from incremental import IncrementalTrainer
from losses import ClassBalancedFocalLoss


# ──────────────────────────────────────────────────────────────────────────
# TRADES adversarial example search (KL-maximising PGD inner loop)
# ──────────────────────────────────────────────────────────────────────────
def _trades_adv(
    model:   nn.Module,
    x:       torch.Tensor,
    epsilon: float,
    alpha:   float,
    steps:   int,
) -> torch.Tensor:
    """Return x_adv that maximises KL( f(x) || f(x+delta) ), ||delta||_inf<=eps.

    Model is switched to eval() during the search (freezes dropout/BN) and
    restored afterwards.  Gradients flow only w.r.t. the input."""
    was_training = model.training
    model.eval()
    # cuDNN's fused RNN/LSTM kernels refuse to run backward() in eval() mode,
    # which the PGD inner loop needs (grad w.r.t. the input while dropout is
    # frozen). Disabling cuDNN here falls back to the native RNN path that
    # supports eval-mode backward; Transformer/DNN backbones are unaffected.
    with torch.backends.cudnn.flags(enabled=False):
        with torch.no_grad():
            p_clean = F.softmax(model(x)["logits"], dim=-1)

        x_orig = x.detach()
        x_adv  = (x_orig + 0.001 * torch.randn_like(x_orig)).detach()
        for _ in range(steps):
            x_adv.requires_grad_(True)
            logp_adv = F.log_softmax(model(x_adv)["logits"], dim=-1)
            kl   = F.kl_div(logp_adv, p_clean, reduction="batchmean")
            grad = torch.autograd.grad(kl, x_adv)[0]
            with torch.no_grad():
                x_adv = x_adv + alpha * grad.sign()
                x_adv = torch.min(torch.max(x_adv, x_orig - epsilon),
                                  x_orig + epsilon)
            x_adv = x_adv.detach()

    if was_training:
        model.train()
    return x_adv


# ──────────────────────────────────────────────────────────────────────────
# Continual + Adversarial trainer
# ──────────────────────────────────────────────────────────────────────────
class ContinualAdversarialTrainer:
    """Wraps an IncrementalTrainer and adds the PGD-TRADES adversarial term.

    Usage mirrors IncrementalTrainer:

        cat = ContinualAdversarialTrainer(epsilon=0.1, beta_trades=6.0, ...)
        for task_id, loader in enumerate(task_loaders):
            cat.train_task(model, loader, optimizer, device, epochs=25)
            cat.cl.prepare_for_new_task(model, loader, device)   # Fisher/KD/buffer
    """

    def __init__(
        self,
        # adversarial-training hyperparameters
        epsilon:     float = 0.10,
        alpha:       float = 0.02,
        pgd_steps:   int   = 7,
        beta_trades: float = 6.0,
        # continual-learning hyperparameters (delegated to IncrementalTrainer)
        ewc_lambda:  float = 1000.0,
        kd_alpha:    float = 0.50,
        kd_temp:     float = 2.0,
        ewc_weight:  float = 1.0,
        kd_weight:   float = 1.0,
        buffer_size: int   = 100,
        random_seed: int   = 42,
        use_ewc:     bool  = True,
        use_kd:      bool  = True,
        use_replay:  bool  = True,
        anomaly_weight: float = 0.30,
        grad_clip:   float = 1.0,
        # class-imbalance handling (applied to ALL backbones for fairness)
        focal_beta:  float = 0.9999,
        focal_gamma: float = 2.0,
        # ── adversary-aware continual mechanisms (Paper-3 novelty) ─────────
        robust_replay: bool = True,   # replay samples pass the PGD inner loop
        robust_fisher: bool = True,   # EWC Fisher estimated on adversarial x
        replay_ratio:  float = 0.5,   # replay batch size = ratio * current batch
    ):
        self.epsilon     = epsilon
        self.alpha       = alpha
        self.pgd_steps   = pgd_steps
        self.beta_trades = beta_trades
        self.anomaly_w   = anomaly_weight
        self.grad_clip   = grad_clip
        self.focal_beta  = focal_beta
        self.focal_gamma = focal_gamma
        self.use_replay    = use_replay
        self.robust_replay = robust_replay
        self.robust_fisher = robust_fisher
        self.replay_ratio  = replay_ratio
        self._seed_rng     = np.random.default_rng(random_seed)
        # Class-balanced focal loss; rebuilt per task via set_class_loss().
        # Until set, falls back to plain cross-entropy.
        self.cls_loss_fn = None

        # The continual-learning state (memory, EWC, KD teacher) lives here.
        self.cl = IncrementalTrainer(
            ewc_lambda=ewc_lambda, kd_alpha=kd_alpha, kd_temp=kd_temp,
            ewc_weight=ewc_weight, kd_weight=kd_weight,
            buffer_size=buffer_size, random_seed=random_seed,
            use_ewc=use_ewc, use_kd=use_kd, use_replay=use_replay,
        )

    # ──────────────────────────────────────────────────────────────────
    def set_class_loss(self, y_train, n_classes: int, device) -> None:
        """Build a class-balanced focal loss for the current task's label
        distribution. Applied to EVERY backbone so the architecture
        comparison is not confounded by imbalance handling. Counts are
        floored at 1 so classes absent from the current task (class-
        incremental setting) do not produce infinite weights."""
        counts = np.bincount(np.asarray(y_train, dtype=np.int64),
                             minlength=n_classes).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        self.cls_loss_fn = ClassBalancedFocalLoss(
            samples_per_class=counts, beta=self.focal_beta,
            gamma=self.focal_gamma).to(device)

    def _ce(self, logits, targets):
        """Class-balanced focal loss if configured, else plain CE."""
        if self.cls_loss_fn is not None:
            return self.cls_loss_fn(logits, targets)
        return F.cross_entropy(logits, targets)

    def compute_total_loss(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor,
        n_current: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """L_clean + beta*KL(clean||adv) + anomaly + KD + EWC.

        ``n_current`` marks how many leading rows of (x, y) are current-task
        samples; the remainder are replay samples. With robust_replay=True
        (Paper-3 novelty) the TRADES inner loop perturbs the WHOLE batch, so
        the memory rehearses past *robustness*, not just past examples; with
        robust_replay=False only the current-task rows enter the adversarial
        term (a "clean replay" ablation).
        """
        # rows that receive the adversarial (TRADES) treatment
        if self.robust_replay or n_current is None:
            x_adv_src = x
        else:
            x_adv_src = x[:n_current]

        # ── adversarial example via TRADES KL-maximising PGD ──────────────
        x_adv = _trades_adv(model, x_adv_src, self.epsilon, self.alpha,
                            self.pgd_steps)

        out_clean = model(x)
        logits    = out_clean["logits"]
        anomaly   = out_clean["anomaly_score"]

        # clean classification loss (class-balanced focal; filter replay
        # labels that exceed current head width)
        n_cls = logits.size(1)
        valid = y < n_cls
        if valid.any():
            ce = self._ce(logits[valid], y[valid])
        else:
            ce = (logits * 0.0).sum()

        # TRADES robustness term (over the adversarially-treated rows only)
        p_clean  = F.softmax(logits[:x_adv_src.size(0)], dim=-1).detach()
        logp_adv = F.log_softmax(model(x_adv)["logits"], dim=-1)
        kl       = F.kl_div(logp_adv, p_clean, reduction="batchmean")

        # auxiliary anomaly head (binary benign-vs-attack on all samples)
        anm = F.binary_cross_entropy(anomaly.squeeze(-1), (y > 0).float())

        total = ce + self.beta_trades * kl + self.anomaly_w * anm
        comp  = {"ce": ce.item(), "kl": kl.item(), "anm": anm.item(),
                 "kd": 0.0, "ewc": 0.0}

        # ── Knowledge distillation (if a frozen teacher exists) ───────────
        if self.cl.use_kd and self.cl._old_model is not None:
            # Transformer backbones expose `.embedding`; DNN/RNN/LSTM do not,
            # so fall back to the current feature width (always matches within
            # a dataset, where KD applies).
            emb    = getattr(self.cl._old_model, "embedding", None)
            old_in = emb.in_features if emb is not None else x.shape[-1]
            if x.shape[-1] == old_in and valid.any():
                with torch.no_grad():
                    old_out = self.cl._old_model(x)
                kd_loss = self.cl.kd.compute_loss(
                    logits[valid], old_out["logits"][valid], y[valid])
                total   = total + self.cl.kd_w * kd_loss
                comp["kd"] = kd_loss.item()

        # ── EWC penalty (only once Fisher exists, i.e. from task 2 on) ────
        # On task 1 EWC.penalty() returns a CPU scalar tensor(0.0); adding it
        # to a CUDA loss raises a device-mismatch error.  Guard on a non-empty
        # Fisher dict and move the penalty onto the loss device.
        if self.cl.use_ewc and self.cl.ewc.fisher_dict:
            ewc_loss = self.cl.ewc.penalty(model).to(total.device)
            total    = total + self.cl.ewc_w * ewc_loss
            comp["ewc"] = ewc_loss.item()

        return total, comp

    # ──────────────────────────────────────────────────────────────────
    def _sample_replay(self, n: int, feat_dim: int, device):
        """Draw up to ``n`` replay samples from the memory buffer, matched to
        the current task's feature dimensionality. Returns (x, y) tensors on
        ``device`` or (None, None) if replay is disabled / buffer empty."""
        if not self.use_replay or self.cl.memory.size == 0 or n <= 0:
            return None, None
        Xb, yb = self.cl.memory.sample(n)            # [n, seq, F'], [n]
        # align feature width to the current task (within-dataset F is equal;
        # this guards the heterogeneous-F edge case defensively)
        if Xb.shape[-1] != feat_dim:
            if Xb.shape[-1] > feat_dim:
                Xb = Xb[..., :feat_dim]
            else:
                pad = list(Xb.shape); pad[-1] = feat_dim - Xb.shape[-1]
                Xb = np.concatenate([Xb, np.zeros(pad, dtype=Xb.dtype)], -1)
        x = torch.as_tensor(Xb, dtype=torch.float32, device=device)
        y = torch.as_tensor(yb, dtype=torch.long, device=device)
        return x, y

    def train_task(
        self, model: nn.Module, loader, optimizer, device,
        epochs: int = 25, log_every: int = 5,
    ) -> None:
        """Train one task to convergence with CL + adversarial loss.

        If replay is enabled and the memory holds past-task samples, each
        mini-batch is augmented with a replay sub-batch (size
        replay_ratio * current batch). The current-task rows lead the batch
        so compute_total_loss can apply robust vs clean replay correctly.
        """
        model.train()
        comp = {"ce": 0.0, "kl": 0.0, "kd": 0.0, "ewc": 0.0}
        for ep in range(1, epochs + 1):
            run_loss = 0.0
            n_batches = 0
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                n_current = x.size(0)

                # ── mix replay samples (Paper-3: robust replay) ───────────
                n_rep = int(round(self.replay_ratio * n_current))
                xr, yr = self._sample_replay(n_rep, x.size(-1), device)
                if xr is not None:
                    x = torch.cat([x, xr], dim=0)
                    y = torch.cat([y, yr], dim=0)

                optimizer.zero_grad(set_to_none=True)
                loss, comp = self.compute_total_loss(
                    model, x, y, n_current=n_current)
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(),
                                             self.grad_clip)
                optimizer.step()
                run_loss += loss.item()
                n_batches += 1
            if ep % log_every == 0 or ep == epochs:
                print(f"      epoch {ep:3d}/{epochs}  "
                      f"loss={run_loss / max(n_batches,1):.4f}  "
                      f"(ce={comp['ce']:.3f} kl={comp['kl']:.3f} "
                      f"kd={comp['kd']:.3f} ewc={comp['ewc']:.3f})")

    # ──────────────────────────────────────────────────────────────────
    def consolidate_task(
        self, model: nn.Module, X_train, y_train, prev_loader, device,
        n_fisher_batches: int = 50,
    ) -> None:
        """Call AFTER finishing a task, BEFORE the next one. Populates the
        replay buffer, freezes the KD teacher, and estimates the EWC Fisher
        — using ADVERSARIAL examples when robust_fisher is set (Paper-3
        novelty: protect the parameters that matter for *robust* predictions,
        not merely clean ones)."""
        # 1) fill replay memory with this task's samples
        if self.use_replay:
            self.cl.memory.add_task_data(np.asarray(X_train),
                                         np.asarray(y_train))
        # 2) freeze KD teacher
        if self.cl.use_kd:
            import copy
            self.cl._old_model = copy.deepcopy(model)
            self.cl._old_model.eval()
            for p in self.cl._old_model.parameters():
                p.requires_grad_(False)
        # 3) EWC Fisher (clean or adversarial)
        if self.cl.use_ewc:
            adv_fn = None
            if self.robust_fisher:
                def adv_fn(m, xb):
                    return _trades_adv(m, xb, self.epsilon, self.alpha,
                                       self.pgd_steps)
            self.cl.ewc.compute_fisher(model, prev_loader, device,
                                       n_fisher_batches, adv_fn=adv_fn)


# ──────────────────────────────────────────────────────────────────────────
# BiC — Bias Correction (Wu et al., CVPR 2019, "Large Scale Incremental
# Learning"). A class-incremental baseline that augments rehearsal +
# distillation with a two-parameter linear correction (alpha, beta) applied
# to the logits of the classes introduced at each task, fit on a class-
# balanced old/new validation split with the backbone frozen. This counters
# the well-known bias of the unified classifier toward recently-seen classes.
# Implemented as a post-training stage so it sits on top of the existing
# replay(+KD) machinery without touching the per-batch objective.
# ──────────────────────────────────────────────────────────────────────────
class BiasCorrection:
    """Stack of per-task linear corrections on disjoint class subsets.

    Each stage stores (new_class_indices, alpha, beta) and rescales the raw
    logits of those classes by ``alpha * logit + beta``. Stages touch disjoint
    class sets (the classes each task introduces), so order is irrelevant and
    the whole stack is applied at inference."""

    def __init__(self):
        self.stages = []  # list of (np.ndarray[int], float alpha, float beta)

    def add_stage(self, new_class_idx, alpha: float, beta: float) -> None:
        self.stages.append((np.asarray(new_class_idx, dtype=np.int64),
                            float(alpha), float(beta)))

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.stages:
            return logits
        out = logits.clone()
        for idx, a, b in self.stages:
            if len(idx) == 0:
                continue
            ix = torch.as_tensor(idx, device=logits.device, dtype=torch.long)
            out[:, ix] = a * logits[:, ix] + b
        return out


class BiCWrapped(nn.Module):
    """Wrap a trained backbone so ``forward(x)['logits']`` returns the
    bias-corrected logits. Used for evaluation/attack only; training uses the
    raw backbone (the correction is fit afterwards with the backbone frozen).
    All other model attributes/outputs pass through unchanged."""

    def __init__(self, model: nn.Module, bias_correction: BiasCorrection):
        super().__init__()
        self.model = model
        self.bias_correction = bias_correction

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, dict) and "logits" in out:
            out = dict(out)
            out["logits"] = self.bias_correction.apply(out["logits"])
        return out


@torch.no_grad()
def _stack_balanced(x_old, y_old, x_new, y_new, device):
    """Concatenate old (replay) and new (current-task) validation samples
    after subsampling the larger side so the two groups are balanced."""
    n = min(len(y_old), len(y_new))
    if n == 0:                       # no old samples (first task): use new only
        Xc = x_new; yc = y_new
    else:
        ro = np.random.default_rng(0).permutation(len(y_old))[:n]
        rn = np.random.default_rng(1).permutation(len(y_new))[:n]
        Xc = np.concatenate([np.asarray(x_old)[ro], np.asarray(x_new)[rn]], 0)
        yc = np.concatenate([np.asarray(y_old)[ro], np.asarray(y_new)[rn]], 0)
    Xc = torch.as_tensor(Xc, dtype=torch.float32, device=device)
    yc = torch.as_tensor(yc, dtype=torch.long, device=device)
    return Xc, yc


def _pgd_calib(model, bias_corr, X, y, eps, alpha, steps):
    """Untargeted PGD against the frozen, bias-corrected backbone, used only to
    build an adversarial copy of the *calibration* set. The backbone and all
    earlier correction stages are frozen; we perturb the inputs to maximise the
    classification loss so the (alpha, beta) fit below sees worst-case logits."""
    X_orig = X.detach()
    X_adv = X_orig.clone()
    X_adv += torch.empty_like(X_adv).uniform_(-eps, eps)
    X_adv = X_adv.detach()
    with torch.backends.cudnn.flags(enabled=False):
        for _ in range(steps):
            X_adv.requires_grad_(True)
            logits = bias_corr.apply(model(X_adv)["logits"])
            loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, X_adv)[0]
            with torch.no_grad():
                X_adv = X_adv + alpha * grad.sign()
                X_adv = torch.min(torch.max(X_adv, X_orig - eps),
                                  X_orig + eps)
            X_adv = X_adv.detach()
    return X_adv


def fit_bias_correction(model, bias_corr, new_class_idx,
                        x_old, y_old, x_new, y_new, device,
                        steps: int = 300, lr: float = 0.01,
                        robust: bool = False, adv_eps: float = 0.10,
                        adv_alpha: float = 0.02, adv_steps: int = 7,
                        adv_w: float = 1.0):
    """Fit the (alpha, beta) of the current task's new classes on a balanced
    old/new validation set, backbone frozen, earlier stages frozen. Appends
    the fitted stage to ``bias_corr``.

    When ``robust=True`` (Robust Bias Correction, RBC), the calibration set is
    augmented with PGD-adversarial copies of the same samples so the scalar
    (alpha, beta) calibration of each task's new classes is fit against
    worst-case logits, not just clean ones. This is the structural advantage
    over vanilla BiC, whose bias-correction stage is fit on clean data only and
    is therefore blind to adversarial logit shifts on the new classes."""
    new_class_idx = np.asarray(new_class_idx, dtype=np.int64)
    if len(new_class_idx) == 0 or len(y_new) == 0:
        bias_corr.add_stage(new_class_idx, 1.0, 0.0)
        return 1.0, 0.0
    Xc, yc = _stack_balanced(x_old, y_old, x_new, y_new, device)
    was_training = model.training
    model.eval()

    # Optionally build an adversarial copy of the calibration set (backbone +
    # earlier stages frozen) and append it so the fit sees worst-case logits.
    if robust:
        Xa_parts = []
        for i in range(0, len(yc), 1024):
            Xa_parts.append(_pgd_calib(model, bias_corr,
                                       Xc[i:i + 1024], yc[i:i + 1024],
                                       adv_eps, adv_alpha, adv_steps))
        Xa = torch.cat(Xa_parts, 0)
    else:
        Xa = None

    with torch.no_grad():                       # base logits + frozen stages
        base = []
        for i in range(0, len(yc), 1024):
            base.append(bias_corr.apply(model(Xc[i:i + 1024])["logits"]))
        base = torch.cat(base, 0)
        if robust:
            base_a = []
            for i in range(0, len(yc), 1024):
                base_a.append(bias_corr.apply(model(Xa[i:i + 1024])["logits"]))
            base_a = torch.cat(base_a, 0)

    ix = torch.as_tensor(new_class_idx, device=device, dtype=torch.long)
    a = torch.ones((), device=device, requires_grad=True)
    b = torch.zeros((), device=device, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = base.clone()
        logits[:, ix] = a * base[:, ix] + b
        loss = F.cross_entropy(logits, yc)
        if robust:
            logits_a = base_a.clone()
            logits_a[:, ix] = a * base_a[:, ix] + b
            loss = loss + adv_w * F.cross_entropy(logits_a, yc)
        loss.backward()
        opt.step()
    if was_training:
        model.train()
    bias_corr.add_stage(new_class_idx, a.item(), b.item())
    return a.item(), b.item()
