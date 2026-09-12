"""
incremental.py — Incremental Learning for Concept Drift Adaptation (Objective 2).

Implements three complementary mechanisms as specified in dissertation Ch.3 §4:

  1. MemoryBuffer    — reservoir sampling of representative past-task samples
  2. KnowledgeDistillation — soft-label transfer from old → new model
  3. EWC             — Elastic Weight Consolidation (Fisher Information penalty)

Total loss:
  L_total = L_CE + β1 * L_KD + β2 * L_EWC

Cross-dataset incremental scenario (concept drift simulation):
  Task 1: CICIoT  (initial training)
  Task 2: CICIoMT (incremental update — new IoMT domain)
  Task 3: CICIoV  (incremental update — new IoV domain)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import copy
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Memory Buffer (Experience Replay)
# ──────────────────────────────────────────────────────────────────────────────

class MemoryBuffer:
    """
    Stores a fixed number of representative samples per class using
    reservoir sampling (Vitter, 1985).

    The buffer ensures the model can replay knowledge of past attack
    patterns during incremental updates.
    """

    def __init__(self, buffer_size_per_class: int = 100, random_seed: int = 42):
        self.buffer_size_per_class = buffer_size_per_class
        self.rng  = np.random.default_rng(random_seed)
        self._X:  List[np.ndarray] = []
        self._y:  List[int]        = []
        self._class_counts: Dict[int, int] = {}

    def add_task_data(
        self,
        X_seq: np.ndarray,     # [N, seq_len, F]
        y:     np.ndarray,     # [N]
    ) -> None:
        """Add samples from a new task using stratified reservoir sampling."""
        classes = np.unique(y)
        for cls in classes:
            mask   = (y == cls)
            X_cls  = X_seq[mask]
            n_new  = len(X_cls)

            # Sample up to buffer_size_per_class
            n_keep = min(n_new, self.buffer_size_per_class)
            idx    = self.rng.choice(n_new, size=n_keep, replace=False)

            self._X.extend(X_cls[idx])
            self._y.extend([cls] * n_keep)
            self._class_counts[int(cls)] = n_keep

        logger.info(
            f"MemoryBuffer updated: {len(self._X)} samples, "
            f"classes={list(self._class_counts.keys())}"
        )

    def sample(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample n_samples from the buffer.

        Hardened (V3.5 hotfix): when the buffer contains samples from CL
        tasks with DIFFERENT feature dimensions, np.stack would crash. We
        now pad the smaller-F samples up to max_F so a uniform-shape batch
        can be returned.
        """
        if len(self._X) == 0:
            raise RuntimeError("Buffer is empty — call add_task_data() first")
        total = len(self._X)
        idx   = self.rng.choice(total, size=min(n_samples, total), replace=False)
        picked = [self._X[i] for i in idx]
        max_F  = max(p.shape[-1] for p in picked)
        padded = []
        for p in picked:
            cur_F = p.shape[-1]
            if cur_F < max_F:
                pad_shape = list(p.shape)
                pad_shape[-1] = max_F - cur_F
                p = np.concatenate([p, np.zeros(pad_shape, dtype=p.dtype)],
                                   axis=-1)
            elif cur_F > max_F:
                p = p[..., :max_F]
            padded.append(p)
        X_buf = np.stack(padded, axis=0)
        y_buf = np.array([self._y[i] for i in idx])
        return X_buf, y_buf

    @property
    def size(self) -> int:
        return len(self._X)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Knowledge Distillation
# ──────────────────────────────────────────────────────────────────────────────

class KnowledgeDistillation:
    """
    Transfers soft predictions from an old (teacher) model to the updated
    (student) model to prevent catastrophic forgetting.

    Loss:
      L_KD = α * L_CE(y, ŷ_new) + (1-α) * KL(ŷ_old_soft || ŷ_new_soft)

    where soft predictions use temperature T for smoothing.
    """

    def __init__(self, alpha: float = 0.5, temperature: float = 2.0):
        self.alpha       = alpha         # balance CE vs distillation
        self.temperature = temperature   # > 1 softens the distribution

    def compute_loss(
        self,
        new_logits: torch.Tensor,    # [B, C_new] from current model
        old_logits: torch.Tensor,    # [B, C_old] from frozen old model
        targets:    torch.Tensor,    # [B] hard labels (current task)
    ) -> torch.Tensor:
        """
        Compute combined cross-entropy + distillation loss.
        Note: C_new may differ from C_old (different #classes per task).
        We distil only on classes present in old_logits.
        """
        T = self.temperature

        # Hard-label CE loss on new task data
        ce_loss = F.cross_entropy(new_logits, targets)

        # Soft-label KL distillation on shared embedding space
        # Use min(C_new, C_old) shared classes for distillation
        n_shared = min(new_logits.size(1), old_logits.size(1))

        soft_old = F.log_softmax(old_logits[:, :n_shared] / T, dim=1)
        soft_new = F.softmax(new_logits[:, :n_shared] / T, dim=1)

        kd_loss  = F.kl_div(soft_old, soft_new, reduction="batchmean") * (T ** 2)

        total = self.alpha * ce_loss + (1.0 - self.alpha) * kd_loss
        return total


# ──────────────────────────────────────────────────────────────────────────────
# 3. Elastic Weight Consolidation (EWC)
# ──────────────────────────────────────────────────────────────────────────────

class EWC:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017).

    Computes Fisher Information for each parameter after a task and adds
    a quadratic penalty to prevent important weights from drifting.

    Fisher:
      F_i = E_{x~D_old}[ (∂ log p(y|x,θ) / ∂θ_i)² ]

    EWC penalty:
      L_EWC = Σ_i (λ/2) * F_i * (θ_i - θ_i*)²
    """

    def __init__(self, ewc_lambda: float = 1000.0):
        self.ewc_lambda     = ewc_lambda
        self.fisher_dict:   Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}

    def compute_fisher(
        self,
        model,
        data_loader,
        device: torch.device,
        n_batches: int = 50,
        adv_fn=None,
    ) -> None:
        """
        Compute diagonal Fisher Information Matrix on a subset of old-task data.
        Stores Fisher values and current (optimal) parameter values.

        Args:
            model:      trained model after completing a task
            data_loader: DataLoader for the completed task's data
            n_batches:  number of mini-batches to average over
            adv_fn:     optional callable (model, x) -> x_adv. When provided
                        (Paper-3 robust-Fisher novelty), the Fisher is
                        estimated on ADVERSARIAL inputs, so EWC protects the
                        parameters important for robust — not merely clean —
                        predictions.
        """
        model.eval()
        fisher = {n: torch.zeros_like(p)
                  for n, p in model.named_parameters() if p.requires_grad}

        for step, (x, y) in enumerate(data_loader):
            if step >= n_batches:
                break
            x, y = x.to(device), y.to(device)
            if adv_fn is not None:
                x = adv_fn(model, x).detach()

            model.zero_grad()
            outputs = model(x)
            # Use log-likelihood of the predicted class
            log_probs = F.log_softmax(outputs["logits"], dim=1)
            # Sample class from model's distribution for Monte-Carlo Fisher estimate
            y_hat = outputs["logits"].argmax(dim=1)
            loss  = F.nll_loss(log_probs, y_hat)
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)

        # Average over batches
        for n in fisher:
            fisher[n] /= max(n_batches, 1)

        self.fisher_dict    = {n: f.clone().detach() for n, f in fisher.items()}
        self.optimal_params = {n: p.data.clone().detach()
                               for n, p in model.named_parameters()
                               if p.requires_grad}
        logger.info("EWC: Fisher Information computed")

    def penalty(self, model) -> torch.Tensor:
        """
        Compute EWC penalty: L_EWC = Σ_i (λ/2) * F_i * (θ_i - θ_i*)²
        Returns scalar tensor.
        """
        if not self.fisher_dict:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0)
        for n, p in model.named_parameters():
            if n in self.fisher_dict:
                f   = self.fisher_dict[n].to(p.device)
                opt = self.optimal_params[n].to(p.device)
                if f.shape != p.shape:
                    # Parameter was resized (e.g. embedding adapted for new input dim)
                    continue
                loss = loss + (f * (p - opt).pow(2)).sum()

        return (self.ewc_lambda / 2.0) * loss


# ──────────────────────────────────────────────────────────────────────────────
# Incremental Trainer — coordinates all three mechanisms
# ──────────────────────────────────────────────────────────────────────────────

class IncrementalTrainer:
    """
    Manages the cross-dataset incremental learning loop.

    Total loss:
      L_total = L_CE + β1 * L_KD + β2 * L_EWC

    (Eq. from dissertation §3.4.4 Adaptive Integration)

    Ablation flags:
      use_ewc    : False → disable EWC penalty (β2 = 0)
      use_kd     : False → disable knowledge distillation (β1 = 0)
      use_replay : False → disable memory buffer replay
    """

    def __init__(
        self,
        ewc_lambda:   float = 1000.0,
        kd_alpha:     float = 0.50,
        kd_temp:      float = 2.0,
        ewc_weight:   float = 1.0,    # β2
        kd_weight:    float = 1.0,    # β1
        buffer_size:  int   = 100,
        random_seed:  int   = 42,
        use_ewc:      bool  = True,
        use_kd:       bool  = True,
        use_replay:   bool  = True,
        use_class_balanced_reservoir: bool = True,
    ):
        # V3.5: use class-balanced reservoir to avoid minority eviction
        if use_class_balanced_reservoir:
            try:
                from imbalance import ClassBalancedReservoir
                self.memory = ClassBalancedReservoir(
                    buffer_size_per_class=buffer_size, random_seed=random_seed,
                )
            except ImportError:
                self.memory = MemoryBuffer(buffer_size, random_seed)
        else:
            self.memory = MemoryBuffer(buffer_size, random_seed)
        self.kd         = KnowledgeDistillation(kd_alpha, kd_temp)
        self.ewc        = EWC(ewc_lambda)
        self.ewc_w      = ewc_weight
        self.kd_w       = kd_weight
        self.use_ewc    = use_ewc
        self.use_kd     = use_kd
        self.use_replay = use_replay
        self._old_model: Optional[nn.Module] = None

    def prepare_for_new_task(
        self,
        current_model,
        prev_loader,
        device: torch.device,
        n_fisher_batches: int = 50,
    ) -> None:
        """
        Call AFTER completing a task and BEFORE starting the next.
        1. Saves frozen copy of old model (for KD — if use_kd)
        2. Computes Fisher Information (for EWC — if use_ewc)
        3. Buffers old-task samples (for replay — if use_replay)
        """
        # Frozen teacher model for KD
        if self.use_kd:
            self._old_model = copy.deepcopy(current_model)
            self._old_model.eval()
            for p in self._old_model.parameters():
                p.requires_grad_(False)

        # Fisher info for EWC
        if self.use_ewc:
            self.ewc.compute_fisher(current_model, prev_loader, device, n_fisher_batches)

        logger.info(
            f"IncrementalTrainer: prepared for new task "
            f"[EWC={self.use_ewc}, KD={self.use_kd}, Replay={self.use_replay}]"
        )

    def compute_total_loss(
        self,
        model,
        x:       torch.Tensor,
        y:       torch.Tensor,
        anomaly_weight: float = 0.3,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute L_total = L_CE + anomaly_aux + β1*L_KD + β2*L_EWC

        Returns (total_loss, component_dict) for logging.
        """
        outputs = model(x)
        logits  = outputs["logits"]
        anomaly = outputs["anomaly_score"]

        # ── Cross-entropy (primary classification) ───────────────────────────
        # Replay samples from earlier tasks may have labels >= current n_classes.
        # Filter them out so CE doesn't trigger an out-of-bounds CUDA assert.
        n_cls  = logits.size(1)
        valid  = y < n_cls
        if valid.any():
            ce_loss = F.cross_entropy(logits[valid], y[valid])
        else:
            ce_loss = (logits * 0.0).sum()  # zero, keeps autograd graph

        # ── Auxiliary anomaly loss (binary: benign vs attack — all samples) ──
        anm_loss = F.binary_cross_entropy(
            anomaly.squeeze(), (y > 0).float()
        )

        total = ce_loss + anomaly_weight * anm_loss

        # ── Knowledge Distillation (if enabled and old model available) ──────
        kd_val = 0.0
        if self.use_kd and self._old_model is not None:
            old_in = self._old_model.embedding.in_features
            if x.shape[-1] != old_in:
                # Heterogeneous input space — KD inapplicable; EWC covers forgetting
                pass
            else:
                with torch.no_grad():
                    old_out = self._old_model(x)
                # kd.compute_loss also calls F.cross_entropy internally —
                # pass only valid-label samples to avoid out-of-bounds assert.
                if valid.any():
                    kd_loss = self.kd.compute_loss(
                        logits[valid], old_out["logits"][valid], y[valid]
                    )
                else:
                    kd_loss = (logits * 0.0).sum()
                total   = total + self.kd_w * kd_loss
                kd_val  = kd_loss.item()

        # ── EWC penalty (if enabled) ──────────────────────────────────────────
        ewc_val  = 0.0
        if self.use_ewc:
            ewc_loss = self.ewc.penalty(model)
            total    = total + self.ewc_w * ewc_loss
            ewc_val  = ewc_loss.item() if isinstance(ewc_loss, torch.Tensor) else float(ewc_loss)

        return total, {
            "ce":  ce_loss.item(),
            "anm": anm_loss.item(),
            "kd":  kd_val,
            "ewc": ewc_val,
        }
