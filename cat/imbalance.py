"""
imbalance.py — Imbalance handling for AAM-TRANS V3.5.

Provides four complementary mechanisms layered on top of the existing
class-balanced focal loss + WeightedRandomSampler + two-stage decoupled
training that V3 already ships:

  1. SMOTE / SMOTEENN / ADASYN sequence oversampling
        Generate synthetic minority-class samples in feature space via
        k-NN interpolation. Operates on the first timestep of the
        [N, L, F] sequence (the "real" features), then re-expands the
        synthetic base samples back to sequences with Gaussian temporal
        noise so the augmented data matches the original distribution.

  2. LDAM Loss (Cao et al., NeurIPS 2019)
        Label-Distribution-Aware Margin loss. Adds per-class additive
        margin Δ_y = C / n_y^(1/4) to the true-class logit BEFORE softmax,
        forcing larger margin on rare classes. Outperforms focal loss for
        severe long-tailed imbalance.

  3. Logit Adjustment (Menon et al., ICLR 2021)
        Subtract tau * log(class_prior) from logits at training time,
        provably equivalent to balanced-error optimal classification.
        One line, zero hyperparameter tuning needed, very effective.

  4. Class-Balanced Reservoir Sampler
        Replaces vanilla reservoir sampling in memory buffer with per-class
        quotas so the buffer is *balanced* even when the data stream is
        heavily imbalanced (critical for CL on minority attack classes).
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. SMOTE / SMOTEENN / ADASYN for sequence data
# ──────────────────────────────────────────────────────────────────────────────

def apply_smote_to_sequences(
    X_seq:              np.ndarray,
    y:                  np.ndarray,
    method:             str   = "smote",
    sampling_strategy = "not majority",
    k_neighbors:        int   = 5,
    random_state:       int   = 42,
    seq_noise:          float = 0.01,
    max_samples_per_class: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE / SMOTEENN / ADASYN to sequence-shaped data [N, L, F].

    Strategy:
        base = X_seq[:, 0, :]              # first timestep ("real" features)
        synthetic_base = SMOTE(base, y)    # 2-D oversampling
        X_aug = expand_to_sequence(synthetic_base, L, noise=seq_noise)
        return X_aug, y_aug

    Args
    ----
    method                : 'smote' | 'smoteenn' | 'adasyn'
    sampling_strategy     : passed through to imblearn (see their docs).
                            'not majority' = oversample everything except
                            the largest class to its size.
    k_neighbors           : k for SMOTE's k-NN interpolation
    seq_noise             : std of Gaussian noise injected when re-expanding
                            synthetic base features into sequences
    max_samples_per_class : optional cap to limit dataset size after
                            oversampling (avoids OOM on huge datasets)
    """
    try:
        from imblearn.over_sampling import SMOTE, ADASYN
        from imblearn.combine        import SMOTEENN
    except ImportError as e:
        logger.warning(
            f"imbalanced-learn not installed ({e}); SMOTE oversampling skipped. "
            f"Install with: pip install imbalanced-learn"
        )
        return X_seq, y

    N, L, F_ = X_seq.shape
    base     = X_seq[:, 0, :]                   # [N, F] — first timestep
    y        = np.asarray(y).astype(int)

    # Guard: SMOTE requires every class to have at least k+1 samples.
    counts        = np.bincount(y)
    min_class_n   = counts[counts > 0].min()
    safe_k        = max(1, min(k_neighbors, min_class_n - 1))
    if safe_k < k_neighbors:
        logger.warning(
            f"SMOTE k_neighbors reduced {k_neighbors}->{safe_k} because the "
            f"smallest class has only {min_class_n} samples."
        )

    # Pick resampler
    if method == "smoteenn":
        sampler = SMOTEENN(
            sampling_strategy=sampling_strategy,
            smote=SMOTE(k_neighbors=safe_k, random_state=random_state),
            random_state=random_state,
        )
    elif method == "adasyn":
        sampler = ADASYN(
            sampling_strategy=sampling_strategy,
            n_neighbors=safe_k,
            random_state=random_state,
        )
    else:
        sampler = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=safe_k,
            random_state=random_state,
        )

    logger.info(f"  Applying {method.upper()} (k={safe_k}) on base features "
                f"[{N},{F_}]; before counts: {counts.tolist()}")

    try:
        base_res, y_res = sampler.fit_resample(base, y)
    except (ValueError, RuntimeError) as e:
        logger.warning(f"  {method.upper()} failed ({e}); falling back to original data")
        return X_seq, y

    # Optional cap per class (avoid blowing up disk / memory)
    if max_samples_per_class is not None:
        keep_idx = []
        for c in np.unique(y_res):
            cls_idx = np.where(y_res == c)[0]
            if len(cls_idx) > max_samples_per_class:
                cls_idx = np.random.default_rng(random_state).choice(
                    cls_idx, size=max_samples_per_class, replace=False
                )
            keep_idx.extend(cls_idx.tolist())
        keep_idx = np.array(keep_idx)
        base_res = base_res[keep_idx]
        y_res    = y_res[keep_idx]

    # Re-expand to sequences with the same temporal noise model
    rng       = np.random.default_rng(random_state)
    noise     = rng.normal(0.0, seq_noise, size=(base_res.shape[0], L, F_))
    X_seq_res = base_res[:, None, :] + noise
    X_seq_res = X_seq_res.astype(X_seq.dtype, copy=False)

    new_counts = np.bincount(y_res.astype(int))
    logger.info(f"  After {method.upper()}: shape={X_seq_res.shape}; "
                f"counts={new_counts.tolist()}")
    return X_seq_res, y_res.astype(int)


# ──────────────────────────────────────────────────────────────────────────────
# 2. LDAM Loss (Cao et al., NeurIPS 2019)
# ──────────────────────────────────────────────────────────────────────────────

class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin loss.

        Δ_y = C / n_y^(1/4)
        L_LDAM(x, y) = -log(exp(s*(z_y - Δ_y)) /
                           [exp(s*(z_y - Δ_y)) + Σ_{j≠y} exp(s*z_j)])

    The per-class margin is *larger* for rare classes (small n_y), so the
    network is forced to leave more separation around them. The scale `s`
    sharpens the softmax; the original paper uses s=30.

    Args
    ----
    samples_per_class : 1-D array of per-class training counts
    max_m             : maximum margin C (paper recommends 0.5)
    s                 : softmax scale (paper recommends 30)
    weight            : optional per-class weight tensor (use with DRW schedule)
    """

    def __init__(
        self,
        samples_per_class,
        max_m:  float = 0.5,
        s:      float = 30.0,
        weight = None,
    ):
        super().__init__()
        spc = np.asarray(samples_per_class, dtype=np.float64)
        spc = np.maximum(spc, 1.0)
        m_list = 1.0 / np.sqrt(np.sqrt(spc))      # 1 / n_y^(1/4)
        m_list = m_list * (max_m / m_list.max())  # normalise so max margin = max_m
        self.register_buffer("m_list", torch.tensor(m_list, dtype=torch.float32))
        self.s      = s
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.unsqueeze(1), True)
        batch_m = self.m_list.to(logits.device)[targets].unsqueeze(1)
        logits_m = logits - batch_m * index.float()
        logits_m = logits_m * self.s
        w = self.weight.to(logits.device) if self.weight is not None else None
        return F.cross_entropy(logits_m, targets, weight=w)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Logit Adjustment (Menon et al., ICLR 2021)
# ──────────────────────────────────────────────────────────────────────────────

class LogitAdjustedCrossEntropy(nn.Module):
    """
    Logit-Adjusted softmax cross-entropy.

        L(x, y) = CE(z + tau * log(π), y)
        where π_k = n_k / N is the empirical class prior.

    At inference time, simply use the raw logits z (no adjustment) — the
    paper proves this is equivalent to balanced-error-rate-optimal
    classification under the assumed prior shift.

    Args
    ----
    samples_per_class : per-class training counts
    tau               : strength of the adjustment (paper uses tau=1)
    weight            : optional per-class weighting
    label_smoothing   : optional smoothing ε for the cross-entropy
    """

    def __init__(
        self,
        samples_per_class,
        tau:             float = 1.0,
        weight                  = None,
        label_smoothing: float  = 0.0,
    ):
        super().__init__()
        spc   = np.asarray(samples_per_class, dtype=np.float64)
        spc   = np.maximum(spc, 1.0)
        prior = spc / spc.sum()
        self.register_buffer("log_prior",
                             torch.tensor(np.log(prior), dtype=torch.float32))
        self.tau             = tau
        self.weight          = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adj_logits = logits + self.tau * self.log_prior.to(logits.device)
        w = self.weight.to(logits.device) if self.weight is not None else None
        return F.cross_entropy(
            adj_logits, targets,
            weight=w, label_smoothing=self.label_smoothing,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 4. Class-Balanced Reservoir Sampler (for CL memory buffer)
# ──────────────────────────────────────────────────────────────────────────────

def _pad_or_truncate_feature_dim(x: np.ndarray, target_F: int) -> np.ndarray:
    """
    Pad or truncate the LAST axis of x (the feature axis) to target_F.

    Used by the buffer classes to make samples from different CL tasks
    (which may have different #features) stackable into a single batch.
    """
    cur_F = x.shape[-1]
    if cur_F == target_F:
        return x
    if cur_F < target_F:
        pad_shape = list(x.shape)
        pad_shape[-1] = target_F - cur_F
        pad = np.zeros(pad_shape, dtype=x.dtype)
        return np.concatenate([x, pad], axis=-1)
    return x[..., :target_F]


class ClassBalancedReservoir:
    """
    Per-class reservoir buffer that keeps `buffer_size_per_class` samples for
    every class seen so far. Replaces the vanilla single-pool reservoir in
    incremental.py to ensure minority-class memory does not get evicted by
    a flood of new-task majority-class samples.

    Hardened (V3.5 hotfix) to handle CL streams with HETEROGENEOUS feature
    dimensionality across tasks (CICIoT 39 → CICIoMT 153 → CICIoV X):
        - Per-class storage uses python lists of np.ndarrays, NEVER
          pre-stacked into a uniform-shape array
        - sample() pads everything to max_feature_dim seen so far before
          stacking → returns a uniform-shape batch consumable by DataLoader

    Public API mirrors the original ReservoirBuffer in incremental.py:
        add_task_data(X, y)
        sample(n_samples) -> (X, y)
        size  (property)
    """

    def __init__(self, buffer_size_per_class: int = 100, random_seed: int = 42):
        self.k                       = buffer_size_per_class
        self.rng                     = np.random.default_rng(random_seed)
        # class_id -> list[np.ndarray]   (heterogeneous shapes allowed)
        self.per_class: dict         = {}
        # class_id -> int  (total samples ever seen of this class — for reservoir)
        self._n_seen_per_class: dict = {}

    @property
    def size(self) -> int:
        return sum(len(v) for v in self.per_class.values())

    def add_task_data(self, X: np.ndarray, y: np.ndarray):
        """
        Stream new task data through the per-class reservoir.

        Samples are stored as separate np.ndarrays in a python list to allow
        the buffer to mix samples whose last-axis (feature) dimensionality
        differs across CL tasks. No conversion to a uniform numpy array
        happens here — that is deferred to sample() with on-the-fly padding.
        """
        y_arr = np.asarray(y).astype(int)
        for c in np.unique(y_arr):
            cls_idx = np.where(y_arr == c)[0]
            X_c     = X[cls_idx]
            c_key   = int(c)
            if c_key not in self.per_class:
                self.per_class[c_key]         = []
                self._n_seen_per_class[c_key] = 0

            for x in X_c:
                self._n_seen_per_class[c_key] += 1
                # `np.array(x, copy=True)` — defensive, in case x shares memory
                x_copy = np.array(x, copy=True)
                if len(self.per_class[c_key]) < self.k:
                    self.per_class[c_key].append(x_copy)
                else:
                    # Standard reservoir: pick j ∈ [0, n_seen);  if j < k, replace.
                    j = int(self.rng.integers(0, self._n_seen_per_class[c_key]))
                    if j < self.k:
                        # Heterogeneous shapes are fine — we just REPLACE the
                        # python list slot with the new ndarray. No broadcast.
                        self.per_class[c_key][j] = x_copy

    def sample(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uniformly sample across classes (then random WITHIN each class).
        Returned X is padded to the max feature-dim across the sampled set
        so that DataLoader can batch them as a single tensor.
        """
        classes = list(self.per_class.keys())
        if not classes:
            return np.empty((0,)), np.empty((0,))

        per_class_n  = max(1, n_samples // len(classes))
        sampled_x:   list = []
        sampled_y:   list = []
        for c in classes:
            buf = self.per_class[c]
            if not buf:
                continue
            n   = min(per_class_n, len(buf))
            idx = self.rng.choice(len(buf), size=n, replace=False)
            for i in idx:
                sampled_x.append(buf[int(i)])
                sampled_y.append(int(c))

        if not sampled_x:
            return np.empty((0,)), np.empty((0,))

        # ── Heterogeneous-shape stack: pad to max feature-dim ─────────────
        max_F   = max(x.shape[-1] for x in sampled_x)
        padded  = [_pad_or_truncate_feature_dim(x, max_F) for x in sampled_x]
        X_out   = np.stack(padded, axis=0)
        y_out   = np.array(sampled_y, dtype=np.int64)

        # Shuffle so classes are not in blocks
        perm    = self.rng.permutation(len(X_out))
        return X_out[perm], y_out[perm]


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: imbalance ratio diagnostic
# ──────────────────────────────────────────────────────────────────────────────

def imbalance_ratio(y) -> float:
    """Compute imbalance ratio = max_count / min_count_nonzero."""
    y = np.asarray(y).astype(int)
    c = np.bincount(y)
    c = c[c > 0]
    if len(c) <= 1:
        return 1.0
    return float(c.max() / c.min())
