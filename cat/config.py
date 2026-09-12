"""
config.py — Central configuration for AAM-TRANS experiment pipeline.
All hyperparameters, dataset paths, and attack settings are defined here.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Base paths (adjust if datasets are moved)
# ──────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PAPER3_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATHS = {
    "CICIoT":  os.path.join(_PROJECT_ROOT, "merged_CICIoT_train_shuffled.csv"),
    "CICIoMT": os.path.join(_PROJECT_ROOT, "merged_train_ciciomt_shuffled.csv"),
    "CICIoV":  os.path.join(_PROJECT_ROOT, "merged_train_binary_shuffled.csv"),
    "TONIoT":  os.path.join(_PAPER3_DIR, "merged_TONIoT_clean.csv"),
}

LABEL_COLUMN = "Label"

# Order matters for incremental learning (Task 1 → Task 2 → Task 3)
DATASET_ORDER = ["CICIoT", "CICIoMT", "CICIoV", "TONIoT"]

# Class representing normal / benign traffic per dataset
BENIGN_LABELS = {
    "CICIoT":  "BENIGN",
    "CICIoMT": "Benign",
    "CICIoV":  "BENIGN",
    "TONIoT":  "normal",
}

# ──────────────────────────────────────────────────────────────────────────────
# Data sampling (dissertation uses 10% for initial results; set to 1.0 for full)
# ──────────────────────────────────────────────────────────────────────────────
DATA_SAMPLE_FRACTION = 0.10   # 0.10 = 10 % as in dissertation baseline
RANDOM_SEED          = 42
TEST_SIZE            = 0.20   # 80/20 train-validation split

# Per-dataset fractions to equalize sample sizes across datasets.
DATASET_SAMPLE_FRACTIONS: dict = {
    "CICIoT":  0.100,
    "CICIoMT": 0.031,
    "CICIoV":  0.153,
    "TONIoT":  0.100,
}


# ──────────────────────────────────────────────────────────────────────────────
# Classification mode
# ──────────────────────────────────────────────────────────────────────────────
CLASSIFICATION_MODE = "both"   # options: "binary", "multiclass", "both"


# ──────────────────────────────────────────────────────────────────────────────
# Sequence expansion (tabular → temporal for Transformer input)
# ──────────────────────────────────────────────────────────────────────────────
SEQ_LEN       = 50    # sequence length per sample
SEQ_NOISE_STD = 0.01  # noise added at each timestep to simulate flow variability


# ──────────────────────────────────────────────────────────────────────────────
# Model hyperparameters (identical for ALL models → fair comparison)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    d_model:     int   = 96            # was 64 — larger for multiclass capacity
    n_heads:     int   = 4
    n_layers:    int   = 3
    d_ff:        int   = 384           # was 256 — keeps d_ff/d_model = 4
    dropout:     float = 0.1
    max_seq_len: int   = SEQ_LEN

    # ── Q1-v1 enhancement flags ──
    use_feature_tokenizer:   bool = True
    use_cls_token:           bool = True
    use_pre_norm:            bool = True
    use_spectral_norm:       bool = True
    use_gradient_aware_gate: bool = True
    use_stochastic_gate:     bool = True

    # ── Q1-v2 enhancement flags ──
    use_swiglu:              bool  = True
    drop_path:               float = 0.10

    # ── Adaptive scaling for difficult multiclass tasks ──
    # When n_classes > this, scale up d_model and epochs.
    adaptive_multiclass_threshold: int = 10
    adaptive_d_model_multiplier:   float = 1.5
    adaptive_epoch_multiplier:     float = 1.5

    # ── Inference-time Bayesian gates (V3 main defense) ──
    # Adversarial robustness comes mainly from this — Bayesian gate noise
    # at inference time obfuscates the gradient signal and creates an
    # implicit ensemble (MC averaging). Defenders not visible to attacker.
    inference_gate_noise: float = 0.15
    inference_mc_samples: int   = 8


# ──────────────────────────────────────────────────────────────────────────────
# Training hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    epochs:         int   = 20      # was 15 — more capacity to escape local optima
    batch_size:     int   = 512
    learning_rate:  float = 1e-3
    weight_decay:   float = 1e-2
    patience:       int   = 7       # was 5 — give SAM/Jacobian more time
    anomaly_weight: float = 0.3

    # ── Optimizer / schedule ──
    use_adamw:           bool  = True
    use_sam:             bool  = True     # NEW Q1-v2
    sam_rho:             float = 0.05
    use_cosine_schedule: bool  = True
    warmup_frac:         float = 0.10
    grad_clip:           float = 1.0

    # ── Loss / imbalance (V3.5 expanded) ──
    # Priority: LDAM > Logit-Adjust > Focal > LS-CE > CE
    use_ldam:            bool  = False    # V3.5 — set True for multiclass severe imbalance
    ldam_max_m:          float = 0.5
    ldam_s:              float = 30.0
    use_logit_adjust:    bool  = True     # V3.5 — cheap, always-on for multiclass
    logit_adjust_tau:    float = 1.0
    use_focal:           bool  = True     # fallback if LDAM disabled
    focal_gamma:         float = 1.0
    focal_beta:          float = 0.9999
    label_smoothing:     float = 0.05

    # ── V3.5 SMOTE-sequence oversampling ──
    use_smote:                  bool  = True
    smote_method:               str   = "smote"        # 'smote' | 'smoteenn' | 'adasyn'
    smote_sampling_strategy           = "not majority"
    smote_k_neighbors:          int   = 5
    smote_min_imbalance_ratio:  float = 5.0            # skip SMOTE if ratio < this
    smote_max_per_class:        int   = 50_000         # cap synthetic samples per class
    smote_only_multiclass:      bool  = True           # skip for binary (already balanced)

    # ── V3.5 class-balanced reservoir buffer ──
    use_class_balanced_reservoir: bool = True

    # ── Augmentation (V3: lighter — V2 was over-regularising) ──
    mixup_alpha:         float = 0.1      # was 0.2 — gentler
    cutmix_alpha:        float = 0.0      # was 1.0 — REMOVED (confused binary)
    cutmix_p:            float = 0.0
    noise_aug_eps:       float = 0.0      # was 0.10 — REMOVED (use at inference)
    noise_aug_p:         float = 0.0

    # ── Smoothness / robustness (V3: lighter; primary defense moved to eval) ──
    use_trades:          bool  = True
    trades_sigma:        float = 0.03     # was 0.10 — gentler
    trades_beta:         float = 2.0      # was 6.0 — gentler
    use_jacobian_reg:    bool  = True
    jacobian_weight:     float = 0.02     # was 0.05 — gentler
    use_confidence_pen:  bool  = False    # was True — REMOVED (was pushing predictions to uniform)
    confidence_weight:   float = 0.0

    # ── EMA ──
    use_ema:             bool  = True
    ema_decay:           float = 0.999

    # ── Two-stage (decoupled) training ──
    use_two_stage:           bool  = True
    two_stage_epochs_s2:     int   = 5
    two_stage_max_classes:   int   = 20      # NEW Q1-v2 — skip for >20 classes

    # ── Inference-time smoothing ──
    use_smoothing_eval:  bool  = False
    smoothing_sigma:     float = 0.05
    smoothing_samples:   int   = 16


# ──────────────────────────────────────────────────────────────────────────────
# Adversarial attack configurations (Table in dissertation Ch.3)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class AttackConfig:
    """Attack configuration.

    V3: PGD (white-box) is REPLACED with two blackbox attacks:
        Square Attack (Andriushchenko 2020) — score-based black-box
        Transfer Attack (FGSM on a surrogate DNN) — zero-query black-box
    This aligns with the realistic IDS threat model (attacker has no
    gradient access to the deployed model).
    """
    fgsm_epsilons: List[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.10, 0.15, 0.20]
    )
    # Multi-step white-box PGD. One-step FGSM is a weak white-box probe and
    # does not bound what a gradient-aware attacker can do, so the suite also
    # carries a proper multi-step attack at the same budget.
    pgd_epsilons: List[float] = field(
        default_factory=lambda: [0.05, 0.10]
    )
    pgd_alpha:  float = 0.01
    pgd_n_iter: int   = 20
    # Square Attack (blackbox score-based)
    square_epsilon:   float = 0.10
    square_n_queries: int   = 200      # V3.5c: was 1000 — 200 is plenty post-subsample
    square_p_init:    float = 0.05
    # Transfer Attack (zero-query blackbox)
    transfer_epsilons: List[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.15]
    )
    # Random noise control
    noise_sigmas:  List[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.10, 0.15, 0.20]
    )
    apply_constraints: bool = True

    # V3.5c: cap adversarial-eval subset size. Crafting attacks (esp. the
    # query-based Square Attack) is O(n_samples × n_queries × mc_samples);
    # on a full ~40 K-row val set this runs for hours. Robustness papers
    # (RobustBench etc.) routinely evaluate on a few-thousand stratified
    # subset — set to 0 to disable capping and use the full val set.
    attack_eval_max_samples: int = 3000

    # Legacy PGD fields kept for backward compatibility with older runs/tests;
    # use_pgd=False disables it from generate_all and evaluate_robustness.
    use_pgd:        bool  = False
    pgd_epsilon:    float = 0.10
    pgd_alpha:      float = 0.01
    pgd_iterations: int   = 10


# ──────────────────────────────────────────────────────────────────────────────
# Incremental learning (Objective 2: Concept Drift)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class IncrementalConfig:
    buffer_size_per_class: int   = 100
    ewc_lambda:            float = 1000.0
    kd_alpha:              float = 0.50
    kd_temperature:        float = 2.0
    ewc_weight:            float = 1.0
    kd_weight:             float = 1.0
    incremental_epochs:    int   = 10


# ──────────────────────────────────────────────────────────────────────────────
# Baseline model names (Phase 1)
# ──────────────────────────────────────────────────────────────────────────────
BASELINE_MODELS = ["DNN", "RNN", "LSTM", "StandardTransformer"]


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: build all configs with defaults
# ──────────────────────────────────────────────────────────────────────────────
def get_default_configs():
    return ModelConfig(), TrainingConfig(), AttackConfig(), IncrementalConfig()
