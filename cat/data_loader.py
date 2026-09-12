"""
data_loader.py — Dataset loading, preprocessing, and PyTorch Dataset/DataLoader creation.
Supports binary and multi-class modes for CICIoT, CICIoMT, CICIoV.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Optional, Dict
import logging
from config import DATASET_SAMPLE_FRACTIONS

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────

class NetworkTrafficDataset(Dataset):
    """
    Wraps (sequences, labels) arrays for DataLoader consumption.
    sequences: ndarray [N, seq_len, n_features]
    labels:    ndarray [N]  (integer class indices)
    """
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.labels    = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Core loading function
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(
    csv_path:    str,
    dataset_name: str,
    label_column: str     = "Label",
    mode:         str     = "binary",     # "binary" | "multiclass"
    sample_frac:  float   = 0.10,
    test_size:    float   = 0.20,
    seq_len:      int     = 50,
    seq_noise:    float   = 0.01,
    random_seed:  int     = 42,
    benign_label: Optional[str] = None,
    force_frac:   bool    = False,
    balance_train_only: bool = False,
    sel_size:     float   = 0.0,
) -> Dict:
    """
    Load one CIC-* dataset CSV and return train/val splits as sequences.

    Returns dict with keys:
      X_train_seq, X_val_seq   : ndarray [N, seq_len, n_features]
      y_train, y_val           : ndarray [N] (int labels)
      n_features               : int
      n_classes                : int
      class_names              : list[str]
      feature_names            : list[str]
      scaler                   : fitted StandardScaler
      label_encoder            : fitted LabelEncoder (multiclass) or None (binary)
    """
    logger.info(f"Loading {dataset_name} from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"  Raw shape: {df.shape}")

    # ── Sample fraction ──────────────────────────────────────────────────────
    # Per-dataset override takes priority over the generic sample_frac argument,
    # UNLESS force_frac=True (used for pre-built balanced subsets, where the
    # caller wants the whole file used and no second sub-sampling applied).
    if force_frac:
        effective_frac = sample_frac
    else:
        effective_frac = DATASET_SAMPLE_FRACTIONS.get(dataset_name, sample_frac)
    if effective_frac < 1.0:
        df = df.sample(frac=effective_frac, random_state=random_seed)
        logger.info(f"  After {effective_frac*100:.1f}% sampling: {df.shape}")

    # ── Separate features and labels ─────────────────────────────────────────
    X_raw = df.drop(columns=[label_column])
    y_raw = df[label_column].values
    feature_names = list(X_raw.columns)

    # ── Clean numeric issues ─────────────────────────────────────────────────
    X_raw = X_raw.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_raw = X_raw.replace([np.inf, -np.inf], 0)

    # ── Encode labels ────────────────────────────────────────────────────────
    label_encoder = None
    if mode == "binary":
        # 0 = benign, 1 = attack
        if benign_label is None:
            unique = np.unique(y_raw)
            candidates = [l for l in unique if str(l).lower() in ("benign", "normal", "0")]
            benign_label = candidates[0] if candidates else unique[0]
        y_encoded = (y_raw != benign_label).astype(int)
        class_names = ["Benign", "Attack"]
        n_classes   = 2
        logger.info(f"  Binary mode — benign label: '{benign_label}' | "
                    f"Benign: {(y_encoded==0).sum()} | Attack: {(y_encoded==1).sum()}")
    else:
        label_encoder = LabelEncoder()
        y_encoded     = label_encoder.fit_transform(y_raw)
        class_names   = list(label_encoder.classes_)
        n_classes     = len(class_names)
        logger.info(f"  Multi-class mode — {n_classes} classes")

    # ── Normalise features ───────────────────────────────────────────────────
    scaler    = StandardScaler()
    X_unscaled = X_raw.values
    X_scaled  = scaler.fit_transform(X_unscaled)
    n_features = X_scaled.shape[1]

    # ── Train / Val split (stratified) ───────────────────────────────────────
    # Indices ride along so the un-normalised features can be split the
    # same way. Feasibility bounds for the attack constraints have to be
    # learned in the units traffic is actually measured in, not in
    # standardised units.
    idx = np.arange(len(X_scaled))
    X_train, X_val, y_train, y_val, idx_train, _ = train_test_split(
        X_scaled, y_encoded, idx,
        test_size=test_size,
        random_state=random_seed,
        stratify=y_encoded,
    )
    X_train_unscaled = X_unscaled[idx_train]
    logger.info(f"  Train: {len(X_train)} | Val: {len(X_val)}")

    # ── Optional selection split, carved out of TRAIN ────────────────────────
    # The val partition above is never trained on and is what results are
    # reported on. Choosing a hyperparameter by looking at it would make it
    # a selection set rather than a held-out one, so anything that needs to
    # be tuned is tuned here instead, on data taken from the training side.
    X_sel = y_sel = X_sel_seq = None
    if sel_size > 0:
        X_train, X_sel, y_train, y_sel = train_test_split(
            X_train, y_train,
            test_size=sel_size,
            random_state=random_seed,
            stratify=y_train if np.bincount(y_train).min() >= 2 else None,
        )
        logger.info(f"  Selection split carved from train: "
                    f"{len(X_sel)} | Train now: {len(X_train)}")

    # ── Class balancing, train partition only ────────────────────────────────
    # Oversampling before the split lets duplicate records land on both sides
    # of it, so validation scores are inflated by rows the model trained on.
    if balance_train_only:
        n_before = len(X_train)
        X_train, y_train = _oversample_to_balance(X_train, y_train, random_seed)
        logger.info(f"  Train balanced (train-only oversampling): "
                    f"{n_before} -> {len(X_train)} | Val left untouched "
                    f"at its natural distribution")

    # ── Expand to sequence ───────────────────────────────────────────────────
    X_train_seq = _expand_to_sequence(X_train, seq_len, seq_noise, random_seed)
    X_val_seq   = _expand_to_sequence(X_val,   seq_len, seq_noise, random_seed + 1)
    if X_sel is not None:
        X_sel_seq = _expand_to_sequence(X_sel, seq_len, seq_noise,
                                        random_seed + 2)

    return {
        "X_train_seq":   X_train_seq,
        "X_val_seq":     X_val_seq,
        "y_train":       y_train,
        "y_val":         y_val,
        "X_sel_seq":     X_sel_seq,
        "y_sel":         y_sel,
        "n_features":    n_features,
        "n_classes":     n_classes,
        "class_names":   class_names,
        "feature_names": feature_names,
        "scaler":        scaler,
        "label_encoder": label_encoder,
        # Normalised, model-space features — what EWC's Fisher estimate wants.
        "X_train_raw":   X_train,
        "X_val_raw":     X_val,
        # Genuinely un-normalised features, in the original measurement
        # units. Fit the attack feasibility bounds on THIS: the projection
        # and the validity check both operate in raw space, so bounds
        # learned from standardised values would describe a different
        # space entirely.
        "X_train_unscaled": X_train_unscaled,
    }


def _oversample_to_balance(
    X: np.ndarray, y: np.ndarray, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample every class up to the largest class count, with replacement."""
    rng = np.random.default_rng(seed)
    counts = np.bincount(y.astype(int))
    target = counts.max()
    parts = []
    for c in np.flatnonzero(counts):
        idx_c = np.flatnonzero(y == c)
        if len(idx_c) < target:
            extra = rng.choice(idx_c, size=target - len(idx_c), replace=True)
            idx_c = np.concatenate([idx_c, extra])
        parts.append(idx_c)
    idx = np.concatenate(parts)
    rng.shuffle(idx)
    return X[idx], y[idx]


def _expand_to_sequence(
    X: np.ndarray, seq_len: int, noise_std: float, seed: int
) -> np.ndarray:
    """
    Expand [N, F] tabular data to [N, seq_len, F] by repeating each row
    with slight Gaussian noise at each timestep — simulating flow variability.
    """
    rng = np.random.default_rng(seed)
    N, F = X.shape
    sequences = np.zeros((N, seq_len, F), dtype=np.float32)
    for t in range(seq_len):
        sequences[:, t, :] = X + rng.normal(0, noise_std, size=(N, F))
    return sequences


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader creation
# ──────────────────────────────────────────────────────────────────────────────

def make_dataloaders(
    data: Dict,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from a loaded dataset dict."""
    train_ds = NetworkTrafficDataset(data["X_train_seq"], data["y_train"])
    val_ds   = NetworkTrafficDataset(data["X_val_seq"],   data["y_val"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True, persistent_workers=True)
    return train_loader, val_loader


def make_dataloader_from_arrays(
    X_seq: np.ndarray,
    y:     np.ndarray,
    batch_size: int = 32,
    shuffle: bool = False,
) -> DataLoader:
    """Convenience helper for any (X_seq, y) pair."""
    ds = NetworkTrafficDataset(X_seq, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=2, pin_memory=True)


def make_class_balanced_loader(
    X_seq:      np.ndarray,
    y:          np.ndarray,
    batch_size: int = 512,
) -> DataLoader:
    """
    Class-balanced training DataLoader using WeightedRandomSampler.

    Each sample's weight is 1 / count[class]. Effectively the sampler will
    draw classes uniformly regardless of natural imbalance — critical for
    Stage 2 of decoupled (cRT) training on the CICIoMT2024 dataset whose
    benign class dominates ~99.7 % of the training set.
    """
    from torch.utils.data import WeightedRandomSampler
    import torch

    y_arr  = np.asarray(y).astype(int)
    counts = np.bincount(y_arr)
    # Avoid div-by-zero for classes that never appear
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts[y_arr]
    weights = torch.as_tensor(weights, dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(y_arr), replacement=True)

    ds = NetworkTrafficDataset(X_seq, y_arr)
    return DataLoader(
        ds, batch_size=batch_size, sampler=sampler,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )
