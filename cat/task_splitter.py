"""
task_splitter.py — Paper 3 continual-learning task construction.
================================================================
Two utilities:

  EXP 1 (attack-incremental CL, single dataset):
    split_into_attack_tasks(data, dataset_name)
      Groups the fine-grained attack labels of one dataset into
      attack FAMILIES (DDoS, DoS, Mirai, Recon, Spoofing, MQTT, ...)
      and produces a sequence of class-incremental tasks. Task t
      introduces family t; benign traffic is present in every task.
      The model head is sized for all families from the start
      (class-incremental protocol). Evaluation is cumulative:
      after task t, test on families {0..t}.

  EXP 2 (cross-dataset CL, union feature space):
    build_union_feature_index(feature_name_lists)
      Builds a shared feature vocabulary across the three datasets
      so a single feature-tokenizer can serve all of them (each
      dataset activates its own subset of feature columns). This
      lets the SAME backbone (including the vanilla Transformer)
      train continually across CICIoT -> CICIoMT -> CICIoV without
      an ad-hoc input-dim adaptation.

Both utilities operate on the dict returned by data_loader.load_dataset.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# Attack-family rules (prefix / keyword → family).  Order matters: the first
# matching rule wins.  Everything benign maps to family "Benign".
# ──────────────────────────────────────────────────────────────────────────
_FAMILY_RULES = {
    "CICIoT": [
        ("BENIGN",      "Benign"),
        ("MIRAI",       "Mirai"),
        ("DDOS",        "DDoS"),
        ("DOS",         "DoS"),
        ("MITM",        "Spoofing"),
        ("SPOOFING",    "Spoofing"),
        ("DNS",         "Spoofing"),
        ("RECON",       "Recon"),
        ("SCAN",        "Recon"),
        ("VULNERABILITY", "Recon"),
        ("WEB",         "Web"),
        ("BRUTE",       "BruteForce"),
        ("BACKDOOR",    "Web"),
        ("XSS",         "Web"),
        ("SQL",         "Web"),
        ("UPLOAD",      "Web"),
        ("COMMAND",     "Web"),
    ],
    "CICIoMT": [
        ("BENIGN",      "Benign"),
        ("MQTT",        "MQTT"),
        ("DDOS",        "DDoS"),
        ("DOS",         "DoS"),
        ("RECON",       "Recon"),
        ("SCAN",        "Recon"),
        ("ARP",         "Spoofing"),
        ("SPOOF",       "Spoofing"),
    ],
    "CICIoV": [
        ("BENIGN",      "Benign"),
        ("DOS",         "DoS"),
        ("SPOOF",       "Spoofing"),
    ],
    # ToN_IoT: ``type`` is already a clean per-attack label. Keep each
    # attack as its own family. NOTE: order matters because matching is
    # substring-based — "DDOS" must precede "DOS" so that "ddos" does not
    # fall through to the "DOS" rule.
    "TONIoT": [
        ("NORMAL",      "Benign"),
        ("DDOS",        "DDoS"),
        ("DOS",         "DoS"),
        ("BACKDOOR",    "Backdoor"),
        ("INJECTION",   "Injection"),
        ("PASSWORD",    "Password"),
        ("SCANNING",    "Scanning"),
        ("RANSOMWARE",  "Ransomware"),
        ("XSS",         "XSS"),
        ("MITM",        "MITM"),
    ],
}

# Order in which families are introduced as sequential tasks (excluding
# Benign, which is always present). Families not listed are appended in
# alphabetical order after the listed ones.
_FAMILY_ORDER = {
    "CICIoT":  ["DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web", "BruteForce"],
    "CICIoMT": ["DDoS", "DoS", "MQTT", "Recon", "Spoofing"],
    "CICIoV":  ["DoS", "Spoofing"],
    "TONIoT":  ["DDoS", "DoS", "Scanning", "Backdoor", "Injection",
                "Password", "Ransomware", "XSS", "MITM"],
}


def label_to_family(class_name: str, dataset_name: str) -> str:
    """Map one fine-grained attack label to its family."""
    name = class_name.upper().replace("_", "").replace("-", "").replace(" ", "")
    rules = _FAMILY_RULES.get(dataset_name, _FAMILY_RULES["CICIoT"])
    for key, fam in rules:
        if key.replace("_", "").replace("-", "").replace(" ", "") in name:
            return fam
    return "Other"


def build_family_mapping(
    class_names: List[str], dataset_name: str,
) -> Tuple[np.ndarray, List[str]]:
    """Return (encoded_class_idx -> family_idx array, ordered family names).

    Family index 0 is always Benign. Subsequent indices follow
    _FAMILY_ORDER for the dataset, with any leftover families appended.
    """
    # Family of each original encoded class
    fam_of_class = [label_to_family(c, dataset_name) for c in class_names]

    present = []
    for f in fam_of_class:
        if f not in present:
            present.append(f)

    # Build ordered family list: Benign first, then _FAMILY_ORDER, then rest
    ordered = ["Benign"]
    for f in _FAMILY_ORDER.get(dataset_name, []):
        if f in present and f not in ordered:
            ordered.append(f)
    for f in sorted(present):
        if f not in ordered:
            ordered.append(f)

    fam_to_idx = {f: i for i, f in enumerate(ordered)}
    class_to_family_idx = np.array(
        [fam_to_idx[fam_of_class[c]] for c in range(len(class_names))],
        dtype=np.int64,
    )
    return class_to_family_idx, ordered


def split_into_attack_tasks(
    data: Dict, dataset_name: str, min_task_samples: int = 50,
) -> Tuple[List[Dict], List[str]]:
    """Build a class-incremental task sequence by attack family.

    Args:
      data: dict from load_dataset(mode='multiclass') — must contain
            X_train_seq, y_train, X_val_seq, y_val, class_names.
      dataset_name: one of CICIoT / CICIoMT / CICIoV.

    Returns:
      tasks: list of dicts, one per attack family (excluding Benign).
             Each dict has:
               'family'        : family name introduced at this task
               'family_idx'    : its family index
               'X_train','y_train' : training data = benign + THIS family
                                     (labels in family-index space)
               'X_val','y_val'     : validation data for THIS family + benign
                                     (for per-task accuracy)
      family_names: ordered list of all family names (index 0 = Benign).
    """
    class_names = data["class_names"]
    c2f, family_names = build_family_mapping(class_names, dataset_name)

    # Remap fine labels -> family labels
    ytr_fam = c2f[data["y_train"]]
    yva_fam = c2f[data["y_val"]]
    Xtr, Xva = data["X_train_seq"], data["X_val_seq"]

    benign_idx = 0  # by construction
    attack_families = [f for f in family_names if f != "Benign"]

    tasks = []
    for fam in attack_families:
        fidx = family_names.index(fam)
        # Training data for this task: benign + this family only
        tr_mask = (ytr_fam == benign_idx) | (ytr_fam == fidx)
        va_mask = (yva_fam == benign_idx) | (yva_fam == fidx)
        if int((ytr_fam == fidx).sum()) < min_task_samples:
            # Too few samples for this family — skip as its own task
            continue
        tasks.append({
            "family":     fam,
            "family_idx": fidx,
            "X_train":    Xtr[tr_mask],
            "y_train":    ytr_fam[tr_mask],
            "X_val":      Xva[va_mask],
            "y_val":      yva_fam[va_mask],
        })

    return tasks, family_names


# ──────────────────────────────────────────────────────────────────────────
# EXP 2 — union feature space across datasets
# ──────────────────────────────────────────────────────────────────────────
def build_union_feature_index(
    feature_name_lists: Dict[str, List[str]],
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Build a shared (union) feature vocabulary across datasets.

    Args:
      feature_name_lists: {dataset_name: [feature_name, ...]} for each dataset.

    Returns:
      union_features: sorted list of all distinct feature names.
      column_index:   {dataset_name: np.ndarray of column positions in the
                       union space that this dataset's features occupy}.

    A dataset's raw feature matrix X[:, :F_d] is scattered into a
    union matrix U[:, len(union_features)] at the returned positions;
    absent features are left at zero. The feature tokenizer is then
    sized for len(union_features) and shared across all datasets, so the
    same backbone (vanilla or AAM-TRANS) trains continually across them.
    """
    union = []
    for names in feature_name_lists.values():
        for n in names:
            if n not in union:
                union.append(n)
    union = sorted(union)
    pos = {u: i for i, u in enumerate(union)}

    column_index = {}
    for ds, names in feature_name_lists.items():
        column_index[ds] = np.array([pos[n] for n in names], dtype=np.int64)
    return union, column_index


def scatter_to_union(
    X: np.ndarray, col_idx: np.ndarray, union_dim: int,
) -> np.ndarray:
    """Scatter a dataset's feature matrix into the union feature space.

    X:        [N, F_d]  (or [N, seq_len, F_d])
    col_idx:  [F_d]     positions in the union space
    union_dim:int       len(union_features)
    Returns:  [N, union_dim] (or [N, seq_len, union_dim]) zero-padded.
    """
    if X.ndim == 2:
        U = np.zeros((X.shape[0], union_dim), dtype=X.dtype)
        U[:, col_idx] = X
        return U
    if X.ndim == 3:
        U = np.zeros((X.shape[0], X.shape[1], union_dim), dtype=X.dtype)
        U[:, :, col_idx] = X
        return U
    raise ValueError(f"Unexpected X.ndim={X.ndim}")
