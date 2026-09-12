"""
detection_metrics.py — Evaluation metrics for Gate-Signature Adversarial
Detection (GSAD).

Provides:
  auroc()                  — Area under ROC curve
  eer()                    — Equal Error Rate (+ its threshold)
  tpr_at_fpr()             — detection rate at a fixed false-alarm rate
  detection_prf1()         — precision / recall / F1 at a threshold
  accuracy_coverage_curve()— system-level accuracy vs coverage trade-off
  summarize_detection()    — convenience: all detection metrics at once

Convention:
  score   : higher  => more "adversarial"
  y_is_adv: 1 = adversarial / should-be-flagged, 0 = clean
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple

# NumPy 2.0 renamed `np.trapz` -> `np.trapezoid`; older NumPy lacks the new name.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ──────────────────────────────────────────────────────────────────────────────
# ROC-based metrics
# ──────────────────────────────────────────────────────────────────────────────

def _roc_points(score: np.ndarray, y_is_adv: np.ndarray):
    """Return (fpr, tpr, thresholds) sorted by threshold descending."""
    score    = np.asarray(score,    dtype=np.float64)
    y_is_adv = np.asarray(y_is_adv, dtype=np.int64)
    order    = np.argsort(-score)
    s        = score[order]
    y        = y_is_adv[order]

    P = max(1, int(y.sum()))
    N = max(1, int((1 - y).sum()))

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / P
    fpr = fp / N
    # prepend the (0,0) origin
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    thr = np.concatenate([[np.inf], s])
    return fpr, tpr, thr


def auroc(score: np.ndarray, y_is_adv: np.ndarray) -> float:
    """Area under the ROC curve (trapezoidal)."""
    fpr, tpr, _ = _roc_points(score, y_is_adv)
    return float(_trapz(tpr, fpr))


def eer(score: np.ndarray, y_is_adv: np.ndarray) -> Tuple[float, float]:
    """
    Equal Error Rate: the point where FPR == FNR (= 1 - TPR).
    Returns (eer_value, threshold_at_eer).
    """
    fpr, tpr, thr = _roc_points(score, y_is_adv)
    fnr = 1.0 - tpr
    diff = np.abs(fpr - fnr)
    i = int(np.argmin(diff))
    eer_val = float((fpr[i] + fnr[i]) / 2.0)
    return eer_val, float(thr[i])


def tpr_at_fpr(score: np.ndarray, y_is_adv: np.ndarray,
               target_fpr: float = 0.05) -> Tuple[float, float]:
    """
    True-positive (detection) rate at a fixed false-positive rate.
    Returns (tpr, threshold).
    """
    fpr, tpr, thr = _roc_points(score, y_is_adv)
    # largest threshold whose FPR <= target
    ok = np.where(fpr <= target_fpr)[0]
    if len(ok) == 0:
        return 0.0, float(thr[0])
    i = int(ok[-1])
    return float(tpr[i]), float(thr[i])


# ──────────────────────────────────────────────────────────────────────────────
# Threshold-based metrics
# ──────────────────────────────────────────────────────────────────────────────

def detection_prf1(score: np.ndarray, y_is_adv: np.ndarray,
                   threshold: float) -> Dict[str, float]:
    """Precision / Recall / F1 / Accuracy of detection at a given threshold."""
    score    = np.asarray(score)
    y        = np.asarray(y_is_adv, dtype=np.int64)
    pred     = (score > threshold).astype(np.int64)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc  = (tp + tn) / max(1, tp + fp + fn + tn)
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ──────────────────────────────────────────────────────────────────────────────
# System-level: accuracy vs coverage
# ──────────────────────────────────────────────────────────────────────────────

def accuracy_coverage_curve(
    score:        np.ndarray,
    correct:      np.ndarray,
    n_points:     int = 21,
) -> Dict[str, np.ndarray]:
    """
    Accuracy-coverage trade-off.

    Args:
        score   : detector score (higher = more likely flagged/rejected)
        correct : 1 if the classifier prediction was correct, else 0
                  (computed on the SAME inputs, after attack)
    Returns dict with arrays: coverage, accuracy, threshold.

    For each threshold, inputs with score <= threshold are 'accepted';
    accuracy is measured on the accepted subset, coverage is its fraction.
    """
    score   = np.asarray(score, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.int64)
    qs      = np.linspace(0.0, 1.0, n_points)
    cov, acc, thr = [], [], []
    for q in qs:
        t = np.quantile(score, q) if q < 1.0 else np.inf
        accepted = score <= t
        c = accepted.mean()
        a = correct[accepted].mean() if accepted.any() else 0.0
        cov.append(c); acc.append(a); thr.append(t)
    return {"coverage": np.array(cov),
            "accuracy": np.array(acc),
            "threshold": np.array(thr)}


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: full detection summary
# ──────────────────────────────────────────────────────────────────────────────

def summarize_detection(score: np.ndarray, y_is_adv: np.ndarray,
                        threshold: float = None,
                        target_fpr: float = 0.05) -> Dict[str, float]:
    """Compute the full set of detection metrics in one call."""
    a          = auroc(score, y_is_adv)
    e, e_thr   = eer(score, y_is_adv)
    t_tpr, t_thr = tpr_at_fpr(score, y_is_adv, target_fpr)
    if threshold is None:
        threshold = t_thr            # default operating point = fixed FPR
    prf = detection_prf1(score, y_is_adv, threshold)
    return {
        "auroc":            a,
        "eer":              e,
        "eer_threshold":    e_thr,
        f"tpr@fpr{target_fpr}": t_tpr,
        "op_threshold":     threshold,
        "det_precision":    prf["precision"],
        "det_recall":       prf["recall"],
        "det_f1":           prf["f1"],
        "det_accuracy":     prf["accuracy"],
    }
