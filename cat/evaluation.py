"""
evaluation.py — Comprehensive metrics for adversarial robustness evaluation.

Standard metrics (dissertation Ch.3 §3.8):
  Accuracy, Precision, Recall, F1-Score  — standard classification
  Attack Success Rate (ASR)              — adversarial effectiveness
  Validity Rate (VR)                     — protocol constraint compliance

Extended metrics (for Q1 journal publication):
  MCC (Matthews Correlation Coefficient) — robust to class imbalance
  Cohen's Kappa                          — inter-rater agreement proxy
  Balanced Accuracy                      — mean per-class recall
  G-Mean                                 — geometric mean of sensitivity & specificity
  FPR / FAR (False Alarm Rate)           — IDS-critical metric
  FNR (Miss Detection Rate)              — IDS-critical metric
  DR  (Detection Rate)                   — = Recall, IDS alias
  ROC-AUC (macro OvR)                    — ranking quality
  PR-AUC  (macro)                        — imbalanced-class ranking
  Inference latency (ms/sample)          — deployment efficiency
  Model parameters                       — model complexity

Also provides:
  evaluate_robustness()         — full adversarial eval loop (one model/dataset)
  evaluate_robustness_extended()— same but collects softmax scores for AUC
  save_results()                — persist results to CSV + classification report TXT
  measure_inference_time()      — benchmarks model latency
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score,
)
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import os
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Core metric functions
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "weighted",
) -> Dict[str, float]:
    """Accuracy, Precision, Recall, F1 (weighted by default)."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall":    recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1":        f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def compute_extended_metrics(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    y_score:   Optional[np.ndarray] = None,   # [N, n_classes] softmax probabilities
    average:   str = "weighted",
    n_classes: Optional[int] = None,
) -> Dict[str, float]:
    """
    Full extended metric suite for Q1 journal reporting.

    IDS-specific metrics included:
      FAR (False Alarm Rate) = macro FPR — fraction of benign traffic misclassified
      FNR (Miss Detection Rate)          — fraction of attacks missed
      DR  (Detection Rate)               — = weighted Recall

    Returns merged dict of standard + extended metrics.
    """
    metrics = compute_metrics(y_true, y_pred, average)

    n_cls = n_classes if n_classes else len(np.unique(np.concatenate([y_true, y_pred])))

    # ── Robust classification metrics ─────────────────────────────────────────
    try:
        metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        metrics["mcc"] = float("nan")

    try:
        metrics["kappa"] = float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        metrics["kappa"] = float("nan")

    try:
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        metrics["balanced_accuracy"] = float("nan")

    # ── IDS-specific: FAR (False Alarm Rate) and FNR (Miss Rate) ─────────────
    # Class 0 = benign, classes 1..K = attack types (convention in data_loader).
    # Binary  : FAR = FP/(FP+TN);  FNR = FN/(FN+TP)  with attack (class 1) as positive.
    # Multiclass: FAR = benign samples classified as any attack / total benign;
    #             FNR = attack samples classified as benign / total attack.
    # Macro-averaging FPR/FNR over all classes gives FAR==FNR for binary (wrong);
    # the IDS-aware formula below avoids this.
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))

    if n_cls == 2:
        TN = int(cm[0, 0]); FP = int(cm[0, 1])
        FN = int(cm[1, 0]); TP = int(cm[1, 1])
        far = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    else:
        benign_total = int(cm[0, :].sum())
        far = int(cm[0, 1:].sum()) / benign_total if benign_total > 0 else 0.0
        attack_total = int(cm[1:, :].sum())
        fnr = int(cm[1:, 0].sum()) / attack_total if attack_total > 0 else 0.0

    metrics["fpr"] = float(far)
    metrics["far"] = float(far)               # IDS alias
    metrics["fnr"] = float(fnr)
    metrics["dr"]  = float(max(0.0, 1.0 - fnr))  # Detection Rate = Sensitivity = 1 - FNR

    # ── G-Mean = sqrt(Sensitivity × Specificity) ──────────────────────────────
    # Sensitivity = DR = 1 - FNR;  Specificity = TNR = 1 - FAR
    metrics["gmean"] = float(np.sqrt(metrics["dr"] * max(0.0, 1.0 - far)))

    # ── Macro F1 (complement to weighted) ────────────────────────────────────
    metrics["f1_macro"] = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )

    # ── AUC metrics (require probability scores) ──────────────────────────────
    if y_score is not None and len(y_score) == len(y_true):
        try:
            if n_cls == 2:
                prob = y_score[:, 1] if y_score.ndim > 1 else y_score
                metrics["roc_auc"] = float(roc_auc_score(y_true, prob))
                metrics["pr_auc"]  = float(average_precision_score(y_true, prob))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
                )
                metrics["pr_auc"] = float(
                    average_precision_score(y_true, y_score, average="macro")
                )
        except Exception as e:
            logger.debug(f"AUC computation skipped: {e}")
            metrics["roc_auc"] = float("nan")
            metrics["pr_auc"]  = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"]  = float("nan")

    return metrics


def compute_asr(
    y_true:       np.ndarray,
    y_pred_clean: np.ndarray,
    y_pred_adv:   np.ndarray,
) -> float:
    """
    Attack Success Rate (ASR):
    ASR = (1/N) Σ 1[f(x_adv_i) ≠ y_i] × 100%

    Only counts samples that were CORRECTLY classified on clean data
    (otherwise the attack doesn't need to 'succeed' — model was already wrong).
    """
    correctly_classified = (y_pred_clean == y_true)
    if correctly_classified.sum() == 0:
        return 0.0
    misclassified_after = (y_pred_adv[correctly_classified] != y_true[correctly_classified])
    return float(misclassified_after.mean()) * 100.0


def compute_validity_rate(
    x_orig_raw: np.ndarray,
    x_adv_raw:  np.ndarray,
    fcc,
) -> float:
    """
    Validity Rate (VR): proportion of adversarial samples satisfying all protocol constraints.
    VR = (1/N) Σ 1[isValid(x_adv_i)] × 100%
    """
    if fcc is None:
        return 100.0
    return fcc.validity_rate(x_orig_raw, x_adv_raw) * 100.0


# ──────────────────────────────────────────────────────────────────────────────
# Efficiency metrics
# ──────────────────────────────────────────────────────────────────────────────

def measure_inference_time(
    model,
    x_tensor: torch.Tensor,
    device:   torch.device,
    n_warmup: int = 10,
    n_repeat: int = 50,
    batch:    int = 32,
) -> Dict[str, float]:
    """
    Measure inference latency in ms/sample and throughput in samples/sec.
    Uses a fixed batch size for stable measurement.
    """
    model.eval()
    x_batch = x_tensor[:min(batch, len(x_tensor))].to(device)
    n_samples = x_batch.shape[0]

    with torch.no_grad():
        for _ in range(n_warmup):
            model(x_batch)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_repeat):
            model(x_batch)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    total_samples = n_repeat * n_samples
    ms_per_sample = (elapsed / total_samples) * 1000.0
    throughput    = total_samples / elapsed

    return {
        "latency_ms_per_sample": ms_per_sample,
        "throughput_samples_sec": throughput,
    }


def count_model_params(model) -> Dict[str, int]:
    """Count total and trainable parameters."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


# ──────────────────────────────────────────────────────────────────────────────
# Full adversarial robustness evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_robustness(
    model,
    val_loader:   DataLoader,
    attacker,
    device:       torch.device,
    attack_config,
    fcc           = None,
    scaler        = None,
    class_names:  Optional[List[str]] = None,
    extended:     bool = True,
) -> Dict[str, Dict]:
    """
    Evaluate model on all attack types specified in attack_config.

    Args:
        extended: if True, compute extended metrics (MCC, AUC, FAR, etc.)

    Returns nested dict:
      results[attack_name] = {
          accuracy, precision, recall, f1,
          [extended: mcc, kappa, roc_auc, pr_auc, far, fnr, gmean, ...],
          asr, validity_rate,
          y_true, y_pred, anomaly_scores
      }
    """
    model.eval()

    all_x_batches = []
    all_y_batches = []
    for x, y in val_loader:
        all_x_batches.append(x)
        all_y_batches.append(y)

    X_val_tensor = torch.cat(all_x_batches, dim=0)   # [N, seq_len, F]
    y_val_tensor = torch.cat(all_y_batches, dim=0)   # [N]
    n_classes    = len(class_names) if class_names else int(y_val_tensor.max().item()) + 1

    # ── V3.5c: cap the adversarial-evaluation subset ─────────────────────────
    # Crafting adversarial examples (esp. the query-based Square Attack) is
    # O(n_samples × n_queries × mc_samples). On a full 40 K-row val set this
    # explodes to many hours. Standard practice in robustness papers
    # (RobustBench, etc.) is to evaluate on a fixed stratified subset of a
    # few thousand samples. We do the same — deterministic, seeded.
    max_atk = getattr(attack_config, "attack_eval_max_samples", 3000)
    if max_atk and len(X_val_tensor) > max_atk:
        import numpy as _np
        _rng = _np.random.default_rng(42)
        y_np = y_val_tensor.cpu().numpy()
        # Stratified subsample — keep class proportions
        idx_keep = []
        per_cls  = max(1, max_atk // max(1, len(_np.unique(y_np))))
        for c in _np.unique(y_np):
            c_idx = _np.where(y_np == c)[0]
            take  = min(len(c_idx), per_cls)
            idx_keep.extend(_rng.choice(c_idx, size=take, replace=False).tolist())
        # Top up to max_atk if stratified quota under-filled
        if len(idx_keep) < max_atk:
            remaining = _np.setdiff1d(_np.arange(len(y_np)), _np.array(idx_keep))
            extra = _rng.choice(remaining,
                                size=min(len(remaining), max_atk - len(idx_keep)),
                                replace=False)
            idx_keep.extend(extra.tolist())
        idx_keep = _np.array(sorted(idx_keep))
        X_val_tensor = X_val_tensor[idx_keep]
        y_val_tensor = y_val_tensor[idx_keep]
        logger.info(
            f"  Adversarial eval subset: {len(X_val_tensor)} samples "
            f"(stratified from full val set)"
        )

    results = {}

    # ── Build attack list ────────────────────────────────────────────────────
    attack_list = [("Clean", X_val_tensor)]
    logger.info(f"  [robustness] Clean set ready ({len(X_val_tensor)} samples)")

    for eps in attack_config.fgsm_epsilons:
        logger.info(f"  [robustness] Crafting FGSM eps={eps} ...")
        x_adv = attacker.fgsm(
            X_val_tensor, y_val_tensor,
            epsilon=eps,
            apply_constraints=attack_config.apply_constraints,
        )   # already returns CPU tensor
        attack_list.append((f"FGSM_eps{eps}", x_adv))

    # ── White-box: multi-step PGD ────────────────────────────────────────────
    for eps in getattr(attack_config, "pgd_epsilons", []):
        n_iter = getattr(attack_config, "pgd_n_iter", 20)
        logger.info(f"  [robustness] Crafting PGD eps={eps} "
                    f"({n_iter} steps) ...")
        x_adv = attacker.pgd(
            X_val_tensor, y_val_tensor,
            epsilon=eps,
            alpha=getattr(attack_config, "pgd_alpha", 0.01),
            n_iter=n_iter,
            apply_constraints=attack_config.apply_constraints,
        )
        attack_list.append((f"PGD_eps{eps}", x_adv))

    # ── Black-box: Square Attack (score-based) ───────────────────────────────
    if getattr(attack_config, "square_n_queries", 0) > 0:
        _sq_q = getattr(attack_config, "square_n_queries", 200)
        logger.info(f"  [robustness] Crafting Square Attack ({_sq_q} queries) ...")
        x_sq = attacker.square_attack(
            X_val_tensor, y_val_tensor,
            epsilon  = getattr(attack_config, "square_epsilon",   0.10),
            n_queries= _sq_q,
            p_init   = getattr(attack_config, "square_p_init",    0.05),
            apply_constraints=attack_config.apply_constraints,
        )
        attack_list.append(
            (f"Square_eps{getattr(attack_config, 'square_epsilon', 0.10)}", x_sq)
        )

    # ── Black-box: Transfer Attack (zero-query) ──────────────────────────────
    for t_eps in getattr(attack_config, "transfer_epsilons", []):
        logger.info(f"  [robustness] Crafting Transfer Attack eps={t_eps} ...")
        x_tr = attacker.transfer_attack(
            X_val_tensor, y_val_tensor,
            epsilon=t_eps,
            apply_constraints=attack_config.apply_constraints,
        )
        attack_list.append((f"Transfer_eps{t_eps}", x_tr))

    # ── Legacy PGD (kept for back-compat; off by default in V3) ──────────────
    if getattr(attack_config, "use_pgd", False):
        x_pgd = attacker.pgd(
            X_val_tensor, y_val_tensor,
            epsilon=attack_config.pgd_epsilon,
            alpha=attack_config.pgd_alpha,
            n_iter=attack_config.pgd_iterations,
            apply_constraints=attack_config.apply_constraints,
        )
        attack_list.append((f"PGD_eps{attack_config.pgd_epsilon}", x_pgd))

    for sigma in attack_config.noise_sigmas:
        x_adv = attacker.gaussian_noise(
            X_val_tensor.to(device), sigma=sigma,
            apply_constraints=attack_config.apply_constraints,
        ).cpu()
        attack_list.append((f"Gaussian_std{sigma}", x_adv))

    # ── Clean predictions for ASR reference ──────────────────────────────────
    clean_preds, clean_labels = _batch_predict(
        model, X_val_tensor, y_val_tensor, device
    )

    # ── Evaluate each attack ──────────────────────────────────────────────────
    for attack_name, x_adv in attack_list:
        logger.info(f"  Evaluating: {attack_name}")

        adv_preds, adv_labels = _batch_predict(
            model, x_adv, y_val_tensor, device
        )
        anomaly_scores = _batch_anomaly_scores(model, x_adv, device)

        if extended:
            y_score = _batch_softmax_scores(model, x_adv, device)
            metrics = compute_extended_metrics(
                adv_labels, adv_preds, y_score, n_classes=n_classes
            )
        else:
            metrics = compute_metrics(adv_labels, adv_preds)

        asr = compute_asr(clean_labels, clean_preds, adv_preds)

        # VR: inverse-transform to raw space if scaler available
        if fcc is not None and scaler is not None and attack_name != "Clean":
            B, L, F = x_adv.shape
            x_raw = scaler.inverse_transform(
                x_adv.numpy().reshape(-1, F)
            ).reshape(B, L, F)
            x_orig_raw = scaler.inverse_transform(
                X_val_tensor.numpy().reshape(-1, F)
            ).reshape(B, L, F)
            vr = compute_validity_rate(
                x_orig_raw.mean(axis=1),
                x_raw.mean(axis=1),
                fcc,
            )
        else:
            vr = 100.0 if attack_name == "Clean" else 0.0

        results[attack_name] = {
            **metrics,
            "asr":            asr,
            "validity_rate":  vr,
            "y_true":         adv_labels,
            "y_pred":         adv_preds,
            "anomaly_scores": anomaly_scores,
        }

        logger.info(
            f"    Acc={metrics['accuracy']*100:.2f}% | "
            f"F1={metrics['f1']*100:.2f}% | "
            f"MCC={metrics.get('mcc', float('nan')):.3f} | "
            f"FAR={metrics.get('far', float('nan'))*100:.2f}% | "
            f"ASR={asr:.2f}% | VR={vr:.2f}%"
        )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Save helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_results(
    results:     Dict,
    save_dir:    str,
    prefix:      str,
    class_names: Optional[List[str]] = None,
) -> str:
    """
    Save robustness results to:
      - {prefix}_all_results.csv   (one row per attack, all metrics)
      - {prefix}_{attack}_report.txt  (classification report per attack)

    Returns path to the summary CSV.
    """
    os.makedirs(save_dir, exist_ok=True)

    rows = []
    for attack_name, res in results.items():
        row = {
            "attack":          attack_name,
            "accuracy":        res["accuracy"],
            "precision":       res["precision"],
            "recall":          res["recall"],
            "f1":              res["f1"],
            "f1_macro":        res.get("f1_macro",        float("nan")),
            "mcc":             res.get("mcc",             float("nan")),
            "kappa":           res.get("kappa",           float("nan")),
            "balanced_acc":    res.get("balanced_accuracy", float("nan")),
            "gmean":           res.get("gmean",           float("nan")),
            "roc_auc":         res.get("roc_auc",         float("nan")),
            "pr_auc":          res.get("pr_auc",          float("nan")),
            "fpr":             res.get("fpr",             float("nan")),
            "far":             res.get("far",             float("nan")),
            "fnr":             res.get("fnr",             float("nan")),
            "dr":              res.get("dr",              float("nan")),
            "asr":             res["asr"],
            "validity_rate":   res["validity_rate"],
        }
        rows.append(row)

        if "y_true" in res and "y_pred" in res:
            report = classification_report(
                res["y_true"], res["y_pred"],
                labels=list(range(len(class_names))),
                target_names=class_names,
                zero_division=0,
            )
            report_path = os.path.join(save_dir, f"{prefix}_{attack_name}_report.txt")
            with open(report_path, "w") as f:
                f.write(f"Attack: {attack_name}\n\n{report}")

    df       = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, f"{prefix}_all_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved: {csv_path}")
    return csv_path


# ──────────────────────────────────────────────────────────────────────────────
# Objective 3 — Anomaly Detection Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_anomaly_metrics(
    y_true_binary:  np.ndarray,   # [N] — 0=benign, 1=any attack
    anomaly_scores: np.ndarray,   # [N] — model's anomaly_score output ∈ [0,1]
    n_thresholds:   int = 200,
) -> Dict[str, object]:
    """
    Standalone evaluation of the Confidence-Based Anomaly Scorer (Objective 3).

    Metrics:
      AUROC   — threshold-independent ranking quality
      EER     — Equal Error Rate (threshold where FPR = FNR)
      d'      — Signal detection theory d-prime (score separation)
      ECE     — Expected Calibration Error (10-bin)
      MCE     — Maximum Calibration Error
      Threshold sensitivity curve  — DataFrame of [threshold, fpr, fnr, tpr, f1]

    y_true_binary: 0 = benign traffic, 1 = any attack class.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    results = {}

    # ── AUROC ─────────────────────────────────────────────────────────────────
    try:
        results["anomaly_auroc"] = float(roc_auc_score(y_true_binary, anomaly_scores))
    except Exception:
        results["anomaly_auroc"] = float("nan")

    # ── Full ROC curve for EER and threshold sensitivity ─────────────────────
    try:
        fpr_arr, tpr_arr, thresh_arr = roc_curve(y_true_binary, anomaly_scores)
        fnr_arr = 1.0 - tpr_arr

        # EER — point where FPR ≈ FNR
        eer_idx = int(np.argmin(np.abs(fpr_arr - fnr_arr)))
        results["eer"]             = float((fpr_arr[eer_idx] + fnr_arr[eer_idx]) / 2)
        results["eer_threshold"]   = float(thresh_arr[eer_idx]) if eer_idx < len(thresh_arr) else float("nan")
        results["eer_fpr"]         = float(fpr_arr[eer_idx])
        results["eer_fnr"]         = float(fnr_arr[eer_idx])

        # Threshold sensitivity DataFrame
        ts_rows = []
        ts_thresholds = np.linspace(0.0, 1.0, n_thresholds)
        for thr in ts_thresholds:
            pred   = (anomaly_scores >= thr).astype(int)
            tp     = int(((pred == 1) & (y_true_binary == 1)).sum())
            fp     = int(((pred == 1) & (y_true_binary == 0)).sum())
            fn     = int(((pred == 0) & (y_true_binary == 1)).sum())
            tn     = int(((pred == 0) & (y_true_binary == 0)).sum())
            fpr_t  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr_t  = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            prec_t = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec_t  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_t   = 2 * prec_t * rec_t / (prec_t + rec_t) if (prec_t + rec_t) > 0 else 0.0
            ts_rows.append({
                "threshold": thr,
                "fpr": fpr_t, "fnr": fnr_t,
                "precision": prec_t, "recall": rec_t, "f1": f1_t,
            })
        results["threshold_curve"] = pd.DataFrame(ts_rows)

        # Optimal threshold by max-F1
        best_idx = results["threshold_curve"]["f1"].idxmax()
        results["optimal_threshold"] = float(results["threshold_curve"].loc[best_idx, "threshold"])
        results["optimal_f1"]        = float(results["threshold_curve"].loc[best_idx, "f1"])
        results["optimal_fpr"]       = float(results["threshold_curve"].loc[best_idx, "fpr"])
        results["optimal_fnr"]       = float(results["threshold_curve"].loc[best_idx, "fnr"])

    except Exception as e:
        logger.debug(f"Threshold curve computation failed: {e}")
        results.setdefault("eer", float("nan"))
        results.setdefault("threshold_curve", pd.DataFrame())

    # ── d' (signal detection theory — score separation) ──────────────────────
    try:
        normal_scores = anomaly_scores[y_true_binary == 0]
        attack_scores = anomaly_scores[y_true_binary == 1]
        mu_n, mu_a    = normal_scores.mean(), attack_scores.mean()
        sigma_n       = normal_scores.std() + 1e-12
        sigma_a       = attack_scores.std() + 1e-12
        pooled_sigma  = np.sqrt((sigma_n ** 2 + sigma_a ** 2) / 2)
        results["d_prime"]        = float((mu_a - mu_n) / pooled_sigma)
        results["mean_normal_score"] = float(mu_n)
        results["mean_attack_score"] = float(mu_a)
        results["std_normal_score"]  = float(sigma_n)
        results["std_attack_score"]  = float(sigma_a)
    except Exception:
        results["d_prime"] = float("nan")

    # ── ECE / MCE (Expected/Maximum Calibration Error) ────────────────────────
    try:
        ece, mce = _compute_ece_mce(y_true_binary, anomaly_scores, n_bins=10)
        results["ece"] = ece
        results["mce"] = mce
    except Exception:
        results["ece"] = float("nan")
        results["mce"] = float("nan")

    return results


def _compute_ece_mce(
    y_true:   np.ndarray,
    y_score:  np.ndarray,
    n_bins:   int = 10,
) -> Tuple[float, float]:
    """
    Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    Bins predictions by confidence score; measures |avg_confidence - avg_accuracy|.
    """
    bin_edges   = np.linspace(0, 1, n_bins + 1)
    ece         = 0.0
    mce         = 0.0
    n           = len(y_true)

    for i in range(n_bins):
        lo, hi   = bin_edges[i], bin_edges[i + 1]
        mask     = (y_score >= lo) & (y_score < hi)
        if mask.sum() == 0:
            continue
        avg_conf = float(y_score[mask].mean())
        avg_acc  = float(y_true[mask].mean())
        gap      = abs(avg_conf - avg_acc)
        ece     += (mask.sum() / n) * gap
        mce      = max(mce, gap)

    return float(ece), float(mce)


def evaluate_anomaly_detection(
    model,
    val_loader,
    device:      torch.device,
    class_names: Optional[List[str]] = None,
    save_dir:    Optional[str]       = None,
    prefix:      str                  = "anomaly",
) -> Dict[str, object]:
    """
    Standalone Objective 3 evaluation: treat model's anomaly_score as a
    binary detector (0=benign, 1=attack) and compute all anomaly metrics.

    Returns compute_anomaly_metrics() results dict + saves CSV.
    """
    model.eval()
    all_scores  = []
    all_labels  = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            out = model(x)
            all_scores.extend(out["anomaly_score"].squeeze().cpu().numpy().flatten())
            all_labels.extend(y.numpy())

    y_true_binary = (np.array(all_labels) > 0).astype(int)   # 0=benign, 1=attack
    anomaly_scores = np.clip(np.array(all_scores), 0.0, 1.0)

    results = compute_anomaly_metrics(y_true_binary, anomaly_scores)

    logger.info(
        f"  Anomaly Detection | AUROC={results.get('anomaly_auroc', float('nan')):.4f} | "
        f"EER={results.get('eer', float('nan'))*100:.2f}% | "
        f"d'={results.get('d_prime', float('nan')):.3f} | "
        f"ECE={results.get('ece', float('nan')):.4f}"
    )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        scalar_keys = ["anomaly_auroc", "eer", "eer_threshold", "d_prime",
                       "ece", "mce", "optimal_threshold", "optimal_f1",
                       "optimal_fpr", "optimal_fnr",
                       "mean_normal_score", "mean_attack_score"]
        scalar_vals = {k: results.get(k, float("nan")) for k in scalar_keys}
        pd.DataFrame([scalar_vals]).to_csv(
            os.path.join(save_dir, f"{prefix}_anomaly_metrics.csv"), index=False
        )
        if "threshold_curve" in results and not results["threshold_curve"].empty:
            results["threshold_curve"].to_csv(
                os.path.join(save_dir, f"{prefix}_threshold_sensitivity.csv"), index=False
            )

    return results


def save_confusion_matrix_data(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    class_names: Optional[List[str]],
    save_dir:    str,
    prefix:      str,
) -> np.ndarray:
    """Save confusion matrix as CSV and return the matrix array."""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(
        cm,
        index=[f"True_{c}" for c in (class_names or range(cm.shape[0]))],
        columns=[f"Pred_{c}" for c in (class_names or range(cm.shape[1]))],
    )
    path = os.path.join(save_dir, f"{prefix}_cm.csv")
    df.to_csv(path)
    return cm


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _batch_predict(
    model,
    X:      torch.Tensor,
    y:      torch.Tensor,
    device: torch.device,
    batch:  int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model over X in batches and return (predictions, labels)."""
    model.eval()
    preds  = []
    labels = y.numpy()
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x_b = X[i : i + batch].to(device)
            out  = model(x_b)
            preds.extend(out["logits"].argmax(dim=1).cpu().numpy())
    return np.array(preds), labels


def _batch_softmax_scores(
    model,
    X:      torch.Tensor,
    device: torch.device,
    batch:  int = 256,
) -> np.ndarray:
    """Collect softmax probability scores [N, n_classes] for AUC computation."""
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x_b = X[i : i + batch].to(device)
            out  = model(x_b)
            prob = F.softmax(out["logits"], dim=1).cpu().numpy()
            scores.append(prob)
    return np.concatenate(scores, axis=0)


def _batch_anomaly_scores(
    model,
    X:      torch.Tensor,
    device: torch.device,
    batch:  int = 256,
) -> np.ndarray:
    """Collect anomaly scores from the model."""
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x_b = X[i : i + batch].to(device)
            out  = model(x_b)
            scores.extend(out["anomaly_score"].squeeze().cpu().numpy().flatten())
    return np.array(scores)
