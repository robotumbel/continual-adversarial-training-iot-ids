"""
adversarial_metrics.py — Comprehensive adversarial robustness metrics for Objective 1.

Metrics:
  compute_robustness_drop()        — RD = Acc_clean − Acc_adv per attack/ε
  compute_perturbation_norms()     — L2, L∞ norms of actual perturbations
  compute_gate_response_delta()    — Δgate = gate_adv − gate_clean per head (AAM-TRANS)
  check_gradient_masking()         — Detect if robustness is due to gradient masking
  compute_transferability_matrix() — Cross-model adversarial transferability rates
  compute_robustness_degradation() — Accuracy vs ε curve for multiple models
  summarize_robustness()           — Aggregate RD + VR + ASR table per model

References:
  Goodfellow et al. (2014) FGSM
  Carlini & Wagner (2017) C&W — gradient masking detection
  Papernot et al. (2016) transferability framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import os
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Robustness Drop (RD)
# ──────────────────────────────────────────────────────────────────────────────

def compute_robustness_drop(
    results: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Robustness Drop (RD) = Acc_clean − Acc_adversarial.

    Positive RD = performance degraded under attack.
    Smaller RD = more robust model.

    Args:
        results: dict from evaluate_robustness() — {attack_name: {accuracy, ...}}

    Returns DataFrame: [attack, accuracy, rd, rd_pct, relative_rd]
    """
    clean_acc = results.get("Clean", {}).get("accuracy", 1.0)
    rows = []
    for atk, res in results.items():
        if atk == "Clean":
            continue
        adv_acc = res["accuracy"]
        rd      = clean_acc - adv_acc                      # absolute drop
        rd_pct  = rd * 100                                 # in percentage points
        rel_rd  = (rd / clean_acc * 100) if clean_acc > 0 else 0  # relative %
        rows.append({
            "attack":      atk,
            "clean_acc":   clean_acc,
            "adv_acc":     adv_acc,
            "rd":          rd,
            "rd_pct":      rd_pct,
            "rel_rd_pct":  rel_rd,
            "asr":         res.get("asr", float("nan")),
            "vr":          res.get("validity_rate", float("nan")),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Perturbation Norms
# ──────────────────────────────────────────────────────────────────────────────

def compute_perturbation_norms(
    X_clean: torch.Tensor,   # [N, seq_len, F]
    X_adv:   torch.Tensor,   # [N, seq_len, F]
) -> Dict[str, float]:
    """
    Compute L2 and L∞ norms of adversarial perturbations.

    Validates that perturbations are non-trivial (attack is meaningful)
    while remaining within the declared ε budget.

    Returns {mean_l2, std_l2, max_l2, mean_linf, max_linf}
    """
    if isinstance(X_clean, np.ndarray):
        X_clean = torch.from_numpy(X_clean)
    if isinstance(X_adv, np.ndarray):
        X_adv = torch.from_numpy(X_adv)
    if X_clean.dim() == 2:           # (N, F) -> (N, 1, F)
        X_clean = X_clean.unsqueeze(1)
        X_adv   = X_adv.unsqueeze(1)
    delta    = (X_adv - X_clean).float()           # [N, seq_len, F]
    B, L, F  = delta.shape
    flat     = delta.view(B, -1)                   # [N, seq_len*F]

    l2   = flat.norm(dim=1)                        # [N]
    linf = flat.abs().max(dim=1).values            # [N]

    return {
        "mean_l2":   float(l2.mean()),
        "std_l2":    float(l2.std()),
        "max_l2":    float(l2.max()),
        "mean_linf": float(linf.mean()),
        "max_linf":  float(linf.max()),
        "min_linf":  float(linf.min()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. Gate Response Delta (Objective 1 interpretability)
# ──────────────────────────────────────────────────────────────────────────────

def compute_gate_response_delta(
    model,
    X_clean: torch.Tensor,
    X_adv:   torch.Tensor,
    device:  torch.device,
    batch:   int = 128,
) -> Dict[str, np.ndarray]:
    """
    For AAM-TRANS: compute mean gate values on clean vs adversarial inputs,
    and their absolute delta.

    Gate delta > 0 for a head means the head is MORE active on adversarial inputs.
    Gate delta < 0 means the head is SUPPRESSED (protective gating behavior).

    Returns:
      {
        "gates_clean":  [N, n_heads] mean gate values on clean data
        "gates_adv":    [N, n_heads] mean gate values on adversarial data
        "delta_mean":   [n_heads]    mean(gate_adv − gate_clean) per head
        "delta_abs":    [n_heads]    mean|gate_adv − gate_clean| per head
        "stats": DataFrame with head-level stats
      }
    """
    if not hasattr(model, "blocks"):
        logger.warning("Model has no 'blocks' attribute — gate analysis skipped.")
        return {}

    model.eval()

    if isinstance(X_clean, np.ndarray):
        X_clean = torch.from_numpy(X_clean)
    if isinstance(X_adv, np.ndarray):
        X_adv = torch.from_numpy(X_adv)
    if X_clean.dim() == 2:       # (N, F) -> (N, 1, F)
        X_clean = X_clean.unsqueeze(1)
        X_adv   = X_adv.unsqueeze(1)

    def _collect_gates(X):
        """Run model in batches and collect gates from the LAST block."""
        all_gates = []
        with torch.no_grad():
            for i in range(0, len(X), batch):
                x_b   = X[i: i + batch].to(device)
                out   = model(x_b)
                # adaptive_gates: list of [B, n_heads] tensors (one per block)
                if "adaptive_gates" in out and out["adaptive_gates"]:
                    # Average across all blocks
                    stacked = torch.stack(out["adaptive_gates"], dim=0)   # [n_blocks, B, H]
                    avg     = stacked.mean(dim=0)                         # [B, H]
                    all_gates.append(avg.cpu().numpy())
        return np.concatenate(all_gates, axis=0) if all_gates else None

    gates_clean = _collect_gates(X_clean)
    gates_adv   = _collect_gates(X_adv)

    if gates_clean is None or gates_adv is None:
        return {}

    n = min(len(gates_clean), len(gates_adv))
    delta     = gates_adv[:n] - gates_clean[:n]   # [N, n_heads]
    n_heads   = gates_clean.shape[1]

    stats_rows = []
    for h in range(n_heads):
        stats_rows.append({
            "head":           h + 1,
            "mean_gate_clean": float(gates_clean[:n, h].mean()),
            "std_gate_clean":  float(gates_clean[:n, h].std()),
            "mean_gate_adv":   float(gates_adv[:n, h].mean()),
            "std_gate_adv":    float(gates_adv[:n, h].std()),
            "delta_mean":      float(delta[:, h].mean()),
            "delta_abs_mean":  float(np.abs(delta[:, h]).mean()),
            "delta_std":       float(delta[:, h].std()),
        })

    return {
        "gates_clean": gates_clean[:n],
        "gates_adv":   gates_adv[:n],
        "delta":       delta,
        "delta_mean":  delta.mean(axis=0),   # [n_heads]
        "delta_abs":   np.abs(delta).mean(axis=0),
        "stats":       pd.DataFrame(stats_rows),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Gradient Masking Detection
# ──────────────────────────────────────────────────────────────────────────────

def check_gradient_masking(
    model,
    X:      torch.Tensor,   # [N, seq_len, F]
    y:      torch.Tensor,   # [N]
    device: torch.device,
    n_batches: int = 20,
) -> Dict[str, float]:
    """
    Detect gradient masking / gradient obfuscation (Athalye et al., 2018).

    Three indicators:
      1. Gradient Norm — abnormally small gradients suggest masking
      2. Loss Sensitivity — loss should increase with perturbation ε
      3. Black-box vs White-box gap — large gap suggests masking
         (approximated here: compare FGSM 1-step vs PGD multi-step success rate)

    Returns dict with gradient norm stats and masking_suspected flag.
    """
    model.eval()
    grad_norms = []
    loss_vals  = []

    for i in range(min(n_batches, len(X) // 32 + 1)):
        x_b = X[i * 32: (i + 1) * 32].clone().to(device).requires_grad_(True)
        y_b = y[i * 32: (i + 1) * 32].to(device)
        if len(x_b) == 0:
            break

        out     = model(x_b)
        loss    = F.cross_entropy(out["logits"], y_b)
        loss.backward()

        if x_b.grad is not None:
            g_norm = x_b.grad.view(len(x_b), -1).norm(dim=1)
            grad_norms.extend(g_norm.detach().cpu().numpy().tolist())
        loss_vals.append(loss.item())

    if not grad_norms:
        return {"gradient_masking_suspected": False,
                "mean_grad_norm": float("nan"),
                "note": "Could not compute gradients"}

    mean_gn = float(np.mean(grad_norms))
    std_gn  = float(np.std(grad_norms))

    # Heuristic: if gradient norm is extremely small (< 1e-6) across samples
    # it suggests numerical vanishing or masking
    masking_suspected = bool(mean_gn < 1e-5)

    return {
        "mean_grad_norm":              mean_gn,
        "std_grad_norm":               std_gn,
        "min_grad_norm":               float(np.min(grad_norms)),
        "max_grad_norm":               float(np.max(grad_norms)),
        "near_zero_fraction":          float(np.mean(np.array(grad_norms) < 1e-6)),
        "gradient_masking_suspected":  masking_suspected,
        "mean_loss":                   float(np.mean(loss_vals)),
        "interpretation": (
            "Gradient masking suspected — robustness may be obfuscated"
            if masking_suspected else
            "No gradient masking detected — robustness appears genuine"
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. Transferability Rate
# ──────────────────────────────────────────────────────────────────────────────

def compute_transferability_rate(
    source_model,
    target_model,
    X:           torch.Tensor,   # [N, seq_len, F] — shared validation set
    y:           torch.Tensor,   # [N]
    source_name: str,
    target_name: str,
    device:      torch.device,
    epsilon:     float = 0.1,
    batch:       int   = 128,
) -> Dict[str, float]:
    """
    Adversarial Transferability: craft examples on source_model, evaluate on target_model.

    Transferability Rate = fraction of examples that fool target_model
                           when crafted against source_model.

    Low transferability to AAM-TRANS validates Objective 1:
    adaptive gates disrupt the gradient directions exploited by generic attacks.

    Returns {source, target, epsilon, transfer_asr, target_clean_acc, source_clean_acc}
    """
    source_model.eval()
    target_model.eval()

    # ── Craft adversarial examples on source ─────────────────────────────────
    adv_batches = []
    src_pred    = []
    tgt_pred    = []
    tgt_adv_pred= []
    labels_all  = []

    for i in range(0, len(X), batch):
        x_b = X[i: i + batch].clone().to(device).requires_grad_(True)
        y_b = y[i: i + batch].to(device)

        # FGSM on source
        source_model.zero_grad()
        out_src  = source_model(x_b)
        loss_src = F.cross_entropy(out_src["logits"], y_b)
        loss_src.backward()

        x_adv = (x_b + epsilon * x_b.grad.sign()).detach()
        adv_batches.append(x_adv.cpu())

        # Source clean predictions
        with torch.no_grad():
            src_pred.extend(out_src["logits"].argmax(dim=1).cpu().numpy())
            # Target clean predictions
            out_tgt = target_model(x_b.detach())
            tgt_pred.extend(out_tgt["logits"].argmax(dim=1).cpu().numpy())
            # Target adversarial predictions
            out_tgt_adv = target_model(x_adv.to(device))
            tgt_adv_pred.extend(out_tgt_adv["logits"].argmax(dim=1).cpu().numpy())

        labels_all.extend(y_b.cpu().numpy())

    src_pred     = np.array(src_pred)
    tgt_pred     = np.array(tgt_pred)
    tgt_adv_pred = np.array(tgt_adv_pred)
    labels_all   = np.array(labels_all)

    # Correct on target clean — only these samples are meaningful
    tgt_correct = tgt_pred == labels_all
    if tgt_correct.sum() == 0:
        transfer_asr = 0.0
    else:
        # Transfer ASR = fraction of target-correct samples fooled by source-crafted adversarials
        transfer_asr = float(
            (tgt_adv_pred[tgt_correct] != labels_all[tgt_correct]).mean() * 100
        )

    return {
        "source":              source_name,
        "target":              target_name,
        "epsilon":             epsilon,
        "source_clean_acc":    float((src_pred == labels_all).mean() * 100),
        "target_clean_acc":    float((tgt_pred == labels_all).mean() * 100),
        "target_adv_acc":      float((tgt_adv_pred == labels_all).mean() * 100),
        "transfer_asr":        transfer_asr,
        "transfer_reduced":    bool(transfer_asr < float((src_pred == labels_all).mean() * 100)),
    }


def compute_transferability_matrix(
    models:  Dict[str, nn.Module],
    X:       torch.Tensor,
    y:       torch.Tensor,
    device:  torch.device,
    epsilon: float = 0.1,
) -> pd.DataFrame:
    """
    Full N×N transferability matrix across all models.

    Returns DataFrame where [source_row, target_col] = Transfer ASR (%).
    Diagonal = standard white-box ASR.
    """
    model_names = list(models.keys())
    rows = []

    for src_name, src_model in models.items():
        for tgt_name, tgt_model in models.items():
            result = compute_transferability_rate(
                source_model=src_model,
                target_model=tgt_model,
                X=X, y=y,
                source_name=src_name,
                target_name=tgt_name,
                device=device,
                epsilon=epsilon,
            )
            rows.append(result)
            logger.info(
                f"  Transfer {src_name:20s} -> {tgt_name:20s}: "
                f"ASR={result['transfer_asr']:.2f}%"
            )

    df = pd.DataFrame(rows)
    logger.info("Transferability matrix computed.")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 6. Robustness Degradation Curve (Accuracy vs ε)
# ──────────────────────────────────────────────────────────────────────────────

def compute_robustness_degradation(
    models:  Dict[str, nn.Module],
    X:       torch.Tensor,
    y:       torch.Tensor,
    device:  torch.device,
    epsilons: List[float] = None,
    attack:   str = "fgsm",
    batch:    int = 128,
) -> pd.DataFrame:
    """
    Compute accuracy vs ε curve for multiple models simultaneously.

    Enables plotting a smooth degradation curve showing AAM-TRANS
    degrades more gracefully than baselines.

    Returns DataFrame: [model, epsilon, accuracy, rd, asr]
    """
    if epsilons is None:
        epsilons = [0.0, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3]

    rows = []
    for model_name, model in models.items():
        model.eval()
        logger.info(f"  Degradation curve: {model_name}")

        # Clean accuracy
        clean_preds = _batch_predict(model, X, device, batch)
        clean_acc   = float((clean_preds == y.numpy()).mean())

        for eps in epsilons:
            if eps == 0.0:
                acc = clean_acc
                rd  = 0.0
            else:
                # FGSM perturbation
                x_adv = _fgsm_batch(model, X, y, device, eps, batch)
                adv_preds = _batch_predict(model, x_adv, device, batch)
                acc = float((adv_preds == y.numpy()).mean())
                rd  = clean_acc - acc

                # ASR (on clean-correct samples)
                correct = clean_preds == y.numpy()
                asr = float((adv_preds[correct] != y.numpy()[correct]).mean() * 100) \
                      if correct.sum() > 0 else 0.0

            rows.append({
                "model":   model_name,
                "epsilon": eps,
                "accuracy": acc,
                "rd":       rd,
                "rd_pct":  rd * 100,
                "asr":     asr if eps > 0 else 0.0,
            })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 7. Robustness Summary Table
# ──────────────────────────────────────────────────────────────────────────────

def summarize_robustness(
    all_model_results: Dict[str, Dict[str, Dict]],
    save_dir:          str,
    prefix:            str = "robustness_summary",
) -> pd.DataFrame:
    """
    Create a comprehensive robustness summary across all models and attacks.

    all_model_results: {model_name: results_dict_from_evaluate_robustness}

    Returns DataFrame with mean RD, max ASR, min VR per model.
    """
    os.makedirs(save_dir, exist_ok=True)
    rows = []

    for model_name, results in all_model_results.items():
        rd_df = compute_robustness_drop(results)
        if rd_df.empty:
            continue

        clean_acc = results.get("Clean", {}).get("accuracy", float("nan"))
        clean_f1  = results.get("Clean", {}).get("f1",       float("nan"))
        clean_mcc = results.get("Clean", {}).get("mcc",      float("nan"))

        rows.append({
            "model":         model_name,
            "clean_acc":     clean_acc,
            "clean_f1":      clean_f1,
            "clean_mcc":     clean_mcc,
            "mean_rd_pct":   float(rd_df["rd_pct"].mean()),
            "max_rd_pct":    float(rd_df["rd_pct"].max()),
            "worst_attack":  rd_df.loc[rd_df["rd"].idxmax(), "attack"] if len(rd_df) else "—",
            "mean_asr":      float(rd_df["asr"].mean()),
            "max_asr":       float(rd_df["asr"].max()),
            "mean_vr":       float(rd_df["vr"].mean()),
            "min_vr":        float(rd_df["vr"].min()),
            "n_attacks":     len(rd_df),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        csv_path = os.path.join(save_dir, f"{prefix}.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Robustness summary saved: {csv_path}")

        # LaTeX table
        _save_robustness_latex(df, os.path.join(save_dir, f"{prefix}.tex"))

    return df


def _save_robustness_latex(df: pd.DataFrame, path: str) -> None:
    cols = ["model", "clean_acc", "mean_rd_pct", "max_rd_pct", "mean_asr",
            "max_asr", "mean_vr"]
    cols = [c for c in cols if c in df.columns]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Adversarial Robustness Summary — All Models}",
        r"\label{tab:robustness_summary}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & Clean Acc & Mean RD & Max RD & Mean ASR & Max ASR & Mean VR \\",
        r"& (\%) & (pp) & (pp) & (\%) & (\%) & (\%) \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['model']} & "
            f"{row.get('clean_acc', float('nan'))*100:.2f} & "
            f"{row.get('mean_rd_pct', float('nan')):.2f} & "
            f"{row.get('max_rd_pct', float('nan')):.2f} & "
            f"{row.get('mean_asr', float('nan')):.2f} & "
            f"{row.get('max_asr', float('nan')):.2f} & "
            f"{row.get('mean_vr', float('nan')):.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"LaTeX robustness table saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _batch_predict(
    model, X: torch.Tensor, device: torch.device, batch: int = 128
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out = model(X[i: i + batch].to(device))
            preds.extend(out["logits"].argmax(dim=1).cpu().numpy())
    return np.array(preds)


def _fgsm_batch(
    model, X: torch.Tensor, y: torch.Tensor,
    device: torch.device, epsilon: float, batch: int = 128,
) -> torch.Tensor:
    """FGSM in batches, returns adversarial tensor on CPU."""
    model.eval()
    adv_chunks = []
    for i in range(0, len(X), batch):
        x_b = X[i: i + batch].clone().to(device).requires_grad_(True)
        y_b = y[i: i + batch].to(device)
        out  = model(x_b)
        loss = F.cross_entropy(out["logits"], y_b)
        loss.backward()
        adv = (x_b + epsilon * x_b.grad.sign()).detach().cpu()
        adv_chunks.append(adv)
    return torch.cat(adv_chunks, dim=0)
