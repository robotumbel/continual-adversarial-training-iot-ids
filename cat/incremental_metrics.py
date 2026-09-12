"""
incremental_metrics.py — Standard continual learning evaluation metrics for Objective 2.

Metrics follow the benchmark protocol from:
  Lopez-Paz & Ranzato (2017) "Gradient Episodic Memory for Continual Learning" (NeurIPS)
  Riemer et al. (2018) "Learning to Learn without Forgetting" (ICLR)
  Díaz-Rodríguez et al. (2018) "Don't Forget, There Are No Shortcuts" (ICLR)

Notation:
  T = number of tasks (3 for CICIoT → CICIoMT → CICIoV)
  acc[i][j] = accuracy on task i AFTER training on task j  (i ≤ j)
  acc[i][i] = accuracy on task i right after it was trained (reference)

Metrics:
  AA   (Average Accuracy)         — how well model performs across all tasks at the end
  BWT  (Backward Transfer)        — negative = forgetting, positive = positive plasticity
  FWT  (Forward Transfer)         — zero-shot generalisation to future tasks
  RR   (Remembering Rate)         — fraction of initial accuracy retained on old tasks
  CFI  (Catastrophic Forgetting Index) — normalized total forgetting
  Intransigence                   — rigidity of model to learning new tasks
  Plasticity                      — accuracy gained on new task during update

Data structure expected:
  task_matrix: np.ndarray of shape [T, T]
               task_matrix[i, j] = accuracy on task i after training on task j
               (only upper triangle i ≤ j is meaningful)
  random_acc:  np.ndarray of shape [T]
               per-task accuracy of a random classifier (baseline for FWT)
"""

import numpy as np
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Core metric functions (operate on task_matrix)
# ──────────────────────────────────────────────────────────────────────────────

def compute_average_accuracy(task_matrix: np.ndarray) -> float:
    """
    AA = (1/T) Σ_{i=1}^{T} acc[i][T]

    Average accuracy on all tasks AFTER the final task is trained.
    Higher is better.
    """
    T = task_matrix.shape[0]
    return float(np.mean([task_matrix[i, T - 1] for i in range(T)]))


def compute_backward_transfer(task_matrix: np.ndarray) -> float:
    """
    BWT = (1/(T-1)) Σ_{i=1}^{T-1} (acc[i][T] − acc[i][i])

    Measures forgetting of old tasks after new tasks are learned.
    BWT < 0 → catastrophic forgetting
    BWT = 0 → no forgetting
    BWT > 0 → positive backward transfer (new tasks helped old ones)
    """
    T = task_matrix.shape[0]
    if T < 2:
        return 0.0
    bwt = np.mean([
        task_matrix[i, T - 1] - task_matrix[i, i]
        for i in range(T - 1)
    ])
    return float(bwt)


def compute_forward_transfer(
    task_matrix:  np.ndarray,
    random_acc:   np.ndarray,
) -> float:
    """
    FWT = (1/(T-1)) Σ_{i=2}^{T} (acc[i][i-1] − random_acc[i])

    Measures how much prior knowledge (from tasks 1..i-1) helps performance
    on task i BEFORE it is trained.
    acc[i][i-1] = accuracy on task i using model trained only up to task i-1.
    random_acc[i] = accuracy of random classifier on task i.

    FWT > 0 → positive transfer from previous tasks
    FWT < 0 → negative transfer (prior tasks hurt generalisation)
    """
    T = task_matrix.shape[0]
    if T < 2:
        return 0.0
    fwt = np.mean([
        task_matrix[i, i - 1] - random_acc[i]
        for i in range(1, T)
    ])
    return float(fwt)


def compute_remembering_rate(task_matrix: np.ndarray) -> float:
    """
    RR = (1/(T-1)) Σ_{i=1}^{T-1} acc[i][T] / acc[i][i]

    Fraction of original per-task accuracy retained after all updates.
    RR = 1.0 → perfect retention; RR < 1.0 → some forgetting.
    """
    T = task_matrix.shape[0]
    if T < 2:
        return 1.0
    rates = []
    for i in range(T - 1):
        if task_matrix[i, i] > 0:
            rates.append(task_matrix[i, T - 1] / task_matrix[i, i])
        else:
            rates.append(0.0)
    return float(np.mean(rates))


def compute_catastrophic_forgetting_index(task_matrix: np.ndarray) -> float:
    """
    CFI = 1 − (AA_final / mean(acc[i][i]))

    Normalised forgetting index.
    CFI = 0 → no forgetting (AA equals isolated per-task accuracy)
    CFI = 1 → complete forgetting

    Positive CFI = forgetting; negative = improvement (unlikely).
    """
    T         = task_matrix.shape[0]
    aa_final  = compute_average_accuracy(task_matrix)
    aa_oracle = float(np.mean([task_matrix[i, i] for i in range(T)]))
    if aa_oracle == 0:
        return 0.0
    return float(1.0 - aa_final / aa_oracle)


def compute_plasticity(task_matrix: np.ndarray) -> np.ndarray:
    """
    Plasticity[i] = acc[i][i]  (accuracy on new task right after it is trained)

    Returns array of shape [T].
    """
    T = task_matrix.shape[0]
    return np.array([task_matrix[i, i] for i in range(T)])


def compute_intransigence(
    task_matrix:      np.ndarray,
    isolated_acc:     np.ndarray,
) -> float:
    """
    Intransigence = (1/T) Σ_i (isolated_acc[i] − acc[i][i])

    Measures failure to learn task i due to constraints from prior tasks.
    Positive = model is too constrained (can't learn new task well).
    Negative = continual model learns better than isolated (positive transfer).

    isolated_acc[i] = accuracy when training ONLY on task i from scratch.
    """
    T = task_matrix.shape[0]
    intrans = np.mean([
        isolated_acc[i] - task_matrix[i, i]
        for i in range(T)
    ])
    return float(intrans)


def compute_stability(task_matrix: np.ndarray) -> np.ndarray:
    """
    Stability[i] = acc[i][T-1]  (final accuracy on task i after all updates)
    Complementary to plasticity.
    """
    T = task_matrix.shape[0]
    return np.array([task_matrix[i, T - 1] for i in range(T)])


def compute_plasticity_stability_gap(task_matrix: np.ndarray) -> float:
    """
    PS-Gap = mean(Plasticity) − mean(Stability on old tasks)

    Positive = good at learning new tasks but forgets old ones.
    Near 0   = balanced plasticity and stability.
    """
    T    = task_matrix.shape[0]
    plas = float(np.mean([task_matrix[i, i] for i in range(T)]))
    stab = float(np.mean([task_matrix[i, T - 1] for i in range(T - 1)])) \
           if T > 1 else plas
    return plas - stab


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build task_matrix from result dicts
# ──────────────────────────────────────────────────────────────────────────────

def build_task_matrix(
    results_by_phase: dict,
    task_names:       list,
    attack:           str = "Clean",
    metric:           str = "accuracy",
) -> np.ndarray:
    """
    Build the T×T task matrix from results_by_phase dict.

    results_by_phase format (from run_aam_trans.py):
      {
        "after_task_1": {"CICIoT": result_dict, ...},
        "after_task_2": {"CICIoT": result_dict, "CICIoMT": result_dict, ...},
        "after_task_3": {"CICIoT": ..., "CICIoMT": ..., "CICIoV": ...},
      }

    Returns np.ndarray [T, T] where mat[i, j] = acc on task i after task j.
    """
    T      = len(task_names)
    phases = sorted(results_by_phase.keys())   # after_task_1, after_task_2, ...
    mat    = np.zeros((T, T))

    for j_idx, phase in enumerate(phases):
        phase_results = results_by_phase[phase]
        for i_idx, task_name in enumerate(task_names):
            if i_idx > j_idx:
                break   # task i not yet seen
            if task_name in phase_results:
                res = phase_results[task_name]
                # res is a results dict from evaluate_robustness
                if attack in res:
                    mat[i_idx, j_idx] = float(res[attack].get(metric, 0.0))
                elif "Clean" in res:
                    mat[i_idx, j_idx] = float(res["Clean"].get(metric, 0.0))

    return mat


# ──────────────────────────────────────────────────────────────────────────────
# Full incremental evaluation report
# ──────────────────────────────────────────────────────────────────────────────

def compile_incremental_report(
    task_matrix:   np.ndarray,
    task_names:    list,
    save_dir:      str,
    prefix:        str          = "incremental",
    random_acc:    np.ndarray   = None,
    isolated_acc:  np.ndarray   = None,
    mode:          str          = "",
) -> pd.DataFrame:
    """
    Compute all continual learning metrics and save to CSV + LaTeX.

    Args:
        task_matrix:   [T, T] accuracy matrix
        task_names:    list of dataset names
        random_acc:    per-task random classifier accuracy (default: 1/n_classes)
        isolated_acc:  per-task accuracy when trained in isolation
        mode:          'binary' or 'multiclass' (for filename prefix)

    Returns DataFrame with all metrics.
    """
    os.makedirs(save_dir, exist_ok=True)
    T = task_matrix.shape[0]

    if random_acc is None:
        random_acc = np.full(T, 1.0 / T)   # rough default

    # ── Scalar metrics ────────────────────────────────────────────────────────
    aa   = compute_average_accuracy(task_matrix)
    bwt  = compute_backward_transfer(task_matrix)
    fwt  = compute_forward_transfer(task_matrix, random_acc)
    rr   = compute_remembering_rate(task_matrix)
    cfi  = compute_catastrophic_forgetting_index(task_matrix)
    plas = compute_plasticity(task_matrix)
    stab = compute_stability(task_matrix)
    ps_gap = compute_plasticity_stability_gap(task_matrix)

    scalar_rows = [
        {"metric": "Average Accuracy (AA)",        "value": aa,     "unit": "%", "higher_better": True},
        {"metric": "Backward Transfer (BWT)",       "value": bwt,    "unit": "pp", "higher_better": True},
        {"metric": "Forward Transfer (FWT)",        "value": fwt,    "unit": "pp", "higher_better": True},
        {"metric": "Remembering Rate (RR)",         "value": rr,     "unit": "ratio", "higher_better": True},
        {"metric": "Catastrophic Forgetting Index (CFI)", "value": cfi, "unit": "ratio", "higher_better": False},
        {"metric": "Plasticity-Stability Gap",      "value": ps_gap, "unit": "pp", "higher_better": False},
    ]

    if isolated_acc is not None:
        intrans = compute_intransigence(task_matrix, isolated_acc)
        scalar_rows.append(
            {"metric": "Intransigence", "value": intrans, "unit": "pp", "higher_better": False}
        )

    scalar_df = pd.DataFrame(scalar_rows)
    scalar_path = os.path.join(save_dir, f"{prefix}_{mode}_cl_metrics.csv")
    scalar_df.to_csv(scalar_path, index=False)
    logger.info(f"CL metrics saved: {scalar_path}")

    # ── Per-task matrix ───────────────────────────────────────────────────────
    mat_df = pd.DataFrame(
        task_matrix,
        index=[f"Task_{i+1}_{n}" for i, n in enumerate(task_names)],
        columns=[f"After_Task_{j+1}" for j in range(T)],
    )
    mat_path = os.path.join(save_dir, f"{prefix}_{mode}_task_matrix.csv")
    mat_df.to_csv(mat_path)
    logger.info(f"Task matrix saved: {mat_path}")

    # ── Per-task plasticity and stability ─────────────────────────────────────
    task_rows = []
    for i, name in enumerate(task_names):
        task_rows.append({
            "task":       f"Task {i+1} ({name})",
            "plasticity": float(plas[i]),
            "stability":  float(stab[i]),
            "forgetting": float(plas[i] - stab[i]),
            "rr":         float(stab[i] / plas[i]) if plas[i] > 0 else 0.0,
        })
    task_df = pd.DataFrame(task_rows)
    task_path = os.path.join(save_dir, f"{prefix}_{mode}_per_task.csv")
    task_df.to_csv(task_path, index=False)

    # ── LaTeX tables ──────────────────────────────────────────────────────────
    _save_cl_latex(scalar_df, task_df, mat_df,
                   os.path.join(save_dir, f"{prefix}_{mode}_cl_latex.tex"), mode)

    # ── Console summary ───────────────────────────────────────────────────────
    logger.info(f"\n  Incremental Learning Metrics ({mode}):")
    logger.info(f"    AA  = {aa*100:.2f}%  | BWT = {bwt*100:+.2f}pp | FWT = {fwt*100:+.2f}pp")
    logger.info(f"    RR  = {rr*100:.2f}% | CFI = {cfi:.4f}        | PS-Gap = {ps_gap*100:+.2f}pp")

    return scalar_df


# ──────────────────────────────────────────────────────────────────────────────
# LaTeX table helper
# ──────────────────────────────────────────────────────────────────────────────

def _save_cl_latex(
    scalar_df: pd.DataFrame,
    task_df:   pd.DataFrame,
    mat_df:    pd.DataFrame,
    path:      str,
    mode:      str,
) -> None:
    lines = [
        r"\section*{Incremental Learning Evaluation — " + mode.title() + " Mode}",
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Continual Learning Metrics (AAM-TRANS)}",
        r"\label{tab:cl_metrics}",
        r"\begin{tabular}{lrll}",
        r"\toprule",
        r"Metric & Value & Unit & Higher Better \\",
        r"\midrule",
    ]
    for _, row in scalar_df.iterrows():
        val     = row["value"]
        unit    = row["unit"]
        # Convert to percentage if unit is fraction-like
        if unit in ("%", "pp"):
            val_str = f"{val*100:.2f}\\%"
        else:
            val_str = f"{val:.4f}"
        hb  = r"\checkmark" if row["higher_better"] else "—"
        lines.append(f"{row['metric']} & {val_str} & {unit} & {hb} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # Task matrix table
    T   = len(mat_df)
    col_spec = "l" + "r" * T
    lines += [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Task Accuracy Matrix $R_{i,j}$ — Accuracy on Task $i$ After Task $j$}",
        r"\label{tab:task_matrix}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        "Task / After Task & " + " & ".join([f"Task {j+1}" for j in range(T)]) + r" \\",
        r"\midrule",
    ]
    for i, (idx, row) in enumerate(mat_df.iterrows()):
        vals = []
        for j in range(T):
            v = row[f"After_Task_{j+1}"]
            if j < i:
                vals.append("—")
            else:
                bold = r"\textbf" if j == i else ""
                vals.append(f"{bold}{{{v*100:.2f}}}")
        lines.append(f"Task {i+1} & " + " & ".join(vals) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"CL LaTeX tables saved: {path}")
