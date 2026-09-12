"""
statistical_eval.py — Statistical significance testing for Q1 journal evaluation.

Tests implemented (following NeurIPS/IEEE TNNLS conventions):
  bootstrap_ci()         — 95% bootstrap confidence interval for any metric
  wilcoxon_signed_rank() — pairwise non-parametric test (paired scores over attacks)
  mcnemar_test()         — compare classification errors of two models (sample-level)
  friedman_nemenyi()     — multiple classifier comparison + Nemenyi post-hoc CD diagram
  cohens_d()             — effect size for pairwise comparison

Output helpers:
  run_statistical_evaluation() — full pipeline; saves CSVs + LaTeX tables
  _save_latex_comparison_table()
  _save_latex_ci_table()
  _save_cd_diagram()           — Critical Difference diagram (Demsar, JMLR 2006)
"""

import os
import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ALPHA = 0.05   # significance level throughout


# ──────────────────────────────────────────────────────────────────────────────
# 1. Bootstrap Confidence Interval
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    metric_fn:   Callable,
    n_bootstrap: int   = 1000,
    ci:          float = 0.95,
    seed:        int   = 42,
) -> Tuple[float, float, float]:
    """
    Non-parametric bootstrap confidence interval.

    Returns (point_estimate, lower_bound, upper_bound).
    """
    rng    = np.random.default_rng(seed)
    n      = len(y_true)
    point  = metric_fn(y_true, y_pred)

    boot_scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            s = metric_fn(y_true[idx], y_pred[idx])
            boot_scores.append(s)
        except Exception:
            continue

    if not boot_scores:
        return point, float("nan"), float("nan")

    alpha  = 1.0 - ci
    lower  = float(np.percentile(boot_scores, 100 * alpha / 2))
    upper  = float(np.percentile(boot_scores, 100 * (1 - alpha / 2)))
    return float(point), lower, upper


# ──────────────────────────────────────────────────────────────────────────────
# 2. Wilcoxon Signed-Rank Test
# ──────────────────────────────────────────────────────────────────────────────

def wilcoxon_signed_rank(
    scores_a:    np.ndarray,
    scores_b:    np.ndarray,
    alternative: str = "two-sided",
) -> Dict:
    """
    Wilcoxon signed-rank test for paired observations (e.g., accuracy over N attacks).

    Preferred over paired t-test because metric distributions are often non-normal.

    Returns dict: {statistic, p_value, significant (p < 0.05), effect_size_r}
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    if len(scores_a) < 5:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "significant": False, "effect_size_r": float("nan")}

    try:
        stat, p = stats.wilcoxon(scores_a, scores_b, alternative=alternative,
                                 zero_method="wilcox")
        # Effect size r = Z / sqrt(N)
        n  = len(scores_a)
        z  = stats.norm.ppf(1 - p / 2) if p < 1.0 else 0.0
        r  = abs(z) / np.sqrt(n)
        return {
            "statistic":     float(stat),
            "p_value":       float(p),
            "significant":   bool(p < ALPHA),
            "effect_size_r": float(r),
        }
    except ValueError as e:
        logger.debug(f"Wilcoxon skipped: {e}")
        return {"statistic": float("nan"), "p_value": float("nan"),
                "significant": False, "effect_size_r": float("nan")}


# ──────────────────────────────────────────────────────────────────────────────
# 3. McNemar's Test
# ──────────────────────────────────────────────────────────────────────────────

def mcnemar_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> Dict:
    """
    McNemar's test for comparing two classifiers at the sample level.

    Contingency table (with continuity correction):
      b = samples correct by B but wrong by A
      c = samples correct by A but wrong by B
      χ² = (|b - c| - 1)² / (b + c)

    Returns dict: {statistic, p_value, significant, b, c, odds_ratio}
    """
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    b = int((~correct_a & correct_b).sum())   # B right, A wrong
    c = int((correct_a & ~correct_b).sum())   # A right, B wrong

    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False,
                "b": b, "c": c, "odds_ratio": 1.0}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p    = 1.0 - stats.chi2.cdf(chi2, df=1)
    odds = (b / c) if c > 0 else float("inf")

    return {
        "statistic":   float(chi2),
        "p_value":     float(p),
        "significant": bool(p < ALPHA),
        "b":           b,
        "c":           c,
        "odds_ratio":  float(odds),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Friedman + Nemenyi Post-Hoc
# ──────────────────────────────────────────────────────────────────────────────

def friedman_nemenyi(
    model_scores: Dict[str, np.ndarray],
    alpha:        float = ALPHA,
) -> Dict:
    """
    Friedman test across k classifiers over N datasets/conditions,
    followed by pairwise Nemenyi post-hoc if Friedman is significant.

    model_scores: {model_name: np.array of shape [N]} (one score per condition)

    Returns:
      friedman_stat, p_value, significant,
      mean_ranks,   cd (critical difference),
      nemenyi:  {f"{A}_vs_{B}": {rank_diff, cd, significant}}
    """
    names  = list(model_scores.keys())
    arrays = [np.asarray(model_scores[n], dtype=float) for n in names]
    k      = len(names)
    n_obs  = len(arrays[0])

    if k < 3 or n_obs < 3 or any(len(a) != n_obs for a in arrays):
        return {
            "friedman_stat": float("nan"),
            "p_value":       float("nan"),
            "significant":   False,
            "mean_ranks":    {},
            "cd":            float("nan"),
            "nemenyi":       {},
        }

    stat, p = stats.friedmanchisquare(*arrays)

    # Rank within each observation (row = observation, col = model)
    data  = np.column_stack(arrays)           # [N, k]
    # Rank each row independently (higher score → lower rank: best=1)
    ranks = np.apply_along_axis(
        lambda row: stats.rankdata(-row), axis=1, arr=data
    )                                          # [N, k]  (1 = best)
    mean_ranks = {n: float(ranks[:, i].mean()) for i, n in enumerate(names)}

    # Nemenyi critical difference (two-tailed, q_alpha from Studentized range)
    # q_alpha table (α=0.05): k=2→2.772, 3→3.314, 4→3.633, 5→3.858, 6→4.030
    q_table = {2: 2.772, 3: 3.314, 4: 3.633, 5: 3.858, 6: 4.030, 7: 4.170,
               8: 4.286, 9: 4.387, 10: 4.477}
    q_alpha = q_table.get(k, 1.96 * np.sqrt(2))   # fallback to normal approx
    cd      = q_alpha * np.sqrt(k * (k + 1) / (6 * n_obs))

    nemenyi = {}
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(mean_ranks[names[i]] - mean_ranks[names[j]])
            nemenyi[f"{names[i]}_vs_{names[j]}"] = {
                "rank_diff":  float(diff),
                "cd":         float(cd),
                "significant": bool(diff > cd),
                "better":     names[i] if mean_ranks[names[i]] < mean_ranks[names[j]]
                              else names[j],
            }

    return {
        "friedman_stat": float(stat),
        "p_value":       float(p),
        "significant":   bool(p < alpha),
        "mean_ranks":    mean_ranks,
        "cd":            float(cd),
        "nemenyi":       nemenyi,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. Cohen's d Effect Size
# ──────────────────────────────────────────────────────────────────────────────

def cohens_d(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> float:
    """
    Cohen's d effect size (positive = A > B).
    Interpretation: small=0.2, medium=0.5, large=0.8.
    """
    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)
    n1, n2  = len(group_a), len(group_b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled_s = np.sqrt(
        ((n1 - 1) * group_a.std(ddof=1) ** 2 + (n2 - 1) * group_b.std(ddof=1) ** 2)
        / (n1 + n2 - 2)
    )
    return float((group_a.mean() - group_b.mean()) / (pooled_s + 1e-12))


# ──────────────────────────────────────────────────────────────────────────────
# 6. t-distribution CI (for aggregate metric tables)
# ──────────────────────────────────────────────────────────────────────────────

def t_ci(values: np.ndarray, ci: float = 0.95) -> Tuple[float, float, float]:
    """
    Confidence interval via Student's t (mean ± t * SEM).
    Returns (mean, lower, upper).
    """
    values = np.asarray(values, dtype=float)
    n      = len(values)
    if n < 2:
        return float(values.mean()), float("nan"), float("nan")
    mean = float(values.mean())
    sem  = values.std(ddof=1) / np.sqrt(n)
    h    = sem * stats.t.ppf((1 + ci) / 2, df=n - 1)
    return mean, mean - h, mean + h


# ──────────────────────────────────────────────────────────────────────────────
# 7. Full statistical evaluation pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_statistical_evaluation(
    results_df:  pd.DataFrame,
    save_dir:    str,
    aam_model:   str   = "AAM-TRANS",
    metric:      str   = "accuracy",
    n_bootstrap: int   = 1000,
) -> Dict[str, pd.DataFrame]:
    """
    Full statistical evaluation suite for Q1 journal submission.

    Produces:
      1. pairwise_comparison.csv      — Δmetric and % improvement for each attack
      2. wilcoxon_results.csv         — Wilcoxon tests: AAM-TRANS vs each baseline
      3. friedman_nemenyi.csv         — Friedman + Nemenyi across all models
      4. confidence_intervals.csv     — Mean ± 95% CI per model (t-distribution)
      5. latex_comparison_{metric}.tex — LaTeX table for paper
      6. latex_ci_{metric}.tex        — LaTeX CI table
      7. latex_nemenyi.tex            — Nemenyi pairwise significance table

    Returns dict of DataFrames: {name: df}
    """
    os.makedirs(save_dir, exist_ok=True)
    outputs = {}

    # ── 1. Pairwise comparison table ─────────────────────────────────────────
    pairwise_rows = []
    for (ds, mode, attack), grp in results_df.groupby(["dataset", "mode", "attack"]):
        if aam_model not in grp["model"].values:
            continue
        aam_val = float(grp.loc[grp["model"] == aam_model, metric].iloc[0])
        for _, row in grp[grp["model"] != aam_model].iterrows():
            bsl_val = float(row[metric])
            pairwise_rows.append({
                "dataset":        ds,
                "mode":           mode,
                "attack":         attack,
                "proposed":       aam_model,
                "baseline":       row["model"],
                "proposed_score": aam_val,
                "baseline_score": bsl_val,
                "delta":          aam_val - bsl_val,
                "pct_improvement": (aam_val - bsl_val) / (bsl_val + 1e-12) * 100,
                "proposed_wins":  int(aam_val > bsl_val),
            })

    pairwise_df = pd.DataFrame(pairwise_rows)
    if not pairwise_df.empty:
        pairwise_df.to_csv(os.path.join(save_dir, "pairwise_comparison.csv"), index=False)
        outputs["pairwise"] = pairwise_df
        logger.info("Pairwise comparison saved.")

    # ── 2. Wilcoxon tests (scores over all attacks per dataset × mode) ────────
    wilcoxon_rows = []
    for (ds, mode), grp in results_df.groupby(["dataset", "mode"]):
        models = grp["model"].unique().tolist()
        if aam_model not in models:
            continue

        aam_scores = grp.loc[grp["model"] == aam_model].sort_values("attack")[metric].values
        for bsl in [m for m in models if m != aam_model]:
            bsl_scores = grp.loc[grp["model"] == bsl].sort_values("attack")[metric].values
            min_len    = min(len(aam_scores), len(bsl_scores))
            w_result   = wilcoxon_signed_rank(
                aam_scores[:min_len], bsl_scores[:min_len]
            )
            d = cohens_d(aam_scores[:min_len], bsl_scores[:min_len])
            d_label = ("small" if abs(d) < 0.5
                       else "medium" if abs(d) < 0.8 else "large")
            wilcoxon_rows.append({
                "dataset":       ds,
                "mode":          mode,
                "model_a":       aam_model,
                "model_b":       bsl,
                "metric":        metric,
                "mean_a":        aam_scores[:min_len].mean(),
                "mean_b":        bsl_scores[:min_len].mean(),
                "delta_mean":    aam_scores[:min_len].mean() - bsl_scores[:min_len].mean(),
                "W_statistic":   w_result["statistic"],
                "p_value":       w_result["p_value"],
                "significant":   w_result["significant"],
                "effect_r":      w_result["effect_size_r"],
                "cohens_d":      d,
                "effect_label":  d_label,
            })

    if wilcoxon_rows:
        wil_df = pd.DataFrame(wilcoxon_rows)
        wil_df.to_csv(os.path.join(save_dir, "wilcoxon_results.csv"), index=False)
        outputs["wilcoxon"] = wil_df
        logger.info("Wilcoxon test results saved.")

    # ── 3. Friedman + Nemenyi (one test per dataset × mode) ──────────────────
    friedman_rows  = []
    nemenyi_rows   = []
    for (ds, mode), grp in results_df.groupby(["dataset", "mode"]):
        model_scores = {}
        for model in grp["model"].unique():
            sub = grp[grp["model"] == model].sort_values("attack")
            model_scores[model] = sub[metric].values

        fr = friedman_nemenyi(model_scores)
        friedman_rows.append({
            "dataset":       ds,
            "mode":          mode,
            "chi2_stat":     fr["friedman_stat"],
            "p_value":       fr["p_value"],
            "significant":   fr["significant"],
            "critical_diff": fr["cd"],
            **{f"rank_{m}": r for m, r in fr.get("mean_ranks", {}).items()},
        })

        for pair, res in fr.get("nemenyi", {}).items():
            nemenyi_rows.append({
                "dataset":    ds,
                "mode":       mode,
                "comparison": pair,
                **res,
            })

    if friedman_rows:
        fr_df = pd.DataFrame(friedman_rows)
        fr_df.to_csv(os.path.join(save_dir, "friedman_test.csv"), index=False)
        outputs["friedman"] = fr_df
        logger.info("Friedman test results saved.")

    if nemenyi_rows:
        nem_df = pd.DataFrame(nemenyi_rows)
        nem_df.to_csv(os.path.join(save_dir, "nemenyi_posthoc.csv"), index=False)
        outputs["nemenyi"] = nem_df

    # ── 4. Confidence intervals (t-distribution, per model × dataset × mode) ─
    ci_rows = []
    for (ds, mode, model), grp in results_df.groupby(["dataset", "mode", "model"]):
        vals        = grp[metric].values
        mean, lo, hi = t_ci(vals)
        ci_rows.append({
            "dataset":   ds,
            "mode":      mode,
            "model":     model,
            "metric":    metric,
            "n":         len(vals),
            "mean":      mean,
            "std":       float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
            "ci_lower":  lo,
            "ci_upper":  hi,
            "ci_width":  hi - lo if not np.isnan(hi) else float("nan"),
        })

    if ci_rows:
        ci_df = pd.DataFrame(ci_rows)
        ci_df.to_csv(os.path.join(save_dir, "confidence_intervals.csv"), index=False)
        outputs["ci"] = ci_df
        logger.info("Confidence intervals saved.")

    # ── 5. LaTeX tables ───────────────────────────────────────────────────────
    if not pairwise_df.empty:
        _save_latex_comparison_table(pairwise_df, save_dir, metric, aam_model)
    if ci_rows:
        _save_latex_ci_table(ci_df, save_dir, metric)
    if nemenyi_rows:
        _save_latex_nemenyi_table(pd.DataFrame(nemenyi_rows), save_dir)

    logger.info(f"Statistical evaluation complete. Results in: {save_dir}")
    return outputs


# ──────────────────────────────────────────────────────────────────────────────
# LaTeX table helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save_latex_comparison_table(
    df:        pd.DataFrame,
    save_dir:  str,
    metric:    str,
    aam_model: str,
) -> None:
    """LaTeX table: proposed vs baseline Δmetric with significance markers."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Performance Comparison: {aam_model} vs Baselines "
        rf"({metric.replace('_', ' ').title()})}}",
        rf"\label{{tab:comparison_{metric}}}",
        r"\small",
        r"\begin{tabular}{llllrrr}",
        r"\toprule",
        r"Dataset & Mode & Attack & Baseline & "
        rf"$\Delta${metric.replace('_', ' ').title()} & Improv. (\%) \\",
        r"\midrule",
    ]

    prev_ds = None
    for _, row in df.iterrows():
        ds_str = row["dataset"] if row["dataset"] != prev_ds else r"\quad"
        prev_ds = row["dataset"]
        sign    = "+" if row["delta"] >= 0 else ""
        bold    = r"\textbf" if row["proposed_wins"] else ""
        lines.append(
            f"{ds_str} & {row['mode']} & {row['attack']} & "
            f"{row['baseline']} & "
            f"{bold}{{{sign}{row['delta']*100:.2f}}} & "
            f"{row['pct_improvement']:.1f} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        rf"\small\item Bold = proposed model outperforms baseline.",
        r"\end{tablenotes}",
        r"\end{table}",
    ]
    path = os.path.join(save_dir, f"latex_comparison_{metric}.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"LaTeX comparison table saved: {path}")


def _save_latex_ci_table(
    ci_df:    pd.DataFrame,
    save_dir: str,
    metric:   str,
) -> None:
    """LaTeX table: mean ± 95% CI per model across attacks."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{metric.replace('_',' ').title()} — Mean $\pm$ 95\% CI Across Attack Conditions}}",
        rf"\label{{tab:ci_{metric}}}",
        r"\begin{tabular}{llllrr}",
        r"\toprule",
        r"Dataset & Mode & Model & N & Mean (\%) & 95\% CI \\",
        r"\midrule",
    ]

    for _, row in ci_df.iterrows():
        lo = row["ci_lower"] * 100 if not np.isnan(row["ci_lower"]) else float("nan")
        hi = row["ci_upper"] * 100 if not np.isnan(row["ci_upper"]) else float("nan")
        ci_str = f"[{lo:.2f}, {hi:.2f}]" if not np.isnan(lo) else "—"
        lines.append(
            f"{row['dataset']} & {row['mode']} & {row['model']} & "
            f"{int(row['n'])} & {row['mean']*100:.2f} & {ci_str} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path = os.path.join(save_dir, f"latex_ci_{metric}.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"LaTeX CI table saved: {path}")


def _save_latex_nemenyi_table(
    nem_df:   pd.DataFrame,
    save_dir: str,
) -> None:
    """LaTeX table: Nemenyi pairwise significance results."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Nemenyi Post-Hoc Pairwise Significance Test}",
        r"\label{tab:nemenyi}",
        r"\begin{tabular}{llllrrr}",
        r"\toprule",
        r"Dataset & Mode & Comparison & Better & Rank Diff. & CD & Sig. \\",
        r"\midrule",
    ]

    for _, row in nem_df.iterrows():
        sig_str = r"\checkmark" if row["significant"] else "—"
        lines.append(
            f"{row['dataset']} & {row['mode']} & "
            f"{row['comparison'].replace('_vs_', ' vs ')} & "
            f"{row.get('better', '—')} & "
            f"{row['rank_diff']:.3f} & {row['cd']:.3f} & {sig_str} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path = os.path.join(save_dir, "latex_nemenyi.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"LaTeX Nemenyi table saved: {path}")
