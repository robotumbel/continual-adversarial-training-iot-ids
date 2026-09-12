"""
analyze_r2.py — statistics for the revised protocol.

Implements what Section V-E of the revised paper promises:

  * two pre-specified PRIMARY hypotheses (average accuracy, clean MCC)
    on the fixed-backbone CAT-vs-sequential-AT contrast, reported
    uncorrected;
  * every other contrast treated as SECONDARY and exploratory, with a
    Holm correction across that family, raw and adjusted p side by side;
  * the dataset as the blocking factor: a per-dataset paired Wilcoxon
    over seeds with a bootstrap CI on the mean difference, plus a
    dataset-stratified permutation test that flips the signs of paired
    differences within each dataset rather than pooling seeds across
    datasets.

Reads one or more paper3_exp1_master.csv files and pairs rows on
(dataset, seed). Emits a summary table and, with --latex, the rows for
the paper's statistics table.

Usage:
  python analyze_r2.py --runs "runs_r2/**/paper3_exp1_master.csv"
  python analyze_r2.py --runs "runs_r2/full/paper3_exp1_master.csv" --latex
"""
from __future__ import annotations
import argparse, glob, itertools

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RNG = np.random.default_rng(20260907)
N_PERM = 20000
N_BOOT = 10000

# The contrast that carries the paper's primary claims. Everything else
# is secondary.
PRIMARY_METRICS = [("AA", "average accuracy"), ("clean_mcc", "clean MCC")]
SECONDARY_METRICS = [("rMCC", "robust MCC")]


def load(patterns):
    frames = []
    for pat in patterns:
        for f in glob.glob(pat, recursive=True):
            frames.append(pd.read_csv(f))
    if not frames:
        raise SystemExit(f"no master CSVs matched {patterns}")
    df = pd.concat(frames, ignore_index=True)
    if "variant" not in df.columns:
        df["variant"] = "legacy"

    # A resumed run appends: a config that failed and was later retried
    # leaves both its error row and its successful row in the master CSV.
    # Keep only successful rows, and only the last one per config.
    n_all = len(df)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    df = df[df["AA"].notna()]

    # The no-adversarial-loop control shares its backbone and variant
    # with CAT and differs only in beta, so beta has to be part of the
    # identity or the two would collapse into one another here.
    df["beta"] = pd.to_numeric(df.get("beta"), errors="coerce").fillna(6.0)
    df["ewc_lambda"] = pd.to_numeric(
        df.get("ewc_lambda"), errors="coerce").fillna(1000.0)
    key = ["dataset", "seed", "backbone", "variant", "beta", "ewc_lambda"]
    df = df.drop_duplicates(subset=key, keep="last").reset_index(drop=True)
    if len(df) != n_all:
        print(f"[load] kept {len(df)} of {n_all} rows "
              f"(dropped failed runs and superseded retries)")
    robust_cols = [c for c in ("fgsm_mcc", "pgd_mcc", "square_mcc",
                               "transfer_mcc") if c in df.columns]
    df["rMCC"] = df[robust_cols].mean(axis=1)
    df["cond"] = (df["backbone"] + "/" + df["variant"].astype(str)
                  + np.where(df["beta"] == 0, "/noAT", "")
                  + np.where(df["ewc_lambda"] != 1000.0,
                             "/ewc" + df["ewc_lambda"].astype(int).astype(str),
                             ""))
    return df


def paired(df, metric, cond_a, cond_b):
    """Return a DataFrame of per-(dataset, seed) paired differences a-b."""
    a = df[df["cond"] == cond_a].set_index(["dataset", "seed"])[metric]
    b = df[df["cond"] == cond_b].set_index(["dataset", "seed"])[metric]
    idx = a.index.intersection(b.index)
    if len(idx) == 0:
        return None
    d = (a.loc[idx] - b.loc[idx]).rename("diff").reset_index()
    return d.dropna()


def boot_ci(x, n=N_BOOT, alpha=0.05):
    if len(x) < 2:
        return (np.nan, np.nan)
    means = [RNG.choice(x, size=len(x), replace=True).mean() for _ in range(n)]
    return tuple(np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def stratified_perm_test(d):
    """Sign-flip permutation test with the dataset as the block.

    Statistic: the mean over datasets of each dataset's mean paired
    difference, so every dataset contributes equally regardless of how
    many seeds it carries. The null flips the sign of each paired
    difference independently, which is the exact randomisation null for
    a paired design.
    """
    groups = [g["diff"].to_numpy() for _, g in d.groupby("dataset")]
    obs = np.mean([g.mean() for g in groups])
    count = 0
    for _ in range(N_PERM):
        stat = np.mean([(g * RNG.choice([-1.0, 1.0], size=len(g))).mean()
                        for g in groups])
        if abs(stat) >= abs(obs) - 1e-15:
            count += 1
    return obs, (count + 1) / (N_PERM + 1)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def analyse(d, label, metric_label):
    """Per-dataset tests plus the stratified combination."""
    rows = []
    for ds, g in d.groupby("dataset"):
        x = g["diff"].to_numpy()
        try:
            p = wilcoxon(x).pvalue if np.any(x != 0) else 1.0
        except ValueError:
            p = 1.0
        lo, hi = boot_ci(x)
        rows.append({"scope": ds, "n": len(x), "mean_diff": x.mean(),
                     "ci_lo": lo, "ci_hi": hi, "p_raw": p})
    obs, p_strat = stratified_perm_test(d)
    consistent = len({np.sign(r["mean_diff"]) for r in rows}) == 1
    rows.append({"scope": "STRATIFIED", "n": len(d), "mean_diff": obs,
                 "ci_lo": np.nan, "ci_hi": np.nan, "p_raw": p_strat})
    out = pd.DataFrame(rows)
    out.insert(0, "metric", metric_label)
    out.insert(0, "contrast", label)
    out["direction_consistent"] = consistent
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["results/s*.csv"])
    ap.add_argument("--cat", default="transformer_cl/cat",
                    help="CAT condition as backbone/variant.")
    ap.add_argument("--control", default="transformer/noCL",
                    help="Sequential-AT control as backbone/variant.")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--out", default="stats_r2.csv")
    args = ap.parse_args()

    df = load(args.runs)
    print("conditions found:",
          ", ".join(sorted(df["cond"].unique())), "\n")

    primary, secondary = [], []

    for metric, mlabel in PRIMARY_METRICS:
        d = paired(df, metric, args.cat, args.control)
        if d is None:
            print(f"!! no pairs for {args.cat} vs {args.control} on {metric}")
            continue
        primary.append(analyse(d, f"{args.cat} - {args.control}", mlabel))

    for metric, mlabel in SECONDARY_METRICS:
        d = paired(df, metric, args.cat, args.control)
        if d is not None:
            secondary.append(analyse(d, f"{args.cat} - {args.control}",
                                     mlabel))

    # Secondary family: every other condition against the CAT condition,
    # on the primary metrics -- the architecture and mechanism contrasts.
    others = [c for c in sorted(df["cond"].unique())
              if c not in (args.cat, args.control)]
    for other, (metric, mlabel) in itertools.product(others, PRIMARY_METRICS):
        d = paired(df, metric, other, args.cat)
        if d is not None:
            secondary.append(analyse(d, f"{other} - {args.cat}", mlabel))

    res_p = pd.concat(primary) if primary else pd.DataFrame()
    res_s = pd.concat(secondary) if secondary else pd.DataFrame()

    if not res_p.empty:
        res_p["family"] = "primary"
        res_p["p_adj"] = res_p["p_raw"]          # pre-specified, uncorrected
    if not res_s.empty:
        res_s["family"] = "secondary"
        mask = res_s["scope"] == "STRATIFIED"    # correct across contrasts
        res_s.loc[mask, "p_adj"] = holm(res_s.loc[mask, "p_raw"].to_numpy())

    res = pd.concat([r for r in (res_p, res_s) if not r.empty],
                    ignore_index=True)
    res.to_csv(args.out, index=False)

    show = res.copy()
    for c in ("mean_diff", "ci_lo", "ci_hi"):
        show[c] = show[c].round(3)
    for c in ("p_raw", "p_adj"):
        show[c] = show[c].round(4)
    print(show.to_string(index=False))
    print(f"\nwrote {args.out}")

    if args.latex:
        print("\n% --- rows for the statistics table ---")
        for _, r in res[res["scope"] == "STRATIFIED"].iterrows():
            star = "$^{\\ast}$" if r["p_adj"] < 0.05 else ""
            print(f"{r['contrast']} & {r['metric']} & "
                  f"${r['mean_diff']:+.3f}$ & "
                  f"${r['p_raw']:.3f}$ & ${r['p_adj']:.3f}${star} \\\\")


if __name__ == "__main__":
    main()
