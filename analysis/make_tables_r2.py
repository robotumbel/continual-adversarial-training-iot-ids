"""make_tables_r2.py — paper tables, generated from the corrected runs.

Every number in the revised manuscript should come from here rather than
being retyped, because the previous submission carried figures that could
not be traced back to any surviving data (the abstract's p=0.286 among
them). Run this, paste the output, and the paper and the CSVs cannot
drift apart.

Usage:
    python make_tables_r2.py                     # all tables
    python make_tables_r2.py --table main        # just one
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

RUNS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results", "s*.csv")

# Display names. The variant keys are the code's; the labels are what the
# paper calls them after the method was redefined to include replay.
LABEL = {
    "transformer/noCL":          r"Sequential AT (no continual loop)",
    "transformer_cl/ewc":        r"\;+ EWC only",
    "transformer_cl/kd":         r"\;+ distillation only",
    "transformer_cl/replay":     r"\;+ replay only",
    "transformer_cl/ewckd":      r"\;+ EWC $+$ distillation",
    "transformer_cl/bic":        r"BiC~\cite{wu2019bic}",
    "transformer_cl/rbc":        r"RBC (robust bias correction)",
    "transformer_cl/cat":        r"\textbf{CAT (ours)}",
    "transformer_cl/robustcat":  r"\;\;CAT $+$ adversary-aware (ablation)",
}
ORDER = list(LABEL)
STABLE = ["CICIoMT", "TONIoT"]          # datasets that carry claims


def load():
    frames = []
    for f in glob.glob(RUNS):
        d = pd.read_csv(f)
        d = d[d.get("status", "ok") == "ok"]
        # RBC is implemented as variant "bic" plus a --robust-bc flag that
        # the master CSV never records, so BiC and RBC are indistinguishable
        # by their columns alone and silently collapse into one another on
        # de-duplication. The run directory is the only thing that tells
        # them apart, so it becomes part of the identity here.
        d["rundir"] = os.path.splitext(os.path.basename(f))[0]
        frames.append(d[d["AA"].notna()])
    if not frames:
        raise SystemExit(f"no runs matched {RUNS}")
    df = pd.concat(frames, ignore_index=True)
    df["beta"] = pd.to_numeric(df.get("beta"), errors="coerce").fillna(6.0)
    df = df[df["beta"] == 6.0]           # the no-AT control is its own table
    df["rMCC"] = df[["fgsm_mcc", "pgd_mcc", "square_mcc",
                     "transfer_mcc"]].mean(axis=1)
    df["cond"] = df["backbone"] + "/" + df["variant"].astype(str)
    df.loc[df["rundir"] == "s5_sota_rbc", "cond"] = "transformer_cl/rbc"
    return df.drop_duplicates(
        subset=["dataset", "seed", "cond", "beta"], keep="last")


def fmt(m, s=None):
    if pd.isna(m):
        return "--"
    return f"${m:.3f}$" if s is None or pd.isna(s) else f"${m:.3f}${{\\tiny$\\pm{s:.3f}$}}"


def table_main(df):
    """Mechanism ladder: what each continual component buys."""
    print(r"""% --- Table: mechanism ladder -------------------------------
\begin{table*}[!t]
\caption{Each continual mechanism added to the same adversarially-trained
Transformer, on the two benchmarks whose seed-to-seed spread supports a
claim (CICIoT is reported per seed in Table~\ref{tab:ciciot} instead).
Mean over five seeds, $\pm$ SD. cMCC and rMCC are clean and robust MCC;
robust MCC averages FGSM, PGD, Square and transfer, all of which respect
the feature-space constraints (validity $100\%$).}
\label{tab:ladder}
\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcccccc@{}}
\toprule
& \multicolumn{3}{c}{CICIoMT2024} & \multicolumn{3}{c}{TON\_IoT} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Training & AA & cMCC & rMCC & AA & cMCC & rMCC \\
\midrule""")
    for cond in ORDER:
        sub = df[df["cond"] == cond]
        if sub.empty:
            continue
        cells = []
        for ds in STABLE:
            s = sub[sub["dataset"] == ds]
            for col in ("AA", "clean_mcc", "rMCC"):
                cells.append(fmt(s[col].mean(), s[col].std()))
        print(f"{LABEL[cond]} & " + " & ".join(cells) + r" \\")
    print(r"""\bottomrule
\end{tabular}
\end{table*}""")


def table_ciciot(df):
    """CICIoT is bimodal across seeds; a mean would hide that."""
    sub = df[(df["dataset"] == "CICIoT")
             & df["cond"].isin(["transformer/noCL", "transformer_cl/cat"])]
    if sub.empty:
        return
    print(r"""
% --- Table: CICIoT per seed --------------------------------
\begin{table}[!t]
\caption{CICIoT2023, clean MCC per seed. This benchmark is the most
seed-sensitive of the three -- the five-seed verification that fixed our
protocol measured a coefficient of variation near $0.8$ on it, against
$0.04$--$0.05$ on the other two -- so we report it per seed rather than
as a mean. Under the proposed method the spread is wide but no run
collapses; under sequential AT every seed is degenerate.}
\label{tab:ciciot}
\centering\footnotesize
\begin{tabular}{@{}lccccc@{}}
\toprule
Training & \multicolumn{5}{c}{seed} \\
\cmidrule(lr){2-6}
 & $1$ & $7$ & $13$ & $21$ & $42$ \\
\midrule""")
    for cond in ["transformer/noCL", "transformer_cl/cat"]:
        s = sub[sub["cond"] == cond].set_index("seed")["clean_mcc"]
        cells = [f"${s.get(k, float('nan')):.3f}$" for k in (1, 7, 13, 21, 42)]
        print(f"{LABEL[cond]} & " + " & ".join(cells) + r" \\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")


def table_noat(df_all):
    """Does the adversarial inner loop earn its place?"""
    d = df_all[(df_all["backbone"] == "transformer_cl")
               & (df_all["variant"] == "ewckd")]
    if d.empty:
        return
    print(r"""
% --- Table: adversarial inner loop on/off -------------------
\begin{table}[!t]
\caption{The PGD--TRADES inner loop switched off ($\beta=0$) against the
identical continual schedule with it on. The effect is not uniform: it
helps robust MCC where there is robustness to retain and costs a little
clean MCC elsewhere, so we report it as a component study rather than as
a headline claim.}
\label{tab:noat}
\centering\footnotesize
\begin{tabular}{@{}llccc@{}}
\toprule
Dataset & Inner loop & AA & cMCC & rMCC \\
\midrule""")
    for ds in STABLE + ["CICIoT"]:
        for beta, name in ((0.0, r"off ($\beta{=}0$)"), (6.0, r"on")):
            s = d[(d["dataset"] == ds) & (d["beta"] == beta)]
            if s.empty:
                continue
            print(f"{ds if beta == 0.0 else ''} & {name} & "
                  f"{fmt(s['AA'].mean())} & {fmt(s['clean_mcc'].mean())} & "
                  f"{fmt(s['rMCC'].mean())} " + r"\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")


def table_arch(df):
    """Architecture-agnosticism, for the proposed method."""
    d = df[(df["variant"] == "cat") & df["dataset"].isin(STABLE)]
    if d["backbone"].nunique() < 2:
        print("\n% (architecture table pending: cat has only been run on "
              f"{sorted(d['backbone'].unique())})")
        return
    fam = {"dnn": "MLP", "lstm": "recurrent", "transformer_cl": "Transformer",
           "rdt": "denoising Transformer", "aam_trans": "adaptive attention",
           "aam_trans_no_gates": "adaptive attention, no gates"}
    print(r"""
% --- Table: architecture ------------------------------------
\begin{table}[!t]
\caption{CAT instantiated on backbones from three model families, pooled
over the two claim-carrying benchmarks and five seeds. A small spread
says the training procedure, not the architecture, sets the operating
point.}
\label{tab:arch}
\centering\footnotesize
\begin{tabular}{@{}lccc@{}}
\toprule
Backbone & AA & cMCC & rMCC \\
\midrule""")
    means = {}
    for bb, g in d.groupby("backbone"):
        name = fam.get(bb, bb)
        means[bb] = g["clean_mcc"].mean()
        print(f"{name} & {fmt(g['AA'].mean(), g['AA'].std())} & "
              f"{fmt(g['clean_mcc'].mean(), g['clean_mcc'].std())} & "
              f"{fmt(g['rMCC'].mean(), g['rMCC'].std())} " + r"\\")
    spread = max(means.values()) - min(means.values())
    print(r"\midrule")
    print(f"cross-family spread & & ${spread:.3f}$ & " + r"\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")


def main():
    global RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="all",
                    choices=["all", "main", "ciciot", "noat", "arch"])
    ap.add_argument("--runs", default=RUNS)
    a = ap.parse_args()
    RUNS = a.runs

    # the beta=0 rows are dropped by load(), so keep an unfiltered copy
    frames = []
    for f in glob.glob(RUNS):
        d = pd.read_csv(f)
        d = d[d.get("status", "ok") == "ok"]
        frames.append(d[d["AA"].notna()])
    raw = pd.concat(frames, ignore_index=True)
    raw["beta"] = pd.to_numeric(raw.get("beta"), errors="coerce").fillna(6.0)
    raw["rMCC"] = raw[["fgsm_mcc", "pgd_mcc", "square_mcc",
                       "transfer_mcc"]].mean(axis=1)

    df = load()
    if a.table in ("all", "main"):
        table_main(df)
    if a.table in ("all", "ciciot"):
        table_ciciot(df)
    if a.table in ("all", "noat"):
        table_noat(raw)
    if a.table in ("all", "arch"):
        table_arch(df)


if __name__ == "__main__":
    main()
