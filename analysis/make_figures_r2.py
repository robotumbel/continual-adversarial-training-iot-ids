"""make_figures_r2.py — result figures, regenerated from the corrected runs.

The figures shipped with the withdrawn version were built from data the
integrity audit later disqualified: two of them showed task matrices on
CICIoV2024, the benchmark this version excludes outright. A figure that
contradicts the text is worse than no figure, and a reader looks at the
figures first, so these are generated from the same master CSVs as the
tables rather than kept as static files.

Outputs (paper3_iotj_clat/figs/):
    fig_mechanism_bar.pdf   what each continual mechanism recovers
    fig_taskmatrix.pdf      retention matrices, baseline vs proposed
    fig_forget_curve.pdf    accuracy over the families seen so far
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(os.path.dirname(HERE), "figs")
RUNS = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "figure.dpi": 300,
    "axes.linewidth": 0.7,
})

STABLE = ["CICIoMT", "TONIoT"]
C_BASE, C_MECH, C_OURS = "#b2182b", "#8aa8c4", "#2c7d3f"


def master():
    frames = []
    for f in glob.glob(os.path.join(RUNS, "s*.csv")):
        d = pd.read_csv(f)
        d = d[(d.get("status", "ok") == "ok") & d["AA"].notna()]
        d["rd"] = os.path.splitext(os.path.basename(f))[0]
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["beta"] = pd.to_numeric(d["beta"], errors="coerce").fillna(6.0)
    d["rMCC"] = d[["fgsm_mcc", "pgd_mcc", "square_mcc",
                   "transfer_mcc"]].mean(axis=1)
    d["m"] = np.where(d["rd"] == "s5_sota_rbc", "rbc", d["variant"])
    d = d[d["beta"] == 6.0]
    return d.drop_duplicates(subset=["dataset", "seed", "m", "backbone"],
                             keep="last")


def fig_mechanism(d):
    """The ladder, as a picture: what each mechanism recovers."""
    order = [("noCL", "transformer", "Sequential\nAT"),
             ("ewc", "transformer_cl", "+EWC"),
             ("kd", "transformer_cl", "+KD"),
             ("replay", "transformer_cl", "+replay"),
             ("cat", "transformer_cl", "CAT"),
             ("robustcat", "transformer_cl", "CAT$^{adv}$")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for ax, (col, title) in zip(axes, [("clean_mcc", "clean MCC"),
                                       ("rMCC", "robust MCC")]):
        means, errs, labels, colours = [], [], [], []
        for m, bb, lab in order:
            q = d[(d.m == m) & (d.backbone == bb) & d.dataset.isin(STABLE)]
            means.append(q[col].mean())
            errs.append(q[col].std())
            labels.append(lab)
            colours.append(C_BASE if m == "noCL"
                           else C_OURS if m in ("cat", "robustcat") else C_MECH)
        x = np.arange(len(order))
        ax.bar(x, means, yerr=errs, capsize=2.5, color=colours,
               edgecolor="black", linewidth=0.5, error_kw={"lw": 0.7})
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.6)
        ax.set_ylabel(title, fontsize=7.5)
        ax.tick_params(axis="y", labelsize=6.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", lw=0.4, alpha=0.35)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig_mechanism_bar.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("written", out)


def _taskmat(stage, tag, dataset):
    """Mean task matrix over seeds for one configuration."""
    pat = os.path.join(RUNS, "taskmat", f"taskmat_{tag}_s*_{dataset}.csv")
    mats, cols = [], None
    for f in sorted(glob.glob(pat)):
        m = pd.read_csv(f)
        cols = list(m.columns)
        mats.append(m.values)
    if not mats:
        return None, None
    n = min(x.shape[0] for x in mats)
    return np.mean([x[:n, :n] for x in mats], axis=0), cols[:n]


def fig_taskmatrix(dataset="CICIoMT"):
    """Retention, baseline against the proposed method, on a benchmark that
    carries a claim -- the withdrawn version showed this on CICIoV2024."""
    panels = [("s1_primary", "transformer_noCL", "Sequential AT"),
              ("s6_method", "transformer_cl_cat", "CAT")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, (stage, tag, title) in zip(axes, panels):
        M, cols = _taskmat(stage, tag, dataset)
        if M is None:
            ax.text(0.5, 0.5, f"no data\n{tag}", ha="center", va="center")
            ax.axis("off")
            continue
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols, fontsize=6)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("after training through", fontsize=6.8)
        if ax is axes[0]:
            ax.set_ylabel("accuracy on task", fontsize=6.8)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if j >= i:
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                            fontsize=5.2,
                            color="white" if M[i, j] < 0.6 else "black")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02).ax.tick_params(
        labelsize=6)
    out = os.path.join(FIGS, "fig_taskmatrix.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("written", out)


def fig_forget(dataset="TONIoT"):
    """Accuracy over the families seen so far, as the stream grows."""
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for stage, tag, label, colour in [
            ("s1_primary", "transformer_noCL", "Sequential AT", C_BASE),
            ("s6_method", "transformer_cl_cat", "CAT", C_OURS)]:
        M, _ = _taskmat(stage, tag, dataset)
        if M is None:
            continue
        # after stage j, mean accuracy over the tasks seen so far
        curve = [np.mean(M[:j + 1, j]) for j in range(M.shape[1])]
        ax.plot(range(1, len(curve) + 1), curve, marker="o", ms=3.2, lw=1.3,
                color=colour, label=label)
    ax.set_xlabel("attack families seen", fontsize=7.5)
    ax.set_ylabel("mean accuracy so far", fontsize=7.5)
    ax.tick_params(labelsize=6.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(lw=0.4, alpha=0.35)
    ax.legend(fontsize=6.8, frameon=False)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig_forget_curve.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("written", out)


if __name__ == "__main__":
    d = master()
    fig_mechanism(d)
    fig_taskmatrix()
    fig_forget()
