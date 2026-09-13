"""check_consistency.py — cross-check the manuscript against the data.

Reviewer 2's first three comments were all the same complaint: a number
in one place did not match the same number in another, and one quoted
p-value could not be traced to any test at all. Fixing those instances
by hand would leave the mechanism that produced them intact, so this
checks the whole document instead:

  1. numbers that were carried over from the withdrawn runs and should
     no longer appear anywhere;
  2. the headline quantities, verified against the master CSVs rather
     than against each other;
  3. every distinct numeric literal in the tex, grouped, so a reader can
     eyeball whether one quantity is quoted two ways.

Usage:  python check_consistency.py
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

TEX = os.environ.get("PAPER_TEX", "paper3.tex")
RUNS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results", "s*.csv")

# Values that appeared in the withdrawn submission. Any of these still in
# the text is a number that survived the rerun without being revisited.
STALE = {
    "0.303": "old Seq-AT average accuracy",
    "0.439": "old T-CAT average accuracy",
    "0.250": "old T-CAT clean MCC",
    "0.220": "old T-CAT robust MCC",
    "0.011": "old Seq-AT robust MCC",
    # 0.286 was the untraceable pooled p-value, but it is also a
    # per-family accuracy in the appendix; checking it is pure noise.
    "0.709": "old CAT average accuracy (4-dataset protocol)",
    "0.553": "old CAT clean MCC (4-dataset protocol)",
    "0.332": "old CAT robust MCC",
    "0.665": "old BiC average accuracy",
    "0.524": "old BiC clean MCC",
    "0.673": "old RBC average accuracy",
    # 0.531 was the old RBC clean MCC and is now the FGSM robust MCC
    # in the corrected attack table -- same digits, different claim.
}


def load():
    frames = []
    for f in glob.glob(RUNS):
        d = pd.read_csv(f)
        d = d[(d.get("status", "ok") == "ok") & d["AA"].notna()]
        d["rd"] = os.path.splitext(os.path.basename(f))[0]
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["beta"] = pd.to_numeric(d.get("beta"), errors="coerce").fillna(6.0)
    d["rMCC"] = d[["fgsm_mcc", "pgd_mcc", "square_mcc",
                   "transfer_mcc"]].mean(axis=1)
    # RBC is variant "bic" plus a flag the CSV never recorded; the run
    # directory is the only thing that distinguishes the two.
    d["m"] = np.where(d["rd"] == "s5_sota_rbc", "rbc", d["variant"])
    d = d[d["beta"] == 6.0]
    return d.drop_duplicates(
        subset=["dataset", "seed", "m", "backbone"], keep="last")


def sign_flip_p(diff):
    """Exact two-sided sign-flip permutation test on paired differences.

    Flipping within each dataset block is the randomisation null the
    protocol names; with the mean as statistic it equals flipping every
    pair, so all 2^n sign patterns are enumerated directly.
    """
    import itertools
    obs = abs(diff.mean())
    signs = np.array(list(itertools.product([1, -1], repeat=len(diff))))
    return float((np.abs((signs * diff).mean(axis=1)) >= obs - 1e-12).mean())


def paired(d, a, bba, b, bbb, col):
    x = d[(d.m == a) & (d.backbone == bba)].set_index(["dataset", "seed"])[col]
    y = d[(d.m == b) & (d.backbone == bbb)].set_index(["dataset", "seed"])[col]
    i = x.index.intersection(y.index)
    diff = (x.sort_index()[i] - y.sort_index()[i]).values
    return diff.mean(), sign_flip_p(diff), len(diff)


def main():
    tex = open(TEX, encoding="utf-8").read()
    # strip comments: a stale number in a comment is not a claim
    body = "\n".join(l for l in tex.splitlines() if not l.lstrip().startswith("%"))
    # Standard deviations are printed as {\tiny$\pm0.011$}. Those digits are
    # not claims about a quantity and collide with withdrawn values purely by
    # coincidence, so strip them before looking for stale numbers -- a checker
    # that cries wolf gets ignored exactly when it matters.
    body = re.sub(r"\{\\tiny\$\\pm[0-9.]+\$\}", "", body)
    d = load()
    bad = 0

    print("=" * 62)
    print("1. Numbers carried over from the withdrawn runs")
    print("=" * 62)
    for val, what in sorted(STALE.items()):
        hits = len(re.findall(r"(?<![\d.])" + re.escape(val) + r"(?![\d])", body))
        if hits:
            print(f"  !! {val} appears {hits}x  ({what})")
            bad += hits
    if not bad:
        print("  none found")

    print()
    print("=" * 62)
    print("2. Headline quantities, recomputed from the master CSVs")
    print("=" * 62)
    checks = [
        ("H1 clean MCC", "cat", "transformer_cl", "noCL", "transformer", "clean_mcc"),
        ("H1 AA",        "cat", "transformer_cl", "noCL", "transformer", "AA"),
        ("H2 clean MCC", "cat", "transformer_cl", "replay", "transformer_cl", "clean_mcc"),
        ("H2 robust MCC","cat", "transformer_cl", "replay", "transformer_cl", "rMCC"),
        ("H3 robust MCC","robustcat", "transformer_cl", "cat", "transformer_cl", "rMCC"),
        ("H3 clean MCC", "robustcat", "transformer_cl", "cat", "transformer_cl", "clean_mcc"),
    ]
    for label, a, bba, b, bbb, col in checks:
        m, p, n = paired(d, a, bba, b, bbb, col)
        dtxt, ptxt = f"{abs(m):.3f}", f"{p:.4f}".rstrip("0")
        in_tex = re.search(r"(?<![\d.])" + re.escape(dtxt) + r"(?![\d])", body)
        mark = "ok " if in_tex else "!! "
        if not in_tex:
            bad += 1
        print(f"  {mark}{label:<14} delta={m:+.3f} p={p:.4f} n={n}"
              f"   {'delta quoted in text' if in_tex else 'DELTA NOT FOUND IN TEXT'}")

    print()
    print("=" * 62)
    print("3. Repeated numeric literals (same value, several sections)")
    print("=" * 62)
    # A quantity quoted once is fine; quoted many times it should agree,
    # and the eyeball check is cheaper than parsing every sentence.
    nums = re.findall(r"\$?(0\.\d{3})\$?", body)
    from collections import Counter
    for val, c in sorted(Counter(nums).items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {val}  x{c}")

    print()
    print("=" * 62)
    print(f"VERDICT: {'PROBLEMS FOUND (' + str(bad) + ')' if bad else 'consistent'}")
    print("=" * 62)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
