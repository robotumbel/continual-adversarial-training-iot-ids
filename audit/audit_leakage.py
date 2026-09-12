"""
audit_leakage.py — record-diversity and train/test overlap audit.

Reports, per dataset subset:
  * how many rows are distinct feature vectors, overall and per class
  * how many validation rows are exact duplicates of a training row, under
    the split the experiment actually uses, averaged over the seeds

Run it on both an old-protocol subset (balanced before splitting) and a
split-safe one to show what the fix changes, and on the raw sources to show
which duplication is inherent to a dataset rather than introduced by
resampling. CICIoV2024 is the case where it is inherent: its attack classes
hold only a handful of distinct CAN-frame vectors each, so no split can keep
train and test disjoint.

Usage:
  python audit_leakage.py --dirs balanced_data balanced_data_ss10k
  python audit_leakage.py --files ../merged_train_binary_shuffled.csv --nrows 400000
"""
from __future__ import annotations
import argparse, glob, os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEEDS = [1, 7, 13, 21, 42]


def label_col(df):
    return [c for c in df.columns if c.strip().lower() == "label"][0]


def audit(path, nrows=None, test_size=0.20, per_class=True):
    df = pd.read_csv(path, nrows=nrows)
    lc = label_col(df)
    feats = [c for c in df.columns if c != lc]
    n = len(df)
    uniq = df.drop_duplicates(subset=feats).shape[0]

    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0).values
    keys = np.array([r.tobytes() for r in np.ascontiguousarray(X)])
    y = pd.factorize(df[lc])[0]

    rates = []
    for s in SEEDS:
        # stratify needs >=2 members per class; fall back to unstratified
        strat = y if np.bincount(y).min() >= 2 else None
        ktr, kva = train_test_split(keys, test_size=test_size,
                                    random_state=s, stratify=strat)
        tr = set(ktr.tolist())
        rates.append(100.0 * sum(k in tr for k in kva.tolist()) / len(kva))

    print(f"\n{os.path.basename(path)}: {n:,} rows | "
          f"{100*uniq/n:.1f}% distinct | "
          f"val rows duplicated from train: {np.mean(rates):.1f}% "
          f"(min {min(rates):.1f}, max {max(rates):.1f})")

    if per_class:
        print(f"  {'class':<28s} {'rows':>8s} {'distinct':>9s} {'%':>6s}")
        for lbl, g in df.groupby(lc):
            d = g.drop_duplicates(subset=feats).shape[0]
            print(f"  {str(lbl):<28s} {len(g):>8,d} {d:>9,d} "
                  f"{100*d/len(g):>5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=[],
                    help="Folders holding balanced_<DS>.csv subsets.")
    ap.add_argument("--files", nargs="*", default=[],
                    help="Individual CSVs (e.g. a raw merged source).")
    ap.add_argument("--nrows", type=int, default=None,
                    help="Read only the first N rows (for the big raw CSVs).")
    ap.add_argument("--no-per-class", action="store_true")
    args = ap.parse_args()

    paths = list(args.files)
    for d in args.dirs:
        paths += sorted(glob.glob(os.path.join(d, "balanced_*.csv")))
    if not paths:
        ap.error("nothing to audit: pass --dirs and/or --files")

    for p in paths:
        if not os.path.exists(p):
            print(f"\n!! missing {p}")
            continue
        audit(p, nrows=args.nrows, per_class=not args.no_per_class)


if __name__ == "__main__":
    main()
