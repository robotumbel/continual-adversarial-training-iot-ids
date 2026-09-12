"""
make_balanced_subset.py — build class-balanced subsets for fast,
publication-defensible Paper 3 training on a laptop GPU.

For each dataset, every class is resampled to the SAME count
(target_per_class = total // n_classes):
  * majority classes  -> random undersample (without replacement)
  * minority classes  -> random oversample  (with replacement)

The result is a perfectly class-balanced CSV of ~`--total` rows per
dataset. Combined with the seq-expansion noise in data_loader, the
oversampled minority duplicates are not identical at training time.

`--no-oversample` caps majority classes but never duplicates minority
rows, so the emitted file holds distinct records only. Pair it with
`load_dataset(balance_train_only=True)`, which restores class balance
after the train/val split: duplicating rows here, before that split,
puts copies of the same record on both sides of it.

Usage:
  python make_balanced_subset.py --total 10000 --out balanced_data
  python make_balanced_subset.py --total 10000 --no-oversample \
         --out balanced_data_splitsafe
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd

FULL = {
    "CICIoT":  "merged_CICIoT_train_shuffled.csv",
    "CICIoMT": "merged_train_ciciomt_shuffled.csv",
    "CICIoV":  "merged_train_binary_shuffled.csv",
    "TONIoT":  "merged_TONIoT_clean.csv",
}


def label_col(df):
    return [c for c in df.columns if c.strip().lower() == "label"][0]


def collect_by_class(path, cap_per_class, seed):
    """Stream the big CSV and keep up to cap_per_class rows per label."""
    keep = {}
    for chunk in pd.read_csv(path, chunksize=200000):
        lc = label_col(chunk)
        for lbl, grp in chunk.groupby(lc):
            cur = keep.get(lbl)
            have = 0 if cur is None else len(cur)
            need = cap_per_class - have
            if need > 0:
                add = grp.head(need)
                keep[lbl] = add if cur is None else pd.concat([cur, add])
        if keep and all(len(v) >= cap_per_class for v in keep.values()):
            break
    return keep, lc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=10000,
                    help="Approx. total rows per dataset (balanced).")
    ap.add_argument("--out", default="balanced_data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-oversample", action="store_true",
                    help="Cap majority classes but keep minority classes at "
                         "their natural count, emitting distinct records "
                         "only. Balance the train split afterwards with "
                         "load_dataset(balance_train_only=True).")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    project = os.path.dirname(here)            # parent holds the full CSVs
    out_dir = os.path.join(here, args.out)
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for name, fname in FULL.items():
        # Most raw sources sit in the parent folder; TON_IoT sits here.
        path = next((p for p in (os.path.join(project, fname),
                                 os.path.join(here, fname))
                     if os.path.exists(p)), None)
        if path is None:
            print(f"!! missing {fname}, skipping {name}")
            continue
        # First pass: how many classes? (peek header + sample)
        peek = pd.read_csv(path, nrows=300000)
        lc = label_col(peek)
        n_classes = peek[lc].nunique()
        target = max(args.total // n_classes, 20)

        # Collect a generous cap so we can undersample majorities cleanly
        keep, lc = collect_by_class(path, cap_per_class=max(target, 2000),
                                    seed=args.seed)

        balanced = []
        for lbl, grp in keep.items():
            g = grp.reset_index(drop=True)
            if len(g) >= target:
                idx = rng.choice(len(g), size=target, replace=False)
            elif args.no_oversample:               # keep minority as-is
                idx = np.arange(len(g))
            else:                                  # oversample minority
                idx = rng.choice(len(g), size=target, replace=True)
            balanced.append(g.iloc[idx])
        out = pd.concat(balanced).sample(frac=1.0, random_state=args.seed)
        op = os.path.join(out_dir, f"balanced_{name}.csv")
        out.to_csv(op, index=False)
        vc = out[lc].value_counts()
        print(f"{name}: {len(out)} rows | {n_classes} classes | "
              f"{target}/class | min={vc.min()} max={vc.max()} -> {op}")


if __name__ == "__main__":
    main()
