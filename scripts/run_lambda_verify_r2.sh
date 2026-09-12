#!/usr/bin/env bash
# run_lambda_verify_r2.sh — is the lambda=10000 result real, or one seed?
#
# WHY. The single-seed sweep at 100k gave CICIoT clean MCC 0.000 at
# lambda=1000, 0.192 at 10000, and 0.052 at 30000, while CICIoMT rose
# gently across the whole range (0.265 -> 0.315). Those two shapes do
# not agree, so no scaling rule is supported. Worse, CICIoT's clean
# accuracy sat at 0.117 -- the collapsed, constant-prediction state --
# for three of the four lambdas, so the run is switching between two
# regimes rather than responding smoothly. With one seed per point we
# cannot tell a real optimum from which side of that switch a run
# happened to land on.
#
# WHAT THIS DECIDES. Not "which lambda is optimal", but "is there ONE
# lambda that works reliably across datasets and seeds". Only 10000 was
# decent on both, so this compares it against the 1000 default over five
# seeds. If 10000 holds around 0.15-0.20 on CICIoT it can be fixed for
# every dataset, which is the no-per-dataset-tuning stance the paper
# already takes elsewhere. If the spread is wide, then CICIoT at 100k is
# simply unstable, and that is the finding to report.
#
# HONEST SELECTION. Runs use --sel-size 0.15, carving a selection split
# out of TRAIN. Choose lambda on sel_mcc; report clean_mcc, which comes
# from a held-out partition that no choice here ever consulted. Note
# this makes training 15% smaller, so these numbers are NOT directly
# comparable to the earlier sweep -- compare within this script only.
#
# RESUMABLE. Re-run to continue; finished configs are skipped.
#
# Usage:
#   bash run_lambda_verify_r2.sh                     # both datasets
#   DATASETS="CICIoT" bash run_lambda_verify_r2.sh   # the volatile one only
#   SEEDS="1 7 13" bash run_lambda_verify_r2.sh      # faster, 3 seeds
set -uo pipefail
cd "$(dirname "$0")"

DIR="${DIR:-balanced_data_ss100k}"
LAMBDAS="${LAMBDAS:-1000 10000}"
DATASETS="${DATASETS:-CICIoT CICIoMT}"   # CICIoT first: it is the open question
SEEDS="${SEEDS:-1 7 13 21 42}"
EPOCHS="${EPOCHS:-10}"
SEL=0.15
FAILED=()

# The subset size is part of a run's identity. Without it in the path,
# the same dataset at a different size would land in the same run dir
# and its already-recorded seeds would be skipped as "done" -- silently
# producing nothing.
TAG="${DIR##*_}"          # balanced_data_ss30k -> ss30k

echo "verify: lambdas=[${LAMBDAS}] datasets=[${DATASETS}] seeds=[${SEEDS}]"
echo "        selection split ${SEL} of train; reporting on held-out"
echo

for DS in $DATASETS; do
  for LAM in $LAMBDAS; do
    echo "=== ${DS}, lambda=${LAM}, ${SEEDS} ==="
    python run_paper3_exp1.py \
      --balanced --balanced-dir "$DIR" --split-safe --sel-size "$SEL" \
      --datasets "$DS" --seeds $SEEDS --epochs "$EPOCHS" \
      --backbones transformer_cl --variant ewckd --ewc-lambda "$LAM" \
      --run-dir "runs_r2/lamverify_${TAG}_${DS}_lam${LAM}"
    [ $? -ne 0 ] && FAILED+=("${DS}/lam${LAM}")
  done
done

echo
echo "=== Verification: mean +/- SD over seeds ==="
python - <<'PY'
import glob, os, re
import pandas as pd

rows = []
for f in glob.glob("runs_r2/lamverify_*/paper3_exp1_master.csv"):
    d = pd.read_csv(f)
    d = d[d.get("status", "ok") == "ok"]
    d = d[d["AA"].notna()]
    if "ewc_lambda" not in d.columns or d["ewc_lambda"].isna().all():
        m = re.search(r"_lam(\d+)", os.path.dirname(f))
        d["ewc_lambda"] = int(m.group(1)) if m else 1000
    m = re.search(r"lamverify_(ss\d+k)_", os.path.dirname(f))
    d["size"] = m.group(1) if m else "?"
    rows.append(d)

if not rows:
    print("no results yet")
else:
    df = pd.concat(rows)
    df["ewc_lambda"] = pd.to_numeric(df["ewc_lambda"], errors="coerce")

    for metric, label in (("sel_mcc", "SELECTION split (choose on this)"),
                          ("clean_mcc", "HELD-OUT split (report this)"),
                          ("AA", "average accuracy, held-out")):
        if metric not in df.columns:
            continue
        g = (df.groupby(["size", "dataset", "ewc_lambda"])[metric]
               .agg(["mean", "std", "min", "max", "count"]).round(3))
        print(f"\n--- {metric}: {label}")
        print(g.to_string())

    # A collapsed run predicts one class; counting them says whether a
    # lambda is reliable or merely lucky on some seeds.
    if "clean_acc" in df.columns:
        df["collapsed"] = df["clean_mcc"].abs() < 0.01
        c = (df.groupby(["size", "dataset", "ewc_lambda"])["collapsed"]
               .agg(["sum", "count"]))
        print("\n--- collapsed runs (clean MCC ~ 0) per config")
        print(c.to_string())

    print("\nRead: a lambda is usable only if it is both high on average "
          "AND collapses on no seed. A high mean with several collapses "
          "is instability, not a working setting.")
PY

if [ ${#FAILED[@]} -gt 0 ]; then
  echo; echo "!! failed: ${FAILED[*]}"
  echo "!! Re-run 'bash run_lambda_verify_r2.sh' to retry only those."
fi
