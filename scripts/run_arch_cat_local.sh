#!/usr/bin/env bash
# run_arch_cat_local.sh — architecture-agnosticism, but for the method
# the paper actually proposes.
#
# THE GAP THIS FILLS. Stage 2 of the main grid compared five backbones,
# but every one of them ran the `ewckd` variant (EWC+KD, no replay) --
# the configuration that turned out to be weak (clean MCC ~0.15 against
# `cat`'s ~0.67). So the existing architecture result is clean but is
# about the wrong method, and a reviewer will ask whether the PROPOSED
# method is architecture-agnostic or only its weak variant is.
#
# SCOPE. The claim is about three model families: multilayer perceptron,
# recurrent, and attention. The attention arm already exists -- `cat` on
# transformer_cl in runs_vast_csv/full_100k/s6_method. Only DNN and LSTM
# are missing, so only those run here. CICIoT is skipped: it is bimodal
# across seeds and carries no claim, and skipping it halves the cost.
#
# COST. ~20 runs at roughly 75 min each on an RTX 4050 laptop, so budget
# about a day. On a rented A100 the same work is well under an hour;
# this script exists for when renting is not convenient.
#
# RESUMABLE. Re-run after a crash or a Ctrl-C and it continues; finished
# configs are skipped.
#
# Usage:
#   bash run_arch_cat_local.sh
#   SEEDS="1 7 13" bash run_arch_cat_local.sh     # faster, wider CIs
set -uo pipefail
cd "$(dirname "$0")"

DIR="${DIR:-balanced_data_ss100k}"
DATASETS="${DATASETS:-CICIoMT TONIoT}"
SEEDS="${SEEDS:-1 7 13 21 42}"
EPOCHS="${EPOCHS:-20}"
SEL=0.15
FAILED=()

for DS in $DATASETS; do
  [ -s "${DIR}/balanced_${DS}.csv" ] || { echo "!! missing ${DIR}/balanced_${DS}.csv"; exit 1; }
done

echo "arch study for the proposed method: cat on dnn + lstm"
echo "datasets=[${DATASETS}] seeds=[${SEEDS}] epochs=${EPOCHS}"
echo

for BB in dnn lstm; do
  echo "=== ${BB} / cat ==="
  python run_paper3_exp1.py \
    --balanced --balanced-dir "$DIR" --split-safe --sel-size "$SEL" \
    --datasets $DATASETS --seeds $SEEDS --epochs "$EPOCHS" \
    --backbones "$BB" --variant cat \
    --run-dir "runs_r2/arch_cat"
  [ $? -ne 0 ] && FAILED+=("$BB")
done

echo
echo "=== cat across model families ==="
python - <<'PY'
import glob
import pandas as pd

rows = []
for f in (glob.glob("runs_r2/arch_cat/paper3_exp1_master.csv")
          + glob.glob("runs_vast_csv/full_100k/s6_method/paper3_exp1_master.csv")):
    d = pd.read_csv(f)
    d = d[(d.get("status", "ok") == "ok") & (d["variant"] == "cat")]
    rows.append(d)

if not rows:
    print("no results yet")
else:
    df = pd.concat(rows)
    df = df[df["AA"].notna()]
    df["rMCC"] = df[["fgsm_mcc", "pgd_mcc", "square_mcc",
                     "transfer_mcc"]].mean(axis=1)
    fam = {"dnn": "MLP", "lstm": "recurrent", "transformer_cl": "attention"}
    df["family"] = df["backbone"].map(fam).fillna(df["backbone"])
    keep = df["dataset"].isin(["CICIoMT", "TONIoT"])
    g = (df[keep].groupby(["family", "dataset"])[["clean_mcc", "rMCC"]]
           .agg(["mean", "std"]).round(3))
    print(g.to_string())
    spread = (df[keep].groupby("family")["clean_mcc"].mean())
    print(f"\ncross-family spread in clean MCC: "
          f"{spread.max() - spread.min():.3f}")
    print("A small spread supports architecture-agnosticism FOR THE "
          "PROPOSED METHOD, which is what Stage 2 could not show.")
PY

if [ ${#FAILED[@]} -gt 0 ]; then
  echo; echo "!! failed: ${FAILED[*]} — re-run to retry"
fi
