#!/usr/bin/env bash
# run_scaletest_r2.sh — does more data per family lift the result?
#
# THE DECISION THIS ANSWERS. After the leakage fix, CAT reaches clean
# MCC 0.14-0.18 against a sequential-AT baseline at 0.00 on the two
# sound datasets. The difference is real but both are weak detectors,
# which is thin for a top-tier venue. The pilot showed the numbers
# climbing with subset size (CICIoT clean MCC 0.000 at 10k -> 0.142 at
# 30k). This script extends that curve to 100k on the same protocol.
#
#   Numbers keep climbing steeply  -> scaling up is the paper's best
#                                     move; a larger-scale run is worth
#                                     the compute and IoTJ stays viable.
#   Numbers flatten near 0.2       -> more data is not the constraint;
#                                     better to retarget the venue than
#                                     to spend weeks on compute.
#
# Same protocol as the pilot (1 seed, 10 epochs, split-safe) so the
# three sizes are directly comparable. Only the data volume changes.
#
# RESUMABLE: re-run after a crash and it continues where it stopped.
#
# Usage:
#   bash run_scaletest_r2.sh                 # CICIoT then CICIoMT
#   DATASETS="CICIoT" bash run_scaletest_r2.sh
set -uo pipefail
cd "$(dirname "$0")"

EPOCHS="${EPOCHS:-10}"      # matches the pilot, so sizes compare cleanly
SEED="${SEED:-1}"
DATASETS="${DATASETS:-CICIoT CICIoMT}"
DIR=balanced_data_ss100k
FAILED=()

echo "=== [1/3] Building the 100k split-safe subset ==="
# Rare families cap out at what the source actually holds, so the file
# lands well under 100k rows; that is the point, not a problem.
MISSING=0
for DS in $DATASETS; do
  [ -s "${DIR}/balanced_${DS}.csv" ] || MISSING=1
done
if [ "$MISSING" -eq 0 ]; then
  echo "  ${DIR} already has the datasets needed, skipping"
else
  echo "  building ${DIR} (reads the full source CSVs — several minutes)"
  python make_balanced_subset.py --total 100000 --no-oversample --out "$DIR" \
    || { echo "!! subset build failed"; exit 1; }
fi

echo
echo "=== [2/3] Runs at 100k: {Seq-AT, CAT} x {${DATASETS}}, seed ${SEED} ==="
for DS in $DATASETS; do
  for SPEC in "transformer noCL" "transformer_cl ewckd"; do
    set -- $SPEC
    BB="$1"; VAR="$2"
    echo "--- ${DS} @ 100k: ${BB} / ${VAR} ---"
    python run_paper3_exp1.py \
      --balanced --balanced-dir "$DIR" --split-safe \
      --datasets "$DS" --seeds "$SEED" --epochs "$EPOCHS" \
      --backbones "$BB" --variant "$VAR" \
      --run-dir "runs_r2/scale100k_${DS}"
    if [ $? -ne 0 ]; then
      echo "!! FAILED: ${DS}/${BB}/${VAR}"
      FAILED+=("${DS}/${BB}/${VAR}")
    fi
  done
done

echo
echo "=== [3/3] Scaling curve: 10k -> 30k -> 100k ==="
python - <<'PY'
import glob, os, re
import pandas as pd

rows = []
for f in glob.glob("runs_r2/pilot_*/paper3_exp1_master.csv") + \
         glob.glob("runs_r2/scale100k_*/paper3_exp1_master.csv"):
    d = pd.read_csv(f)
    base = os.path.basename(os.path.dirname(f))
    m = re.match(r"(?:pilot|scale)(\d+k)?_?(\w+)?", base)
    d["size"] = "100k" if base.startswith("scale") else base.split("_")[1]
    rows.append(d)

if not rows:
    print("no results yet")
else:
    df = pd.concat(rows)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    df = df[df["AA"].notna()].drop_duplicates(
        subset=["size", "dataset", "seed", "backbone", "variant"], keep="last")
    df["method"] = df["variant"].map({"noCL": "Seq-AT", "ewckd": "CAT"})
    order = {"10k": 0, "30k": 1, "100k": 2}
    df = df[df["size"].isin(order)]
    df["_o"] = df["size"].map(order)

    for metric in ("clean_mcc", "AA"):
        piv = (df.pivot_table(index=["dataset", "_o", "size"],
                              columns="method", values=metric)
                 .reset_index().sort_values(["dataset", "_o"]))
        if {"CAT", "Seq-AT"} <= set(piv.columns):
            piv["gap"] = piv["CAT"] - piv["Seq-AT"]
        print(f"\n--- {metric} ---")
        print(piv.drop(columns="_o").round(3).to_string(index=False))

    print("\nRead the gap column down each dataset. Still climbing at "
          "100k means data volume is the binding constraint; flat means "
          "it is not.")
PY

if [ ${#FAILED[@]} -gt 0 ]; then
  echo
  echo "!! ${#FAILED[@]} config(s) failed: ${FAILED[*]}"
  echo "!! Re-run 'bash run_scaletest_r2.sh' to retry only those."
fi
