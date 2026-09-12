#!/usr/bin/env bash
# run_full_r2.sh — the corrected full grid for the resubmission.
#
# RESUMABLE. Each stage writes into a fixed --run-dir and
# run_paper3_exp1.py skips configs already recorded there with status
# ok, so re-running this script after a crash, an out-of-memory kill,
# or a Ctrl-C continues where it stopped. Configs that FAILED are
# retried rather than skipped, and one failing config does not abort
# the rest of the grid. Nothing needs deleting between attempts.
#
# ORDERED BY PRIORITY. The stages run most-important first, so if the
# machine is needed for something else after a few hours, what has
# finished is still a coherent result set:
#   Stage 1  primary claim: CAT vs sequential AT on a fixed backbone
#   Stage 2  architecture-agnosticism across model families
#   Stage 3  factorial mechanism ablation (what Reviewer 1 asked for)
#   Stage 4  no-adversarial-loop control
# Stage 1 alone supports the paper's two primary hypotheses.
#
# CAT is variant `ewckd` (EWC + KD + class-balanced focal, NO replay),
# matching Eq. (3) and the rehearsal-free claim. Every condition,
# sequential AT included, uses the identical class-balanced focal loss
# and the identical PGD-TRADES budget, so contrasts isolate the
# mechanism and not the loss function.
#
# Usage:
#   bash run_full_r2.sh                              # everything, 100k
#   STAGES="1" bash run_full_r2.sh                   # primary contrast only
#   DATASETS="CICIoMT TONIoT" bash run_full_r2.sh    # the two stable ones
#   SIZE=30k SEEDS="1 7 13" bash run_full_r2.sh      # smaller/faster
set -uo pipefail          # deliberately NOT -e: see run_cfg below
cd "$(dirname "$0")"

SIZE="${SIZE:-100k}"
SEEDS="${SEEDS:-1 7 13 21 42}"
EPOCHS="${EPOCHS:-20}"
# Settled by the five-seed verification in runs_r2/lamverify_*:
#   CICIoMT  clean MCC 0.270 +/- 0.011, CV 0.04 -- stable, carries claims
#   TONIoT   clean MCC 0.149 +/- 0.007, CV 0.05 -- stable, and non-CIC, so
#                                                  it carries generality
#   CICIoT   CV 0.74-0.87 at BOTH 30k and 100k  -- bimodal across seeds
#                                                  (a run either retains or
#                                                  collapses); reported per
#                                                  seed, carries no claim
#   CICIoV   99.7% duplicate records            -- excluded as degenerate
DATASETS="${DATASETS:-CICIoMT TONIoT CICIoT}"
# Which stages to run, e.g. STAGES="1" for the primary contrast only.
# Useful for CICIoV, which carries no claim and is worth running for
# stage 1 alone rather than across the whole grid.
STAGES="${STAGES:-1 2 3 4 5}"
# EWC lambda stays at its 1000 default, confirmed over five seeds on both
# stable datasets: raising it to 10000 cut CICIoMT's clean MCC by a third
# and multiplied its spread elevenfold. No per-dataset tuning is needed,
# which is the stance the paper already takes. The selection split still
# carries any choice, so the reported partition stays untouched by one.
SEL="${SEL:-0.15}"
DIR="balanced_data_ss${SIZE}"
FAILED=()

want_stage () { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

if [ ! -s "${DIR}/balanced_CICIoT.csv" ]; then
  echo "!! ${DIR} missing. Build it first:"
  echo "   python make_balanced_subset.py --total ${SIZE%k}000 \\"
  echo "          --no-oversample --out ${DIR}"
  exit 1
fi

echo "grid: size=${SIZE} seeds=[${SEEDS}] epochs=${EPOCHS} stages=[${STAGES}]"
echo "      datasets=[${DATASETS}]"
echo

run_cfg () {   # run_cfg <stage> <backbone> <variant>
  local STAGE="$1" BB="$2" VAR="$3"
  echo "=== [${STAGE}] ${BB} / ${VAR} ==="
  python run_paper3_exp1.py \
    --balanced --balanced-dir "$DIR" --split-safe --sel-size "$SEL" \
    --datasets $DATASETS --seeds $SEEDS --epochs "$EPOCHS" \
    --backbones "$BB" --variant "$VAR" \
    --run-dir "runs_r2/full_${SIZE}/${STAGE}"
  if [ $? -ne 0 ]; then
    echo "!! FAILED: ${STAGE}/${BB}/${VAR} (re-run this script to retry)"
    FAILED+=("${STAGE}/${BB}/${VAR}")
  fi
}

# --- Stage 1: the primary contrast -----------------------------------
# Same architecture, same loss, same adversarial budget; the only
# difference is the continual loop. This is the comparison the paper's
# two pre-specified hypotheses rest on.
if want_stage 1; then
run_cfg s1_primary transformer    noCL     # sequential AT
run_cfg s1_primary transformer_cl ewckd    # CAT
fi

# --- Stage 2: architecture-agnosticism -------------------------------
if want_stage 2; then
for BB in dnn lstm rdt aam_trans aam_trans_no_gates; do
  run_cfg s2_arch "$BB" ewckd
done
for BB in dnn lstm; do
  run_cfg s2_arch "$BB" noCL              # their own sequential-AT controls
done
fi

# --- Stage 3: factorial mechanism ablation ---------------------------
# Each mechanism added to the SAME baseline as Stage 1's noCL rung.
if want_stage 3; then
for VAR in ewc kd replay; do
  run_cfg s3_ablation transformer_cl "$VAR"
done
fi

# --- Stage 4: is the adversarial inner loop doing anything? ----------
# Same continual schedule with TRADES switched off.
if want_stage 4; then
echo "=== [s4_noat] transformer_cl / ewckd, beta=0 ==="
python run_paper3_exp1.py \
  --balanced --balanced-dir "$DIR" --split-safe --sel-size "$SEL" \
  --datasets $DATASETS --seeds $SEEDS --epochs "$EPOCHS" \
  --backbones transformer_cl --variant ewckd --beta 0.0 \
  --run-dir "runs_r2/full_${SIZE}/s4_noat"
[ $? -ne 0 ] && FAILED+=("s4_noat/transformer_cl/ewckd")
fi

# --- Stage 5: rehearsal-based references for the SOTA table ----------
# BiC and our robust variant of it. These are the comparison the paper's
# rehearsal-free claim is measured against, so they run over the same
# four datasets as that table, TON_IoT included. Skipped automatically
# if the TON_IoT subset is absent.
if want_stage 5; then
S5_DATASETS="$DATASETS"
if [ -s "${DIR}/balanced_TONIoT.csv" ]; then
  case " $S5_DATASETS " in *" TONIoT "*) ;; *) S5_DATASETS="$S5_DATASETS TONIoT";; esac
else
  echo "note: ${DIR}/balanced_TONIoT.csv absent — stage 5 runs without it"
fi

echo "=== [s5_sota] BiC (transformer_cl / bic) ==="
python run_paper3_exp1.py \
  --balanced --balanced-dir "$DIR" --split-safe --sel-size "$SEL" \
  --datasets $S5_DATASETS --seeds $SEEDS --epochs "$EPOCHS" \
  --backbones transformer_cl --variant bic --bias-correct \
  --run-dir "runs_r2/full_${SIZE}/s5_sota_bic"
[ $? -ne 0 ] && FAILED+=("s5_sota/bic")

echo "=== [s5_sota] RBC, robust bias correction w=4 ==="
python run_paper3_exp1.py \
  --balanced --balanced-dir "$DIR" --split-safe --sel-size "$SEL" \
  --datasets $S5_DATASETS --seeds $SEEDS --epochs "$EPOCHS" \
  --backbones transformer_cl --variant bic --bias-correct \
  --robust-bc --bc-adv-w 4 \
  --run-dir "runs_r2/full_${SIZE}/s5_sota_rbc"
[ $? -ne 0 ] && FAILED+=("s5_sota/rbc")

# (Stage 1 covers TON_IoT directly now: it is a primary dataset.)
fi

# ---------------------------------------------------------------------
echo
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "!! ${#FAILED[@]} config(s) failed:"
  printf '   %s\n' "${FAILED[@]}"
  echo "!! Re-run 'bash run_full_r2.sh' to retry only those."
else
  echo "=== grid complete ==="
fi

echo
echo "Next: statistics for the revised protocol"
echo "  python analyze_r2.py --runs \"runs_r2/full_${SIZE}/**/paper3_exp1_master.csv\" --latex"
