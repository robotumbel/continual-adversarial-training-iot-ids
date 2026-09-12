#!/usr/bin/env bash
# deploy_vast.sh — put this project on a rented Vast.ai instance and run it.
#
# WHAT YOU DO: rent the instance on vast.ai (that is billing on your own
# account, so it has to be you) and copy the SSH line Vast shows on the
# instance card. It looks like:
#     ssh -p 41234 root@ssh5.vast.ai -L 8080:localhost:8080
# You only need the port and the host from it.
#
# WHAT THIS DOES: bundles code + the 100k subsets, uploads them, installs
# what the NGC PyTorch image is missing, checks the GPU is actually
# visible, and then either benchmarks one config or launches a phase
# under nohup so it survives your laptop disconnecting.
#
# Usage:
#   bash deploy_vast.sh <port> <host> setup       # upload + install + check
#   bash deploy_vast.sh <port> <host> bench       # time ONE config, ~30-60 min
#   bash deploy_vast.sh <port> <host> phaseA      # stage 1, 3 datasets
#   bash deploy_vast.sh <port> <host> phaseB      # stages 2-5, 2 datasets
#   bash deploy_vast.sh <port> <host> status      # progress so far
#   bash deploy_vast.sh <port> <host> fetch       # pull results back here
#
# Run `bench` BEFORE committing to a phase. It measures the real speedup
# on that specific machine. A CICIoMT config at 20 epochs takes 47 min on
# the RTX 4050 this was developed on; if the rented box is not clearly
# faster, its CPU is the bottleneck (this pipeline does heavy numpy work
# in the constraint projection) and a different listing is the fix.
set -uo pipefail
cd "$(dirname "$0")"

PORT="${1:?usage: deploy_vast.sh <port> <host> <command>}"
HOST="${2:?usage: deploy_vast.sh <port> <host> <command>}"
CMD="${3:?usage: deploy_vast.sh <port> <host> <command>}"

SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
REMOTE="root@$HOST"
RDIR="/workspace/aamtrans"
BUNDLE="/tmp/aamtrans_bundle.tar.gz"

case "$CMD" in

setup)
  echo "=== bundling code + data ==="
  tar czf "$BUNDLE" \
    --exclude='__pycache__' --exclude='*.pyc' \
    run_paper3_exp1.py run_paper3_exp2.py data_loader.py config.py \
    evaluation.py continual_adversarial.py incremental.py \
    incremental_metrics.py adversarial_training.py task_splitter.py \
    losses.py imbalance.py checkpoint.py detection_metrics.py \
    statistical_eval.py adversarial_metrics.py gate_detector.py \
    training.py analyze_r2.py benchmark_complexity.py audit_leakage.py \
    requirements.txt models attacks \
    run_full_r2.sh run_lambda_verify_r2.sh \
    balanced_data_ss100k
  echo "bundle: $(du -h "$BUNDLE" | cut -f1)"

  echo "=== uploading ==="
  $SSH "$REMOTE" "mkdir -p $RDIR"
  scp -P "$PORT" -o StrictHostKeyChecking=accept-new "$BUNDLE" "$REMOTE:$RDIR/"
  $SSH "$REMOTE" "cd $RDIR && tar xzf $(basename $BUNDLE) && rm $(basename $BUNDLE)"

  # These scripts are authored on Windows, so they carry CRLF endings.
  # Linux bash reads the trailing \r as part of the command and dies with
  # errors like "cd: $'.\r': No such file or directory".
  echo "=== normalising line endings ==="
  $SSH "$REMOTE" "cd $RDIR && sed -i 's/\r\$//' *.sh && echo 'shell scripts converted to LF'"

  echo "=== installing what the image lacks ==="
  # The NGC PyTorch image already has torch+CUDA; do not reinstall it.
  $SSH "$REMOTE" "pip install -q scikit-learn pandas scipy imbalanced-learn 2>&1 | tail -2"

  echo "=== environment check ==="
  $SSH "$REMOTE" "cd $RDIR && python -c \"
import torch, sklearn, pandas, scipy, os
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
print('gpu  ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
print('vcpu ', os.cpu_count(), '(<8 means the CPU-bound parts will drag)')
import glob; print('data ', len(glob.glob('balanced_data_ss100k/*.csv')), 'subset files')
\""
  echo
  echo "Setup done. Next: bash deploy_vast.sh $PORT $HOST bench"
  ;;

bench)
  echo "=== timing ONE CICIoMT config, 5 seeds, 20 epochs ==="
  echo "    (47 min per seed on the RTX 4050 baseline)"
  $SSH "$REMOTE" "cd $RDIR && rm -rf runs_r2/_bench && \
    /usr/bin/time -f 'TOTAL %e s' python run_paper3_exp1.py \
      --balanced --balanced-dir balanced_data_ss100k --split-safe \
      --sel-size 0.15 --datasets CICIoMT --seeds 1 --epochs 20 \
      --backbones transformer_cl --variant ewckd \
      --run-dir runs_r2/_bench 2>&1 | tail -5"
  echo
  echo "Compare against 47 min. 3x or better -> commit to the phases."
  ;;

phaseA|phaseB)
  if [ "$CMD" = phaseA ]; then
    ENVS='SIZE=100k DATASETS="CICIoMT TONIoT CICIoT" STAGES="1"'
    LOG=phaseA.log
  else
    # All three datasets: on an A100 the grid runs ~12x faster than the
    # laptop it was budgeted for, so the reason CICIoT was cut from these
    # stages (it was 53% of a 200-hour estimate) no longer applies. Having
    # the ablation on every dataset also answers, in advance, why a
    # mechanism study would cover fewer datasets than the main result.
    ENVS='SIZE=100k DATASETS="CICIoMT TONIoT CICIoT" STAGES="2 3 4 5"'
    LOG=phaseB.log
  fi
  echo "=== launching $CMD under nohup (survives disconnection) ==="
  # setsid plus closing stdin/stdout/stderr fully detaches the job. Without
  # that, ssh holds the session open waiting on the inherited descriptors
  # and this command appears to hang for as long as the job runs.
  $SSH "$REMOTE" "cd $RDIR && setsid env $ENVS nohup bash run_full_r2.sh \
      > $LOG 2>&1 < /dev/null & disown; sleep 5; echo launched"
  sleep 2
  $SSH "$REMOTE" "cd $RDIR && tail -3 $LOG"
  echo
  echo "Monitor:  bash deploy_vast.sh $PORT $HOST status"
  ;;

arch)
  # Architecture-agnosticism for the method the paper actually proposes.
  # The main grid's Stage 2 compared these same five backbones but ran
  # `ewckd` on all of them -- the weak variant (clean MCC ~0.15 vs
  # `cat`'s ~0.67), so it cannot support a claim about the proposed
  # method. Same backbones, same datasets, `cat` instead. CICIoT is
  # excluded: bimodal across seeds, carries no claim, and doubles cost.
  echo "=== launching arch study: cat on 5 backbones ==="
  $SSH "$REMOTE" "cd $RDIR && cat > runarch.sh <<'EOS'
#!/usr/bin/env bash
set -uo pipefail
cd /workspace/aamtrans
for BB in dnn lstm rdt aam_trans aam_trans_no_gates; do
  echo \"=== \\$BB / cat ===\"
  python run_paper3_exp1.py \\
    --balanced --balanced-dir balanced_data_ss100k --split-safe --sel-size 0.15 \\
    --datasets CICIoMT TONIoT --seeds 1 7 13 21 42 --epochs 20 \\
    --backbones \\$BB --variant cat \\
    --run-dir runs_r2/full_100k/s7_arch_cat
done
echo '=== arch study complete ==='
EOS
    sed -i 's/\r\$//' runarch.sh
    setsid env nohup bash runarch.sh > arch.log 2>&1 < /dev/null & disown
    sleep 5; echo launched"
  sleep 2
  $SSH "$REMOTE" "cd $RDIR && tail -3 arch.log"
  echo
  echo "Monitor:  bash deploy_vast.sh $PORT $HOST status"
  ;;

status)
  $SSH "$REMOTE" "cd $RDIR && \
    echo '--- running? ---'; pgrep -af run_paper3_exp1.py | head -2 || echo 'no run in progress'; \
    echo '--- configs finished ---'; \
    for f in runs_r2/full_100k/*/paper3_exp1_master.csv; do \
      [ -f \"\$f\" ] && echo \"\$(( \$(grep -c ',ok\$' \"\$f\") )) ok  \$f\"; done; \
    echo '--- last log lines ---'; tail -4 phase*.log 2>/dev/null"
  ;;

fetch)
  echo "=== pulling results back ==="
  mkdir -p runs_vast
  scp -P "$PORT" -o StrictHostKeyChecking=accept-new -r \
    "$REMOTE:$RDIR/runs_r2/full_100k" runs_vast/ 2>/dev/null
  scp -P "$PORT" -o StrictHostKeyChecking=accept-new \
    "$REMOTE:$RDIR/phase*.log" runs_vast/ 2>/dev/null
  echo "landed in runs_vast/. Analyse with:"
  echo "  python analyze_r2.py --runs \"runs_vast/full_100k/**/paper3_exp1_master.csv\" --latex"
  ;;

*)
  echo "unknown command: $CMD (setup|bench|phaseA|phaseB|status|fetch)"
  exit 1
  ;;
esac
