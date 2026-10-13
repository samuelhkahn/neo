#!/usr/bin/env bash
#SBATCH --job-name=neo-train-pair
#SBATCH --account=comp-astro
#SBATCH --qos=comp-astro
#SBATCH --partition=comp-astro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=21-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/train-pair-%j.out
# Two NEO training runs side by side on one comp-astro node (2x V100), one config per GPU: by
# default the generator ablation arms, lux_neo_relu.ini (Arm A) on GPU 0 and lux_neo_lrtrunk.ini
# (Arm B) on GPU 1, each to 2M steps like the parity run (lux.ini, train.sbatch). The parity run
# makes ~5,400 steps/h on a V100 (all 5 VGG blocks), i.e. ~15.4 days for 2M steps; the arms run
# only VGG blocks 1-3 and are estimated at ~7-11 days (Arm A) and ~9-13 days (Arm B), hence 21 days
# (comp-astro's MaxTime is unlimited). train.sbatch's 14 days stop the parity control near 1.8M
# steps, and a time-limit stop is not requeued: resubmit it to reach 2M (it resumes from latest.pt):
#   sbatch --time=21-00:00:00 scripts/lux/train.sbatch
# Each arm resumes exactly like train.sbatch (latest.pt every [CHECKPOINT] latest_every steps,
# --resume) and writes its own log, logs/train-<config>-<job id>.out (appended to, so a requeued
# job keeps the earlier segments); this file has only the job's own output. Within the job, an arm
# whose train.py fails is rerun with --resume RETRY_PAUSE seconds later (default 60), up to RETRIES
# times in a row (default 5); an attempt that ran at least RETRY_RESET seconds (default 3600)
# starts the count again (scripts/lux/retry.sh). A transient crash such as the DataLoader
# shared-memory bus error that ended a diffusion arm at 600k then costs at most latest_every steps,
# not the run. Each config has its own DataLoader workers (lux.ini: 8 train + 8 persistent val, so
# 32 worker processes on the node); the node has 40 cores and 192 GB.
# Lux: account, qos and partition must all be given (they share a name).
# Submit from the repo root:  sbatch scripts/lux/train_pair.sbatch
#   (CONFIG_A=... CONFIG_B=... for other configs, RETRIES, RETRY_PAUSE, RETRY_RESET for the reruns)
# Comet: COMET_ML_ASTRO_API_KEY must be exported when submitting (scripts/lux/comet_check.sh).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1   # print() lines reach logs/ as they happen, not at exit
: "${NEO_DATA:?set NEO_DATA}"
CONFIG_A="${CONFIG_A:-neo/configs/lux_neo_relu.ini}"
CONFIG_B="${CONFIG_B:-neo/configs/lux_neo_lrtrunk.ini}"
RETRIES="${RETRIES:-5}"
RETRY_PAUSE="${RETRY_PAUSE:-60}"
export RETRY_RESET="${RETRY_RESET:-3600}"   # read by run_with_retries
if [[ "$CONFIG_A" == "$CONFIG_B" ]]; then
  echo "CONFIG_A and CONFIG_B are the same file ($CONFIG_A): both runs would share one ckpt_dir" >&2
  exit 1
fi
source scripts/lux/job_tmp.sh
# job_tmp.sh's EXIT trap plus `|| true`: scancel signals every process of the job, so the cleaner
# may already be gone, and under set -e the failed kill would end the trap before rm (leaving
# TMPDIR behind and turning exit 143 into 1). Set here so job_tmp.sh stays the same on all branches.
trap 'kill "$JOB_TMP_CLEANER" 2>/dev/null || true; rm -rf "$TMPDIR"' EXIT
source scripts/lux/comet_check.sh
source scripts/lux/retry.sh
# refuse to resume checkpoints trained on another pairs build (neo/preprocess/manifest.py); both
# are checked before either run starts
uv run python -m neo.preprocess.manifest guard "$CONFIG_A"
uv run python -m neo.preprocess.manifest guard "$CONFIG_B"

nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
# train.py falls back to the CPU without a GPU, so check that both are there
if (( $(nvidia-smi -L | wc -l) < 2 )); then
  echo "need 2 GPUs on this node, found: $(nvidia-smi -L)" >&2
  exit 1
fi
LOG_A="logs/train-$(basename "$CONFIG_A" .ini)-${SLURM_JOB_ID}.out"
LOG_B="logs/train-$(basename "$CONFIG_B" .ini)-${SLURM_JOB_ID}.out"
echo "GPU 0: $CONFIG_A -> $LOG_A"
echo "GPU 1: $CONFIG_B -> $LOG_B"

# On scancel or preemption stop both arms: each loop stops its own train.py and does not rerun it.
# Replaces job_tmp.sh's TERM trap; the EXIT trap set above removes TMPDIR after both have exited.
PID_A=""
PID_B=""
stop_arms() {
  kill -TERM ${PID_A:+"$PID_A"} ${PID_B:+"$PID_B"} 2>/dev/null || true
  wait ${PID_A:+"$PID_A"} ${PID_B:+"$PID_B"} 2>/dev/null || true
  exit 143
}
trap stop_arms TERM INT

(
  export CUDA_VISIBLE_DEVICES=0
  run_with_retries "$LOG_A" "$RETRIES" "$RETRY_PAUSE" uv run python train.py "$CONFIG_A" --resume
) &
PID_A=$!
(
  export CUDA_VISIBLE_DEVICES=1
  run_with_retries "$LOG_B" "$RETRIES" "$RETRY_PAUSE" uv run python train.py "$CONFIG_B" --resume
) &
PID_B=$!

# Wait for both even if one fails: under set -e a failing bare `wait` would end the script (and
# the job, killing the other run) at once, so the statuses are collected with `||`
STATUS_A=0
STATUS_B=0
wait "$PID_A" || STATUS_A=$?
wait "$PID_B" || STATUS_B=$?
echo "exit status: $CONFIG_A $STATUS_A, $CONFIG_B $STATUS_B"
if (( STATUS_A != 0 || STATUS_B != 0 )); then
  exit 1
fi
