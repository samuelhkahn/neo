#!/usr/bin/env bash
#SBATCH --job-name=neo-diffusion-pair
#SBATCH --account=comp-astro
#SBATCH --qos=comp-astro
#SBATCH --partition=comp-astro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=14-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/diffusion-pair-%j.out
# Two diffusion runs side by side on one comp-astro node (2x V100), one config per GPU: by default
# the two source-focus arms, lux_diffusion_lrfocus.ini on GPU 0 and lux_diffusion_hstfocus.ini on
# GPU 1. Each is resumable exactly like train_diffusion.sbatch (latest.pt every [CHECKPOINT]
# latest_every steps, --resume) and writes its own log, logs/diffusion-<config>-<job id>.out
# (appended to, so a requeued job keeps the earlier segments); this file only has the job's own
# output. Each config uses 8 DataLoader workers; the node has 40 cores and 192 GB.
# Lux: account, qos and partition must all be given (they share a name).
# Submit from the repo root:  sbatch scripts/lux/train_diffusion_pair.sbatch
#   (CONFIG_A=... CONFIG_B=... for other configs)
# Comet: COMET_ML_ASTRO_API_KEY must be exported when submitting (scripts/lux/comet_check.sh).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1   # print() lines reach logs/ as they happen, not at exit
: "${NEO_DATA:?set NEO_DATA}"
CONFIG_A="${CONFIG_A:-neo/configs/lux_diffusion_lrfocus.ini}"
CONFIG_B="${CONFIG_B:-neo/configs/lux_diffusion_hstfocus.ini}"
source scripts/lux/job_tmp.sh
source scripts/lux/comet_check.sh
# refuse to resume checkpoints trained on another pairs build (neo/preprocess/manifest.py); both
# are checked before either run starts
uv run python -m neo.preprocess.manifest guard "$CONFIG_A"
uv run python -m neo.preprocess.manifest guard "$CONFIG_B"

nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
# train_diffusion.py falls back to the CPU without a GPU, so check that both are there
if (( $(nvidia-smi -L | wc -l) < 2 )); then
  echo "need 2 GPUs on this node, found: $(nvidia-smi -L)" >&2
  exit 1
fi
LOG_A="logs/diffusion-$(basename "$CONFIG_A" .ini)-${SLURM_JOB_ID}.out"
LOG_B="logs/diffusion-$(basename "$CONFIG_B" .ini)-${SLURM_JOB_ID}.out"
echo "GPU 0: $CONFIG_A -> $LOG_A"
echo "GPU 1: $CONFIG_B -> $LOG_B"
CUDA_VISIBLE_DEVICES=0 uv run python train_diffusion.py "$CONFIG_A" --resume >>"$LOG_A" 2>&1 &
PID_A=$!
CUDA_VISIBLE_DEVICES=1 uv run python train_diffusion.py "$CONFIG_B" --resume >>"$LOG_B" 2>&1 &
PID_B=$!
# On scancel or preemption also stop both runs (replaces job_tmp.sh's TERM trap; its EXIT trap
# still removes TMPDIR)
trap 'kill "$PID_A" "$PID_B" 2>/dev/null; exit 143' TERM INT

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
