#!/usr/bin/env bash
#SBATCH --job-name=neo-diffusion
#SBATCH --account=comp-astro
#SBATCH --qos=comp-astro
#SBATCH --partition=comp-astro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=14-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/diffusion-%j.out
# Diffusion SR experiment (train_diffusion.py), resumable exactly like train.sbatch: latest.pt every
# [CHECKPOINT] latest_every steps and --resume continues from it after a preemption or requeue.
# Lux: account, qos and partition must all be given (they share a name); --exclusive takes a whole
# comp-astro node (2x V100) and training uses GPU 0.
# Submit from the repo root:  sbatch scripts/lux/train_diffusion.sbatch     (CONFIG=... for another ini)
# Comet: COMET_ML_ASTRO_API_KEY must be exported when submitting (scripts/lux/comet_check.sh).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1   # print() lines reach logs/ as they happen, not at exit
: "${NEO_DATA:?set NEO_DATA}"
CONFIG="${CONFIG:-neo/configs/lux_diffusion.ini}"
source scripts/lux/job_tmp.sh
source scripts/lux/comet_check.sh
# refuse to resume checkpoints trained on another pairs build (neo/preprocess/manifest.py)
uv run python -m neo.preprocess.manifest guard "$CONFIG"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uv run python train_diffusion.py "$CONFIG" --resume
