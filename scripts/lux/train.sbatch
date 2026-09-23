#!/usr/bin/env bash
#SBATCH --job-name=neo-train
#SBATCH --account=comp-astro
#SBATCH --qos=comp-astro
#SBATCH --partition=comp-astro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=14-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/train-%j.out
# Long training run, resumable: train.py writes <ckpt_dir>/latest.pt every [CHECKPOINT] latest_every
# steps and --resume continues from it, so a preempted or requeued job picks up where it stopped.
# Lux: account, qos and partition must all be given (they share a name). No GPU GRES is configured, so
# --exclusive takes a whole comp-astro node (2x V100); training uses GPU 0, leaving GPU 1 free for a
# second config in the same job (CUDA_VISIBLE_DEVICES=1) if you want an ablation alongside.
# Submit from the repo root:  sbatch scripts/lux/train.sbatch        (CONFIG=... to use another ini)
# COMET_ML_ASTRO_API_KEY should be exported in ~/.bashrc; without it Comet logs offline.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
: "${NEO_DATA:?set NEO_DATA}"
CONFIG="${CONFIG:-neo/configs/lux.ini}"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uv run python train.py "$CONFIG" --resume
