#!/usr/bin/env bash
#SBATCH --job-name=neo-smoke
#SBATCH --account=comp-astro
#SBATCH --qos=comp-astro
#SBATCH --partition=comp-astro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke-%j.out
# 12-step end-to-end check on one GPU: env, CUDA, data loading, losses, Comet. Run before train.sbatch.
# Lux: account, qos and partition must all be given (they share a name). No GPU GRES is configured, so
# --exclusive takes a whole comp-astro node (2x V100) and training uses GPU 0.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
: "${NEO_DATA:?set NEO_DATA}"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uv run python -c "import torch; print('torch', torch.__version__, torch.version.cuda, '|', torch.cuda.get_device_name(0), 'sm', torch.cuda.get_device_capability(0))"
uv run python train.py neo/configs/lux_smoke.ini
