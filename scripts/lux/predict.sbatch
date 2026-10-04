#!/usr/bin/env bash
#SBATCH --job-name=neo-predict
#SBATCH --account=comp-astro
#SBATCH --qos=comp-astro
#SBATCH --partition=comp-astro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/predict-%j.out
# Super-resolve a pairs split with one trained model, for the morphology comparison (compare.sbatch).
# Submit from the repo root of the branch that trained the model, e.g.
#   CONFIG=neo/configs/lux.ini MODEL=neo sbatch scripts/lux/predict.sbatch
# Knobs: CONFIG (training .ini), MODEL (output name), CKPT (required: the selected checkpoint,
# e.g. step_00500000.pt), SPLIT (default val), LIMIT, BATCH.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
: "${NEO_DATA:?set NEO_DATA}"
: "${CONFIG:?set CONFIG to the training .ini of the model}"
: "${MODEL:?set MODEL to a short name, e.g. neo, swinir, diffusion}"
: "${CKPT:?set CKPT to the checkpoint to evaluate, e.g. \$NEO_DATA/checkpoints/<run>/step_00500000.pt}"
SPLIT="${SPLIT:-val}"

nvidia-smi --query-gpu=name,memory.total --format=csv
uv run python -m neo.eval.predict \
  --config "$CONFIG" --checkpoint "$CKPT" \
  --split-dir "$NEO_DATA/pairs/cosmos_web_i/$SPLIT" \
  --out "$NEO_DATA/predictions/$MODEL/$SPLIT" \
  --batch-size "${BATCH:-4}" ${LIMIT:+--limit "$LIMIT"}
