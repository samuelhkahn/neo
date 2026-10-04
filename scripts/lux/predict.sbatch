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
# Super-resolve val pairs with one trained model, for the morphology comparison (compare.sbatch).
# Submit from the repo root of the branch that trained the model, e.g.
#   CONFIG=neo/configs/lux.ini MODEL=neo CKPT=$NEO_DATA/checkpoints/lux_cosmosweb_i/step_00500000.pt \
#     sbatch scripts/lux/predict.sbatch
# Protocol (no report data in model selection): predict each candidate checkpoint with SUBSET=select
# (the default) and compare them with compare.sbatch SUBSET=select; then predict only the chosen
# checkpoint with SUBSET=report. Use numbered step_*.pt checkpoints (latest.pt keeps changing).
# Output: $NEO_DATA/predictions/<MODEL>/<run>/<checkpoint>/<SPLIT>-<SUBSET> (printed below).
# Knobs: CONFIG, MODEL, CKPT (required), SUBSET (select|report|all), SPLIT (default val), LIMIT, BATCH.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1   # print() lines reach logs/ as they happen, not at exit
: "${NEO_DATA:?set NEO_DATA}"
source scripts/lux/comet_check.sh
: "${CONFIG:?set CONFIG to the training .ini of the model}"
: "${MODEL:?set MODEL to a short name, e.g. neo, swinir, diffusion}"
: "${CKPT:?set CKPT to the checkpoint to evaluate, e.g. \$NEO_DATA/checkpoints/<run>/step_00500000.pt}"
SPLIT="${SPLIT:-val}"
SUBSET="${SUBSET:-select}"
OUT="$NEO_DATA/predictions/$MODEL/$(basename "$(dirname "$CKPT")")/$(basename "$CKPT" .pt)/$SPLIT-$SUBSET"
echo "predictions -> $OUT"

nvidia-smi --query-gpu=name,memory.total --format=csv
uv run python -m neo.eval.predict \
  --config "$CONFIG" --checkpoint "$CKPT" \
  --split-dir "$NEO_DATA/pairs/cosmos_web_i/$SPLIT" \
  --subset "$SUBSET" --out "$OUT" \
  --batch-size "${BATCH:-4}" ${LIMIT:+--limit "$LIMIT"}
