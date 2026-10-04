#!/usr/bin/env bash
#SBATCH --job-name=neo-compare
#SBATCH --account=cpuq
#SBATCH --qos=cpuq
#SBATCH --partition=cpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/compare-%j.out
# NEO-paper morphology comparison (Table 4 statistics, gains vs LR, model-vs-model) of every model
# predicted with predict.sbatch. CPU only; run from any experiment branch (all share neo/eval).
#   MODELS="neo swinir diffusion" SUBSET=report sbatch scripts/lux/compare.sbatch
# SUBSET=select scores the 20% of pairs reserved for checkpoint selection; report scores the rest.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1   # print() lines reach logs/ as they happen, not at exit
: "${NEO_DATA:?set NEO_DATA}"
source scripts/lux/comet_check.sh
SPLIT="${SPLIT:-val}"
SUBSET="${SUBSET:-report}"

PRED_ARGS=()
for m in ${MODELS:-neo swinir diffusion}; do
  PRED_ARGS+=(--pred "$m=$NEO_DATA/predictions/$m/$SPLIT")
done
uv run python -m neo.eval.compare \
  --split-dir "$NEO_DATA/pairs/cosmos_web_i/$SPLIT" "${PRED_ARGS[@]}" \
  --subset "$SUBSET" --workers "${SLURM_CPUS_PER_TASK:-8}" \
  --out "$NEO_DATA/comparison/$SPLIT-$SUBSET" ${LIMIT:+--limit "$LIMIT"}
