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
# NEO-paper morphology comparison (Table 4 statistics, gains vs LR, model-vs-model) of models
# predicted with predict.sbatch. CPU only; run from any experiment branch (all share neo/eval).
#   PREDS="neo=<dir> swinir=<dir> diffusion=<dir>" SUBSET=report sbatch scripts/lux/compare.sbatch
# with the directories predict.sbatch printed. SUBSET=select compares candidate checkpoints on the
# 20% of val reserved for selection; report scores the rest, once the checkpoints are chosen.
# compare.py stops if a directory mixes checkpoints or holds predictions of other cutouts.
# PAPER_MODE=1 measures as the code behind the paper's Table 4 did (compare.py --paper-mode: clip
# at 0 and p99.999, no background subtraction, kernel and minimum area matched on the sky, no
# ellipticity cut, +/- = std); the output directory gets a -paper suffix.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1   # print() lines reach logs/ as they happen, not at exit
: "${NEO_DATA:?set NEO_DATA}"
source scripts/lux/comet_check.sh
SPLIT="${SPLIT:-val}"
SUBSET="${SUBSET:-report}"
case "${PAPER_MODE:-0}" in
  0) PAPER_FLAG=""; SUFFIX="" ;;
  1) PAPER_FLAG="--paper-mode"; SUFFIX="-paper" ;;
  *) echo "PAPER_MODE must be 0 or 1, not $PAPER_MODE" >&2; exit 1 ;;
esac
: "${PREDS:?set PREDS to MODEL=DIR pairs, e.g. neo=\$NEO_DATA/predictions/neo/<run>/<checkpoint>/val-report}"

PRED_ARGS=()
for p in $PREDS; do
  PRED_ARGS+=(--pred "$p")
done
uv run python -m neo.eval.compare \
  --split-dir "$NEO_DATA/pairs/cosmos_web_i/$SPLIT" "${PRED_ARGS[@]}" ${PAPER_FLAG:+"$PAPER_FLAG"} \
  --subset "$SUBSET" --workers "${SLURM_CPUS_PER_TASK:-8}" \
  --out "$NEO_DATA/comparison/$SPLIT-$SUBSET$SUFFIX-${SLURM_JOB_ID:-local}" ${LIMIT:+--limit "$LIMIT"}
