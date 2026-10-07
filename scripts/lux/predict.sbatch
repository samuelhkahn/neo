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
# GEN_MODE=train runs the generator as the paper generated its SR images: never put in eval mode
# (dropout on, batch norm on each image's own statistics) at batch size 1, so BATCH is forced to 1
# (as the paper's neo/analysis/generate_sr_images.py did). Default eval: deterministic.
# SAMPLES=K (K > 1; diffusion, or GEN_MODE=train) draws K samples per pair and writes the first
# draw and the stacks of all K as single/, mean/, median/ (each a directory for compare.sbatch) and
# std/ (per-pixel spread) under the output directory (neo/eval/stack.py); K times slower.
# Output: $NEO_DATA/predictions/<MODEL>/<run>/<checkpoint>/<SPLIT>-<SUBSET>[-gentrain][-k<K>]
# (printed below), in nJy per HR pixel.
# Knobs: CONFIG, MODEL, CKPT (required), SUBSET (select|report|all), SPLIT (default val), LIMIT, BATCH,
# GEN_MODE (eval|train), SAMPLES (default 1).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1   # print() lines reach logs/ as they happen, not at exit
: "${NEO_DATA:?set NEO_DATA}"
source scripts/lux/job_tmp.sh
source scripts/lux/comet_check.sh
: "${CONFIG:?set CONFIG to the training .ini of the model}"
: "${MODEL:?set MODEL to a short name, e.g. neo, swinir, diffusion}"
: "${CKPT:?set CKPT to the checkpoint to evaluate, e.g. \$NEO_DATA/checkpoints/<run>/step_00500000.pt}"
SPLIT="${SPLIT:-val}"
SUBSET="${SUBSET:-select}"
GEN_MODE="${GEN_MODE:-eval}"
case "$GEN_MODE" in
  eval) BATCH="${BATCH:-4}"; SUFFIX="" ;;
  train)
    if [ "${BATCH:-1}" != 1 ]; then echo "GEN_MODE=train: BATCH=$BATCH -> 1"; fi
    BATCH=1; SUFFIX="-gentrain" ;;
  *) echo "GEN_MODE must be eval or train, not $GEN_MODE" >&2; exit 1 ;;
esac
SAMPLES="${SAMPLES:-1}"
if [ "$SAMPLES" -gt 1 ]; then SUFFIX="$SUFFIX-k$SAMPLES"; fi
OUT="$NEO_DATA/predictions/$MODEL/$(basename "$(dirname "$CKPT")")/$(basename "$CKPT" .pt)/$SPLIT-$SUBSET$SUFFIX"
echo "predictions -> $OUT (generator mode $GEN_MODE, batch $BATCH, $SAMPLES draw(s) per pair)"
if [ "$SAMPLES" -gt 1 ]; then echo "  stacks: $OUT/{single,mean,median,std}"; fi

nvidia-smi --query-gpu=name,memory.total --format=csv
uv run python -m neo.eval.predict \
  --config "$CONFIG" --checkpoint "$CKPT" \
  --split-dir "$NEO_DATA/pairs/cosmos_web_i/$SPLIT" \
  --subset "$SUBSET" --out "$OUT" --gen-mode "$GEN_MODE" \
  --batch-size "$BATCH" --samples "$SAMPLES" ${LIMIT:+--limit "$LIMIT"}
