#!/usr/bin/env bash
#SBATCH --job-name=neo-mosaic
#SBATCH --account=gpuq
#SBATCH --qos=gpuq
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/mosaic-%j.out
# Out-of-sample super-resolved mosaic of a field the model never saw (default: GOODS-S in ECDFS),
# with the NEO paper's morpheus-core recipe (neo/eval/mosaic.py: 100 px windows, stride 20,
# MEAN_VAR aggregation). Runs on gpuq (1-day limit).
# First fetch the LSST patches on a login node: scripts/lux/download_ecdfs.sh. Then, from the repo
# root of the branch that trained the model:
#   CONFIG=neo/configs/lux_neo_relu.ini CKPT=$NEO_DATA/checkpoints/<run>/step_00500000.pt \
#     sbatch scripts/lux/mosaic.sbatch
# TTA=both (default) makes two mosaics from the same LR image and checkpoint: tta none (one pass per
# window) and tta rot4 (each window at 0/90/180/270 degrees, rotated back, per-pixel median). With
# two GPUs they run side by side, one per GPU; with one they run in turn.
# Timing (Apple M-series GPU, 2026-11-03): ~33 ms per window single, ~80 ms rot4. The default
# 24' box is 7200 x 7200 LR px = 126,736 windows: roughly 1-2 h single, 3-4 h rot4.
# Output: $NEO_DATA/mosaics/<FIELD>/lr_ra<RA>_dec<DEC>_<SIZE>arcmin_<key>.fits (the LR image,
#   built once per box and patch set, <key> a hash of the patch file names and sizes, shared by
#   every run) and $NEO_DATA/mosaics/<FIELD>/<run>/<checkpoint>/<tta>[_<reduce>][_gen<mode>]
#   [_seed<seed>]/ (suffixes only for non-default settings) holding
#   {sr_mean,sr_std,sr_n,sr_paper,sr_flag}.fits (nJy per 0.0333" px; sr_flag 2 = bad LR pixel,
#   NaN in the SR images, 1 = a window over it held a bad LR pixel) + output.fits/n.fits
#   (morpheus's raw arrays) + mosaic.json + preview.png.
#   Each mosaic is 43200^2 = 1.87 Gpix: output.fits 15 GB (float32 x 2), n.fits 7.5 GB, sr_mean,
#   sr_std, sr_n, sr_paper 7.5 GB each, sr_flag 1.9 GB: ~52 GB per tta directory, ~105 GB for both.
#   After checking the products, output.fits and n.fits (22 GB) can be deleted.
# Knobs: CONFIG, CKPT (required), FIELD (goodss), RA/DEC (deg, GOODS-S centre), SIZE_ARCMIN (24),
#   PATCHES (glob of deep_coadd files), TTA (none|rot4|both), REDUCE (median|mean), GEN_MODE
#   (train|eval; train = the paper's per-window batch norm), SEED, OVERWRITE=1 (redo a mosaic),
#   MAX_BADFRAC (0.02: refuse an LR box with more bad/uncovered pixels than this), REBUILD_LR=1
#   (rebuild the LR image even if one exists for this box and patch set).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=60  # morpheus's progress bar: one line a minute in the logs
: "${NEO_DATA:?set NEO_DATA}"
source scripts/lux/job_tmp.sh
source scripts/lux/comet_check.sh
: "${CONFIG:?set CONFIG to the training .ini of the model}"
: "${CKPT:?set CKPT to the checkpoint, e.g. \$NEO_DATA/checkpoints/<run>/step_00500000.pt}"
FIELD="${FIELD:-goodss}"
RA="${RA:-53.122}"
DEC="${DEC:--27.805}"
SIZE_ARCMIN="${SIZE_ARCMIN:-24}"
PATCHES="${PATCHES:-$NEO_DATA/rubin/dp2_ecdfs/deep_coadd_5063_*_i.fits}"
TTA="${TTA:-both}"
REDUCE="${REDUCE:-median}"
GEN_MODE="${GEN_MODE:-train}"
SEED="${SEED:-0}"
MAX_BADFRAC="${MAX_BADFRAC:-0.02}"
OVERWRITE_FLAG=()
[ "${OVERWRITE:-0}" = 1 ] && OVERWRITE_FLAG=(--overwrite)
case "$TTA" in
  none|rot4) RUNS=("$TTA") ;;
  both) RUNS=(none rot4) ;;
  *) echo "TTA must be none, rot4 or both, not $TTA" >&2; exit 1 ;;
esac

BASE="$NEO_DATA/mosaics/$FIELD"
# the LR image is keyed by the box and by the patch set (file names and sizes), so another band,
# tract or a completed download builds a new one instead of reusing a stale image
# shellcheck disable=SC2086  # PATCHES is a glob
PATCH_FILES=$(ls -1 $PATCHES 2>/dev/null || true)
[ -n "$PATCH_FILES" ] || { echo "no patch files match $PATCHES (run download_ecdfs.sh)" >&2; exit 1; }
PATCH_KEY=$(while read -r f; do echo "$(basename "$f") $(stat -Lc %s "$f")"; done <<<"$PATCH_FILES" \
  | sort | sha1sum | cut -c1-8)
echo "$(wc -l <<<"$PATCH_FILES") patches, key $PATCH_KEY"
LR="$BASE/lr_ra${RA}_dec${DEC}_${SIZE_ARCMIN}arcmin_${PATCH_KEY}.fits"
OUT_BASE="$BASE/$(basename "$(dirname "$CKPT")")/$(basename "$CKPT" .pt)"
# non-default settings get their own directories, so no run overwrites another's products
SUFFIX=""
[ "$GEN_MODE" = train ] || SUFFIX="${SUFFIX}_gen${GEN_MODE}"
[ "$SEED" = 0 ] || SUFFIX="${SUFFIX}_seed${SEED}"
run_dir() {  # $1 = tta
  if [ "$1" = rot4 ] && [ "$REDUCE" != median ]; then echo "$OUT_BASE/$1_${REDUCE}${SUFFIX}"
  else echo "$OUT_BASE/$1${SUFFIX}"; fi
}
mkdir -p "$OUT_BASE"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
NGPU=$(nvidia-smi -L | wc -l)

# one sync for both runs (they then start with --no-sync, so they never race on the environment)
uv sync --inexact --quiet --extra train --extra rubin --extra eval

# built under a lock into a temporary file and renamed into place, so concurrent jobs never read
# a half-written LR image and a failed build never leaves one behind
(
  flock 9
  if [ "${REBUILD_LR:-0}" = 1 ]; then rm -f "$LR"; fi
  if [ ! -f "$LR" ]; then
    tmp="$LR.tmp.${SLURM_JOB_ID}.fits"
    trap 'rm -f "$tmp"' EXIT
    # shellcheck disable=SC2086  # PATCHES is a glob
    uv run --no-sync python -m neo.eval.mosaic lr --patches $PATCHES \
      --ra "$RA" --dec "$DEC" --size-arcmin "$SIZE_ARCMIN" --out "$tmp"
    mv "$tmp" "$LR"
  else
    echo "LR image exists: $LR"
  fi
) 9>"$LR.lock"

mosaic() {  # $1 = tta, $2 = GPU index
  local log="logs/mosaic-${SLURM_JOB_ID}-$1.log"
  echo "mosaic tta=$1 on GPU $2 -> $(run_dir "$1") (log $log)"
  CUDA_VISIBLE_DEVICES="$2" uv run --no-sync python -m neo.eval.mosaic sr \
    --config "$CONFIG" --checkpoint "$CKPT" --lr "$LR" --out "$(run_dir "$1")" \
    --tta "$1" --reduce "$REDUCE" --gen-mode "$GEN_MODE" --seed "$SEED" \
    --max-badfrac "$MAX_BADFRAC" --tag "$FIELD" ${OVERWRITE_FLAG[@]+"${OVERWRITE_FLAG[@]}"} >"$log" 2>&1
}

status=0
if [ "${#RUNS[@]}" -eq 2 ] && [ "$NGPU" -ge 2 ]; then
  mosaic none 0 & pid_none=$!
  mosaic rot4 1 & pid_rot4=$!
  wait "$pid_none" || { echo "tta none failed (see its log)"; status=1; }
  wait "$pid_rot4" || { echo "tta rot4 failed (see its log)"; status=1; }
else
  for tta in "${RUNS[@]}"; do
    mosaic "$tta" 0 || { echo "tta $tta failed (see its log)"; status=1; }
  done
fi
for tta in "${RUNS[@]}"; do
  echo "== $tta"; tail -n 25 "logs/mosaic-${SLURM_JOB_ID}-$tta.log" || true
done
exit "$status"
