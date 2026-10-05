#!/usr/bin/env bash
#SBATCH --job-name=neo-pairs
#SBATCH --account=cpuq
#SBATCH --qos=cpuq
#SBATCH --partition=cpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pairs-%j.out
# Build the LSST-i x COSMOS-Web F814W training pairs (CPU only). Needs download_data.sh to have finished.
# Submit from the repo root:  sbatch scripts/lux/make_pairs.sbatch
# Split: global Dec stripes (val = 2' of every 10', VAL_FRAC of the sky) with a 6" guard, so train and
# val share no sky; then neo.preprocess.leakage verifies that on the written cutouts, quarantines
# anything that still conflicts, and writes val/groups.csv for compare.py.
# Units follow the NEO paper (--units paper), so its fixed log stretch sees the paper's values: HST
# resampled with reproject_interp in e-/s per 0.03" pixel, sky pedestal removed (CANDELS was sky-
# subtracted), LSST converted to HSC counts (ZP 27) per 0.168" pixel; NJYPERPX headers give nJy.
# Saturated LSST windows are kept and no flux-sum filter is applied, as in the paper's training set.
# Knobs: NEO_DATA (required), TRAIN_DENSITY (train windows per train pixel; 4 covers ~98% of the
# train sky; ~3 MB per pair), VAL_FRAC, SEED.
# Resumable: if the job dies or times out, submit it again with the same settings and it continues
# where it stopped (finished LSST images are skipped). An output directory holding pairs from any
# other run or settings is refused; delete it first.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1   # print() lines reach logs/ as they happen, not at exit
: "${NEO_DATA:?set NEO_DATA}"
PAIRS="$NEO_DATA/pairs/cosmos_web_i"
# reproject's parallel mode spills ~6 GB of zarr chunks per LSST image to $TMPDIR; a compute node's
# /tmp can be too small, so use a per-job directory on the data filesystem instead
export TMPDIR="$NEO_DATA/tmp/pairs-${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
df -h /tmp "$NEO_DATA" | sed 's/^/disk: /'

uv run python -m neo.preprocess.pairs \
  --lr-dir "$NEO_DATA/rubin/dp2" \
  --hr-mosaic "$NEO_DATA"/hst/cosmos_web/*_30mas_*_sci.fits \
  --hr-zeropoint 25.94 \
  --out "$PAIRS" \
  --split sky --val-frac "${VAL_FRAC:-0.2}" --val-period 10 --guard 6 \
  --train-density "${TRAIN_DENSITY:-4}" --val-tile --seed "${SEED:-0}" \
  --units paper --hr-sky-subtract --reject NO_DATA \
  --parallel "${SLURM_CPUS_PER_TASK:-8}" --resume

uv run python -m neo.preprocess.leakage --pairs "$PAIRS" --margin 5 --apply
uv run python -m neo.preprocess.leakage --pairs "$PAIRS" --margin 5   # must report no leakage

echo "pairs written to $PAIRS"
du -sh "$PAIRS"
