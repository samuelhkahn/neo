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
# Knobs: NEO_DATA (required), TRAIN_DENSITY (train windows per train pixel; 4 covers ~98% of the
# train sky; ~3 MB per pair), VAL_FRAC, SEED. The output directory must not already hold pairs.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1   # print() lines reach logs/ as they happen, not at exit
: "${NEO_DATA:?set NEO_DATA}"
PAIRS="$NEO_DATA/pairs/cosmos_web_i"

uv run python -m neo.preprocess.pairs \
  --lr-dir "$NEO_DATA/rubin/dp2" \
  --hr-mosaic "$NEO_DATA"/hst/cosmos_web/*_30mas_*_sci.fits \
  --hr-zeropoint 25.94 \
  --out "$PAIRS" \
  --split sky --val-frac "${VAL_FRAC:-0.2}" --val-period 10 --guard 6 \
  --train-density "${TRAIN_DENSITY:-4}" --val-tile --seed "${SEED:-0}" \
  --parallel "${SLURM_CPUS_PER_TASK:-8}"

uv run python -m neo.preprocess.leakage --pairs "$PAIRS" --margin 5 --apply
uv run python -m neo.preprocess.leakage --pairs "$PAIRS" --margin 5   # must report no leakage

echo "pairs written to $PAIRS"
du -sh "$PAIRS"
