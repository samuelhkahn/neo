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
# Submit from the repo root:  PER_PATCH=500 sbatch scripts/lux/make_pairs.sbatch
# Knobs: NEO_DATA (required), PER_PATCH (cutouts per LSST patch; ~3 MB each), SEED
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
: "${NEO_DATA:?set NEO_DATA}"
PER_PATCH="${PER_PATCH:-500}"

uv run python -m neo.preprocess.pairs \
  --lr-dir "$NEO_DATA/rubin/dp2" \
  --hr-mosaic "$NEO_DATA"/hst/cosmos_web/*_30mas_*_sci.fits \
  --hr-zeropoint 25.94 \
  --out "$NEO_DATA/pairs/cosmos_web_i" \
  --per-patch "$PER_PATCH" --val-frac 0.2 --seed "${SEED:-0}" \
  --parallel "${SLURM_CPUS_PER_TASK:-8}"

echo "pairs written to $NEO_DATA/pairs/cosmos_web_i"
du -sh "$NEO_DATA/pairs/cosmos_web_i"
