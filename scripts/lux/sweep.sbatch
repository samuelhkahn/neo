#!/usr/bin/env bash
#SBATCH --job-name=neo-sweep
#SBATCH --account=comp-astro
#SBATCH --qos=comp-astro
#SBATCH --partition=comp-astro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/sweep-%j.out
# Fixed-set evaluation of every saved checkpoint of a NEO run (neo/eval/sweep.py): the same val
# images with the same dropout masks for each step_*.pt, giving the training losses without the
# batch-of-8 noise of the Comet curves, plus the fine-structure metric. Evaluation only: it reads
# checkpoints and never touches the training run.
# Submit from the repo root of the branch that trained the run, e.g.
#   AVERAGES="650000:850000 750000:850000" sbatch scripts/lux/sweep.sbatch          (parity)
#   CONFIG=neo/configs/lux_rot90.ini AVERAGES="550000:650000" sbatch scripts/lux/sweep.sbatch
# Knobs: CONFIG (default lux.ini), AVERAGES (A:B ranges whose step files are averaged and scored
# too), EVERY (only steps divisible by it), LIMIT (first N val images), MODE (train, the paper's
# deployment mode with dropout on and per-image batch norm, or eval: deterministic), ENSEMBLE (K
# passes per image with their own dropout draws, MODE=train only; the single pass is scored as
# always and the median and mean stacks of the K passes get extra rows <checkpoint>:medianK /
# :meanK; costs K generator passes plus the discriminator, VGG and fine-structure metric three
# times per image instead of once; default 1), KEEP_STARS=1 (count HST sources inside the Gaia
# star masks in the fine-structure metric; by default they are left out, fine_n_star; the stars
# are counted per image either way; writes to its own -stars directory so both sweeps coexist),
# TAG (a suffix for the output directory, to rescore everything after a metric change). Output
# goes to $NEO_DATA/sweeps/<config name>[-eval][-kK][-stars][-TAG]/ (sweep.csv, per_image.csv,
# fine_structure.json, sweep.png); a rerun into the same directory scores only checkpoints saved
# since the last one, and a directory written by an older version of the metric (other sweep.csv
# columns) is refused before anything is loaded: use a TAG. The Gaia catalog (neo/data/
# gaia_dr3_cosmos_stars.csv) must be in the checkout: the sweep stops at once without it.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1   # print() lines reach logs/ as they happen, not at exit
: "${NEO_DATA:?set NEO_DATA}"
source scripts/lux/job_tmp.sh
source scripts/lux/comet_check.sh
CONFIG="${CONFIG:-neo/configs/lux.ini}"
MODE="${MODE:-train}"
ENSEMBLE="${ENSEMBLE:-1}"
OUT="$NEO_DATA/sweeps/$(basename "$CONFIG" .ini)"
if [ "$MODE" = eval ]; then OUT="$OUT-eval"; fi
if [ "$ENSEMBLE" -gt 1 ]; then OUT="$OUT-k$ENSEMBLE"; fi
EXTRA=(--mode "$MODE" --ensemble "$ENSEMBLE")
# its own directory, as compare.sbatch does: settings.json records --keep-stars, so a sweep with
# and one without the stars cannot share one
if [ "${KEEP_STARS:-0}" = 1 ]; then EXTRA+=(--keep-stars); OUT="$OUT-stars"; fi
if [ -n "${TAG:-}" ]; then OUT="$OUT-$TAG"; fi
for range in ${AVERAGES:-}; do
  EXTRA+=(--average "$range")
done
if [ -n "${EVERY:-}" ]; then EXTRA+=(--every "$EVERY"); fi
if [ -n "${LIMIT:-}" ]; then EXTRA+=(--limit "$LIMIT"); fi
echo "sweep of $CONFIG -> $OUT"

nvidia-smi --query-gpu=name,memory.total --format=csv
# ${EXTRA[@]+...}: an empty array is "unbound" under set -u in Lux's bash 4.2
uv run python -m neo.eval.sweep --config "$CONFIG" --out "$OUT" ${EXTRA[@]+"${EXTRA[@]}"}
