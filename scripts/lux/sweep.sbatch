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
# deployment mode with dropout on and per-image batch norm, or eval: deterministic). Output goes
# to $NEO_DATA/sweeps/<config name>[-eval]/ (sweep.csv, per_image.csv, fine_structure.json,
# sweep.png); a rerun scores only checkpoints saved since the last one.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1   # print() lines reach logs/ as they happen, not at exit
: "${NEO_DATA:?set NEO_DATA}"
source scripts/lux/job_tmp.sh
source scripts/lux/comet_check.sh
CONFIG="${CONFIG:-neo/configs/lux.ini}"
MODE="${MODE:-train}"
OUT="$NEO_DATA/sweeps/$(basename "$CONFIG" .ini)"
if [ "$MODE" = eval ]; then OUT="$OUT-eval"; fi
EXTRA=(--mode "$MODE")
for range in ${AVERAGES:-}; do
  EXTRA+=(--average "$range")
done
if [ -n "${EVERY:-}" ]; then EXTRA+=(--every "$EVERY"); fi
if [ -n "${LIMIT:-}" ]; then EXTRA+=(--limit "$LIMIT"); fi
echo "sweep of $CONFIG -> $OUT"

nvidia-smi --query-gpu=name,memory.total --format=csv
# ${EXTRA[@]+...}: an empty array is "unbound" under set -u in Lux's bash 4.2
uv run python -m neo.eval.sweep --config "$CONFIG" --out "$OUT" ${EXTRA[@]+"${EXTRA[@]}"}
