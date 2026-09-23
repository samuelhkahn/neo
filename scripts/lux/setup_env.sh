#!/usr/bin/env bash
# One-time environment setup on a Lux LOGIN node (compute nodes may not have internet).
# Usage, from anywhere:  bash scripts/lux/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

uv python install 3.12
# pyproject pins torch 2.6.0 CUDA 12.4 wheels on Linux (the newest that run on Lux's glibc 2.17 and the
# V100s); the wheels bundle the CUDA runtime, so nothing is needed from `module`.
uv sync --extra train --extra rubin

mkdir -p logs   # Slurm will not create the --output directory itself

uv run python - <<'EOF'
import torch
print("torch", torch.__version__, "| CUDA build", torch.version.cuda)
print("GPU access is verified on a compute node by smoke.sbatch (login nodes have no GPU)")
EOF

cat <<EOF

Environment ready. Next steps:
  export NEO_DATA=/data/<your area>/neo        # add to ~/.bashrc
  export COMET_ML_ASTRO_API_KEY=...            # add to ~/.bashrc (optional; offline logging otherwise)
  nohup bash scripts/lux/download_data.sh > logs/download.out 2>&1 &
  sbatch scripts/lux/make_pairs.sbatch
  sbatch scripts/lux/smoke.sbatch
  sbatch scripts/lux/train.sbatch
EOF
