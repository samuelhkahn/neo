# Sourced by the Lux job scripts so that every run is tracked on Comet (workspace samkahn-astro,
# project neo-rubin-lsst): the job stops unless COMET_ML_ASTRO_API_KEY is set and Comet answers
# from this node. sbatch passes the submitting shell's environment, so export the key there
# (e.g. in ~/.bashrc on Lux).
# COMET_OFFLINE=1 runs anyway, writing offline archives to comet_offline/ instead; upload them
# from a login node afterwards:
#   COMET_API_KEY="$COMET_ML_ASTRO_API_KEY" uv run comet upload comet_offline/*.zip
# (an offline archive is only written when the process exits normally, so a preempted segment's
# logs are lost).
if [[ "${COMET_OFFLINE:-0}" == 1 ]]; then
  echo "comet: offline (COMET_OFFLINE=1); archives go to comet_offline/"
  unset COMET_ML_ASTRO_API_KEY
else
  : "${COMET_ML_ASTRO_API_KEY:?export COMET_ML_ASTRO_API_KEY so the run is tracked on Comet (or set COMET_OFFLINE=1)}"
  timeout 120 uv run python - <<'PY' || { echo "comet: check failed on $(hostname); see above (or set COMET_OFFLINE=1)"; exit 1; }
import os
import sys

import comet_ml

try:
    workspaces = comet_ml.API(api_key=os.environ["COMET_ML_ASTRO_API_KEY"]).get_workspaces()
except Exception as exc:  # report the failure type only, never the request (it carries the key)
    sys.exit(f"comet: cannot reach Comet from this node ({type(exc).__name__})")
if "samkahn-astro" not in workspaces:
    sys.exit("comet: this API key has no access to the samkahn-astro workspace")
print("comet: online, workspace samkahn-astro reachable")
PY
fi
