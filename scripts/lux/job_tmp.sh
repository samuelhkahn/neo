# Sourced by the Lux job scripts: temporary files on the data filesystem, cleaned as the job runs.
# - A compute node's /tmp is only 2 GB. TMPDIR becomes a per-job directory under $NEO_DATA/tmp.
# - comet_ml (3.58) writes every logged figure to a temp file (~0.5 MB) and deletes it only when the
#   run ends, so a long training run fills any disk. Figures are uploaded within seconds; a
#   background loop deletes those older than 15 minutes.
# - Comet's offline fallback mirrors every message (figures included) into TMPDIR for the whole run;
#   it is switched off.
# - The directory is removed when the job exits (also on scancel or requeue); directories of jobs
#   killed without cleanup are removed after two days.
: "${NEO_DATA:?set NEO_DATA}"
mkdir -p "$NEO_DATA/tmp"
find "$NEO_DATA/tmp" -mindepth 1 -maxdepth 1 -type d -mtime +2 -exec rm -rf {} + 2>/dev/null || true
export TMPDIR="$NEO_DATA/tmp/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"
export COMET_FALLBACK_STREAMER_FALLBACK_TO_OFFLINE_MIN_BACKEND_VERSION=999.0.0
(
  while sleep "${JOB_TMP_SWEEP_SECONDS:-300}"; do
    find "$TMPDIR" -maxdepth 1 -type f \( -name 'tmp*.svg' -o -name 'tmp*.png' \) \
      -mmin +"${JOB_TMP_MAX_AGE_MINUTES:-15}" -delete 2>/dev/null
  done
) &
JOB_TMP_CLEANER=$!
trap 'kill "$JOB_TMP_CLEANER" 2>/dev/null; rm -rf "$TMPDIR"' EXIT
trap 'exit 143' TERM INT
