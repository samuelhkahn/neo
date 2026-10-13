# Sourced by the Lux job scripts: rerun a training command that fails, so that a transient crash
# does not end a two-week run. train.py --resume continues from the newest readable checkpoint
# (neo/checkpoint.py), so a rerun repeats at most [CHECKPOINT] latest_every steps. Example: the
# DataLoader shared-memory bus error that ended a diffusion arm at step 600k (2026-10-10).
#
#   ( run_with_retries LOG RETRIES PAUSE COMMAND [ARGS...] ) &
#
# Runs COMMAND with stdout and stderr appended to LOG; after a failure, PAUSE seconds later, runs it
# again, at most RETRIES times in a row. An attempt that ran for at least RETRY_RESET seconds
# (default 3600) before failing made progress, so the count starts again after it: RETRIES limits
# failures in quick succession (a persistent fault), not the transient crashes of a two-week run.
# Each attempt and its exit status are noted in LOG and on stdout. Returns 0 once an attempt
# succeeds, else the last attempt's exit status.
# Run it in a (background) subshell: it sets a TERM/INT trap there. On TERM it stops the running
# attempt, including that command's children (python under uv run), and exits 143 without
# retrying. An attempt that itself ends by TERM or INT (exit 143 or 130: scancel, preemption) was
# stopped on purpose, so it is not retried either.
# Plain bash for Lux (4.2) and macOS (3.2): no arrays (empty ones are "unbound" under set -u
# before bash 4.4) and no wait -n.

_retry_note() {
  local line
  line="$(date '+%Y-%m-%d %H:%M:%S') run_with_retries: $2"
  echo "$line" >>"$1"
  echo "$1: $line"
}

_retry_stop() {
  if [[ -n "${_retry_child:-}" ]]; then
    # children first: once uv exits, python would be reparented and no longer found by -P
    pkill -TERM -P "$_retry_child" 2>/dev/null || true
    kill -TERM "$_retry_child" 2>/dev/null || true
    wait "$_retry_child" 2>/dev/null || true
  fi
  _retry_note "$_retry_log" "stopped by TERM/INT; not retrying"
  exit 143
}

run_with_retries() {
  _retry_log=$1
  local retries=$2 pause=$3 reset=${RETRY_RESET:-3600} attempt=0 failures=0 status started ran
  shift 3
  _retry_child=""
  trap _retry_stop TERM INT
  while true; do
    attempt=$((attempt + 1))
    _retry_note "$_retry_log" "attempt $attempt: $*"
    started=$SECONDS
    # in the background so that the trap runs as soon as TERM arrives, not when the command ends
    "$@" >>"$_retry_log" 2>&1 &
    _retry_child=$!
    status=0
    wait "$_retry_child" || status=$?
    _retry_child=""
    ran=$((SECONDS - started))
    if ((status == 0)); then
      _retry_note "$_retry_log" "attempt $attempt succeeded"
      return 0
    fi
    if ((status == 130 || status == 143)); then
      _retry_note "$_retry_log" "attempt $attempt was stopped (exit $status); not retrying"
      return "$status"
    fi
    if ((ran >= reset)); then
      failures=0
      _retry_note "$_retry_log" "attempt $attempt ran $ran s, at least RETRY_RESET: count reset"
    fi
    if ((failures >= retries)); then
      _retry_note "$_retry_log" \
        "attempt $attempt failed (exit $status) after $ran s; no retries left ($retries in a row)"
      return "$status"
    fi
    failures=$((failures + 1))
    _retry_note "$_retry_log" \
      "attempt $attempt failed (exit $status) after $ran s; rerun $failures/$retries in $pause s"
    sleep "$pause" &
    _retry_child=$!
    wait "$_retry_child" || true
    _retry_child=""
  done
}
