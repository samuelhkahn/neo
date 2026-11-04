#!/usr/bin/env bash
# Fetch the LSST DP2 i-band deep_coadd patches over GOODS-S (ECDFS) for the out-of-sample mosaic
# (scripts/lux/mosaic.sbatch). Run on a LOGIN node (needs internet), ideally under nohup.
# The default search (0.35 deg around GOODS-S) returns 23 patches, all in tract 5063 (checked
# 2026-11-03 against the RSP SIA service); about 1-2 GB.
# Requires:
#   NEO_DATA        data directory on Lux
#   RSP_TOKEN_FILE  file holding a Rubin Science Platform token (default ~/.rsp_token); keep it out of git
# Knobs: RA, DEC, RADIUS (deg).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
: "${NEO_DATA:?set NEO_DATA}"
RSP_TOKEN_FILE="${RSP_TOKEN_FILE:-$HOME/.rsp_token}"
RA="${RA:-53.12}"
DEC="${DEC:--27.81}"
RADIUS="${RADIUS:-0.35}"
DEST="$NEO_DATA/rubin/dp2_ecdfs"
mkdir -p "$DEST"

uv run --extra rubin python -m neo.surveys.rubin.sia --token-file "$RSP_TOKEN_FILE" \
  --collection dp2 --ra "$RA" --dec "$DEC" --radius "$RADIUS" --band i \
  --out "$NEO_DATA/ecdfs_goodss_i.csv"
uv run --extra rubin python -m neo.surveys.rubin.download --table "$NEO_DATA/ecdfs_goodss_i.csv" \
  --dest "$DEST" --token-file "$RSP_TOKEN_FILE" --workers 4
ls "$DEST" | wc -l | sed "s|^|patches in $DEST: |"
