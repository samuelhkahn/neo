#!/usr/bin/env bash
# Fetch the raw inputs onto Lux. Run on a LOGIN node (needs internet), ideally under nohup.
# Requires:
#   NEO_DATA        destination directory on /data
#   RSP_TOKEN_FILE  file holding a Rubin Science Platform token (default ~/.rsp_token); keep it out of git
# The LSST patches can instead be rsync'd from a machine that already has data/rubin/dp2 (same layout).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
: "${NEO_DATA:?set NEO_DATA to a directory on /data, e.g. /data/groups/comp-astro/$USER/neo}"
RSP_TOKEN_FILE="${RSP_TOKEN_FILE:-$HOME/.rsp_token}"
mkdir -p "$NEO_DATA"/{rubin/dp2,hst/cosmos_web,pairs,checkpoints}

echo "== 1/2  LSST DP2 COSMOS i-band deep_coadd patches (185 files, ~6.4 GB) =="
uv run python -m neo.surveys.rubin.sia --token-file "$RSP_TOKEN_FILE" --band i \
  --out "$NEO_DATA/cosmos_i.csv"
uv run python -m neo.surveys.rubin.download --table "$NEO_DATA/cosmos_i.csv" \
  --dest "$NEO_DATA/rubin/dp2" --token-file "$RSP_TOKEN_FILE" --workers 4

echo "== 2/2  COSMOS-Web DR1 HST/ACS F814W, 30 mas science tiles (20 tiles, ~36 GB unzipped) =="
uv run python -m neo.surveys.hst.cosmos_web --dest "$NEO_DATA/hst/cosmos_web" --scale 30mas --kind sci

echo "downloads complete under $NEO_DATA"
