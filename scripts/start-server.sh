#!/usr/bin/env bash
# Start the mymps server on the Mac (run this via SSH on the Mac)
set -euo pipefail

export PYTORCH_ENABLE_MPS_FALLBACK=1

MYMPS_HOST="${MYMPS_HOST:-127.0.0.1}"
MYMPS_PORT="${MYMPS_PORT:-5555}"

echo "→ Starting mymps server on ${MYMPS_HOST}:${MYMPS_PORT} …"
exec python -m mymps.server
