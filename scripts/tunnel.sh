#!/usr/bin/env bash
# Open an SSH tunnel forwarding localhost:5555 → Mac MPS server
set -euo pipefail

REMOTE_USER="${MYMPS_USER:-edoardo.tedesco}"
REMOTE_HOST="${MYMPS_HOST:-openport.io}"
REMOTE_PORT="${MYMPS_SSH_PORT:-26996}"
LOCAL_PORT="${MYMPS_PORT:-5555}"

echo "→ Opening SSH tunnel  localhost:${LOCAL_PORT} → ${REMOTE_HOST}:${REMOTE_PORT} → localhost:${LOCAL_PORT}"
exec ssh -N -L "${LOCAL_PORT}:localhost:${LOCAL_PORT}" \
    -p "${REMOTE_PORT}" \
    "${REMOTE_USER}@${REMOTE_HOST}"
