#!/usr/bin/env bash
# rsync the project to the Mac and install it there
set -euo pipefail

REMOTE_USER="${MYMPS_USER:-edoardo.tedesco}"
REMOTE_HOST="${MYMPS_HOST:-openport.io}"
REMOTE_PORT="${MYMPS_SSH_PORT:-26996}"
REMOTE_DIR="${MYMPS_REMOTE_DIR:-~/mymps}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "→ Syncing project to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR} …"
rsync -avz --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.egg-info' \
    --exclude '.venv' \
    -e "ssh -p ${REMOTE_PORT}" \
    "${PROJECT_DIR}/" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

echo "→ Installing on remote …"
ssh -p "${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}" \
    "cd ${REMOTE_DIR} && pip install -e '.[server]'"

echo "✓ Deploy complete"
