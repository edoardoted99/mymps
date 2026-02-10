"""python -m mymps.server"""

import logging
import os
import sys

import uvicorn

from mymps.protocol import DEFAULT_HOST, DEFAULT_PORT

# Ensure MPS fallback is enabled for operators not yet on MPS
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

host = os.environ.get("MYMPS_HOST", DEFAULT_HOST)
port = int(os.environ.get("MYMPS_PORT", DEFAULT_PORT))

uvicorn.run(
    "mymps.server.app:create_app",
    factory=True,
    host=host,
    port=port,
    log_level="info",
)
