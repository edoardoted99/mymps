"""python -m mymps.dashboard"""

import os

from mymps.dashboard.app import create_app

app = create_app()
app.run(
    host=os.environ.get("MYMPS_DASH_HOST", "127.0.0.1"),
    port=int(os.environ.get("MYMPS_DASH_PORT", "8080")),
    debug=os.environ.get("MYMPS_DASH_DEBUG", "0") == "1",
)
