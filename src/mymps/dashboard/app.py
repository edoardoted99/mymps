"""Minimal Flask dashboard for monitoring the mymps compute coprocessor."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, render_template

from mymps.client.client import MympsClient
from mymps.protocol import DEFAULT_HOST, DEFAULT_PORT

TEMPLATE_DIR = Path(__file__).parent / "templates"


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

    def _client() -> MympsClient:
        host = os.environ.get("MYMPS_SERVER_HOST", DEFAULT_HOST)
        port = int(os.environ.get("MYMPS_SERVER_PORT", DEFAULT_PORT))
        return MympsClient(host=host, port=port, timeout=30.0)

    # ── Pages ────────────────────────────────────────────────────────

    @app.get("/")
    def index():
        return render_template("index.html")

    # ── HTMX partials ────────────────────────────────────────────────

    @app.get("/partials/health")
    def partial_health():
        try:
            with _client() as c:
                info = c.health()
            return render_template("_health.html", info=info, ok=True)
        except Exception as exc:
            return render_template("_health.html", info={}, ok=False, error=str(exc))

    @app.get("/partials/models")
    def partial_models():
        try:
            with _client() as c:
                models = c.list_models()
            return render_template("_models.html", models=models)
        except Exception:
            return render_template("_models.html", models=[])

    return app
