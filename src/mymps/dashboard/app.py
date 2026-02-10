"""Minimal Flask dashboard for monitoring the mymps server."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, render_template, request, jsonify, Response

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

    # ── Actions ──────────────────────────────────────────────────────

    @app.post("/actions/load")
    def action_load():
        name = request.form.get("name", "").strip()
        dtype = request.form.get("dtype", "float16").strip()
        if not name:
            return "<p class='err'>Model name required</p>", 400
        try:
            with _client() as c:
                result = c.load_model(name, dtype=dtype)
            return f"<p class='ok'>Loaded <b>{result['name']}</b> on {result['device']}</p>"
        except Exception as exc:
            return f"<p class='err'>{exc}</p>", 500

    @app.post("/actions/unload")
    def action_unload():
        name = request.form.get("name", "").strip()
        if not name:
            return "<p class='err'>Model name required</p>", 400
        try:
            with _client() as c:
                c.unload_model(name)
            return f"<p class='ok'>Unloaded <b>{name}</b></p>"
        except Exception as exc:
            return f"<p class='err'>{exc}</p>", 500

    @app.post("/actions/generate")
    def action_generate():
        model = request.form.get("model", "").strip()
        prompt = request.form.get("prompt", "").strip()
        max_tokens = int(request.form.get("max_new_tokens", 128))
        temperature = float(request.form.get("temperature", 0.7))
        if not model or not prompt:
            return "<p class='err'>Model and prompt required</p>", 400
        try:
            with _client() as c:
                result = c.generate(
                    model, prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
            return render_template("_generate_result.html", result=result, prompt=prompt)
        except Exception as exc:
            return f"<p class='err'>{exc}</p>", 500

    return app
