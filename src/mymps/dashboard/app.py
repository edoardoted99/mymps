"""Minimal Flask dashboard for monitoring the mymps compute coprocessor."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request

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
        model_type = request.form.get("model_type", "huggingface").strip()
        dtype = request.form.get("dtype", "float16").strip()
        model_path = request.form.get("model_path", "").strip() or None
        model_class = request.form.get("model_class", "").strip() or None
        if not name:
            return "<p class='err'>Model name required</p>", 400
        try:
            with _client() as c:
                result = c.load_model(
                    name,
                    model_type=model_type,
                    dtype=dtype,
                    model_path=model_path,
                    model_class=model_class,
                )
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

    @app.post("/actions/exec")
    def action_exec():
        op = request.form.get("op", "").strip()
        tensor_json = request.form.get("tensors", "").strip()
        kwargs_json = request.form.get("kwargs", "").strip()
        if not op:
            return "<p class='err'>Operation name required</p>", 400
        if not tensor_json:
            return "<p class='err'>Tensor data required (JSON)</p>", 400
        try:
            raw_tensors = json.loads(tensor_json)
            inputs = {k: np.array(v, dtype=np.float32) for k, v in raw_tensors.items()}
            kwargs = json.loads(kwargs_json) if kwargs_json else {}
            with _client() as c:
                result = c.exec(op, inputs, **kwargs)
            # Format result for display
            formatted = {k: v.tolist() for k, v in result.items()}
            return render_template("_exec_result.html", op=op, result=formatted)
        except Exception as exc:
            return f"<p class='err'>{exc}</p>", 500

    return app
