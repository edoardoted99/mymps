"""All HTTP + WebSocket endpoints."""

from __future__ import annotations

import time

import numpy as np
from fastapi import APIRouter, Request, Response, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mymps import tensor
from mymps.protocol import EP_GENERATE, EP_HEALTH, EP_INFER, EP_MODELS, EP_MODELS_LOAD, WS_GENERATE
from mymps.server.device import device_info
from mymps.server.inference import run_inference
from mymps.server.streaming import stream_generate

router = APIRouter()


# --- Pydantic request bodies ------------------------------------------------

class LoadModelRequest(BaseModel):
    name: str
    dtype: str = "float16"
    trust_remote_code: bool = False


class GenerateRequest(BaseModel):
    model: str
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0


# --- Routes ------------------------------------------------------------------

@router.get(EP_HEALTH)
async def health(request: Request):
    info = device_info()
    info["models_loaded"] = len(request.app.state.registry.list())
    return info


@router.get(EP_MODELS)
async def list_models(request: Request):
    return request.app.state.registry.list()


@router.post(EP_MODELS_LOAD)
async def load_model(body: LoadModelRequest, request: Request):
    registry = request.app.state.registry
    entry = registry.load(body.name, dtype=body.dtype, trust_remote_code=body.trust_remote_code)
    return {"status": "ok", "name": entry.name, "device": str(entry.device)}


@router.delete(EP_MODELS + "/{name:path}")
async def unload_model(name: str, request: Request):
    ok = request.app.state.registry.unload(name)
    if not ok:
        return JSONResponse({"error": f"model {name!r} not loaded"}, status_code=404)
    return {"status": "ok", "name": name}


@router.post(EP_INFER)
async def infer(request: Request):
    body = await request.body()
    inputs = tensor.decode_batch(body)
    model_name = request.headers.get("x-model", "")
    try:
        outputs = run_inference(request.app.state.registry, model_name, inputs)
    except KeyError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    return Response(content=tensor.encode_batch(outputs), media_type="application/x-msgpack")


@router.post(EP_GENERATE)
async def generate(body: GenerateRequest, request: Request):
    import torch
    registry = request.app.state.registry
    entry = registry.get(body.model)
    if entry is None:
        return JSONResponse({"error": f"model {body.model!r} not loaded"}, status_code=404)

    tokenizer = entry.tokenizer
    input_ids = tokenizer.encode(body.prompt, return_tensors="pt").to(entry.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = entry.model.generate(
            input_ids,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature if body.temperature > 0 else None,
            top_p=body.top_p if body.top_p < 1.0 else None,
            do_sample=body.temperature > 0,
        )
    elapsed = time.perf_counter() - t0

    new_tokens = output_ids[0][input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return {
        "text": text,
        "tokens_generated": len(new_tokens),
        "time_s": round(elapsed, 3),
        "tokens_per_s": round(len(new_tokens) / elapsed, 2) if elapsed > 0 else 0,
    }


@router.websocket(WS_GENERATE)
async def ws_generate(ws: WebSocket, request: Request = None):
    # Access registry from the app state via the websocket's app
    await stream_generate(ws, ws.app.state.registry)
