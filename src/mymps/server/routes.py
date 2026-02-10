"""All HTTP endpoints for the mymps compute coprocessor."""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mymps import tensor
from mymps.protocol import EP_EXEC, EP_HEALTH, EP_INFER, EP_MODELS, EP_MODELS_LOAD, EP_OPS, EP_STATS
from mymps.server.device import device_info
from mymps.server.inference import run_inference
from mymps.server.ops import get_op, list_ops

router = APIRouter()


# --- Pydantic request bodies ------------------------------------------------

class LoadModelRequest(BaseModel):
    name: str
    model_type: str = "huggingface"
    dtype: str = "float16"
    trust_remote_code: bool = False
    model_path: str | None = None
    model_class: str | None = None


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
    try:
        entry = registry.load(
            body.name,
            model_type=body.model_type,
            dtype=body.dtype,
            trust_remote_code=body.trust_remote_code,
            model_path=body.model_path,
            model_class=body.model_class,
        )
    except (ValueError, ImportError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return {"status": "ok", "name": entry.name, "device": str(entry.device)}


@router.delete(EP_MODELS + "/{name:path}")
async def unload_model(name: str, request: Request):
    ok = request.app.state.registry.unload(name)
    if not ok:
        return JSONResponse({"error": f"model {name!r} not loaded"}, status_code=404)
    return {"status": "ok", "name": name}


@router.get(EP_MODELS + "/{name:path}/info")
async def model_info(name: str, request: Request):
    info = request.app.state.registry.info(name)
    if info is None:
        return JSONResponse({"error": f"model {name!r} not loaded"}, status_code=404)
    return info


@router.post(EP_INFER)
async def infer(request: Request):
    body = await request.body()
    inputs = tensor.decode_batch(body)
    model_name = request.headers.get("x-model", "")
    try:
        outputs = run_inference(request.app.state.registry, model_name, inputs)
    except KeyError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except TypeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    return Response(content=tensor.encode_batch(outputs), media_type="application/x-msgpack")


@router.get(EP_STATS)
async def stats():
    """System resource usage — CPU, memory, MPS memory."""
    import os
    import psutil

    proc = psutil.Process(os.getpid())
    vm = psutil.virtual_memory()

    info: dict = {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "cpu_count": psutil.cpu_count(),
        "load_avg": list(os.getloadavg()),
        "memory_total_gb": round(vm.total / (1024 ** 3), 2),
        "memory_used_gb": round(vm.used / (1024 ** 3), 2),
        "memory_percent": vm.percent,
        "process_rss_mb": round(proc.memory_info().rss / (1024 ** 2), 1),
        "process_cpu_percent": proc.cpu_percent(interval=None),
    }

    # MPS memory stats (available on macOS with MPS)
    if torch.backends.mps.is_available():
        try:
            info["mps_allocated_mb"] = round(torch.mps.current_allocated_memory() / (1024 ** 2), 1)
            info["mps_driver_mb"] = round(torch.mps.driver_allocated_memory() / (1024 ** 2), 1)
        except Exception:
            pass

    return info


@router.get(EP_OPS)
async def ops():
    return {"ops": list_ops()}


@router.post(EP_EXEC)
async def exec_op(request: Request):
    """Execute a whitelisted torch operation on MPS.

    - Op name in header ``x-op``
    - Scalar kwargs in header ``x-kwargs`` (JSON)
    - Input tensors in body (msgpack, same format as /infer)
    - Positional convention: keys "0", "1", "2" for positional args
    """
    op_name = request.headers.get("x-op", "")
    if not op_name:
        return JSONResponse({"error": "x-op header is required"}, status_code=400)

    try:
        op_fn = get_op(op_name)
    except KeyError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    # Parse scalar kwargs from header
    kwargs_raw = request.headers.get("x-kwargs", "")
    scalar_kwargs: dict = json.loads(kwargs_raw) if kwargs_raw else {}

    # Parse tensor inputs from body
    body = await request.body()
    from mymps.server.device import get_device
    device = get_device()

    np_inputs = tensor.decode_batch(body) if body else {}
    torch_inputs: dict[str, torch.Tensor] = {
        k: torch.from_numpy(v).to(device) for k, v in np_inputs.items()
    }

    # Separate positional ("0", "1", …) from keyword tensor args
    positional: list[tuple[int, torch.Tensor]] = []
    kw_tensors: dict[str, torch.Tensor] = {}
    for k, t in torch_inputs.items():
        if k.isdigit():
            positional.append((int(k), t))
        else:
            kw_tensors[k] = t

    positional.sort(key=lambda x: x[0])
    pos_args = [t for _, t in positional]

    t0 = time.perf_counter()
    with torch.no_grad():
        result = op_fn(*pos_args, **kw_tensors, **scalar_kwargs)
    elapsed = time.perf_counter() - t0

    # Normalise result to dict[str, numpy]
    out: dict[str, np.ndarray] = {}
    if isinstance(result, torch.Tensor):
        out["output"] = result.detach().cpu().float().numpy()
    elif isinstance(result, (tuple, list)):
        for i, item in enumerate(result):
            if isinstance(item, torch.Tensor):
                out[str(i)] = item.detach().cpu().float().numpy()
    else:
        out["output"] = np.array(result)

    payload = tensor.encode_batch(out)
    return Response(
        content=payload,
        media_type="application/x-msgpack",
        headers={"x-time": str(round(elapsed, 6))},
    )
