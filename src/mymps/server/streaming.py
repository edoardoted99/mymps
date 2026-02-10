"""Streaming text generation via WebSocket."""

from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator

import torch
from starlette.websockets import WebSocket

from mymps.protocol import MSG_DONE, MSG_ERROR, MSG_TOKEN
from mymps.server.models import ModelRegistry

log = logging.getLogger(__name__)


async def stream_generate(ws: WebSocket, registry: ModelRegistry) -> None:
    """Read a generation request from *ws*, stream tokens back, then close."""
    await ws.accept()
    try:
        raw = await ws.receive_text()
        req = json.loads(raw)

        model_name: str = req["model"]
        prompt: str = req["prompt"]
        max_new_tokens: int = req.get("max_new_tokens", 256)
        temperature: float = req.get("temperature", 1.0)
        top_p: float = req.get("top_p", 1.0)

        entry = registry.get(model_name)
        if entry is None:
            await ws.send_text(json.dumps(
                {"type": MSG_ERROR, "error": f"model {model_name!r} not loaded"},
            ))
            await ws.close()
            return

        async for chunk in _generate_tokens(
            entry, prompt, max_new_tokens, temperature, top_p,
        ):
            await ws.send_text(json.dumps(chunk))

    except Exception as exc:
        log.exception("stream_generate error")
        try:
            await ws.send_text(json.dumps({"type": MSG_ERROR, "error": str(exc)}))
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


async def _generate_tokens(
    entry,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> AsyncIterator[dict]:
    """Yield token dicts one by one (runs blocking torch code in-thread)."""
    tokenizer = entry.tokenizer
    model = entry.model
    device = entry.device

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids
    t0 = time.perf_counter()
    tokens_generated = 0
    stop_reason = "max_tokens"

    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated)
        logits = outputs.logits[:, -1, :]

        if temperature > 0:
            logits = logits / temperature
            if top_p < 1.0:
                logits = _top_p_filter(logits, top_p)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = logits.argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, next_id], dim=-1)
        token_text = tokenizer.decode(next_id[0], skip_special_tokens=True)
        tokens_generated += 1

        yield {"type": MSG_TOKEN, "text": token_text}

        if next_id.item() == eos_id:
            stop_reason = "eos"
            break

    elapsed = time.perf_counter() - t0
    yield {
        "type": MSG_DONE,
        "tokens_generated": tokens_generated,
        "time_s": round(elapsed, 3),
        "tokens_per_s": round(tokens_generated / elapsed, 2) if elapsed > 0 else 0,
        "stop_reason": stop_reason,
    }


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    mask = cumulative - torch.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits[mask] = float("-inf")
    return sorted_logits.scatter(1, sorted_idx, sorted_logits)
