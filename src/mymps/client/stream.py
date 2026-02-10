"""TokenStream — async iterator over streaming generation tokens."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import AsyncIterator

import websockets

from mymps.protocol import MSG_DONE, MSG_ERROR, MSG_TOKEN


@dataclass
class StreamResult:
    """Accumulated result after the stream finishes."""
    text: str = ""
    tokens_generated: int = 0
    time_s: float = 0.0
    tokens_per_s: float = 0.0
    stop_reason: str = ""


class TokenStream:
    """Async iterator that yields token strings from a WebSocket stream.

    Usage::

        async with client.stream_generate("model", "prompt") as stream:
            async for token in stream:
                print(token, end="", flush=True)
        print(stream.result)
    """

    def __init__(self, ws_url: str, payload: dict) -> None:
        self._ws_url = ws_url
        self._payload = payload
        self._ws: websockets.WebSocketClientProtocol | None = None
        self.result: StreamResult = StreamResult()

    async def __aenter__(self) -> TokenStream:
        self._ws = await websockets.connect(self._ws_url)
        await self._ws.send(json.dumps(self._payload))
        return self

    async def __aexit__(self, *exc) -> None:
        if self._ws is not None:
            await self._ws.close()

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iter_tokens()

    async def _iter_tokens(self) -> AsyncIterator[str]:
        assert self._ws is not None
        async for raw in self._ws:
            msg = json.loads(raw)
            mtype = msg.get("type")
            if mtype == MSG_TOKEN:
                text = msg["text"]
                self.result.text += text
                yield text
            elif mtype == MSG_DONE:
                self.result.tokens_generated = msg.get("tokens_generated", 0)
                self.result.time_s = msg.get("time_s", 0.0)
                self.result.tokens_per_s = msg.get("tokens_per_s", 0.0)
                self.result.stop_reason = msg.get("stop_reason", "")
                return
            elif mtype == MSG_ERROR:
                raise RuntimeError(msg.get("error", "unknown server error"))
