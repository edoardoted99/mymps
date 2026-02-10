"""MympsClient — sync HTTP + async WebSocket client SDK."""

from __future__ import annotations

from typing import Any

import httpx
import numpy as np

from mymps import tensor
from mymps.protocol import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    EP_GENERATE,
    EP_HEALTH,
    EP_INFER,
    EP_MODELS,
    EP_MODELS_LOAD,
    WS_GENERATE,
)
from mymps.client.stream import TokenStream


class MympsClient:
    """Thin client for the mymps remote inference server.

    All REST calls are synchronous (httpx).
    Streaming generation returns an async ``TokenStream``.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: float = 300.0,
    ) -> None:
        self._base = f"http://{host}:{port}"
        self._ws_base = f"ws://{host}:{port}"
        self._http = httpx.Client(base_url=self._base, timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- REST helpers --------------------------------------------------------

    def health(self) -> dict:
        return self._http.get(EP_HEALTH).raise_for_status().json()

    def list_models(self) -> list[dict]:
        return self._http.get(EP_MODELS).raise_for_status().json()

    def load_model(
        self,
        name: str,
        *,
        dtype: str = "float16",
        trust_remote_code: bool = False,
    ) -> dict:
        return self._http.post(
            EP_MODELS_LOAD,
            json={"name": name, "dtype": dtype, "trust_remote_code": trust_remote_code},
        ).raise_for_status().json()

    def unload_model(self, name: str) -> dict:
        return self._http.delete(f"{EP_MODELS}/{name}").raise_for_status().json()

    def infer(self, model: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        resp = self._http.post(
            EP_INFER,
            content=tensor.encode_batch(inputs),
            headers={"content-type": "application/x-msgpack", "x-model": model},
        )
        resp.raise_for_status()
        return tensor.decode_batch(resp.content)

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> dict:
        return self._http.post(
            EP_GENERATE,
            json={
                "model": model,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        ).raise_for_status().json()

    # --- WebSocket streaming -------------------------------------------------

    def stream_generate(
        self,
        model: str,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> TokenStream:
        """Return a ``TokenStream`` (use as ``async with``).

        Example::

            async with client.stream_generate("model", "Hello") as stream:
                async for token in stream:
                    print(token, end="")
        """
        return TokenStream(
            ws_url=f"{self._ws_base}{WS_GENERATE}",
            payload={
                "model": model,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        )
