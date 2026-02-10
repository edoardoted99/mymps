"""MympsClient — sync HTTP client SDK for the MPS compute coprocessor."""

from __future__ import annotations

import json
from typing import Any

import httpx
import numpy as np

from mymps import tensor
from mymps.protocol import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    EP_EXEC,
    EP_HEALTH,
    EP_INFER,
    EP_MODELS,
    EP_MODELS_LOAD,
    EP_OPS,
)


class MympsClient:
    """Thin client for the mymps MPS compute coprocessor.

    All calls are synchronous (httpx).
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: float = 300.0,
    ) -> None:
        self._base = f"http://{host}:{port}"
        self._http = httpx.Client(base_url=self._base, timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- Server info --------------------------------------------------------

    def health(self) -> dict:
        return self._http.get(EP_HEALTH).raise_for_status().json()

    def list_ops(self) -> list[str]:
        return self._http.get(EP_OPS).raise_for_status().json()["ops"]

    # --- Model management ---------------------------------------------------

    def list_models(self) -> list[dict]:
        return self._http.get(EP_MODELS).raise_for_status().json()

    def load_model(
        self,
        name: str,
        *,
        model_type: str = "huggingface",
        dtype: str = "float16",
        trust_remote_code: bool = False,
        model_path: str | None = None,
        model_class: str | None = None,
    ) -> dict:
        return self._http.post(
            EP_MODELS_LOAD,
            json={
                "name": name,
                "model_type": model_type,
                "dtype": dtype,
                "trust_remote_code": trust_remote_code,
                "model_path": model_path,
                "model_class": model_class,
            },
        ).raise_for_status().json()

    def unload_model(self, name: str) -> dict:
        return self._http.delete(f"{EP_MODELS}/{name}").raise_for_status().json()

    def model_info(self, name: str) -> dict:
        return self._http.get(f"{EP_MODELS}/{name}/info").raise_for_status().json()

    # --- Inference (model forward pass) -------------------------------------

    def infer(self, model: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        resp = self._http.post(
            EP_INFER,
            content=tensor.encode_batch(inputs),
            headers={"content-type": "application/x-msgpack", "x-model": model},
        )
        resp.raise_for_status()
        return tensor.decode_batch(resp.content)

    # --- Exec (arbitrary torch op) ------------------------------------------

    def exec(
        self,
        op: str,
        inputs: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        """Execute a whitelisted torch operation on the remote MPS device.

        Args:
            op: Operation name (e.g. "matmul", "softmax").
            inputs: Named tensors. Use "0", "1", … for positional args.
            **kwargs: Scalar keyword arguments forwarded to the op.

        Returns:
            Dict of output tensors as numpy arrays.
        """
        headers: dict[str, str] = {
            "content-type": "application/x-msgpack",
            "x-op": op,
        }
        if kwargs:
            headers["x-kwargs"] = json.dumps(kwargs)

        resp = self._http.post(
            EP_EXEC,
            content=tensor.encode_batch(inputs),
            headers=headers,
        )
        resp.raise_for_status()
        return tensor.decode_batch(resp.content)
