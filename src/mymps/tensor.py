"""Tensor serialization: numpy ↔ msgpack bytes."""

from __future__ import annotations

import msgpack
import numpy as np


def encode(arr: np.ndarray) -> bytes:
    """Pack a numpy array into msgpack bytes (shape + dtype + raw data)."""
    return msgpack.packb({
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data": arr.tobytes(),
    })


def decode(buf: bytes) -> np.ndarray:
    """Unpack msgpack bytes back into a numpy array."""
    msg = msgpack.unpackb(buf, raw=True)
    # msgpack may return keys as bytes or str depending on version
    shape = msg.get("shape") or msg.get(b"shape")
    dtype = msg.get("dtype") or msg.get(b"dtype")
    data = msg.get("data") or msg.get(b"data")
    if isinstance(dtype, bytes):
        dtype = dtype.decode()
    return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)


def encode_batch(arrays: dict[str, np.ndarray]) -> bytes:
    """Pack a named dict of arrays."""
    return msgpack.packb({
        name: {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "data": arr.tobytes(),
        }
        for name, arr in arrays.items()
    })


def decode_batch(buf: bytes) -> dict[str, np.ndarray]:
    """Unpack a named dict of arrays."""
    msg = msgpack.unpackb(buf, raw=True)
    out: dict[str, np.ndarray] = {}
    for key, val in msg.items():
        name = key.decode() if isinstance(key, bytes) else key
        shape = val.get("shape") or val.get(b"shape")
        dtype = val.get("dtype") or val.get(b"dtype")
        data = val.get("data") or val.get(b"data")
        if isinstance(dtype, bytes):
            dtype = dtype.decode()
        out[name] = np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)
    return out
