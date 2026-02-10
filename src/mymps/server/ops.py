"""Whitelist of safe torch operations and dispatcher for /exec."""

from __future__ import annotations

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Each entry maps an op name to the callable that implements it.
# Only operations in this dict can be invoked via /exec.
# ---------------------------------------------------------------------------

OPERATIONS: dict[str, callable] = {
    # ── Element-wise math ─────────────────────────────────────────────
    "abs": torch.abs,
    "neg": torch.neg,
    "exp": torch.exp,
    "log": torch.log,
    "sqrt": torch.sqrt,
    "rsqrt": torch.rsqrt,
    "sin": torch.sin,
    "cos": torch.cos,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "sign": torch.sign,
    "floor": torch.floor,
    "ceil": torch.ceil,
    "round": torch.round,
    "clamp": torch.clamp,
    "pow": torch.pow,
    "add": torch.add,
    "sub": torch.sub,
    "mul": torch.mul,
    "div": torch.div,
    "remainder": torch.remainder,
    "maximum": torch.maximum,
    "minimum": torch.minimum,
    # ── Reductions ────────────────────────────────────────────────────
    "sum": torch.sum,
    "mean": torch.mean,
    "prod": torch.prod,
    "amax": torch.amax,
    "amin": torch.amin,
    "argmax": torch.argmax,
    "argmin": torch.argmin,
    "norm": torch.norm,
    # ── Linear algebra ────────────────────────────────────────────────
    "matmul": torch.matmul,
    "mm": torch.mm,
    "bmm": torch.bmm,
    "mv": torch.mv,
    "dot": torch.dot,
    "einsum": torch.einsum,
    "transpose": torch.transpose,
    "svd": torch.linalg.svd,
    "eig": torch.linalg.eig,
    "inv": torch.linalg.inv,
    "det": torch.linalg.det,
    "solve": torch.linalg.solve,
    "cholesky": torch.linalg.cholesky,
    "qr": torch.linalg.qr,
    # ── Activations (functional) ──────────────────────────────────────
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "softmax": F.softmax,
    "log_softmax": F.log_softmax,
    "layer_norm": F.layer_norm,
    # ── Convolution / Pooling ─────────────────────────────────────────
    "conv1d": F.conv1d,
    "conv2d": F.conv2d,
    "conv3d": F.conv3d,
    "avg_pool2d": F.avg_pool2d,
    "max_pool2d": F.max_pool2d,
    "adaptive_avg_pool2d": F.adaptive_avg_pool2d,
    # ── Shape / indexing ──────────────────────────────────────────────
    "reshape": torch.reshape,
    "flatten": torch.flatten,
    "squeeze": torch.squeeze,
    "unsqueeze": torch.unsqueeze,
    "cat": torch.cat,
    "stack": torch.stack,
    "split": torch.split,
    "chunk": torch.chunk,
    "permute": lambda x, dims: x.permute(*dims),
    "contiguous": lambda x: x.contiguous(),
    # ── Creation helpers ──────────────────────────────────────────────
    "zeros_like": torch.zeros_like,
    "ones_like": torch.ones_like,
    "rand_like": torch.rand_like,
    # ── FFT ───────────────────────────────────────────────────────────
    "fft": torch.fft.fft,
    "ifft": torch.fft.ifft,
    "rfft": torch.fft.rfft,
    "irfft": torch.fft.irfft,
    # ── Comparison ────────────────────────────────────────────────────
    "eq": torch.eq,
    "ne": torch.ne,
    "gt": torch.gt,
    "lt": torch.lt,
    "ge": torch.ge,
    "le": torch.le,
    "where": torch.where,
    # ── Other ─────────────────────────────────────────────────────────
    "sort": torch.sort,
    "topk": torch.topk,
    "cumsum": torch.cumsum,
    "cumprod": torch.cumprod,
}


def list_ops() -> list[str]:
    """Return sorted list of available operation names."""
    return sorted(OPERATIONS.keys())


def get_op(name: str) -> callable:
    """Look up an operation by name. Raises KeyError if not whitelisted."""
    try:
        return OPERATIONS[name]
    except KeyError:
        raise KeyError(f"operation {name!r} is not available; see /ops for the full list")
