"""Generic forward-pass inference (tensor-in → tensor-out).

Handles any model type: HuggingFace, TorchScript, or checkpoint.
Output is normalised to dict[str, numpy] regardless of what the model returns.
"""

from __future__ import annotations

import numpy as np
import torch

from mymps.server.models import ModelRegistry


def _tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def _normalise_output(raw: object) -> dict[str, np.ndarray]:
    """Convert model output (Tensor | tuple | dict / ModelOutput) to named numpy arrays."""
    if isinstance(raw, torch.Tensor):
        return {"output": _tensor_to_numpy(raw)}

    if isinstance(raw, (tuple, list)):
        return {
            str(i): _tensor_to_numpy(t) if isinstance(t, torch.Tensor) else np.array(t)
            for i, t in enumerate(raw)
        }

    # dict-like (includes HuggingFace ModelOutput)
    if hasattr(raw, "items"):
        result: dict[str, np.ndarray] = {}
        for key, val in raw.items():
            if isinstance(val, torch.Tensor):
                result[str(key)] = _tensor_to_numpy(val)
        return result

    # If we have known attributes, try them
    for attr in ("logits", "last_hidden_state", "hidden_states"):
        val = getattr(raw, attr, None)
        if isinstance(val, torch.Tensor):
            return {attr: _tensor_to_numpy(val)}

    raise TypeError(f"cannot convert model output of type {type(raw).__name__}")


def run_inference(
    registry: ModelRegistry,
    model_name: str,
    inputs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Run a forward pass and return output tensors as numpy arrays."""
    entry = registry.get(model_name)
    if entry is None:
        raise KeyError(f"model {model_name!r} not loaded")

    # Separate positional args ("0", "1", …) from keyword args
    positional: list[tuple[int, torch.Tensor]] = []
    keyword: dict[str, torch.Tensor] = {}

    for k, v in inputs.items():
        t = torch.from_numpy(v).to(entry.device)
        if k.isdigit():
            positional.append((int(k), t))
        else:
            keyword[k] = t

    with torch.no_grad():
        if positional:
            positional.sort(key=lambda x: x[0])
            args = [t for _, t in positional]
            outputs = entry.model(*args, **keyword)
        else:
            outputs = entry.model(**keyword)

    return _normalise_output(outputs)
