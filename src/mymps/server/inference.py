"""Generic forward-pass inference (tensor-in → tensor-out)."""

from __future__ import annotations

import numpy as np
import torch

from mymps.server.models import ModelRegistry


def run_inference(
    registry: ModelRegistry,
    model_name: str,
    inputs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Run a forward pass and return output tensors as numpy arrays."""
    entry = registry.get(model_name)
    if entry is None:
        raise KeyError(f"model {model_name!r} not loaded")

    torch_inputs = {
        k: torch.from_numpy(v).to(entry.device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = entry.model(**torch_inputs)

    # outputs may be a ModelOutput dataclass — convert to dict of numpy
    result: dict[str, np.ndarray] = {}
    if hasattr(outputs, "logits"):
        result["logits"] = outputs.logits.cpu().float().numpy()
    elif hasattr(outputs, "last_hidden_state"):
        result["last_hidden_state"] = outputs.last_hidden_state.cpu().float().numpy()
    else:
        # generic: try to convert whatever we get
        for key in outputs:
            val = outputs[key]
            if isinstance(val, torch.Tensor):
                result[key] = val.cpu().float().numpy()
    return result
