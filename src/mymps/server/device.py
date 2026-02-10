"""MPS device detection and info."""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Return the MPS device, falling back to CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info() -> dict:
    """Return a dict describing the current accelerator."""
    mps_available = torch.backends.mps.is_available()
    return {
        "mps_available": mps_available,
        "mps_built": torch.backends.mps.is_built(),
        "device": "mps" if mps_available else "cpu",
        "torch_version": torch.__version__,
    }
