"""Model registry — load / unload / list models on MPS.

Supports three loading strategies:
  - huggingface : AutoModel.from_pretrained (generic, not CausalLM)
  - torchscript : torch.jit.load from a local .pt file
  - checkpoint  : state_dict + model class via importlib
"""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from mymps.server.device import get_device

log = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    name: str
    model: Any
    device: torch.device
    model_type: str  # "huggingface" | "torchscript" | "checkpoint"
    metadata: dict = field(default_factory=dict)
    loaded_at: float = field(default_factory=time.time)


class ModelRegistry:
    """Thread-safe (GIL-safe) registry of loaded models."""

    def __init__(self) -> None:
        self._models: dict[str, LoadedModel] = {}

    # ------------------------------------------------------------------
    def load(
        self,
        name: str,
        *,
        model_type: str = "huggingface",
        dtype: str = "float16",
        trust_remote_code: bool = False,
        model_path: str | None = None,
        model_class: str | None = None,
    ) -> LoadedModel:
        if name in self._models:
            log.info("model %s already loaded", name)
            return self._models[name]

        device = get_device()
        torch_dtype = getattr(torch, dtype, torch.float16)

        if model_type == "huggingface":
            model = self._load_huggingface(name, torch_dtype, trust_remote_code)
        elif model_type == "torchscript":
            model = self._load_torchscript(model_path or name, device)
        elif model_type == "checkpoint":
            model = self._load_checkpoint(
                model_path or name, model_class, torch_dtype,
            )
        else:
            raise ValueError(f"unknown model_type {model_type!r}")

        if model_type != "torchscript":
            model = model.to(device)
        model.eval()

        metadata: dict[str, Any] = {
            "dtype": dtype,
            "device": str(device),
        }
        if model_type == "huggingface":
            metadata["hf_model"] = name
        if model_path:
            metadata["model_path"] = model_path

        entry = LoadedModel(
            name=name,
            model=model,
            device=device,
            model_type=model_type,
            metadata=metadata,
        )
        self._models[name] = entry
        log.info("loaded %s (type=%s) → %s", name, model_type, device)
        return entry

    # ------------------------------------------------------------------
    # Loading strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _load_huggingface(
        name: str,
        torch_dtype: torch.dtype,
        trust_remote_code: bool,
    ) -> Any:
        from transformers import AutoModel

        log.info("loading HuggingFace model %s (dtype=%s) …", name, torch_dtype)
        return AutoModel.from_pretrained(
            name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )

    @staticmethod
    def _load_torchscript(path: str, device: torch.device) -> Any:
        log.info("loading TorchScript model from %s …", path)
        return torch.jit.load(path, map_location=device)

    @staticmethod
    def _load_checkpoint(
        path: str,
        model_class: str | None,
        torch_dtype: torch.dtype,
    ) -> Any:
        if not model_class:
            raise ValueError("model_class is required for checkpoint loading")
        module_name, cls_name = model_class.rsplit(".", 1)
        mod = importlib.import_module(module_name)
        cls = getattr(mod, cls_name)
        log.info("loading checkpoint %s → %s …", path, model_class)
        model = cls()
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model.to(torch_dtype)

    # ------------------------------------------------------------------
    def unload(self, name: str) -> bool:
        entry = self._models.pop(name, None)
        if entry is None:
            return False
        del entry.model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        log.info("unloaded %s", name)
        return True

    # ------------------------------------------------------------------
    def get(self, name: str) -> LoadedModel | None:
        return self._models.get(name)

    def info(self, name: str) -> dict | None:
        entry = self._models.get(name)
        if entry is None:
            return None
        return {
            "name": entry.name,
            "model_type": entry.model_type,
            "device": str(entry.device),
            "loaded_at": entry.loaded_at,
            "metadata": entry.metadata,
        }

    def list(self) -> list[dict]:
        return [
            {
                "name": e.name,
                "model_type": e.model_type,
                "device": str(e.device),
                "loaded_at": e.loaded_at,
            }
            for e in self._models.values()
        ]
