"""Model registry — load / unload / list HuggingFace models on MPS."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mymps.server.device import get_device

log = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    name: str
    model: Any
    tokenizer: Any
    device: torch.device
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
        dtype: str = "float16",
        trust_remote_code: bool = False,
    ) -> LoadedModel:
        if name in self._models:
            log.info("model %s already loaded", name)
            return self._models[name]

        device = get_device()
        torch_dtype = getattr(torch, dtype, torch.float16)

        log.info("loading %s (dtype=%s) → %s …", name, dtype, device)
        tokenizer = AutoTokenizer.from_pretrained(
            name, trust_remote_code=trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        ).to(device)
        model.eval()

        entry = LoadedModel(
            name=name, model=model, tokenizer=tokenizer, device=device,
        )
        self._models[name] = entry
        log.info("loaded %s", name)
        return entry

    # ------------------------------------------------------------------
    def unload(self, name: str) -> bool:
        entry = self._models.pop(name, None)
        if entry is None:
            return False
        del entry.model
        del entry.tokenizer
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        log.info("unloaded %s", name)
        return True

    # ------------------------------------------------------------------
    def get(self, name: str) -> LoadedModel | None:
        return self._models.get(name)

    def list(self) -> list[dict]:
        return [
            {"name": e.name, "device": str(e.device), "loaded_at": e.loaded_at}
            for e in self._models.values()
        ]
