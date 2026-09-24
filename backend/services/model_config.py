import config
"""Ollama capability discovery, independent of model name or parameter count."""

import json
import os
import urllib.request
from functools import lru_cache


@lru_cache(maxsize=16)
def model_metadata(base_url: str, model: str) -> dict:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/show",
        data=json.dumps({"model": model}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def thinking_setting(base_url: str, model: str, requested: bool | None = None):
    value = config.OLLAMA_THINK.strip().lower()
    metadata = model_metadata(base_url, model)
    if config.OFFLINE_MODE and metadata.get("remote_host"):
        raise ValueError(f"{model} is a remote model. Offline mode requires locally downloaded weights.")
    capabilities = metadata.get("capabilities", [])
    if capabilities and "completion" not in capabilities:
        raise ValueError(f"{model} is not a chat/completion model. Set OLLAMA_CHAT_MODEL to a chat model.")
    if requested is None and value != "auto":
        if value in {"default", "none"}:
            return None
        if value in {"true", "false"}:
            return value == "true"
        return value  # Model-specific effort level, explicitly requested.
    if "thinking" not in capabilities:
        return None  # Older servers/non-thinking models: omit the option.
    enabled = requested if requested is not None else config.QA_REASONING_ENABLED
    thinking = metadata.get("thinking", {})
    if isinstance(thinking, dict):
        values = thinking.get("values", [])
        if values:
            if any(value is enabled for value in values):
                return enabled
            if not enabled:
                for effort in ("none", "minimal", "low"):
                    if effort in values:
                        return effort
            return thinking.get("default")
    return enabled
