import config
"""Ollama capability discovery, independent of model name or parameter count."""

import json
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
    config.validate_config()
    metadata = model_metadata(base_url, model)
    if config.OFFLINE_MODE and metadata.get("remote_host"):
        raise ValueError(f"{model} is a remote model. Offline mode requires locally downloaded weights.")
    capabilities = metadata.get("capabilities", [])
    if not capabilities:
        raise ValueError(f"{model} has no capability metadata; cannot verify non-reasoning mode.")
    if "completion" not in capabilities:
        raise ValueError(f"{model} is not a chat/completion model. Set OLLAMA_CHAT_MODEL to a chat model.")
    thinking = metadata.get("thinking")
    advertised_thinking = "thinking" in capabilities
    if isinstance(thinking, dict):
        values = thinking.get("values", [])
        advertised_thinking = advertised_thinking or any(value not in (False, None, "false", "none") for value in values)
        advertised_thinking = advertised_thinking or thinking.get("default") not in (False, None, "false", "none")
    elif thinking:
        advertised_thinking = True
    if requested is True or advertised_thinking:
        raise ValueError(f"{model} supports thinking; configure a standard non-reasoning chat model.")
    return False
