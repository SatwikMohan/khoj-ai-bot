"""Pull the configured Ollama models without loading a second config source."""

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def provision_models() -> None:
    base_url = config.OLLAMA_BASE_URL.rstrip("/")
    with urllib.request.urlopen(
        f"{base_url}/api/tags", timeout=config.OLLAMA_REQUEST_TIMEOUT_SECONDS
    ) as response:
        installed = {item["name"] for item in json.load(response).get("models", [])}

    for model in dict.fromkeys((config.OLLAMA_CHAT_MODEL, config.OLLAMA_EMBED_MODEL)):
        if model in installed or (":" not in model and f"{model}:latest" in installed):
            print(f"Already available: {model}", flush=True)
            continue
        print(f"Downloading configured model: {model}", flush=True)
        request = urllib.request.Request(
            f"{base_url}/api/pull",
            data=json.dumps({"model": model, "stream": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        status = ""
        with urllib.request.urlopen(request, timeout=config.OLLAMA_REQUEST_TIMEOUT_SECONDS) as response:
            for line in response:
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("error"):
                    raise RuntimeError(f"Could not pull {model}: {event['error']}")
                current = event.get("status", "")
                if current and current != status:
                    print(f"{model}: {current}", flush=True)
                    status = current
        if status != "success":
            raise RuntimeError(f"Download ended without success: {model}")


if __name__ == "__main__":
    provision_models()
