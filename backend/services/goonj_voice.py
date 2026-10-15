"""Offline Indian English/Hindi voice using one local Goonj speaker pack."""
import os
import re
from functools import lru_cache
from pathlib import Path

import config


def required_files() -> tuple[Path, ...]:
    root = Path(config.GOONJ_MODEL_DIR)
    return root / "config.json", root / "kokoro_hindi_final.pth", root / "voices" / (config.GOONJ_VOICE + ".pt")


@lru_cache(maxsize=1)
def _runtime():
    files = required_files()
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError("Missing offline Goonj files: " + ", ".join(missing))
    if os.name == "nt":
        import espeakng_loader
        os.environ["ESPEAK_DATA_PATH"] = str(Path(espeakng_loader.get_data_path()).parent)
    import torch
    from kokoro import KModel, KPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KModel(repo_id="hexgrad/Kokoro-82M", config=str(files[0]), model=str(files[1])).to(device).eval()
    english = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=model, device=device)
    hindi = KPipeline(lang_code="h", repo_id="hexgrad/Kokoro-82M", model=model, device=device)
    return model, {"en": english, "hi": hindi}, str(files[2]), device


def _pieces(text: str, limit: int = 150):
    words = re.findall(r"\S+\s*", text)
    chunk = ""
    for word in words:
        if chunk and len(chunk) + len(word) > limit:
            yield chunk.strip()
            chunk = ""
        chunk += word
    if chunk.strip():
        yield chunk.strip()


def synthesize(text: str, language: str, speed: float, cancelled=None):
    import numpy as np
    import torch

    _, pipelines, voice, device = _runtime()
    pipeline = pipelines["hi" if language == "hi" else "en"]
    parts = []
    with torch.inference_mode():
        for piece in _pieces(text):
            if cancelled is not None and cancelled.is_set():
                raise RuntimeError("Speech generation was cancelled.")
            for result in pipeline(piece, voice=voice, speed=speed):
                if cancelled is not None and cancelled.is_set():
                    raise RuntimeError("Speech generation was cancelled.")
                if result.audio is not None:
                    parts.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
    if not parts:
        raise RuntimeError("Goonj returned no audio.")
    pause = np.zeros(2400, dtype=np.float32)
    return np.concatenate([item for index, part in enumerate(parts)
                           for item in ((pause, part) if index else (part,))])
