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
    remaining = text.strip()
    while len(remaining) > limit:
        prefix = remaining[:limit + 1]
        # Prefer a clause boundary near the limit so prosody does not restart
        # in the middle of a thought. Never split a word to satisfy the limit.
        clauses = [match.end() for match in re.finditer(r"[,;:.!?\u0964](?=\s|$)", prefix)
                   if match.end() >= limit * 0.65]
        cut = clauses[-1] if clauses else prefix.rfind(" ")
        if cut <= 0:
            following = re.search(r"\s", remaining[limit:])
            if following is None:
                break
            cut = limit + following.start()
        yield remaining[:cut].strip()
        remaining = remaining[cut:].strip()
    if remaining:
        yield remaining


def synthesize(text: str, language: str, speed: float, cancelled=None):
    import numpy as np
    import torch

    _, pipelines, voice, device = _runtime()
    pipeline = pipelines["hi" if language == "hi" else "en"]
    parts = []
    previous_piece = ""
    with torch.inference_mode():
        for piece in _pieces(text):
            if cancelled is not None and cancelled.is_set():
                raise RuntimeError("Speech generation was cancelled.")
            piece_audio = []
            for result in pipeline(piece, voice=voice, speed=speed):
                if cancelled is not None and cancelled.is_set():
                    raise RuntimeError("Speech generation was cancelled.")
                if result.audio is not None:
                    piece_audio.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
            if piece_audio:
                if parts:
                    # Ordinary word splits should not sound like a hesitation.
                    pause = 2400 if re.search(r"[.!?\u0964]$", previous_piece) else 240
                    parts.append(np.zeros(pause, dtype=np.float32))
                parts.extend(piece_audio)
                previous_piece = piece
    if not parts:
        raise RuntimeError("Goonj returned no audio.")
    return np.concatenate(parts)
