"""Generate optional Indian English Piper comparison clips from local model files.

This does not change the active production voice in config.py.
"""

import io
import json
import sys
import time
import wave
from pathlib import Path
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import config
from helpers.request_models import TTSRequest
from services.tts_service import synthesize_speech

SAMPLES = {
    "english_indian_candidate": ("en", "Hello! How are you doing today? I can help you find the information you need."),
    "hinglish_indian_candidate": ("hinglish", "Yaar, mujhe kal office jaana hai. Can you remind me at 9 AM?"),
}


def main() -> None:
    output = Path(__file__).resolve().parent / "voice_samples"
    output.mkdir(parents=True, exist_ok=True)
    name = config.PIPER_INDIAN_ENGLISH_VOICE
    model = Path(config.PIPER_MODEL_DIR) / (name + ".onnx")
    if not model.is_file() or not Path(str(model) + ".json").is_file():
        raise SystemExit(f"Missing candidate Piper model: {model}")
    results = []
    with patch.object(config, "PIPER_ENGLISH_VOICE", name):
        for label, (language, text) in SAMPLES.items():
            started = time.perf_counter()
            audio, mime = synthesize_speech(TTSRequest(
                text=text, language=language, response_format="wav"
            ))
            with wave.open(io.BytesIO(audio), "rb") as wav:
                duration = wav.getnframes() / wav.getframerate()
                rate = wav.getframerate()
            (output / (label + ".wav")).write_bytes(audio)
            result = {"name": label, "voice": name, "language": language,
                      "sample_rate": rate, "duration_seconds": round(duration, 2),
                      "generation_ms": round((time.perf_counter() - started) * 1000, 1),
                      "mime": mime, "file": label + ".wav"}
            results.append(result)
            print(json.dumps(result, ensure_ascii=True), flush=True)
    (output / "candidate_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

