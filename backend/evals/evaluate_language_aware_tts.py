"""Generate a small offline Hindi/English/Hinglish listening set.

Run from any directory with: python backend/evals/evaluate_language_aware_tts.py
The WAV files are local review artifacts; acoustic success is not a listening score.
"""

import argparse
import io
import json
import sys
import time
import wave
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from helpers.request_models import TTSRequest
from services.speech_language import speech_phrases, normalize_speech_text
from services.tts_service import synthesize_speech

SAMPLES = {
    "english": ("en", "Hello! How are you doing today? I can help you find the information you need."),
    "hindi": ("hi", "नमस्ते! आज आप कैसे हैं? मैं आपकी जानकारी खोजने में मदद कर सकता हूँ।"),
    "hinglish": ("hinglish", "Yaar, mujhe kal office jaana hai. Can you remind me at 9 AM?"),
    "mixed_script": ("hi", "मुझे आज एक Python project complete करना है। Can you help me?"),
    "technical": ("hi", "FastAPI backend, vector database, GPU acceleration और machine learning model को configure करना है।"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "voice_samples")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for name, (language, text) in SAMPLES.items():
        started = time.perf_counter()
        speech_text = normalize_speech_text(text, language)
        entry = {"name": name, "language": language, "display_text": text,
                 "speech_text": speech_text,
                 "phrases": speech_phrases(speech_text, language)}
        try:
            audio, mime = synthesize_speech(TTSRequest(
                text=text, language=language, response_format="wav", max_words=140
            ))
            with wave.open(io.BytesIO(audio), "rb") as wav:
                entry["sample_rate"] = wav.getframerate()
                entry["duration_seconds"] = round(wav.getnframes() / wav.getframerate(), 2)
            entry["mime"] = mime
            entry["bytes"] = len(audio)
            (output / (name + ".wav")).write_bytes(audio)
            entry["file"] = name + ".wav"
        except Exception as exc:
            entry["error"] = str(exc)
        entry["generation_ms"] = round((time.perf_counter() - started) * 1000, 1)
        results.append(entry)
        print(json.dumps(entry, ensure_ascii=True), flush=True)
    (output / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

