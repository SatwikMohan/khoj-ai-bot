"""Run the configured offline STT model on a real audio fixture.

Usage: python backend/scripts/transcribe_sample.py recording.wav
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.stt_service import transcribe_audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--language", help="Optional ISO language code; default is automatic")
    args = parser.parse_args()
    path = args.audio.resolve(strict=True)
    if path.suffix.lower() not in {".wav", ".webm", ".ogg", ".m4a", ".mp3"}:
        parser.error("Expected WAV, WebM, Ogg, M4A, or MP3 audio")
    result = transcribe_audio(path.read_bytes(), path.suffix.lower(), args.language)
    print(json.dumps(result, ensure_ascii=False, indent=2))
