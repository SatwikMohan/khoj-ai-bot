"""Measure cold/warm offline speech without loading Ollama, STT, or an index."""
import argparse
import io
import json
import platform
import sys
import time
import uuid
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from helpers.request_models import TTSRequest
from services.request_lifecycle import registry
from services.tts_service import synthesize_speech

SAMPLES = (
    ("en", "The assistant searches your local documents and explains the answer clearly. You can ask a follow-up question."),
    ("hi", "यह सहायक आपके सवाल का जवाब देता है। आप एक और सवाल पूछ सकते हैं।"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "voice_samples" / "tts_speed")
    parser.add_argument("--runs", type=int, default=3, help="Runs per language, including one cold run.")
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2")
    if not config.OFFLINE_MODE:
        raise SystemExit("This benchmark requires OFFLINE_MODE=True.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for language, text in SAMPLES:
        for run in range(args.runs):
            session = "tts-benchmark-" + uuid.uuid4().hex
            request = registry.begin(session)
            try:
                started = time.perf_counter()
                audio, media = synthesize_speech(TTSRequest(
                    text=text, language=language, session_id=session,
                    request_id=request.request_id, response_format="wav",
                ))
                elapsed = time.perf_counter() - started
                if media != "audio/wav":
                    raise RuntimeError("Benchmark expects WAV audio.")
                with wave.open(io.BytesIO(audio), "rb") as wav:
                    duration = wav.getnframes() / wav.getframerate()
                if duration <= 0:
                    raise RuntimeError("Empty speech output.")
                row = {"language": language, "run": run,
                       "phase": "cold" if run == 0 else "warm",
                       "generation_ms": round(elapsed * 1000, 1),
                       "audio_seconds": round(duration, 3),
                       "rtf": round(elapsed / duration, 3)}
                (args.output_dir / f"{language}_{run}.wav").write_bytes(audio)
                results.append(row)
                print(json.dumps(row), flush=True)
            except Exception as exc:
                results.append({"language": language, "run": run, "error": str(exc)})
                raise
            finally:
                report = {"engine": config.TTS_ENGINE, "platform": platform.platform(),
                          "machine": platform.machine(), "results": results}
                (args.output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                registry.cancel(session)



if __name__ == "__main__":
    main()
