import os
import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIBE_SEMAPHORE = threading.BoundedSemaphore(
    max(1, int(os.getenv("STT_MAX_CONCURRENT", "2")))
)


class STTEngineError(RuntimeError):
    pass


def _load_environment() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


@lru_cache(maxsize=1)
def _model():
    _load_environment()
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise STTEngineError(
            "Offline speech recognition is not installed. Install faster-whisper."
        ) from exc

    model_name = os.getenv("WHISPER_MODEL", "large-v3-turbo")
    device = os.getenv("WHISPER_DEVICE", "cuda")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
    try:
        return WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=int(os.getenv("WHISPER_CPU_THREADS", "4")),
            num_workers=int(os.getenv("WHISPER_WORKERS", "1")),
            download_root=os.getenv("WHISPER_MODEL_DIR") or None,
            local_files_only=os.getenv("OFFLINE_MODE", "true").lower() in {"1", "true", "yes", "on"},
        )
    except Exception as exc:
        raise STTEngineError(f"Could not load offline ASR model '{model_name}': {exc}") from exc


def warm_up_stt() -> None:
    _load_environment()
    if os.getenv("STT_WARMUP_ON_STARTUP", "true").lower() not in {"1", "true", "yes", "on"}:
        return
    _model()


def stt_runtime_status() -> dict:
    try:
        _model()
        return {
            "status": "ready",
            "model": os.getenv("WHISPER_MODEL", "large-v3-turbo"),
            "device": os.getenv("WHISPER_DEVICE", "cuda"),
        }
    except STTEngineError as exc:
        return {"status": "unavailable", "error": str(exc)}


def transcribe_audio(
    audio: bytes,
    suffix: str = ".webm",
    language: str | None = None,
) -> dict:
    _load_environment()
    max_bytes = int(os.getenv("STT_MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))
    if not audio:
        raise STTEngineError("No audio was received.")
    if len(audio) > max_bytes:
        raise STTEngineError(f"Audio exceeds the {max_bytes // (1024 * 1024)} MB limit.")

    safe_suffix = suffix if suffix in {".webm", ".wav", ".mp3", ".ogg", ".m4a"} else ".webm"
    path = ""
    started_at = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(suffix=safe_suffix, delete=False) as audio_file:
            audio_file.write(audio)
            path = audio_file.name

        with TRANSCRIBE_SEMAPHORE:
            segments, info = _model().transcribe(
                path,
                language=language or None,
                beam_size=int(os.getenv("WHISPER_BEAM_SIZE", "1")),
                best_of=int(os.getenv("WHISPER_BEST_OF", "1")),
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": int(os.getenv("WHISPER_VAD_SILENCE_MS", "350")),
                    "speech_pad_ms": int(os.getenv("WHISPER_VAD_SPEECH_PAD_MS", "180")),
                },
                condition_on_previous_text=False,
                word_timestamps=False,
            )
            text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
    except STTEngineError:
        raise
    except Exception as exc:
        raise STTEngineError(f"Offline transcription failed: {exc}") from exc
    finally:
        if path:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass

    if not text:
        raise STTEngineError("Speech was not detected in the recording.")

    return {
        "text": text,
        "language": getattr(info, "language", language or "unknown"),
        "language_probability": round(float(getattr(info, "language_probability", 0.0)), 4),
        "duration_seconds": round(float(getattr(info, "duration", 0.0)), 3),
        "latency_ms": round((time.perf_counter() - started_at) * 1000, 1),
    }
