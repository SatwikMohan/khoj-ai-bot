"""Voice request lifecycle shared by Streamlit and the optional HTTP API."""

import config
import json
import time
from services.request_lifecycle import registry
from services.stt_service import STTEngineError, transcribe_audio


SUFFIXES = {
    "audio/webm": ".webm", "audio/ogg": ".ogg", "audio/wav": ".wav",
    "audio/x-wav": ".wav", "audio/mp4": ".m4a", "audio/mpeg": ".mp3",
}


def begin_voice_interaction(session_id: str, request_id: str) -> dict:
    if not session_id or len(session_id) > 128 or not request_id or len(request_id) > 128:
        raise ValueError("Invalid voice session or request ID.")
    request = registry.begin(session_id, request_id, input_type="audio", status="recording")
    return {"session_id": request.session_id, "request_id": request.request_id, "status": request.status}


def cancel_interaction(session_id: str, request_id: str | None = None) -> bool:
    return registry.cancel(session_id, request_id)


def transcribe_interaction(
    audio: bytes, mime: str, session_id: str, request_id: str,
    language: str | None = None,
) -> dict:
    base_mime = mime.split(";", 1)[0].lower()
    suffix = SUFFIXES.get(base_mime)
    if suffix is None:
        raise ValueError("Unsupported microphone audio format.")
    if len(audio) > int(config.STT_MAX_AUDIO_BYTES):
        raise STTEngineError("Microphone audio exceeds the configured size limit.")
    request = registry.current(session_id, request_id)
    if request is None or request.input_type != "audio":
        raise STTEngineError("This voice interaction was replaced.")
    if not registry.set_status(request, "transcribing"):
        raise STTEngineError("This voice interaction was replaced.")
    started = time.perf_counter()
    print(json.dumps({"event": "stt_started", "request_id": request_id,
                      "bytes": len(audio), "mime": base_mime}), flush=True)
    try:
        result = transcribe_audio(audio, suffix, language, request.cancelled)
    except STTEngineError as exc:
        detail = str(exc).lower()
        stage = ("audio_decode" if "decode" in detail or "audio was received" in detail
                 else "model_load" if "load offline" in detail or "not installed" in detail
                 else "signal_validation" if "speech was not detected" in detail or "background noise" in detail
                 else "inference")
        print(json.dumps({"event": "stt_failed", "request_id": request_id,
                          "stage": stage, "category": type(exc).__name__,
                          "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}), flush=True)
        if request.cancelled.is_set():
            raise STTEngineError("This voice interaction was replaced.") from exc
        raise
    print(json.dumps({"event": "stt_completed", "request_id": request_id,
                      "language": result.get("language"), "audio_duration_s": result.get("duration_seconds"),
                      "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}), flush=True)
    if not registry.set_status(request, "transcribed"):
        raise STTEngineError("This voice interaction was replaced.")
    return result
