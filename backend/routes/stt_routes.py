from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

import config
from services.request_lifecycle import registry
from services.stt_service import STTEngineError, transcribe_audio


router = APIRouter(prefix="/stt", tags=["stt"])
SUFFIXES = {
    "audio/webm": ".webm", "audio/ogg": ".ogg", "audio/wav": ".wav",
    "audio/x-wav": ".wav", "audio/mp4": ".m4a", "audio/mpeg": ".mp3",
}


@router.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str | None = Form(None),
    session_id: str | None = Form(None),
    request_id: str | None = Form(None),
) -> dict:
    if bool(session_id) != bool(request_id):
        raise HTTPException(status_code=422, detail="Session ID and request ID must be supplied together.")
    mime = (audio.content_type or "").split(";", 1)[0].lower()
    suffix = Path(audio.filename or "").suffix.lower()
    if mime not in SUFFIXES or suffix != SUFFIXES[mime]:
        raise HTTPException(status_code=415, detail="Unsupported or mismatched microphone audio format.")
    request = registry.current(session_id, request_id) if session_id and request_id else None
    if session_id and (request is None or request.input_type != "audio"):
        raise HTTPException(status_code=409, detail="This voice interaction was replaced.")
    if request and not registry.set_status(request, "transcribing"):
        raise HTTPException(status_code=409, detail="This voice interaction was replaced.")
    try:
        content = await audio.read(int(config.STT_MAX_AUDIO_BYTES) + 1)
        result = await run_in_threadpool(
            transcribe_audio, content, suffix, language or None,
            request.cancelled if request else None,
        )
        if request and not registry.set_status(request, "transcribed"):
            raise HTTPException(status_code=409, detail="This voice interaction was replaced.")
        return result
    except STTEngineError as exc:
        if request and request.cancelled.is_set():
            raise HTTPException(status_code=409, detail="This voice interaction was replaced.") from exc
        raise HTTPException(status_code=422, detail=str(exc)) from exc
