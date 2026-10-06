from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

import config
from services.stt_service import STTEngineError, transcribe_audio
from services.voice_interaction import SUFFIXES, transcribe_interaction


router = APIRouter(prefix="/stt", tags=["stt"])


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
    try:
        content = await audio.read(int(config.STT_MAX_AUDIO_BYTES) + 1)
        if session_id and request_id:
            return await run_in_threadpool(
                transcribe_interaction, content, mime, session_id, request_id, language or None,
            )
        return await run_in_threadpool(transcribe_audio, content, suffix, language or None)
    except STTEngineError as exc:
        if "replaced" in str(exc).lower():
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise HTTPException(status_code=422, detail=str(exc)) from exc
