from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from services.stt_service import STTEngineError, transcribe_audio


router = APIRouter(prefix="/stt", tags=["stt"])


@router.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str | None = Form(None),
) -> dict:
    try:
        content = await audio.read()
        suffix = Path(audio.filename or "recording.webm").suffix.lower()
        return transcribe_audio(content, suffix=suffix, language=language)
    except STTEngineError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
