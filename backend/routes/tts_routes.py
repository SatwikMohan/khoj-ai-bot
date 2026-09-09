from fastapi import APIRouter, HTTPException, Response

from helpers.request_models import TTSRequest
from services.tts_service import TTSEngineError, synthesize_speech


router = APIRouter(prefix="/tts", tags=["tts"])


@router.post("/speech")
def text_to_speech(payload: TTSRequest) -> Response:
    try:
        audio, media_type = synthesize_speech(payload)
    except TTSEngineError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return Response(content=audio, media_type=media_type)
