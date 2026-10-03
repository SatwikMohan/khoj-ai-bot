import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

from helpers.request_models import QARequest, QAResponse
from services.qa_service import QAEngineError, answer_question, stream_answer_events
from services.request_lifecycle import registry


router = APIRouter(prefix="/qa", tags=["qa"])


@router.post("/ask", response_model=QAResponse)
def ask_question(payload: QARequest) -> QAResponse:
    try:
        return answer_question(
            question=payload.question,
            top_k=payload.top_k,
            temperature=payload.temperature,
            chat_history=payload.chat_history,
            session_id=payload.session_id,
            request_id=payload.request_id,
            input_type=payload.input_type,
        )
    except QAEngineError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/ask/stream")
def stream_question(payload: QARequest) -> StreamingResponse:
    def event_stream():
        try:
            for event in stream_answer_events(
                question=payload.question,
                top_k=payload.top_k,
                temperature=payload.temperature,
                chat_history=payload.chat_history,
                session_id=payload.session_id,
                request_id=payload.request_id,
                input_type=payload.input_type,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except QAEngineError as exc:
            error_event = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            error_event = {"type": "error", "message": f"Streaming failed: {exc}"}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


class VoiceInteraction(BaseModel):
    request_id: str = Field(..., min_length=1, max_length=128)


@router.post("/sessions/{session_id}/interactions")
def begin_voice_interaction(session_id: str, payload: VoiceInteraction) -> dict:
    if len(session_id) > 128:
        raise HTTPException(status_code=422, detail="Session ID is too long.")
    request = registry.begin(session_id, payload.request_id, input_type="audio", status="recording")
    return {"session_id": request.session_id, "request_id": request.request_id, "status": request.status}


@router.post("/sessions/{session_id}/cancel")
def cancel_session(session_id: str, request_id: str | None = None) -> dict:
    return {"cancelled": registry.cancel(session_id, request_id)}
