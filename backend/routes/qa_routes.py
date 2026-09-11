import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from helpers.request_models import QARequest, QAResponse
from services.qa_service import QAEngineError, answer_question, stream_answer_events


router = APIRouter(prefix="/qa", tags=["qa"])


@router.post("/ask", response_model=QAResponse)
def ask_question(payload: QARequest) -> QAResponse:
    try:
        return answer_question(
            question=payload.question,
            top_k=payload.top_k,
            temperature=payload.temperature,
            chat_history=payload.chat_history,
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
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except QAEngineError as exc:
            error_event = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            error_event = {"type": "error", "message": f"Streaming failed: {exc}"}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
