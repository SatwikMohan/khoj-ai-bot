import config

config.configure_runtime_environment()

from fastapi import FastAPI
import threading
from fastapi.responses import JSONResponse

from routes.qa_routes import router as qa_router
from routes.stt_routes import router as stt_router
from routes.tts_routes import router as tts_router
from services.qa_service import runtime_status, warm_up_qa_engine
from services.reranker_service import reranker_runtime_status
from services.stt_service import stt_runtime_status, warm_up_stt
from services.tts_service import tts_runtime_status, warm_up_tts


app = FastAPI(title="Texmin AI QA Bot")
app.include_router(qa_router)
app.include_router(stt_router)
app.include_router(tts_router)


def system_status() -> dict:
    status = runtime_status()
    status["stt"] = stt_runtime_status()
    status["tts"] = tts_runtime_status()
    status["reranker"] = reranker_runtime_status()
    if (
        status["stt"]["status"] != "ready"
        or status["tts"]["status"] != "ready"
        or status["reranker"]["status"] not in {"ready", "disabled"}
    ):
        status["status"] = "degraded"
    return status


@app.on_event("startup")
def warm_up_local_models() -> None:
    threading.Thread(target=_warm_up_local_models, daemon=True, name="model-warmup").start()


def _warm_up_local_models() -> None:
    try:
        warm_up_qa_engine()
    except Exception as exc:
        print(f"QA warm-up skipped: {exc}")
    try:
        warm_up_stt()
    except Exception as exc:
        print(f"STT warm-up skipped: {exc}")
    try:
        warm_up_tts()
    except Exception as exc:
        print(f"TTS warm-up skipped: {exc}")
    reranker_status = reranker_runtime_status()
    if reranker_status["status"] == "unavailable":
        print(f"Reranker warm-up skipped: {reranker_status.get('error')}")


@app.get("/health")
def health_check() -> dict:
    return system_status()


@app.get("/ready")
def readiness_check():
    # Optional speech/reranking must not make document chat unreachable.
    # Deep component diagnostics remain available through /health.
    status = runtime_status()
    if status["status"] != "ok":
        return JSONResponse(status_code=503, content=status)
    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT, log_level=config.API_LOG_LEVEL)
