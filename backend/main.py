from fastapi import FastAPI

from routes.qa_routes import router as qa_router
from routes.tts_routes import router as tts_router
from services.qa_service import warm_up_qa_engine


app = FastAPI(title="Texmin AI QA Bot")
app.include_router(qa_router)
app.include_router(tts_router)


@app.on_event("startup")
def warm_up_local_models() -> None:
    try:
        warm_up_qa_engine()
    except Exception as exc:
        print(f"QA warm-up skipped: {exc}")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
