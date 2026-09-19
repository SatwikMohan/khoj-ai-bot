"""Run FastAPI and Streamlit in one process so they share backend model caches."""

import os
import sys
import threading
from pathlib import Path

import uvicorn


APP_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = APP_ROOT / "backend"
FRONTEND_APP = APP_ROOT / "frontend" / "app.py"

sys.path.insert(0, str(BACKEND_DIR))

from main import app as fastapi_app  # noqa: E402


def main() -> None:
    api_server = uvicorn.Server(
        uvicorn.Config(
            fastapi_app,
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            log_level=os.getenv("API_LOG_LEVEL", "info"),
        )
    )
    api_thread = threading.Thread(
        target=api_server.run,
        name="texmin-fastapi",
        daemon=True,
    )
    api_thread.start()

    from streamlit.web import cli as streamlit_cli

    sys.argv = [
        "streamlit",
        "run",
        str(FRONTEND_APP),
        "--server.address=0.0.0.0",
        "--server.port=8501",
        "--browser.gatherUsageStats=false",
    ]
    try:
        streamlit_cli.main()
    finally:
        api_server.should_exit = True
        api_thread.join(timeout=10)


if __name__ == "__main__":
    main()
