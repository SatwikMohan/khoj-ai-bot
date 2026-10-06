"""One browser audio owner per Streamlit conversation session."""

import json
from pathlib import Path


_SOURCE = (Path(__file__).resolve().parent / "assets" / "audio_manager.js").read_text(encoding="utf-8")


def audio_manager_script(session_id: str, request_id: str, queue_id: str = "") -> str:
    settings = json.dumps({"sessionId": session_id, "requestId": request_id, "queueId": queue_id}).replace("</", "<\\/")
    return f"<script>{_SOURCE}\nwindow.texminAudioManager=createTexminAudioManager({settings});</script>"
