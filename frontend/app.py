import os
import base64
import json
import mimetypes
import re
import sys
import uuid
import time
from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from html import escape

import markdown
import requests
import streamlit as st
import streamlit.components.v1 as components

# Import the same configuration used by the backend, also in local Streamlit runs.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import config
from services.audio_chunks import join_audio_chunks
from services.language_service import detect_language, response_language
from services.speech_chunker import IncrementalSpeechSegments
config.configure_runtime_environment()


DEFAULT_API_URL = config.QA_API_URL
BACKEND_CALL_MODE = config.BACKEND_CALL_MODE.strip().lower()
ASSISTANT_NAME = config.ASSISTANT_NAME.strip() or "Khoj"
DEFAULT_QA_TEMPERATURE = max(0.0, min(1.0, float(config.QA_TEMPERATURE)))
DEFAULT_TTS_ENGINE = config.TTS_ENGINE.strip().lower()
DEFAULT_TTS_VOICE = config.TTS_VOICE
if DEFAULT_TTS_ENGINE in {"auto", "piper", "espeak"}:
    DEFAULT_TTS_VOICE = "configured"
DEFAULT_AUDIO_MIME = "audio/mpeg" if DEFAULT_TTS_ENGINE in {"edge", "edge-tts"} else "audio/wav"
APP_DIR = Path(__file__).resolve().parent
DEFAULT_AVATAR_MODEL_PATH = APP_DIR / "assets" / "avatar.glb"
VOICE_QUERY_COMPONENT_DIR = APP_DIR / "voice_query_component"
voice_query_component = components.declare_component(
    "texmin_voice_query",
    path=str(VOICE_QUERY_COMPONENT_DIR),
)
MODEL_MIME_TYPES = {
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".fbx": "application/octet-stream",
    ".obj": "text/plain",
}
MODEL_SEARCH_PATTERNS = (
    "avatar.glb",
    "avatar.gltf",
    "avatar.fbx",
    "avatar.obj",
    "*.glb",
    "*.gltf",
    "*.fbx",
    "*.obj",
)

ONLINE_VOICE_OPTIONS = {
    "English India - Neerja": "en-IN-NeerjaNeural",
    "English India - Prabhat": "en-IN-PrabhatNeural",
    "English multilingual - Ava": "en-US-AvaMultilingualNeural",
    "Hindi India - Swara": "hi-IN-SwaraNeural",
    "Hindi India - Madhur": "hi-IN-MadhurNeural",
    "Tamil India - Pallavi": "ta-IN-PallaviNeural",
    "Telugu India - Shruti": "te-IN-ShrutiNeural",
    "Marathi India - Aarohi": "mr-IN-AarohiNeural",
    "Bengali India - Tanishaa": "bn-IN-TanishaaNeural",
    "Gujarati India - Dhwani": "gu-IN-DhwaniNeural",
    "Spanish Spain - Elvira": "es-ES-ElviraNeural",
    "French France - Denise": "fr-FR-DeniseNeural",
    "German Germany - Katja": "de-DE-KatjaNeural",
    "Arabic Saudi - Zariyah": "ar-SA-ZariyahNeural",
    "Chinese Mandarin - Xiaoxiao": "zh-CN-XiaoxiaoNeural",
    "Japanese Japan - Nanami": "ja-JP-NanamiNeural",
}
OFFLINE_VOICE_OPTIONS = {
    "Automatic offline voice": "configured",
    "Natural English - Heart": "af_heart",
    "Natural English - Bella": "af_bella",
    "Natural English - Nicole": "af_nicole",
    "Natural English - Adam": "am_adam",
    "Natural British English - Emma": "bf_emma",
    "Natural British English - George": "bm_george",
}
VOICE_OPTIONS = (
    OFFLINE_VOICE_OPTIONS
    if DEFAULT_TTS_ENGINE in {"auto", "piper", "espeak", "kokoro"}
    else ONLINE_VOICE_OPTIONS
)
CUSTOM_VOICE_LABEL = "Custom voice name"
if DEFAULT_TTS_ENGINE in {"auto", "piper", "espeak"}:
    VOICE_OPTIONS = {"Automatic offline voice": "configured"}
TONE_OPTIONS = ["neutral", "warm", "cheerful", "calm", "serious", "energetic", "custom"]


st.set_page_config(
    page_title=ASSISTANT_NAME,
    page_icon="K",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    :root {
        --app-bg: #0b0f19;
        --panel: #101827;
        --panel-soft: #141f31;
        --border: #263449;
        --text: #edf3ff;
        --muted: #9fb0c7;
        --accent: #52d6b4;
        --accent-strong: #23b99a;
        --user-bg: #1e3a5f;
        --assistant-bg: #111c2e;
        --assistant-edge: #37506d;
        --danger-bg: #341b24;
        --danger-border: #7f334a;
    }

    .stApp {
        background:
            radial-gradient(circle at 20% 0%, rgba(82, 214, 180, 0.16), transparent 28%),
            radial-gradient(circle at 82% 12%, rgba(99, 141, 255, 0.15), transparent 30%),
            var(--app-bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(11, 15, 25, 0.78);
        backdrop-filter: blur(14px);
    }

    .main .block-container {
        max-width: 980px;
        padding: 2rem 1.2rem 7rem;
    }

    .chat-shell {
        border: 1px solid rgba(159, 176, 199, 0.18);
        background: rgba(16, 24, 39, 0.72);
        box-shadow: 0 28px 80px rgba(0, 0, 0, 0.34);
        border-radius: 18px;
        padding: 1.1rem;
    }

    .app-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1rem;
    }

    .brand-lockup {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .brand-mark {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        display: grid;
        place-items: center;
        background: linear-gradient(145deg, #52d6b4, #5c7cff);
        color: #06101c;
        font-weight: 800;
        font-size: 1.15rem;
        box-shadow: 0 10px 24px rgba(82, 214, 180, 0.18);
    }

    .brand-title {
        color: var(--text);
        font-size: 1.28rem;
        line-height: 1.2;
        font-weight: 760;
        margin: 0;
    }

    .brand-subtitle {
        color: var(--muted);
        font-size: 0.92rem;
        margin-top: 0.16rem;
    }

    .status-pill {
        color: #b7ffeb;
        background: rgba(82, 214, 180, 0.1);
        border: 1px solid rgba(82, 214, 180, 0.28);
        border-radius: 999px;
        padding: 0.42rem 0.72rem;
        font-size: 0.82rem;
        white-space: nowrap;
    }

    .message-row {
        display: flex;
        margin: 0.82rem 0;
    }

    .message-row.user {
        justify-content: flex-end;
    }

    .message-row.assistant {
        justify-content: flex-start;
    }

    .bubble {
        max-width: min(760px, 88%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.92rem 1rem;
        line-height: 1.58;
        font-size: 0.98rem;
        color: var(--text);
        overflow-wrap: anywhere;
    }

    .bubble.user {
        background: linear-gradient(145deg, var(--user-bg), #244a78);
        border-color: rgba(116, 167, 225, 0.36);
        border-bottom-right-radius: 6px;
    }

    .bubble.assistant {
        background: linear-gradient(145deg, var(--assistant-bg), #16243a);
        border-color: rgba(82, 214, 180, 0.24);
        border-bottom-left-radius: 6px;
    }

    .bubble.error {
        background: var(--danger-bg);
        border-color: var(--danger-border);
    }

    .markdown-body {
        color: var(--text);
    }

    .markdown-body p {
        margin: 0.35rem 0 0.55rem;
        color: var(--text);
        line-height: 1.62;
    }

    .markdown-body h1,
    .markdown-body h2,
    .markdown-body h3 {
        color: #f5f9ff;
        margin: 0.8rem 0 0.45rem;
        line-height: 1.25;
        letter-spacing: 0;
    }

    .markdown-body h1 {
        font-size: 1.28rem;
    }

    .markdown-body h2 {
        font-size: 1.14rem;
    }

    .markdown-body h3 {
        font-size: 1.03rem;
    }

    .markdown-body strong {
        color: #b7ffeb;
        font-weight: 760;
    }

    .markdown-body ul,
    .markdown-body ol {
        margin: 0.35rem 0 0.75rem 1.2rem;
        padding-left: 0.6rem;
    }

    .markdown-body li {
        margin: 0.22rem 0;
        color: var(--text);
        line-height: 1.55;
    }

    .markdown-body table {
        width: 100%;
        border-collapse: collapse;
        margin: 0.8rem 0;
        overflow: hidden;
        border-radius: 10px;
        border: 1px solid rgba(159, 176, 199, 0.22);
    }

    .markdown-body th {
        background: rgba(82, 214, 180, 0.12);
        color: #dffdf6;
        font-weight: 740;
        padding: 0.55rem 0.62rem;
        border-bottom: 1px solid rgba(159, 176, 199, 0.18);
    }

    .markdown-body td {
        background: rgba(8, 13, 22, 0.28);
        color: var(--text);
        padding: 0.52rem 0.62rem;
        border-bottom: 1px solid rgba(159, 176, 199, 0.1);
    }

    .markdown-body code {
        color: #f4d38a;
        background: rgba(244, 211, 138, 0.12);
        border-radius: 5px;
        padding: 0.12rem 0.28rem;
    }

    .role-label {
        display: block;
        color: var(--muted);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.38rem;
    }

    .audio-note {
        color: var(--muted);
        font-size: 0.78rem;
        margin-top: 0.48rem;
    }

    .empty-state {
        text-align: center;
        padding: 4rem 1rem 3.4rem;
        color: var(--muted);
    }

    .empty-state h2 {
        color: var(--text);
        font-size: 1.55rem;
        margin: 0 0 0.5rem;
    }

    .empty-state p {
        max-width: 560px;
        margin: 0 auto;
        line-height: 1.65;
    }

    .typing-card {
        border: 1px solid rgba(82, 214, 180, 0.28);
        background: rgba(17, 28, 46, 0.86);
        color: #c9fff1;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        margin: 0.85rem 0;
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }

    .pulse-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: var(--accent);
        box-shadow: 0 0 0 rgba(82, 214, 180, 0.6);
        animation: pulse 1.35s infinite;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(82, 214, 180, 0.5); }
        70% { box-shadow: 0 0 0 10px rgba(82, 214, 180, 0); }
        100% { box-shadow: 0 0 0 0 rgba(82, 214, 180, 0); }
    }

    .stChatInput textarea {
        background: #0f1726 !important;
        color: var(--text) !important;
        border: 1px solid rgba(159, 176, 199, 0.26) !important;
    }

    .stButton > button {
        background: rgba(82, 214, 180, 0.12);
        color: #cffff4;
        border: 1px solid rgba(82, 214, 180, 0.36);
        border-radius: 10px;
    }

    .stButton > button:hover {
        background: rgba(82, 214, 180, 0.18);
        color: #ffffff;
        border-color: rgba(82, 214, 180, 0.58);
    }

    [data-testid="stSidebar"] {
        background: #0d1422;
    }

    [data-testid="stElementContainer"]:has(iframe[height="460"]) {
        position: fixed;
        right: 1rem;
        bottom: 5.2rem;
        width: 520px !important;
        z-index: 1000;
        pointer-events: auto;
    }

    iframe[height="460"] {
        width: 520px !important;
        height: 460px !important;
        border: 0 !important;
        background: transparent !important;
    }

    @media (max-width: 700px) {
        .main .block-container {
            padding-left: 0.7rem;
            padding-right: 0.7rem;
        }

        .app-title {
            align-items: flex-start;
            flex-direction: column;
        }

        .bubble {
            max-width: 96%;
        }

        [data-testid="stElementContainer"]:has(iframe[height="460"]) {
            right: 0.55rem;
            bottom: 4.75rem;
            width: min(520px, calc(100vw - 1.1rem)) !important;
        }

        iframe[height="460"] {
            width: min(520px, calc(100vw - 1.1rem)) !important;
            height: 460px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_session_id" not in st.session_state:
        st.session_state.conversation_session_id = uuid.uuid4().hex
    if "api_url" not in st.session_state:
        st.session_state.api_url = DEFAULT_API_URL
    if "top_k" not in st.session_state:
        st.session_state.top_k = config.QA_TOP_K
    if "temperature" not in st.session_state:
        st.session_state.temperature = DEFAULT_QA_TEMPERATURE
    if "tts_enabled" not in st.session_state:
        st.session_state.tts_enabled = True
    if "tts_voice_id" not in st.session_state:
        st.session_state.tts_voice_id = DEFAULT_TTS_VOICE
    if "tts_voice_choice" not in st.session_state:
        matching_voice = next(
            (label for label, voice in VOICE_OPTIONS.items() if voice == st.session_state.tts_voice_id),
            CUSTOM_VOICE_LABEL,
        )
        st.session_state.tts_voice_choice = matching_voice
    if "tts_tone" not in st.session_state:
        st.session_state.tts_tone = config.TTS_TONE.strip().lower()
    if "tts_rate" not in st.session_state:
        st.session_state.tts_rate = config.TTS_RATE
    if "tts_pitch" not in st.session_state:
        st.session_state.tts_pitch = config.TTS_PITCH
    if "tts_max_words" not in st.session_state:
        st.session_state.tts_max_words = 140
    if "avatar_enabled" not in st.session_state:
        st.session_state.avatar_enabled = True
    if "last_voice_query_id" not in st.session_state:
        st.session_state.last_voice_query_id = ""


def markdown_to_html(content: str) -> str:
    safe_markdown = escape(content or "")
    return markdown.markdown(
        safe_markdown,
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )


def latest_assistant_audio() -> tuple[str | None, str, bool, str]:
    for message in reversed(st.session_state.messages):
        if message.get("role") == "assistant" and message.get("audio_b64"):
            return (
                message.get("audio_b64"),
                message.get("content", ""),
                bool(message.get("audio_autoplay", True)),
                message.get("audio_mime", DEFAULT_AUDIO_MIME),
            )
    return None, "", True, DEFAULT_AUDIO_MIME


def _resolved_avatar_model_path() -> Path:
    configured_path = config.AVATAR_MODEL_FILE.strip()
    if not configured_path:
        configured_path = str(DEFAULT_AVATAR_MODEL_PATH)

    model_path = Path(configured_path)
    if not model_path.is_absolute():
        model_path = APP_DIR / model_path
    if model_path.exists():
        return model_path

    asset_dir = DEFAULT_AVATAR_MODEL_PATH.parent
    for pattern in MODEL_SEARCH_PATTERNS:
        match = next(asset_dir.glob(pattern), None)
        if match:
            return match

    return model_path


def _file_data_uri(path: Path, default_mime: str = "application/octet-stream") -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or default_mime
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _inline_gltf_assets(model_path: Path) -> str:
    gltf = json.loads(model_path.read_text(encoding="utf-8"))
    base_dir = model_path.parent

    for buffer in gltf.get("buffers", []):
        uri = buffer.get("uri", "")
        if uri and not uri.startswith("data:"):
            buffer["uri"] = _file_data_uri(base_dir / uri)

    for image in gltf.get("images", []):
        uri = image.get("uri", "")
        if uri and not uri.startswith("data:"):
            image["uri"] = _file_data_uri(base_dir / uri, "image/png")

    encoded = base64.b64encode(json.dumps(gltf).encode("utf-8")).decode("utf-8")
    return f"data:model/gltf+json;base64,{encoded}"


def avatar_model_source() -> dict | None:
    model_path = _resolved_avatar_model_path()
    extension = model_path.suffix.lower()
    if extension not in MODEL_MIME_TYPES or not model_path.exists():
        return None

    try:
        if extension == ".gltf":
            uri = _inline_gltf_assets(model_path)
        else:
            uri = _file_data_uri(model_path, MODEL_MIME_TYPES[extension])
    except (OSError, json.JSONDecodeError, ValueError):
        # Audio playback and chat should remain usable if an optional avatar
        # file or one of its external glTF resources was not packaged.
        return None

    return {
        "name": model_path.name,
        "format": extension.lstrip("."),
        "uri": uri,
    }


def render_lip_sync_avatar(
    audio_b64: str,
    text: str,
    autoplay: bool = True,
    resume_listener_on_end: bool = True,
    queue_id: str | None = None,
    request_id: str | None = None,
    audio_mime: str = DEFAULT_AUDIO_MIME,
    thinking: bool = False,
) -> None:
    spoken_text = " ".join((text or "").split())
    safe_text = escape(spoken_text[:120] or ("Thinking" if thinking else ""))
    model_json = json.dumps(avatar_model_source())
    speech_text_json = json.dumps(spoken_text[:4000])
    autoplay_attr = "autoplay" if autoplay else ""
    autoplay_js = "true" if autoplay else "false"
    resume_listener_js = "true" if resume_listener_on_end else "false"
    queue_id_json = json.dumps(queue_id or "")
    request_id_json = json.dumps(request_id or "")
    audio_mime_json = json.dumps(audio_mime or DEFAULT_AUDIO_MIME)
    thinking_js = "true" if thinking else "false"
    thinking_class = " thinking" if thinking else ""
    badge_text = "THINKING" if thinking else "3D AVATAR"
    audio_src_attr = f'src="data:{audio_mime};base64,{audio_b64}"' if audio_b64 else ""
    components.html(
        f"""
        <style>
            html,
            body {{
                margin: 0;
                background: transparent;
                font-family: Inter, Segoe UI, Arial, sans-serif;
                overflow: hidden;
            }}

            .texmin-avatar {{
                position: fixed;
                right: 14px;
                bottom: 14px;
                width: min(500px, calc(100vw - 28px));
                border: 1px solid rgba(82, 214, 180, 0.35);
                border-radius: 14px;
                background: rgba(10, 16, 27, 0.94);
                box-shadow: 0 18px 50px rgba(0, 0, 0, 0.36);
                color: #edf3ff;
                padding: 12px;
                box-sizing: border-box;
            }}

            .avatar-stage {{
                display: grid;
                grid-template-columns: minmax(250px, 1fr) 160px;
                gap: 14px;
                align-items: center;
            }}

            #avatarScene {{
                width: 100%;
                height: 360px;
                position: relative;
                overflow: hidden;
                border-radius: 12px;
                background: radial-gradient(circle at 50% 20%, #172033, #0c121e 74%);
                box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.14);
            }}

            #avatarScene canvas {{
                width: 100% !important;
                height: 100% !important;
                display: block;
            }}

            .model-status {{
                position: absolute;
                inset: 0;
                display: grid;
                place-items: center;
                padding: 14px;
                color: #9fb0c7;
                font-size: 0.72rem;
                line-height: 1.35;
                text-align: center;
                z-index: 2;
            }}

            .model-status.loaded {{
                display: none;
            }}

            .model-status.warning {{
                inset: auto 10px 10px 10px;
                display: block;
                padding: 8px 10px;
                border-radius: 10px;
                background: rgba(9, 14, 24, 0.78);
                color: #ffdca8;
                box-shadow: inset 0 0 0 1px rgba(255, 220, 168, 0.22);
            }}

            .avatar-badge {{
                position: absolute;
                left: 8px;
                top: 8px;
                z-index: 3;
                padding: 4px 7px;
                border-radius: 999px;
                background: rgba(9, 14, 24, 0.74);
                color: #b7ffeb;
                font-size: 0.62rem;
                font-weight: 760;
                letter-spacing: 0;
            }}

            .texmin-avatar.thinking .avatar-badge {{
                background: rgba(92, 124, 255, 0.2);
                color: #dce4ff;
                animation: thinkingPulse 1.4s ease-in-out infinite;
            }}

            .thinking-dots {{
                display: none;
                gap: 5px;
                align-items: center;
                margin-top: 10px;
            }}

            .texmin-avatar.thinking .thinking-dots {{
                display: flex;
            }}

            .thinking-dots span {{
                width: 6px;
                height: 6px;
                border-radius: 999px;
                background: #7f9cff;
                animation: thinkingDot 1.05s ease-in-out infinite;
            }}

            .thinking-dots span:nth-child(2) {{ animation-delay: 0.16s; }}
            .thinking-dots span:nth-child(3) {{ animation-delay: 0.32s; }}

            @keyframes thinkingPulse {{
                0%, 100% {{ opacity: 0.62; transform: scale(0.98); }}
                50% {{ opacity: 1; transform: scale(1.03); }}
            }}

            @keyframes thinkingDot {{
                0%, 100% {{ opacity: 0.3; transform: translateY(0); }}
                50% {{ opacity: 1; transform: translateY(-4px); }}
            }}

            .avatar-copy {{
                min-width: 0;
            }}

            .avatar-title {{
                font-size: 0.78rem;
                font-weight: 760;
                color: #b7ffeb;
                line-height: 1.2;
                margin-bottom: 5px;
            }}

            .avatar-line {{
                font-size: 0.72rem;
                color: #9fb0c7;
                line-height: 1.35;
                height: 12.15em;
                overflow: hidden;
            }}

            audio {{
                width: 100%;
                height: 32px;
                margin-top: 10px;
                accent-color: #52d6b4;
            }}

        </style>

        <div class="texmin-avatar{thinking_class}" id="avatar">
            <div class="avatar-stage">
                <div id="avatarScene" aria-label="{escape(ASSISTANT_NAME)} 3D presenter avatar">
                    <div class="model-status" id="modelStatus">Loading 3D avatar...</div>
                    <div class="avatar-badge" id="avatarBadge">{badge_text}</div>
                </div>
                <div class="avatar-copy">
                    <div class="avatar-title">{escape(ASSISTANT_NAME)}</div>
                    <div class="avatar-line" id="avatarLine">{safe_text}</div>
                    <div class="thinking-dots" aria-label="Thinking"><span></span><span></span><span></span></div>
                </div>
            </div>
            <audio id="voice" controls {autoplay_attr} playsinline preload="auto" {audio_src_attr}></audio>
        </div>

        <script type="importmap">
            {{
                "imports": {{
                    "three": "https://unpkg.com/three@0.160.1/build/three.module.js",
                    "three/addons/": "https://unpkg.com/three@0.160.1/examples/jsm/"
                }}
            }}
        </script>
        <script type="module">
            import * as THREE from "three";
            import {{ GLTFLoader }} from "three/addons/loaders/GLTFLoader.js";
            import {{ OBJLoader }} from "three/addons/loaders/OBJLoader.js";
            import {{ FBXLoader }} from "three/addons/loaders/FBXLoader.js";

            const modelSource = {model_json};
            let speechText = {speech_text_json};
            const shouldAutoplay = {autoplay_js};
            const shouldResumeListenerOnEnd = {resume_listener_js};
            const queueId = {queue_id_json};
            const requestId = {request_id_json};
            const avatarStartedAt = performance.now();
            let firstPlaybackReported = false;
            const initialAudioMime = {audio_mime_json};
            const queuedPlayback = Boolean(queueId);
            let thinking = {thinking_js};
            const storageInterruptKey = "texmin_voice_interrupt_at";
            const storageAvatarSpeakingKey = "texmin_voice_avatar_speaking_at";
            const storageWaitingKey = "texmin_voice_waiting_since";
            const storageInputSuppressedUntilKey = "texmin_voice_input_suppressed_until";
            const waitingCueText = "";
            const avatar = document.getElementById("avatar");
            const audio = document.getElementById("voice");
            const avatarLine = document.getElementById("avatarLine");
            const avatarBadge = document.getElementById("avatarBadge");
            const sceneHost = document.getElementById("avatarScene");
            const modelStatus = document.getElementById("modelStatus");
            let context;
            let analyser;
            let data;
            let source;
            let mouthOpen = 0;
            let mouthWide = 0;
            let mouthRound = 0;
            let mouthPress = 0;
            let targetMouthOpen = 0;
            let targetMouthWide = 0;
            let targetMouthRound = 0;
            let targetMouthPress = 0;
            let speechEnergy = 0;
            let speechBrightness = 0;
            let analyserReady = false;
            let speaking = false;
            let avatarRoot = null;
            let headNode = null;
            let mixer = null;
            const audioQueue = [];
            const pendingQueueItems = new Map();
            let nextQueueSequence = 0;
            let finalQueueSequence = null;
            let queueFinalized = !queuedPlayback;
            let queuePlaying = false;
            let queueInterrupted = false;
            let speakingHeartbeat = null;
            let lastInterruptAt = "";
            let cueUtterance = null;
            let waitingCueStarted = false;
            const clock = new THREE.Clock();
            const avatarFitHeight = 2.25;
            const avatarVerticalCenter = 0.52;
            const avatarBaseYaw = 0.5;
            const avatarFaceLookAtY = 1.15;
            const avatarFaceCameraY = 1.17;
            const avatarFaceCameraDistance = 1.9;
            const morphTargets = [];
            const jawNodes = [];
            const tongueNodes = [];

            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(24, 1, 0.1, 100);
            camera.position.set(0, 0.66, 2.6);
            camera.lookAt(0, 0.62, 0);

            const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
            renderer.outputColorSpace = THREE.SRGBColorSpace;
            sceneHost.appendChild(renderer.domElement);

            const keyLight = new THREE.DirectionalLight(0xffffff, 2.55);
            keyLight.position.set(1.4, 2.8, 2.8);
            scene.add(keyLight);
            scene.add(new THREE.HemisphereLight(0xb8d7ff, 0x192033, 1.7));

            function resizeScene() {{
                const rect = sceneHost.getBoundingClientRect();
                const width = Math.max(1, rect.width);
                const height = Math.max(1, rect.height);
                renderer.setSize(width, height, false);
                camera.aspect = width / height;
                camera.updateProjectionMatrix();
            }}
            window.addEventListener("resize", resizeScene);
            resizeScene();

            function setupAudioGraph() {{
                if (analyser) return;
                try {{
                    const AudioContext = window.AudioContext || window.webkitAudioContext;
                    context = new AudioContext();
                    analyser = context.createAnalyser();
                    analyser.fftSize = 256;
                    analyser.smoothingTimeConstant = 0.68;
                    data = new Uint8Array(analyser.frequencyBinCount);
                    source = context.createMediaElementSource(audio);
                    source.connect(analyser);
                    analyser.connect(context.destination);
                    analyserReady = true;
                }} catch (error) {{
                    analyserReady = false;
                    console.info("Audio analyser unavailable; using timed lip-sync fallback.", error);
                }}
            }}

            function isMouthMorph(name) {{
                const lower = name.toLowerCase();
                return (
                    lower.includes("jawopen") ||
                    lower.includes("mouthopen") ||
                    lower.includes("mouth_open") ||
                    lower.includes("viseme_aa") ||
                    lower.includes("viseme_e") ||
                    lower.includes("viseme_ih") ||
                    lower.includes("viseme_oh") ||
                    lower.includes("viseme_ou") ||
                    lower.includes("viseme_u") ||
                    lower.includes("viseme") ||
                    lower.includes("v_aa") ||
                    lower.includes("v_e") ||
                    lower.includes("v_ih") ||
                    lower.includes("v_oh") ||
                    lower.includes("v_u")
                );
            }}

            function classifyMouthMorph(name) {{
                const lower = name.toLowerCase();
                if (
                    lower.includes("round") ||
                    lower.includes("pucker") ||
                    lower.includes("funnel") ||
                    lower.includes("viseme_oh") ||
                    lower.includes("viseme_ou") ||
                    lower.includes("viseme_u") ||
                    lower.includes("v_oh") ||
                    lower.includes("v_u")
                ) {{
                    return "round";
                }}
                if (
                    lower.includes("wide") ||
                    lower.includes("smile") ||
                    lower.includes("stretch") ||
                    lower.includes("viseme_e") ||
                    lower.includes("viseme_ih") ||
                    lower.includes("v_e") ||
                    lower.includes("v_ih")
                ) {{
                    return "wide";
                }}
                if (
                    lower.includes("close") ||
                    lower.includes("press") ||
                    lower.includes("viseme_m") ||
                    lower.includes("viseme_b") ||
                    lower.includes("viseme_p") ||
                    lower.includes("v_m") ||
                    lower.includes("v_b") ||
                    lower.includes("v_p")
                ) {{
                    return "press";
                }}
                return "open";
            }}

            function isJawNode(name) {{
                const lower = name.toLowerCase();
                return lower.includes("jaw") || lower.includes("mandible");
            }}

            function isTongueNode(name) {{
                return name.toLowerCase().includes("tongue");
            }}

            function collectRigControls(root) {{
                morphTargets.length = 0;
                jawNodes.length = 0;
                tongueNodes.length = 0;
                headNode = null;
                root.traverse((node) => {{
                    if (node.morphTargetDictionary && node.morphTargetInfluences) {{
                        Object.entries(node.morphTargetDictionary).forEach(([name, index]) => {{
                            if (isMouthMorph(name)) {{
                                morphTargets.push({{
                                    mesh: node,
                                    index,
                                    shape: classifyMouthMorph(name),
                                }});
                            }}
                        }});
                    }}
                    if (!node.isMesh && isJawNode(node.name || "")) {{
                        node.userData.restRotationX = node.rotation.x;
                        node.userData.restRotationY = node.rotation.y;
                        node.userData.restRotationZ = node.rotation.z;
                        node.userData.restPosition = node.position.clone();
                        jawNodes.push(node);
                    }}
                    if (!node.isMesh && isTongueNode(node.name || "")) {{
                        node.userData.restRotationX = node.rotation.x;
                        node.userData.restRotationY = node.rotation.y;
                        node.userData.restRotationZ = node.rotation.z;
                        node.userData.restPosition = node.position.clone();
                        tongueNodes.push(node);
                    }}
                    if (!headNode && !node.isMesh && node.name && node.name.toLowerCase().includes("head")) {{
                        node.userData.restRotationX = node.rotation.x;
                        node.userData.restRotationY = node.rotation.y;
                        node.userData.restRotationZ = node.rotation.z;
                        headNode = node;
                    }}
                }});
            }}

            function frameModel(root) {{
                root.updateMatrixWorld(true);
                const rootBox = new THREE.Box3().setFromObject(root);
                if (rootBox.isEmpty()) {{
                    camera.position.set(0, 0.9, 4.2);
                    camera.lookAt(0, 0.9, 0);
                    return;
                }}

                const rootSize = rootBox.getSize(new THREE.Vector3());
                const rootCenter = rootBox.getCenter(new THREE.Vector3());
                const rootHeight = Math.max(rootSize.y, rootSize.x, rootSize.z, 0.001);
                const scale = avatarFitHeight / rootHeight;
                root.scale.setScalar(scale);
                root.position.set(
                    -rootCenter.x * scale,
                    avatarVerticalCenter - rootCenter.y * scale,
                    -rootCenter.z * scale
                );
                root.userData.basePosition = root.position.clone();

                root.updateMatrixWorld(true);
                const framedBox = new THREE.Box3().setFromObject(root);
                const framedCenter = framedBox.getCenter(new THREE.Vector3());
                const lookAtY = framedCenter.y + avatarFaceLookAtY;
                const cameraDistance = avatarFaceCameraDistance / Math.max(0.82, Math.min(camera.aspect, 1.2));
                camera.position.set(framedCenter.x, framedCenter.y + avatarFaceCameraY, framedCenter.z + cameraDistance);
                camera.near = 0.01;
                camera.far = 100;
                camera.lookAt(framedCenter.x, lookAtY, framedCenter.z);
                camera.updateProjectionMatrix();
            }}

            function loadModel() {{
                if (!modelSource) {{
                    modelStatus.textContent = "No 3D avatar model found. Put a rigged GLB or GLTF in frontend/assets.";
                    return;
                }}

                const onLoad = (root) => {{
                    avatarRoot = root;
                    collectRigControls(avatarRoot);
                    frameModel(avatarRoot);
                    scene.add(avatarRoot);
                    if (root.animations && root.animations.length) {{
                        mixer = new THREE.AnimationMixer(avatarRoot);
                        const idleClip = root.animations[0];
                        const action = mixer.clipAction(idleClip);
                        action.play();
                    }}
                    if (morphTargets.length || jawNodes.length) {{
                        modelStatus.classList.add("loaded");
                    }} else {{
                        modelStatus.classList.add("warning");
                        modelStatus.textContent = "Avatar loaded. This Mixamo rig has no facial lip/jaw controls, so lip sync uses body motion only.";
                    }}
                }};

                const onError = (error) => {{
                    console.error(error);
                    modelStatus.textContent = "Could not load " + modelSource.name + ". Check GLTF dependencies and format.";
                }};

                if (modelSource.format === "fbx") {{
                    new FBXLoader().load(modelSource.uri, onLoad, undefined, onError);
                }} else if (modelSource.format === "obj") {{
                    new OBJLoader().load(modelSource.uri, onLoad, undefined, onError);
                }} else {{
                    new GLTFLoader().load(modelSource.uri, (gltf) => {{
                        gltf.scene.animations = gltf.animations || [];
                        onLoad(gltf.scene);
                    }}, undefined, onError);
                }}
            }}

            function updateMouthFromAudio() {{
                if (audio.paused || audio.ended) {{
                    speaking = false;
                    targetMouthOpen = 0;
                    return;
                }}

                if (!analyser) {{
                    try {{
                        setupAudioGraph();
                    }} catch (error) {{
                        analyserReady = false;
                    }}
                }}

                if (!analyser || !analyserReady || (context && context.state === "suspended")) {{
                    speaking = true;
                    const t = performance.now() / 1000;
                    speechEnergy = 0.46 + 0.32 * Math.abs(Math.sin(t * 9.3));
                    speechBrightness = 0.52;
                    return;
                }}

                analyser.getByteFrequencyData(data);
                const speechBands = data.slice(2, 42);
                const average = speechBands.reduce((sum, value) => sum + value, 0) / speechBands.length;
                speaking = average > 6;
                speechEnergy = Math.min(1, Math.max(0, (average - 6) / 108));
                const highBands = data.slice(32, 72);
                const highAverage = highBands.reduce((sum, value) => sum + value, 0) / highBands.length;
                speechBrightness = Math.min(1, Math.max(0, highAverage / Math.max(average, 1)));

                if (!speaking && audio.currentTime > 0 && !audio.paused) {{
                    const t = performance.now() / 1000;
                    speaking = true;
                    speechEnergy = 0.38 + 0.24 * Math.abs(Math.sin(t * 10.8));
                    speechBrightness = 0.5;
                }}
            }}

            function clamp01(value) {{
                return Math.min(1, Math.max(0, value));
            }}

            function smooth(current, target, attack, release) {{
                return current + (target - current) * (target > current ? attack : release);
            }}

            function textShapeAtProgress(progress) {{
                if (!speechText) {{
                    return {{ open: 0.7, wide: 0.25, round: 0.18, press: 0 }};
                }}

                const compact = speechText.toLowerCase().replace(/[^a-z0-9\\u0900-\\u097F]+/g, " ");
                const index = Math.min(compact.length - 1, Math.max(0, Math.floor(progress * compact.length)));
                const windowText = compact.slice(Math.max(0, index - 2), Math.min(compact.length, index + 3));
                const current = compact[index] || "";
                const previous = compact[Math.max(0, index - 1)] || "";
                const next = compact[Math.min(compact.length - 1, index + 1)] || "";
                const letter = previous + current + next;

                const shape = {{ open: 0.55, wide: 0.08, round: 0.08, press: 0 }};
                if (/[bmp]/.test(windowText)) {{
                    shape.open = 0.06;
                    shape.press = 0.96;
                }} else if (/[fv]/.test(letter)) {{
                    shape.open = 0.24;
                    shape.wide = 0.44;
                    shape.press = 0.2;
                }} else if (/[oquuw]/.test(letter)) {{
                    shape.open = 0.46;
                    shape.round = 0.86;
                }} else if (/[ei]/.test(letter)) {{
                    shape.open = 0.34;
                    shape.wide = 0.78;
                }} else if (/a/.test(letter)) {{
                    shape.open = 0.98;
                    shape.wide = 0.16;
                }} else if (current === " " || /[tdkgcszrlmn]/.test(current)) {{
                    shape.open = 0.28;
                    shape.wide = 0.18;
                }}
                return shape;
            }}

            function morphValueForShape(shape) {{
                if (shape === "wide") return mouthWide;
                if (shape === "round") return mouthRound;
                if (shape === "press") return mouthPress;
                return mouthOpen;
            }}

            function setThinking(active) {{
                thinking = Boolean(active);
                avatar.classList.toggle("thinking", thinking);
                avatarBadge.textContent = thinking ? "THINKING" : "3D AVATAR";
                if (thinking) {{
                    avatarLine.textContent = "Thinking";
                }} else if (avatarLine.textContent === "Thinking") {{
                    avatarLine.textContent = speechText || "Ready";
                }}
            }}

            function drawAvatar(now) {{
                const delta = clock.getDelta();
                if (mixer) {{
                    mixer.update(delta);
                }}
                updateMouthFromAudio();
                const t = now / 1000;
                if (speaking) {{
                    const progress = audio.duration ? audio.currentTime / audio.duration : 0;
                    const textShape = textShapeAtProgress(progress);
                    const syllablePulse = 0.28 + 0.48 * Math.abs(Math.sin(t * 15.5)) + 0.24 * Math.abs(Math.sin(t * 23.7 + 0.8));
                    const energy = clamp01(speechEnergy * 1.22);
                    targetMouthOpen = clamp01(energy * syllablePulse * textShape.open * (1 - textShape.press * 0.82));
                    targetMouthWide = clamp01(energy * (0.18 + speechBrightness * 0.45) * textShape.wide);
                    targetMouthRound = clamp01(energy * (0.24 + (1 - speechBrightness) * 0.46) * textShape.round);
                    targetMouthPress = clamp01(energy * textShape.press);
                }} else {{
                    targetMouthOpen = 0;
                    targetMouthWide = 0;
                    targetMouthRound = 0;
                    targetMouthPress = 0;
                }}
                mouthOpen = smooth(mouthOpen, targetMouthOpen, 0.72, 0.38);
                mouthWide = smooth(mouthWide, targetMouthWide, 0.48, 0.3);
                mouthRound = smooth(mouthRound, targetMouthRound, 0.5, 0.34);
                mouthPress = smooth(mouthPress, targetMouthPress, 0.86, 0.54);

                morphTargets.forEach((target) => {{
                    target.mesh.morphTargetInfluences[target.index] = morphValueForShape(target.shape);
                }});
                jawNodes.forEach((jaw) => {{
                    const restPosition = jaw.userData.restPosition;
                    jaw.rotation.x = jaw.userData.restRotationX + mouthRound * 0.025;
                    jaw.rotation.y = jaw.userData.restRotationY + mouthWide * 0.018;
                    jaw.rotation.z = jaw.userData.restRotationZ - mouthOpen * 0.17 + mouthPress * 0.018;
                    if (restPosition) {{
                        jaw.position.set(
                            restPosition.x,
                            restPosition.y - mouthOpen * 0.018 + mouthPress * 0.006,
                            restPosition.z + mouthRound * 0.01
                        );
                    }}
                }});
                tongueNodes.forEach((tongue, index) => {{
                    const restPosition = tongue.userData.restPosition;
                    const weight = Math.max(0.15, 1 - index * 0.18);
                    tongue.rotation.x = tongue.userData.restRotationX + mouthOpen * 0.025 * weight;
                    tongue.rotation.z = tongue.userData.restRotationZ + (mouthWide - mouthRound) * 0.035 * weight;
                    if (restPosition) {{
                        tongue.position.set(
                            restPosition.x,
                            restPosition.y + mouthOpen * 0.01 * weight,
                            restPosition.z
                        );
                    }}
                }});

                if (avatarRoot) {{
                    const thinkingTurn = thinking && !speaking ? Math.sin(t * 1.15) * 0.085 : 0;
                    avatarRoot.rotation.y = avatarBaseYaw + Math.sin(t * 0.72) * 0.06 + thinkingTurn + speechEnergy * Math.sin(t * 5.2) * 0.012;
                    const base = avatarRoot.userData.basePosition || new THREE.Vector3(0, -0.04, 0);
                    const thinkingLift = thinking && !speaking ? Math.sin(t * 2.1) * 0.008 : 0;
                    avatarRoot.position.set(base.x, base.y + Math.sin(t * 1.3) * 0.018 + thinkingLift, base.z);
                }}
                if (headNode) {{
                    const thinkingNod = thinking && !speaking ? -0.035 + Math.sin(t * 1.8) * 0.025 : 0;
                    headNode.rotation.x = headNode.userData.restRotationX + thinkingNod + speechEnergy * Math.sin(t * 3.1 + 0.4) * 0.006;
                    headNode.rotation.y = headNode.userData.restRotationY + (thinking && !speaking ? Math.sin(t * 0.9) * 0.035 : 0);
                    headNode.rotation.z = headNode.userData.restRotationZ + Math.sin(t * 0.82) * 0.018 + speechEnergy * Math.sin(t * 4.4) * 0.012;
                }}

                avatar.classList.toggle("speaking", speaking);
                renderer.render(scene, camera);
                requestAnimationFrame(drawAvatar);
            }}

            async function prepareAudioGraph() {{
                setupAudioGraph();
                if (context && context.state === "suspended") {{
                    try {{
                        await context.resume();
                    }} catch (error) {{
                        console.info("AudioContext resume was blocked; lip-sync fallback remains active.", error);
                    }}
                }}
            }}

            function signalListenerResume() {{
                if (!shouldResumeListenerOnEnd) return;
                try {{
                    window.localStorage.setItem("texmin_voice_resume_at", String(Date.now()));
                }} catch (error) {{
                    console.info("Could not signal voice listener resume.", error);
                }}
            }}

            function suppressVoiceInput(durationMs) {{
                try {{
                    window.localStorage.setItem(
                        storageInputSuppressedUntilKey,
                        String(Date.now() + durationMs)
                    );
                }} catch (error) {{
                    console.info("Could not suppress microphone input during cue.", error);
                }}
            }}

            function stopWaitingCue() {{
                if (!cueUtterance) return;
                try {{
                    if (window.speechSynthesis) {{
                        window.speechSynthesis.cancel();
                    }}
                }} catch (error) {{
                    console.info("Could not stop waiting cue.", error);
                }}
                cueUtterance = null;
            }}

            function speakWaitingCue() {{
                if (waitingCueStarted || !waitingCueText || !shouldAutoplay || queueInterrupted) return;
                if (!audio.paused || audio.currentSrc) return;
                if (!window.speechSynthesis || !window.SpeechSynthesisUtterance) return;
                waitingCueStarted = true;
                try {{
                    const utterance = new SpeechSynthesisUtterance(waitingCueText);
                    cueUtterance = utterance;
                    utterance.lang = navigator.language || "en-IN";
                    if (!["en", "hi"].includes(utterance.lang.slice(0, 2).toLowerCase())) {{
                        utterance.lang = "en-IN";
                    }}
                    utterance.rate = 1.04;
                    utterance.pitch = 1;
                    utterance.volume = 0.85;
                    utterance.onstart = () => suppressVoiceInput(1900);
                    utterance.onend = () => {{
                        if (cueUtterance === utterance) {{
                            cueUtterance = null;
                        }}
                    }};
                    utterance.onerror = utterance.onend;
                    window.speechSynthesis.cancel();
                    window.speechSynthesis.speak(utterance);
                }} catch (error) {{
                    cueUtterance = null;
                    console.info("Browser waiting cue could not play.", error);
                }}
            }}

            function setAvatarSpeaking(active) {{
                try {{
                    if (speakingHeartbeat) {{
                        window.clearInterval(speakingHeartbeat);
                        speakingHeartbeat = null;
                    }}
                    if (active) {{
                        const markSpeaking = () => {{
                            window.localStorage.setItem(storageAvatarSpeakingKey, String(Date.now()));
                        }};
                        markSpeaking();
                        speakingHeartbeat = window.setInterval(markSpeaking, 600);
                    }} else {{
                        window.localStorage.removeItem(storageAvatarSpeakingKey);
                    }}
                }} catch (error) {{
                    console.info("Could not update avatar speaking state.", error);
                }}
            }}

            function resetSpeechMotion() {{
                speaking = false;
                targetMouthOpen = 0;
                targetMouthWide = 0;
                targetMouthRound = 0;
                targetMouthPress = 0;
            }}

            function stopAvatarPlaybackForInterrupt() {{
                queueInterrupted = true;
                setThinking(false);
                stopWaitingCue();
                audioQueue.length = 0;
                pendingQueueItems.clear();
                queuePlaying = false;
                queueFinalized = true;
                finalQueueSequence = nextQueueSequence;
                try {{
                    audio.pause();
                    audio.removeAttribute("src");
                    audio.load();
                }} catch (error) {{
                    console.info("Avatar audio was already stopped.", error);
                }}
                setAvatarSpeaking(false);
                resetSpeechMotion();
                avatarLine.textContent = "Listening...";
                signalListenerResume();
            }}

            function currentInterruptSignal() {{
                try {{
                    const interruptAt = Number(window.localStorage.getItem(storageInterruptKey) || 0);
                    const waitingSince = Number(window.localStorage.getItem(storageWaitingKey) || 0);
                    if (!interruptAt) return "";
                    if (waitingSince && interruptAt < waitingSince - 250) return "";
                    return String(interruptAt);
                }} catch (error) {{
                    return "";
                }}
            }}

            function maybeHandleAvatarInterrupt() {{
                const interruptAt = currentInterruptSignal();
                if (!interruptAt || interruptAt === lastInterruptAt) return;
                lastInterruptAt = interruptAt;
                stopAvatarPlaybackForInterrupt();
            }}

            async function playVoiceAutomatically() {{
                try {{
                    stopWaitingCue();
                    audio.volume = 1;
                    await prepareAudioGraph();
                    if (audio.paused) {{
                        await audio.play();
                    }}
                }} catch (error) {{
                    console.info("Automatic avatar audio playback was blocked by the browser.", error);
                    setAvatarSpeaking(false);
                    signalListenerResume();
                }}
            }}

            async function playNextQueuedAudio() {{
                if (queueInterrupted) {{
                    queuePlaying = false;
                    return;
                }}
                if (!audioQueue.length) {{
                    queuePlaying = false;
                    if (
                        queueFinalized &&
                        (finalQueueSequence === null || nextQueueSequence >= finalQueueSequence)
                    ) signalListenerResume();
                    return;
                }}

                const item = audioQueue.shift();
                stopWaitingCue();
                setThinking(false);
                queuePlaying = true;
                speechText = item.text || "";
                avatarLine.textContent = speechText;
                audio.src = `data:${{item.mime || initialAudioMime}};base64,${{item.audio}}`;
                audio.load();
                try {{
                    audio.volume = 1;
                    await prepareAudioGraph();
                    await audio.play();
                }} catch (error) {{
                    console.info("Queued avatar audio playback was blocked.", error);
                    queuePlaying = false;
                    setAvatarSpeaking(false);
                    signalListenerResume();
                }}
            }}

            window.addEventListener("message", (event) => {{
                const message = event.data || {{}};
                if (message.type !== "texmin:avatar-queue" || message.queueId !== queueId) return;
                if (queueInterrupted) return;
                if ((message.audio || message.skip) && message.sequence >= nextQueueSequence) {{
                    setThinking(false);
                    pendingQueueItems.set(message.sequence, {{
                        audio: message.audio,
                        text: message.text || "",
                        mime: message.mime || initialAudioMime,
                    }});
                    while (pendingQueueItems.has(nextQueueSequence)) {{
                        const readyItem = pendingQueueItems.get(nextQueueSequence);
                        if (readyItem.audio) audioQueue.push(readyItem);
                        pendingQueueItems.delete(nextQueueSequence);
                        nextQueueSequence += 1;
                    }}
                }}
                if (message.final) {{
                    queueFinalized = true;
                    finalQueueSequence = message.finalSequence;
                    if (!message.audio && !audioQueue.length && pendingQueueItems.size === 0) {{
                        setThinking(false);
                    }}
                }}
                if (!queuePlaying && audio.paused) playNextQueuedAudio();
            }});

            let autoplayAttempted = false;
            function scheduleAutoplay() {{
                if (!shouldAutoplay) return;
                if (autoplayAttempted) return;
                autoplayAttempted = true;
                window.setTimeout(playVoiceAutomatically, 250);
            }}

            audio.addEventListener("play", async () => {{
                if (queuedPlayback && !firstPlaybackReported) {{
                    firstPlaybackReported = true;
                    console.info(JSON.stringify({{
                        event: "voice_first_playback",
                        request_id: requestId,
                        elapsed_from_avatar_ms: Math.round(performance.now() - avatarStartedAt),
                    }}));
                }}
                setAvatarSpeaking(true);
                try {{
                    await prepareAudioGraph();
                }} catch (error) {{
                    console.info("Audio analyser could not start yet.", error);
                }}
            }});

            audio.addEventListener("pause", () => {{
                setAvatarSpeaking(false);
                resetSpeechMotion();
            }});
            audio.addEventListener("ended", () => {{
                setAvatarSpeaking(false);
                resetSpeechMotion();
                if (queuedPlayback) {{
                    playNextQueuedAudio();
                }} else {{
                    signalListenerResume();
                }}
            }});
            window.addEventListener("storage", maybeHandleAvatarInterrupt);
            window.setInterval(maybeHandleAvatarInterrupt, 250);
            loadModel();
            requestAnimationFrame(drawAvatar);
            if (!queuedPlayback) {{
                audio.addEventListener("canplay", scheduleAutoplay, {{ once: true }});
                audio.addEventListener("loadedmetadata", scheduleAutoplay, {{ once: true }});
                window.addEventListener("load", scheduleAutoplay, {{ once: true }});
                window.setTimeout(scheduleAutoplay, 650);
            }} else {{
                window.setTimeout(speakWaitingCue, 180);
            }}
        </script>
        """,
        height=460,
    )


def render_message(message: dict) -> None:
    role = message.get("role", "assistant")
    label = "You" if role == "user" else ASSISTANT_NAME
    bubble_class = "user" if role == "user" else "assistant"
    if message.get("error"):
        bubble_class += " error"

    if role == "assistant":
        content_html = markdown_to_html(message.get("content", ""))
    else:
        content_html = f"<p>{escape(message.get('content', ''))}</p>"

    st.markdown(
        f"""
        <div class="message-row {role}">
            <div class="bubble {bubble_class}">
                <span class="role-label">{label}</span>
                <div class="markdown-body">{content_html}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    audio_b64 = message.get("audio_b64")
    if role == "assistant" and audio_b64 and not st.session_state.avatar_enabled:
        st.audio(base64.b64decode(audio_b64), format=message.get("audio_mime", DEFAULT_AUDIO_MIME))

    audio_error = message.get("audio_error")
    if role == "assistant" and audio_error:
        st.markdown(f'<div class="audio-note">{escape(audio_error)}</div>', unsafe_allow_html=True)


def render_voice_query_component() -> dict | None:
    value = voice_query_component(
        default=None,
        key="texmin_voice_query",
        height=92,
        server_stt=True,
        browser_api_path=config.BROWSER_API_PATH,
        no_speech_timeout_seconds=config.RECORDING_NO_SPEECH_TIMEOUT_SECONDS,
        max_recording_seconds=config.RECORDING_MAX_SECONDS,
        silence_ms=config.RECORDING_SILENCE_MS,
        recording_buffer_ms=config.RECORDING_BUFFER_MS,
        speech_confirm_ms=config.RECORDING_SPEECH_CONFIRM_MS,
        stt_timeout_seconds=config.STT_REQUEST_TIMEOUT_SECONDS,
        session_id=st.session_state.conversation_session_id,
    )
    if not value:
        return None

    if isinstance(value, str):
        query_id = value
        text = value
        input_type = "text"
    else:
        input_type = "audio"
        query_id = str(value.get("id", ""))
        text = str(value.get("text", "")).strip()
        audio_b64 = str(value.get("audio_b64", ""))
        audio_mime = str(value.get("audio_mime", "audio/webm"))
        if audio_b64 and query_id != st.session_state.last_voice_query_id:
            input_type = "text"  # Legacy component did not begin a voice request.
            text, transcription_error = ask_stt(audio_b64, audio_mime)
            if transcription_error:
                st.warning(transcription_error)

    if not text or query_id == st.session_state.last_voice_query_id:
        return None

    st.session_state.last_voice_query_id = query_id
    return {"text": text, "request_id": query_id, "input_type": input_type}


@lru_cache(maxsize=1)
def _inprocess_backend() -> dict:
    """Load the backend service layer without going through FastAPI/HTTP."""
    backend_dir = APP_DIR.parent / "backend"
    if not backend_dir.exists():
        configured = config.BACKEND_SOURCE_DIR.strip()
        backend_dir = Path(configured) if configured else backend_dir
    backend_path = str(backend_dir.resolve())
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    from helpers.request_models import ChatMessage, TTSRequest
    from services.qa_service import answer_question, stream_answer_events
    from services.stt_service import transcribe_audio
    from services.tts_service import synthesize_speech

    return {
        "ChatMessage": ChatMessage,
        "TTSRequest": TTSRequest,
        "answer_question": answer_question,
        "stream_answer_events": stream_answer_events,
        "transcribe_audio": transcribe_audio,
        "synthesize_speech": synthesize_speech,
    }


def _qa_events(url: str, payload: dict):
    if BACKEND_CALL_MODE == "inprocess":
        services = _inprocess_backend()
        history = [services["ChatMessage"](**item) for item in payload["chat_history"]]
        yield from services["stream_answer_events"](
            question=payload["question"],
            top_k=payload["top_k"],
            temperature=payload["temperature"],
            chat_history=history,
            session_id=payload.get("session_id"),
            request_id=payload.get("request_id"),
            input_type=payload.get("input_type", "text"),
        )
        return

    with requests.post(url, json=payload, timeout=180, stream=True) as response:
        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(f"API returned {response.status_code}: {detail}")
        for line in response.iter_lines(decode_unicode=True, chunk_size=1):
            if not line or not line.startswith("data:"):
                continue
            try:
                yield json.loads(line.removeprefix("data:").strip())
            except json.JSONDecodeError:
                continue


def ask_stt(audio_b64: str, audio_mime: str) -> tuple[str, str | None]:
    url = st.session_state.api_url.rstrip("/") + "/stt/transcribe"
    try:
        audio = base64.b64decode(audio_b64, validate=True)
    except (ValueError, TypeError):
        return "", "The microphone recording was invalid. Please try again."

    extension_by_mime = {
        "audio/webm": ".webm",
        "audio/ogg": ".ogg",
        "audio/wav": ".wav",
        "audio/mp4": ".m4a",
    }
    base_mime = audio_mime.split(";", 1)[0].lower()
    extension = extension_by_mime.get(base_mime, ".webm")
    if BACKEND_CALL_MODE == "inprocess":
        try:
            services = _inprocess_backend()
            result = services["transcribe_audio"](audio, suffix=extension, language=None)
            return str(result.get("text", "")).strip(), None
        except Exception as exc:
            return "", f"Offline speech recognition failed: {exc}"

    try:
        response = requests.post(
            url,
            files={"audio": (f"recording{extension}", audio, base_mime)},
            data={"language": ""},
            timeout=120,
        )
    except requests.RequestException as exc:
        return "", f"Offline speech recognition could not connect to {url}. {exc}"
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return "", f"Offline speech recognition failed: {detail}"
    return str(response.json().get("text", "")).strip(), None


def resume_voice_listener_without_audio() -> None:
    components.html(
        """
        <script>
            window.localStorage.setItem("texmin_voice_resume_at", String(Date.now()));
        </script>
        """,
        height=0,
    )


def clear_voice_interrupt_signal() -> None:
    components.html(
        """
        <script>
            window.localStorage.removeItem("texmin_voice_interrupt_at");
        </script>
        """,
        height=0,
    )


def _chat_history_payload(exclude_latest_user: bool = False) -> list[dict]:
    messages = st.session_state.messages[:-1] if exclude_latest_user else st.session_state.messages
    history = []
    for message in messages[-6:]:
        role = message.get("role")
        content = " ".join(str(message.get("content", "")).split())
        if role not in {"user", "assistant"} or not content or message.get("error"):
            continue
        history.append(
            {
                "role": role,
                "content": content[:420],
            }
        )
    return history


def ask_api(question: str) -> tuple[str, list[dict], str | None]:
    url = st.session_state.api_url.rstrip("/") + "/qa/ask"
    payload = {
        "question": question,
        "top_k": st.session_state.top_k,
        "temperature": st.session_state.temperature,
        "chat_history": _chat_history_payload(exclude_latest_user=True),
    }

    if BACKEND_CALL_MODE == "inprocess":
        try:
            services = _inprocess_backend()
            history = [services["ChatMessage"](**item) for item in payload["chat_history"]]
            result = services["answer_question"](
                question=question,
                top_k=payload["top_k"],
                temperature=payload["temperature"],
                chat_history=history,
            )
            data = result.model_dump() if hasattr(result, "model_dump") else result.dict()
            return data.get("answer", ""), data.get("sources", []), None
        except Exception as exc:
            return "", [], f"In-process QA failed: {exc}"

    try:
        response = requests.post(url, json=payload, timeout=120)
    except requests.RequestException as exc:
        return "", [], f"Could not reach the FastAPI server at {url}. {exc}"

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return "", [], f"API returned {response.status_code}: {detail}"

    data = response.json()
    return data.get("answer", ""), data.get("sources", []), None


def render_streaming_message(container, content: str, status: str = "") -> None:
    display_content = content.strip() or status.strip()
    if not display_content:
        container.empty()
        return
    content_html = markdown_to_html(display_content)
    container.markdown(
        f"""
        <div class="message-row assistant">
            <div class="bubble assistant">
                <span class="role-label">{escape(ASSISTANT_NAME)}</span>
                <div class="markdown-body">{content_html}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _send_avatar_queue_event(
    sender_container,
    queue_id: str,
    sequence: int,
    audio_bytes: bytes | None = None,
    spoken_text: str = "",
    audio_mime: str = DEFAULT_AUDIO_MIME,
    final: bool = False,
) -> None:
    event = {
        "type": "texmin:avatar-queue",
        "queueId": queue_id,
        "sequence": sequence,
        "audio": base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else "",
        "skip": not audio_bytes and not final,
        "mime": audio_mime,
        "text": spoken_text,
        "final": final,
        "finalSequence": sequence if final else None,
    }
    event_json = json.dumps(event, ensure_ascii=False).replace("</", "<\\/")
    with sender_container:
        components.html(
            f"""
            <script>
                const queueEvent = {event_json};
                const broadcast = () => {{
                    for (let index = 0; index < window.parent.frames.length; index += 1) {{
                        window.parent.frames[index].postMessage(queueEvent, "*");
                    }}
                }};
                broadcast();
                window.setTimeout(broadcast, 250);
                window.setTimeout(broadcast, 750);
                window.setTimeout(broadcast, 1500);
            </script>
            """,
            height=0,
        )


def render_audio_queue_player(queue_id: str, request_id: str) -> None:
    queue_json = json.dumps(queue_id)
    request_json = json.dumps(request_id)
    components.html(
        f"""
        <audio id="voice" controls playsinline style="width:100%;height:42px"></audio>
        <script>
          const queueId = {queue_json};
          const requestId = {request_json};
          const audio = document.getElementById("voice");
          const ready = new Map();
          const clips = [];
          let nextSequence = 0;
          let finalSequence = null;
          let interrupted = false;
          let firstPlayback = true;
          const startedAt = performance.now();
          const startedEpoch = Date.now();
          function resumeListener() {{
            window.localStorage.setItem("texmin_voice_resume_at", String(Date.now()));
          }}
          function playNext() {{
            if (interrupted) return;
            if (!clips.length) {{
              if (finalSequence !== null && nextSequence >= finalSequence) resumeListener();
              return;
            }}
            const item = clips.shift();
            audio.src = `data:${{item.mime}};base64,${{item.audio}}`;
            audio.load();
            audio.play().catch(() => {{}});
          }}
          window.addEventListener("message", (event) => {{
            const message = event.data || {{}};
            if (message.type !== "texmin:avatar-queue" || message.queueId !== queueId || interrupted) return;
            if ((message.audio || message.skip) && message.sequence >= nextSequence) {{
              ready.set(message.sequence, message);
              while (ready.has(nextSequence)) {{
                const item = ready.get(nextSequence);
                if (item.audio) clips.push(item);
                ready.delete(nextSequence++);
              }}
            }}
            if (message.final) finalSequence = message.finalSequence;
            if (audio.paused && !audio.src) playNext();
          }});
          audio.addEventListener("play", () => {{
            if (firstPlayback) {{
              firstPlayback = false;
              console.info(JSON.stringify({{event:"voice_first_playback",request_id:requestId,
                elapsed_from_player_ms:Math.round(performance.now()-startedAt)}}));
            }}
          }});
          audio.addEventListener("ended", () => {{
            audio.removeAttribute("src");
            playNext();
          }});
          function interrupt() {{
            const interruptAt = Number(window.localStorage.getItem("texmin_voice_interrupt_at") || 0);
            if (!interruptAt || interruptAt < startedEpoch) return;
            interrupted = true;
            clips.length = 0;
            ready.clear();
            audio.pause();
            audio.removeAttribute("src");
            resumeListener();
          }}
          window.addEventListener("storage", interrupt);
          window.setInterval(interrupt, 250);
        </script>
        """,
        height=48,
    )


def _render_streaming_avatar_audio(
    avatar_container,
    audio_bytes: bytes,
    spoken_text: str,
    audio_mime: str = DEFAULT_AUDIO_MIME,
    resume_listener_on_end: bool = False,
) -> None:
    if avatar_container is None or not audio_bytes or not st.session_state.avatar_enabled:
        return

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    with avatar_container:
        render_lip_sync_avatar(
            audio_b64,
            spoken_text,
            autoplay=True,
            resume_listener_on_end=resume_listener_on_end,
            audio_mime=audio_mime,
        )


def ask_api_stream(
    question: str,
    container,
    avatar_container=None,
    request_id: str | None = None,
    input_type: str = "text",
) -> tuple[str, list[dict], str | None, bytes | None, str, str | None]:
    url = st.session_state.api_url.rstrip("/") + "/qa/ask/stream"
    payload = {
        "question": question,
        "top_k": st.session_state.top_k,
        "temperature": st.session_state.temperature,
        "chat_history": _chat_history_payload(exclude_latest_user=True),
        "session_id": st.session_state.conversation_session_id,
        "request_id": request_id or uuid.uuid4().hex,
        "input_type": input_type,
    }
    answer = ""
    sources = []
    status = ""
    first_audio = None
    replay_chunks = []
    speech_segments = IncrementalSpeechSegments()
    first_audio_started_at = None
    response_started_at = time.perf_counter()
    audio_mime = DEFAULT_AUDIO_MIME
    audio_error = None
    done_received = False
    queue_sequence = 0
    queue_id = uuid.uuid4().hex if avatar_container is not None and st.session_state.tts_enabled else None
    st.session_state.audio_queue_active = bool(queue_id)
    queue_sender = st.container() if queue_id else None
    tts_executor: ThreadPoolExecutor | None = None
    tts_futures: dict[int, tuple[Future, str]] = {}
    next_audio_sequence = 0
    tts_config = _current_tts_config(question)
    tts_config["session_id"] = payload["session_id"]
    tts_config["request_id"] = payload["request_id"]
    spoken_words = 0

    def submit_tts(segment: str) -> None:
        nonlocal queue_sequence, tts_executor, spoken_words
        if not queue_id or not tts_config["enabled"] or not segment.strip():
            return
        available_words = max(0, int(tts_config["max_words"]) - spoken_words)
        if not available_words:
            return
        words = segment.split()
        segment = " ".join(words[:available_words])
        spoken_words += min(len(words), available_words)
        if len(tts_futures) >= int(config.TTS_STREAM_CONCURRENCY) * 2:
            flush_tts(block=True)
        if tts_executor is None:
            tts_executor = ThreadPoolExecutor(max_workers=int(config.TTS_STREAM_CONCURRENCY), thread_name_prefix="texmin-tts")
        sequence = queue_sequence
        queue_sequence += 1
        tts_futures[sequence] = (tts_executor.submit(_request_tts, segment, tts_config), segment)

    def flush_tts(block: bool = False) -> None:
        nonlocal next_audio_sequence, audio_error, audio_mime, first_audio, first_audio_started_at
        if queue_sender is None:
            return
        while next_audio_sequence in tts_futures:
            future, segment = tts_futures[next_audio_sequence]
            if not block and not future.done():
                break
            try:
                segment_audio, segment_mime, segment_error = future.result(
                    timeout=float(config.TTS_RESPONSE_TIMEOUT_SECONDS)
                )
            except TimeoutError:
                audio_error = "Speech generation timed out. Your text answer is available above."
                future.cancel()
                for skipped_sequence in range(next_audio_sequence, queue_sequence):
                    _send_avatar_queue_event(queue_sender, queue_id, skipped_sequence)
                tts_futures.clear()
                next_audio_sequence = queue_sequence
                break
            except Exception as exc:
                segment_audio, segment_mime, segment_error = None, DEFAULT_AUDIO_MIME, str(exc)
            del tts_futures[next_audio_sequence]
            if segment_error:
                audio_error = segment_error
                _send_avatar_queue_event(queue_sender, queue_id, next_audio_sequence)
            elif segment_audio:
                if first_audio is None:
                    first_audio = segment_audio
                    first_audio_started_at = round((time.perf_counter() - response_started_at) * 1000, 1)
                    print(json.dumps({"event": "voice_first_audio_ready", "request_id": payload["request_id"], "first_audio_ms": first_audio_started_at}), flush=True)
                if not st.session_state.avatar_enabled:
                    replay_chunks.append(segment_audio)
                audio_mime = segment_mime
                _send_avatar_queue_event(
                    queue_sender,
                    queue_id,
                    next_audio_sequence,
                    segment_audio,
                    segment,
                    segment_mime,
                )
            next_audio_sequence += 1

    if queue_id:
        with avatar_container:
            if st.session_state.avatar_enabled:
                render_lip_sync_avatar(
                    "", "", autoplay=True, resume_listener_on_end=True,
                    queue_id=queue_id, request_id=payload["request_id"], thinking=True,
                )
            else:
                render_audio_queue_player(queue_id, payload["request_id"])

    try:
        render_streaming_message(container, "", status)
        for event in _qa_events(url, payload):
            flush_tts()
            event_type = event.get("type")
            if event_type == "status":
                status = event.get("message") or status
                if not answer:
                    render_streaming_message(container, "", status)
            elif event_type == "token":
                token = event.get("text", "")
                answer += token
                for segment in speech_segments.push(token):
                    submit_tts(segment)
                render_streaming_message(container, answer, status)
            elif event_type == "done":
                done_received = True
                answer = event.get("answer") or answer
                sources = event.get("sources", [])
                status = ""
                render_streaming_message(container, answer, status)
            elif event_type == "error":
                if tts_executor is not None:
                    tts_executor.shutdown(wait=False, cancel_futures=True)
                if queue_id and queue_sender is not None:
                    _send_avatar_queue_event(queue_sender, queue_id, queue_sequence, final=True)
                return "", [], event.get("message", "Streaming failed."), None, audio_mime, audio_error
            elif event_type == "cancelled":
                if tts_executor is not None:
                    tts_executor.shutdown(wait=False, cancel_futures=True)
                if queue_id and queue_sender is not None:
                    _send_avatar_queue_event(queue_sender, queue_id, queue_sequence, final=True)
                return "", [], "__cancelled__", None, audio_mime, None
    except Exception as exc:
        if tts_executor is not None:
            tts_executor.shutdown(wait=False, cancel_futures=True)
        if queue_id and queue_sender is not None:
            _send_avatar_queue_event(queue_sender, queue_id, queue_sequence, final=True)
        transport = "in-process backend" if BACKEND_CALL_MODE == "inprocess" else url
        return "", [], f"Could not use {transport}. {exc}", None, audio_mime, audio_error
    except BaseException:
        if tts_executor is not None:
            tts_executor.shutdown(wait=False, cancel_futures=True)
        raise

    if not done_received or not answer.strip():
        if tts_executor is not None:
            tts_executor.shutdown(wait=False, cancel_futures=True)
        if queue_id and queue_sender is not None:
            _send_avatar_queue_event(queue_sender, queue_id, queue_sequence, final=True)
        return (
            "",
            [],
            "The answer stream ended without a complete response. Check the app and Ollama logs.",
            None,
            audio_mime,
            audio_error,
        )

    if queue_id and queue_sender is not None:
        for segment in speech_segments.finish():
            submit_tts(segment)
        flush_tts(block=True)
        if tts_executor is not None:
            tts_executor.shutdown(wait=False, cancel_futures=True)
        _send_avatar_queue_event(
            queue_sender,
            queue_id,
            next_audio_sequence,
            final=True,
        )
        audio_bytes = join_audio_chunks(replay_chunks, audio_mime) if not st.session_state.avatar_enabled else None
    else:
        audio_bytes = None
        if st.session_state.tts_enabled and answer.strip():
            audio_bytes, audio_mime, audio_error = ask_tts(answer)
        if audio_bytes:
            _render_streaming_avatar_audio(
                avatar_container,
                audio_bytes,
                answer,
                audio_mime,
                resume_listener_on_end=True,
            )

    return answer.strip(), sources, None, audio_bytes, audio_mime, audio_error


def _current_tts_config(question: str = "") -> dict:
    if not question:
        question = next((str(item.get("content", "")) for item in reversed(st.session_state.messages) if item.get("role") == "user"), "")
    return {
        "enabled": bool(st.session_state.tts_enabled),
        "voice_id": st.session_state.tts_voice_id.strip(),
        "tone": st.session_state.tts_tone,
        "rate": st.session_state.tts_rate.strip() or "+0%",
        "pitch": st.session_state.tts_pitch.strip() or "+0Hz",
        "max_words": st.session_state.tts_max_words,
        "api_url": st.session_state.api_url.rstrip("/"),
        "language": response_language(question),
    }


def _request_tts(text: str, tts_options: dict) -> tuple[bytes | None, str, str | None]:
    if not tts_options["enabled"]:
        return None, DEFAULT_AUDIO_MIME, None

    voice_id = tts_options["voice_id"]
    if not voice_id:
        return None, DEFAULT_AUDIO_MIME, "Voice is off: add a voice name in the sidebar to hear replies."

    url = tts_options["api_url"] + "/tts/speech"
    payload = {
        "text": text,
        "session_id": tts_options.get("session_id"),
        "request_id": tts_options.get("request_id"),
        "voice_id": voice_id,
        "language": detect_language(text, fallback=tts_options["language"]),
        "tone": tts_options["tone"],
        "rate": tts_options["rate"],
        "pitch": tts_options["pitch"],
        "volume": "+0%",
        "response_format": "mp3" if DEFAULT_TTS_ENGINE in {"edge", "edge-tts"} else "wav",
        "max_words": tts_options["max_words"],
    }

    if BACKEND_CALL_MODE == "inprocess":
        try:
            services = _inprocess_backend()
            request = services["TTSRequest"](**payload)
            audio, audio_mime = services["synthesize_speech"](request)
            return audio, audio_mime, None
        except Exception as exc:
            return None, DEFAULT_AUDIO_MIME, f"Voice is unavailable: {exc}"

    try:
        response = requests.post(url, json=payload, timeout=180)
    except requests.RequestException as exc:
        return None, DEFAULT_AUDIO_MIME, f"Voice could not connect to {url}. {exc}"

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return None, DEFAULT_AUDIO_MIME, f"Voice is unavailable: {detail}"

    audio_mime = response.headers.get("content-type", DEFAULT_AUDIO_MIME).split(";", 1)[0]
    return response.content, audio_mime or DEFAULT_AUDIO_MIME, None


def ask_tts(text: str) -> tuple[bytes | None, str, str | None]:
    return _request_tts(text, _current_tts_config())


def get_voice_query_param() -> str | None:
    voice_query = st.query_params.get("voice_query")
    if isinstance(voice_query, list):
        voice_query = voice_query[0] if voice_query else ""
    voice_query = (voice_query or "").strip()
    if not voice_query:
        return None

    return voice_query


init_state()
voice_query_param = get_voice_query_param()

with st.sidebar:
    st.markdown("### Settings")
    if BACKEND_CALL_MODE == "http":
        st.session_state.api_url = st.text_input("API URL", value=st.session_state.api_url)
    else:
        st.caption("Backend: in-process (offline)")
    st.session_state.top_k = st.slider("Context chunks", 1, 10, st.session_state.top_k)
    st.session_state.temperature = st.slider(
        "Response warmth", 0.0, 1.0, st.session_state.temperature, 0.05
    )
    st.markdown("### Voice")
    st.session_state.tts_enabled = st.toggle(
        "Read AI replies aloud",
        value=st.session_state.tts_enabled,
    )
    st.session_state.avatar_enabled = st.toggle(
        "Show lip-sync avatar",
        value=st.session_state.avatar_enabled,
    )
    voice_labels = [*VOICE_OPTIONS.keys()]
    if DEFAULT_TTS_ENGINE not in {"auto", "piper", "espeak"}:
        voice_labels.append(CUSTOM_VOICE_LABEL)
    if st.session_state.tts_voice_choice not in voice_labels:
        st.session_state.tts_voice_choice = voice_labels[0]
    st.session_state.tts_voice_choice = st.selectbox(
        "Preferred voice",
        voice_labels,
        index=voice_labels.index(st.session_state.tts_voice_choice),
    )
    if st.session_state.tts_voice_choice == CUSTOM_VOICE_LABEL:
        st.session_state.tts_voice_id = st.text_input(
            "Voice name",
            value=st.session_state.tts_voice_id,
            placeholder="For example, Hindi or en-IN-NeerjaNeural",
        )
    else:
        st.session_state.tts_voice_id = VOICE_OPTIONS[st.session_state.tts_voice_choice]

    if st.session_state.tts_tone not in TONE_OPTIONS:
        st.session_state.tts_tone = "neutral"
    st.session_state.tts_tone = st.selectbox(
        "Tone and expression",
        TONE_OPTIONS,
        index=TONE_OPTIONS.index(st.session_state.tts_tone),
    )
    if st.session_state.tts_tone == "custom":
        st.session_state.tts_rate = st.text_input(
            "Speech rate",
            value=st.session_state.tts_rate,
            help="Use values such as +10% or -10%.",
        )
        st.session_state.tts_pitch = st.text_input(
            "Speech pitch",
            value=st.session_state.tts_pitch,
            help="Use values such as +2Hz or -2Hz.",
        )
    else:
        st.caption("Tone presets adjust rate, pitch, and volume automatically.")
    st.session_state.tts_max_words = st.slider(
        "Spoken length",
        80,
        700,
        st.session_state.tts_max_words,
        20,
        help="Long answers are shortened for smoother, more natural speech.",
    )
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="app-title">
        <div class="brand-lockup">
            <div class="brand-mark">K</div>
            <div>
                <div class="brand-title">{escape(ASSISTANT_NAME)}</div>
                <div class="brand-subtitle">Grounded answers from your indexed mining documents</div>
            </div>
        </div>
        <div class="status-pill">Mining Trained</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
            <h2>Ask your documents.</h2>
            <p>
                Short, grounded answers with less waiting between question and reply.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for message in st.session_state.messages:
        render_message(message)

live_response_container = st.container()
st.markdown("</div>", unsafe_allow_html=True)

voice_component_query = render_voice_query_component()
typed_question = st.chat_input(f"Message {ASSISTANT_NAME}...")
question = typed_question or (voice_component_query or {}).get("text") or voice_query_param

latest_audio_b64, latest_audio_text, latest_audio_autoplay, latest_audio_mime = latest_assistant_audio()
if not question and st.session_state.avatar_enabled and latest_audio_b64:
    render_lip_sync_avatar(
        latest_audio_b64,
        latest_audio_text,
        autoplay=latest_audio_autoplay,
        audio_mime=latest_audio_mime,
    )

if question:
    if st.session_state.messages and st.session_state.messages[-1].get("role") == "user":
        # A rerun interrupted its previous answer; omit that unfinished turn.
        st.session_state.messages.pop()
    if typed_question:
        components.html(
            '<script>window.localStorage.setItem("texmin_voice_interrupt_at", String(Date.now()));</script>',
            height=0,
        )
    else:
        clear_voice_interrupt_signal()
    with live_response_container:
        user_message = {"role": "user", "content": question}
        st.session_state.messages.append(user_message)
        render_message(user_message)

        stream_container = st.empty()
        avatar_stream_container = st.empty()
        answer, sources, error, audio_bytes, audio_mime, audio_error = ask_api_stream(
            question,
            stream_container,
            avatar_stream_container,
            request_id=voice_component_query["request_id"] if voice_component_query and not typed_question else None,
            input_type=voice_component_query["input_type"] if voice_component_query and not typed_question else "text",
        )

    if error == "__cancelled__":
        if st.session_state.messages and st.session_state.messages[-1].get("role") == "user":
            st.session_state.messages.pop()
        stream_container.empty()
    elif error:
        stream_container.error(error)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error,
                "sources": [],
                "error": True,
            }
        )
    else:
        if audio_error:
            st.warning(audio_error)
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else None
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "audio_b64": audio_b64,
                "audio_mime": audio_mime,
                "audio_error": audio_error,
                "audio_autoplay": not bool(audio_bytes),
            }
        )
    if (voice_component_query or voice_query_param) and not st.session_state.get("audio_queue_active", False):
        resume_voice_listener_without_audio()
    if voice_query_param and "voice_query" in st.query_params:
        del st.query_params["voice_query"]
