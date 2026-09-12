import os
import base64
import json
import mimetypes
import re
import uuid
from pathlib import Path
from html import escape

import markdown
import requests
import streamlit as st
import streamlit.components.v1 as components


DEFAULT_API_URL = os.getenv("QA_API_URL", "http://127.0.0.1:8000")
DEFAULT_TTS_VOICE = os.getenv("TTS_VOICE", os.getenv("EDGE_TTS_VOICE", "en-IN-NeerjaNeural"))
DEFAULT_TTS_ENGINE = os.getenv("TTS_ENGINE", "edge").strip().lower()
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

VOICE_OPTIONS = {
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
CUSTOM_VOICE_LABEL = "Custom voice name"
TONE_OPTIONS = ["neutral", "warm", "cheerful", "calm", "serious", "energetic", "custom"]


st.set_page_config(
    page_title="KHOJ ChatBot",
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
    if "api_url" not in st.session_state:
        st.session_state.api_url = DEFAULT_API_URL
    if "top_k" not in st.session_state:
        st.session_state.top_k = 3
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.35
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
        st.session_state.tts_tone = os.getenv("TTS_TONE", os.getenv("EDGE_TTS_TONE", "neutral")).strip().lower()
    if "tts_rate" not in st.session_state:
        st.session_state.tts_rate = os.getenv("TTS_RATE", os.getenv("EDGE_TTS_RATE", "+0%"))
    if "tts_pitch" not in st.session_state:
        st.session_state.tts_pitch = os.getenv("TTS_PITCH", os.getenv("EDGE_TTS_PITCH", "+0Hz"))
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
    configured_path = os.getenv("AVATAR_MODEL_FILE", "").strip()
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

    if extension == ".gltf":
        uri = _inline_gltf_assets(model_path)
    else:
        uri = _file_data_uri(model_path, MODEL_MIME_TYPES[extension])

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
    audio_mime: str = DEFAULT_AUDIO_MIME,
) -> None:
    spoken_text = " ".join((text or "").split())
    safe_text = escape(spoken_text[:120])
    model_json = json.dumps(avatar_model_source())
    speech_text_json = json.dumps(spoken_text[:4000])
    autoplay_attr = "autoplay" if autoplay else ""
    autoplay_js = "true" if autoplay else "false"
    resume_listener_js = "true" if resume_listener_on_end else "false"
    queue_id_json = json.dumps(queue_id or "")
    audio_mime_json = json.dumps(audio_mime or DEFAULT_AUDIO_MIME)
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

        <div class="texmin-avatar" id="avatar">
            <div class="avatar-stage">
                <div id="avatarScene" aria-label="KHOJ ChatBOT 3D presenter avatar">
                    <div class="model-status" id="modelStatus">Loading 3D avatar...</div>
                    <div class="avatar-badge">3D AVATAR</div>
                </div>
                <div class="avatar-copy">
                    <div class="avatar-title">KHOJ ChatBOT</div>
                    <div class="avatar-line" id="avatarLine">{safe_text}</div>
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
            const initialAudioMime = {audio_mime_json};
            const queuedPlayback = Boolean(queueId);
            const storageInterruptKey = "texmin_voice_interrupt_at";
            const storageAvatarSpeakingKey = "texmin_voice_avatar_speaking_at";
            const storageWaitingKey = "texmin_voice_waiting_since";
            const storageInputSuppressedUntilKey = "texmin_voice_input_suppressed_until";
            const waitingCueText = "";
            const avatar = document.getElementById("avatar");
            const audio = document.getElementById("voice");
            const avatarLine = document.getElementById("avatarLine");
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
                    avatarRoot.rotation.y = avatarBaseYaw + Math.sin(t * 0.72) * 0.06 + speechEnergy * Math.sin(t * 5.2) * 0.012;
                    const base = avatarRoot.userData.basePosition || new THREE.Vector3(0, -0.04, 0);
                    avatarRoot.position.set(base.x, base.y + Math.sin(t * 1.3) * 0.018, base.z);
                }}
                if (headNode) {{
                    headNode.rotation.x = headNode.userData.restRotationX + speechEnergy * Math.sin(t * 3.1 + 0.4) * 0.006;
                    headNode.rotation.y = headNode.userData.restRotationY;
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
                if (message.audio) {{
                    pendingQueueItems.set(message.sequence, {{
                        audio: message.audio,
                        text: message.text || "",
                        mime: message.mime || initialAudioMime,
                    }});
                    while (pendingQueueItems.has(nextQueueSequence)) {{
                        audioQueue.push(pendingQueueItems.get(nextQueueSequence));
                        pendingQueueItems.delete(nextQueueSequence);
                        nextQueueSequence += 1;
                    }}
                }}
                if (message.final) {{
                    queueFinalized = true;
                    finalQueueSequence = message.finalSequence;
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
    label = "You" if role == "user" else "KHOJ ChatBOT"
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


def render_speech_to_text_control() -> None:
    components.html(
        """
        <style>
            html,
            body {
                margin: 0;
                background: transparent;
                font-family: Inter, Segoe UI, Arial, sans-serif;
                color: #edf3ff;
            }

            .voice-query {
                border: 1px solid rgba(82, 214, 180, 0.24);
                border-radius: 12px;
                background: rgba(16, 24, 39, 0.7);
                padding: 10px;
                display: grid;
                grid-template-columns: auto minmax(120px, 1fr);
                gap: 10px;
                align-items: center;
                box-sizing: border-box;
            }

            .mic-button {
                border: 1px solid rgba(82, 214, 180, 0.42);
                border-radius: 10px;
                background: rgba(82, 214, 180, 0.1);
                color: #b7ffeb;
                height: 38px;
                padding: 0 12px;
                display: inline-flex;
                align-items: center;
                gap: 8px;
                font-weight: 760;
                cursor: pointer;
                white-space: nowrap;
            }

            .mic-button.recording {
                border-color: rgba(255, 101, 132, 0.65);
                background: rgba(255, 101, 132, 0.14);
                color: #ffd5de;
            }

            .mic-dot {
                width: 9px;
                height: 9px;
                border-radius: 999px;
                background: #52d6b4;
                box-shadow: 0 0 0 4px rgba(82, 214, 180, 0.16);
            }

            .mic-button.recording .mic-dot {
                background: #ff6584;
                box-shadow: 0 0 0 4px rgba(255, 101, 132, 0.18);
            }

            .voice-panel {
                min-width: 0;
                display: grid;
                grid-template-columns: minmax(90px, 1fr) minmax(140px, 1.8fr);
                gap: 10px;
                align-items: center;
            }

            canvas {
                width: 100%;
                height: 38px;
                border-radius: 8px;
                background: rgba(9, 14, 24, 0.54);
                box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.08);
            }

            .voice-text {
                min-width: 0;
            }

            .voice-status {
                color: #9fb0c7;
                font-size: 0.75rem;
                line-height: 1.25;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }

            .voice-preview {
                color: #edf3ff;
                font-size: 0.78rem;
                line-height: 1.3;
                min-height: 1.3em;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                margin-top: 2px;
            }

            @media (max-width: 560px) {
                .voice-query,
                .voice-panel {
                    grid-template-columns: 1fr;
                }

                .mic-button {
                    justify-content: center;
                    width: 100%;
                }
            }
        </style>

        <div class="voice-query">
            <button class="mic-button" id="micButton" type="button" aria-label="Start voice query">
                <span class="mic-dot"></span>
                <span id="micLabel">Start recording</span>
            </button>
            <div class="voice-panel">
                <canvas id="waveCanvas" width="260" height="76" aria-label="Input volume waves"></canvas>
                <div class="voice-text">
                    <div class="voice-status" id="voiceStatus">Audio is transcribed locally by your browser.</div>
                    <div class="voice-preview" id="voicePreview"></div>
                </div>
            </div>
        </div>

        <script>
            const button = document.getElementById("micButton");
            const label = document.getElementById("micLabel");
            const statusText = document.getElementById("voiceStatus");
            const preview = document.getElementById("voicePreview");
            const canvas = document.getElementById("waveCanvas");
            const canvasContext = canvas.getContext("2d");
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            let recognition = null;
            let listening = false;
            let mediaStream = null;
            let audioContext = null;
            let analyser = null;
            let waveData = null;
            let animationId = null;
            let finalTranscript = "";
            let interimTranscript = "";
            let latestTranscript = "";
            let heardSpeech = false;
            let lastVoiceAt = 0;
            let lastTranscriptAt = 0;
            const noTranscriptSilenceLimitMs = 1850;
            const transcriptSilenceLimitMs = 850;
            const finalTranscriptSilenceLimitMs = 520;
            const continuedSpeechThreshold = 0.055;
            const voiceThreshold = 0.025;

            function drawIdleWave() {
                const width = canvas.width;
                const height = canvas.height;
                canvasContext.clearRect(0, 0, width, height);
                canvasContext.fillStyle = "rgba(9, 14, 24, 0.45)";
                canvasContext.fillRect(0, 0, width, height);
                const bars = 28;
                const gap = 3;
                const barWidth = (width - gap * (bars - 1)) / bars;
                for (let index = 0; index < bars; index += 1) {
                    const phase = index / bars;
                    const barHeight = 5 + Math.sin(phase * Math.PI) * 10;
                    const x = index * (barWidth + gap);
                    const y = (height - barHeight) / 2;
                    canvasContext.fillStyle = "rgba(82, 214, 180, 0.22)";
                    canvasContext.fillRect(x, y, barWidth, barHeight);
                }
            }

            function drawLiveWave() {
                if (!analyser || !waveData) {
                    drawIdleWave();
                    return;
                }

                analyser.getByteTimeDomainData(waveData);
                let sum = 0;
                for (let index = 0; index < waveData.length; index += 1) {
                    const value = (waveData[index] - 128) / 128;
                    sum += value * value;
                }
                const volume = Math.min(1, Math.sqrt(sum / waveData.length) * 4.2);
                if (listening) {
                    const now = Date.now();
                    const transcript = (latestTranscript || finalTranscript || interimTranscript).trim();
                    if (transcript) {
                        const quietLimit = interimTranscript ? transcriptSilenceLimitMs : finalTranscriptSilenceLimitMs;
                        if (volume > continuedSpeechThreshold) {
                            lastTranscriptAt = now;
                        } else if (now - (lastTranscriptAt || lastVoiceAt) > quietLimit) {
                            stopRecording("silence");
                        }
                    } else if (volume > voiceThreshold) {
                        heardSpeech = true;
                        lastVoiceAt = now;
                    } else if (heardSpeech && now - lastVoiceAt > noTranscriptSilenceLimitMs) {
                        stopRecording("silence");
                    }
                }
                const width = canvas.width;
                const height = canvas.height;
                const bars = 34;
                const gap = 3;
                const barWidth = (width - gap * (bars - 1)) / bars;

                canvasContext.clearRect(0, 0, width, height);
                canvasContext.fillStyle = "rgba(9, 14, 24, 0.45)";
                canvasContext.fillRect(0, 0, width, height);

                for (let index = 0; index < bars; index += 1) {
                    const sampleIndex = Math.floor((index / bars) * waveData.length);
                    const sample = Math.abs((waveData[sampleIndex] - 128) / 128);
                    const pulse = 0.35 + sample * 1.8 + volume * 1.2;
                    const barHeight = Math.max(4, Math.min(height - 8, pulse * height * 0.45));
                    const x = index * (barWidth + gap);
                    const y = (height - barHeight) / 2;
                    const intensity = Math.min(1, 0.38 + volume * 0.62);
                    canvasContext.fillStyle = `rgba(${Math.round(82 + 45 * intensity)}, ${Math.round(214 + 22 * intensity)}, ${Math.round(180 + 20 * intensity)}, ${0.35 + intensity * 0.55})`;
                    canvasContext.fillRect(x, y, barWidth, barHeight);
                }

                animationId = requestAnimationFrame(drawLiveWave);
            }

            function setButtonState(recording) {
                button.classList.toggle("recording", recording);
                label.textContent = recording ? "Stop recording" : "Start recording";
            }

            function findChatInput() {
                try {
                    const parentDocument = window.parent.document;
                    const textareas = Array.from(parentDocument.querySelectorAll("textarea"));
                    return textareas.find((node) => {
                        const placeholder = node.getAttribute("placeholder") || "";
                        return placeholder.toLowerCase().includes("message khoj");
                    }) || textareas[textareas.length - 1] || null;
                } catch (error) {
                    console.error(error);
                    return null;
                }
            }

            function findSendButton(input) {
                try {
                    const parentDocument = window.parent.document;
                    const chatInput = input ? input.closest("[data-testid='stChatInput']") : null;
                    const localButton = chatInput ? chatInput.querySelector("button") : null;
                    if (localButton) return localButton;

                    return parentDocument.querySelector("button[data-testid='stChatInputSubmitButton']")
                        || parentDocument.querySelector("button[aria-label='Send message']")
                        || parentDocument.querySelector("button[title='Send message']");
                } catch (error) {
                    console.error(error);
                    return null;
                }
            }

            function submitViaChatInput(text) {
                const cleaned = text.trim();
                if (!cleaned) return false;

                try {
                    const input = findChatInput();
                    if (!input) return false;

                    const setter = Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype, "value").set;
                    setter.call(input, cleaned);
                    input.dispatchEvent(new InputEvent("input", {
                        bubbles: true,
                        inputType: "insertText",
                        data: cleaned,
                    }));
                    input.dispatchEvent(new Event("change", { bubbles: true }));
                    input.focus();

                    let attempts = 0;
                    const submitWhenReady = () => {
                        attempts += 1;
                        const sendButton = findSendButton(input);
                        if (sendButton && !sendButton.disabled && sendButton.getAttribute("aria-disabled") !== "true") {
                            sendButton.click();
                            return;
                        }

                        if (attempts < 8) {
                            window.setTimeout(submitWhenReady, 120);
                            return;
                        }

                        if (input.value.trim()) {
                            input.dispatchEvent(new KeyboardEvent("keydown", {
                                key: "Enter",
                                code: "Enter",
                                bubbles: true,
                                cancelable: true,
                            }));
                        }
                    };
                    window.setTimeout(submitWhenReady, 120);
                    return true;
                } catch (error) {
                    console.error(error);
                    return false;
                }
            }

            function sendTranscriptAsQuery(text) {
                try {
                    const encoded = encodeURIComponent(text.trim());
                    if (!encoded) return false;
                    window.parent.location.href = `?voice_query=${encoded}`;
                    return true;
                } catch (error) {
                    console.error(error);
                    return false;
                }
            }

            function submitTranscript(text) {
                const cleaned = text.trim();
                if (!cleaned) return false;
                return submitViaChatInput(cleaned) || sendTranscriptAsQuery(cleaned);
            }

            async function startMeter() {
                mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                const AudioContext = window.AudioContext || window.webkitAudioContext;
                audioContext = new AudioContext();
                analyser = audioContext.createAnalyser();
                analyser.fftSize = 512;
                waveData = new Uint8Array(analyser.fftSize);
                const source = audioContext.createMediaStreamSource(mediaStream);
                source.connect(analyser);
                drawLiveWave();
            }

            function stopMeter() {
                if (animationId) {
                    cancelAnimationFrame(animationId);
                    animationId = null;
                }
                if (mediaStream) {
                    mediaStream.getTracks().forEach((track) => track.stop());
                    mediaStream = null;
                }
                if (audioContext) {
                    audioContext.close();
                    audioContext = null;
                }
                analyser = null;
                waveData = null;
                drawIdleWave();
            }

            function buildRecognition() {
                const instance = new SpeechRecognition();
                instance.lang = navigator.language || "en-IN";
                if (!["en", "hi"].includes(instance.lang.slice(0, 2).toLowerCase())) {{
                    instance.lang = "en-IN";
                }}
                instance.continuous = true;
                instance.interimResults = true;

                instance.onresult = (event) => {
                    interimTranscript = "";
                    for (let index = event.resultIndex; index < event.results.length; index += 1) {
                        const transcript = event.results[index][0].transcript;
                        if (event.results[index].isFinal) {
                            finalTranscript = `${finalTranscript} ${transcript}`.trim();
                        } else {
                            interimTranscript = `${interimTranscript} ${transcript}`.trim();
                        }
                    }

                    const visibleText = [finalTranscript, interimTranscript].filter(Boolean).join(" ");
                    latestTranscript = visibleText.trim();
                    if (latestTranscript) {{
                        heardSpeech = true;
                        const now = Date.now();
                        lastVoiceAt = now;
                        lastTranscriptAt = now;
                    }}
                    preview.textContent = visibleText;
                    statusText.textContent = interimTranscript ? "Listening..." : "Speech recognized.";
                };

                instance.onerror = (event) => {
                    statusText.textContent = event.error === "not-allowed"
                        ? "Microphone permission was blocked."
                        : `Speech recognition stopped: ${event.error}`;
                    listening = false;
                    setButtonState(false);
                    stopMeter();
                };

                instance.onend = () => {
                    if (listening) {
                        try {
                            instance.start();
                            return;
                        } catch (error) {
                            console.error(error);
                        }
                    }

                    const transcript = (latestTranscript || finalTranscript || interimTranscript).trim();
                    const submitted = submitTranscript(transcript);
                    statusText.textContent = submitted
                        ? "Sending voice query..."
                        : transcript
                            ? "Transcript captured, but the query could not be sent."
                            : heardSpeech
                                ? "I heard audio, but the browser did not return text. Try Chrome or Edge over HTTPS."
                                : "Recording stopped. No transcript was captured.";
                    setButtonState(false);
                    stopMeter();
                };

                return instance;
            }

            async function startRecording() {
                if (!SpeechRecognition) {
                    statusText.textContent = "Speech recognition is supported in Chrome or Edge.";
                    return;
                }
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    statusText.textContent = "Microphone access is not available in this browser.";
                    return;
                }

                finalTranscript = "";
                interimTranscript = "";
                latestTranscript = "";
                heardSpeech = false;
                lastVoiceAt = Date.now();
                lastTranscriptAt = 0;
                preview.textContent = "";
                listening = true;
                setButtonState(true);
                statusText.textContent = "Requesting microphone permission...";

                try {
                    await startMeter();
                    recognition = buildRecognition();
                    recognition.start();
                    statusText.textContent = "Listening...";
                } catch (error) {
                    console.error(error);
                    listening = false;
                    setButtonState(false);
                    stopMeter();
                    statusText.textContent = "Microphone could not start.";
                }
            }

            function stopRecording(reason = "manual") {
                if (!listening && !recognition) return;
                listening = false;
                statusText.textContent = reason === "silence"
                    ? "Silence detected. Sending query..."
                    : "Sending voice query...";
                if (recognition) {
                    recognition.stop();
                } else {
                    setButtonState(false);
                    stopMeter();
                }
            }

            button.addEventListener("click", () => {
                if (listening) {
                    stopRecording();
                } else {
                    startRecording();
                }
            });

            drawIdleWave();
        </script>
        """,
        height=92,
    )


def render_voice_query_component() -> str | None:
    value = voice_query_component(default=None, key="texmin_voice_query", height=92)
    if not value:
        return None

    if isinstance(value, str):
        query_id = value
        text = value
    else:
        query_id = str(value.get("id", ""))
        text = str(value.get("text", "")).strip()

    if not text or query_id == st.session_state.last_voice_query_id:
        return None

    st.session_state.last_voice_query_id = query_id
    return text


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
                <span class="role-label">KHOJ ChatBOT</span>
                <div class="markdown-body">{content_html}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _split_speakable_prefix(buffer: str) -> tuple[list[str], str]:
    normalized = buffer.strip()
    if not normalized:
        return [], ""

    segments = []
    cursor = 0
    for match in re.finditer(r"(.+?[.!?\u0964])(\s+|$)", normalized, flags=re.DOTALL):
        segment = match.group(1).strip()
        if segment:
            segments.append(segment)
        cursor = match.end()

    remainder = normalized[cursor:].strip()
    if not segments and len(normalized) >= 170:
        split_at = max(
            normalized.rfind(",", 0, 165),
            normalized.rfind(";", 0, 165),
            normalized.rfind(":", 0, 165),
            normalized.rfind(" - ", 0, 165),
        )
        if split_at <= 80 and len(normalized) >= 230:
            split_at = normalized.rfind(" ", 0, 205)
        if split_at > 80:
            split_end = split_at
            if normalized[split_at : split_at + 3] == " - ":
                split_end = split_at + 3
            elif normalized[split_at] in ",;:":
                split_end = split_at + 1
            segments.append(normalized[:split_end].strip(" ,;:-"))
            remainder = normalized[split_end:].strip()

    return segments, remainder


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
) -> tuple[str, list[dict], str | None, bytes | None, str, str | None]:
    url = st.session_state.api_url.rstrip("/") + "/qa/ask/stream"
    payload = {
        "question": question,
        "top_k": st.session_state.top_k,
        "temperature": st.session_state.temperature,
        "chat_history": _chat_history_payload(exclude_latest_user=True),
    }
    answer = ""
    sources = []
    status = ""
    speech_buffer = ""
    audio_chunks = []
    audio_mime = DEFAULT_AUDIO_MIME
    audio_error = None
    queue_sequence = 0
    queue_id = (
        uuid.uuid4().hex
        if avatar_container is not None
        and st.session_state.tts_enabled
        and st.session_state.avatar_enabled
        else None
    )
    queue_sender = st.container() if queue_id else None

    if queue_id:
        with avatar_container:
            render_lip_sync_avatar(
                "",
                "",
                autoplay=True,
                resume_listener_on_end=True,
                queue_id=queue_id,
            )

    try:
        with requests.post(url, json=payload, timeout=180, stream=True) as response:
            if response.status_code == 404:
                fallback_answer, fallback_sources, fallback_error = ask_api(question)
                if fallback_answer:
                    render_streaming_message(container, fallback_answer)
                fallback_audio = None
                fallback_audio_mime = DEFAULT_AUDIO_MIME
                fallback_audio_error = None
                if fallback_answer and not fallback_error and st.session_state.tts_enabled:
                    fallback_audio, fallback_audio_mime, fallback_audio_error = ask_tts(fallback_answer)
                    if fallback_audio:
                        if queue_id and queue_sender is not None:
                            _send_avatar_queue_event(
                                queue_sender,
                                queue_id,
                                queue_sequence,
                                fallback_audio,
                                fallback_answer,
                                fallback_audio_mime,
                            )
                            queue_sequence += 1
                        else:
                            _render_streaming_avatar_audio(
                                avatar_container,
                                fallback_audio,
                                fallback_answer,
                                fallback_audio_mime,
                            )
                if queue_id and queue_sender is not None:
                    _send_avatar_queue_event(
                        queue_sender,
                        queue_id,
                        queue_sequence,
                        final=True,
                    )
                return fallback_answer, fallback_sources, fallback_error, fallback_audio, fallback_audio_mime, fallback_audio_error

            if not response.ok:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text
                return "", [], f"API returned {response.status_code}: {detail}", None, DEFAULT_AUDIO_MIME, None

            render_streaming_message(container, "", status)
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue

                try:
                    event = json.loads(line.removeprefix("data:").strip())
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")
                if event_type == "status":
                    status = event.get("message") or status
                    if not answer:
                        render_streaming_message(container, "", status)
                elif event_type == "token":
                    token = event.get("text", "")
                    answer += token
                    speech_buffer += token
                    render_streaming_message(container, answer, status)
                    if queue_id and not audio_error:
                        segments, speech_buffer = _split_speakable_prefix(speech_buffer)
                        for segment in segments:
                            segment_audio, segment_mime, segment_error = ask_tts(segment)
                            if segment_error:
                                audio_error = segment_error
                                break
                            if segment_audio and queue_sender is not None:
                                audio_chunks.append(segment_audio)
                                audio_mime = segment_mime
                                _send_avatar_queue_event(
                                    queue_sender,
                                    queue_id,
                                    queue_sequence,
                                    segment_audio,
                                    segment,
                                    segment_mime,
                                )
                                queue_sequence += 1
                elif event_type == "done":
                    answer = event.get("answer") or answer
                    sources = event.get("sources", [])
                    render_streaming_message(container, answer, status)
                elif event_type == "error":
                    return "", [], event.get("message", "Streaming failed."), None, audio_mime, audio_error
    except requests.RequestException as exc:
        return "", [], f"Could not reach the FastAPI server at {url}. {exc}", None, audio_mime, audio_error

    if queue_id and queue_sender is not None:
        remaining_speech = speech_buffer.strip()
        if not remaining_speech and not audio_chunks:
            remaining_speech = answer.strip()
        if remaining_speech and not audio_error:
            segment_audio, segment_mime, segment_error = ask_tts(remaining_speech)
            if segment_error:
                audio_error = segment_error
            elif segment_audio:
                audio_chunks.append(segment_audio)
                audio_mime = segment_mime
                _send_avatar_queue_event(
                    queue_sender,
                    queue_id,
                    queue_sequence,
                    segment_audio,
                    remaining_speech,
                    segment_mime,
                )
                queue_sequence += 1
        _send_avatar_queue_event(
            queue_sender,
            queue_id,
            queue_sequence,
            final=True,
        )
        audio_bytes = audio_chunks[0] if len(audio_chunks) == 1 else None
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


def ask_tts(text: str) -> tuple[bytes | None, str, str | None]:
    if not st.session_state.tts_enabled:
        return None, DEFAULT_AUDIO_MIME, None

    voice_id = st.session_state.tts_voice_id.strip()
    if not voice_id:
        return None, DEFAULT_AUDIO_MIME, "Voice is off: add a voice name in the sidebar to hear replies."

    url = st.session_state.api_url.rstrip("/") + "/tts/speech"
    payload = {
        "text": text,
        "voice_id": voice_id,
        "tone": st.session_state.tts_tone,
        "rate": st.session_state.tts_rate.strip() or "+0%",
        "pitch": st.session_state.tts_pitch.strip() or "+0Hz",
        "volume": "+0%",
        "response_format": "mp3" if DEFAULT_TTS_ENGINE in {"edge", "edge-tts"} else "wav",
        "max_words": st.session_state.tts_max_words,
    }

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
    st.session_state.api_url = st.text_input("API URL", value=st.session_state.api_url)
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
    voice_labels = [*VOICE_OPTIONS.keys(), CUSTOM_VOICE_LABEL]
    if st.session_state.tts_voice_choice not in voice_labels:
        st.session_state.tts_voice_choice = CUSTOM_VOICE_LABEL
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
    """
    <div class="app-title">
        <div class="brand-lockup">
            <div class="brand-mark">K</div>
            <div>
                <div class="brand-title">KHOJ ChatBOT</div>
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
question = st.chat_input("Message KHOJ ChatBOT...") or voice_component_query or voice_query_param

latest_audio_b64, latest_audio_text, latest_audio_autoplay, latest_audio_mime = latest_assistant_audio()
if not question and st.session_state.avatar_enabled and latest_audio_b64:
    render_lip_sync_avatar(
        latest_audio_b64,
        latest_audio_text,
        autoplay=latest_audio_autoplay,
        audio_mime=latest_audio_mime,
    )

if question:
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
        )

    if error:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error,
                "sources": [],
                "error": True,
            }
        )
    else:
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
    if (voice_component_query or voice_query_param) and not (
        audio_bytes and st.session_state.avatar_enabled
    ):
        resume_voice_listener_without_audio()
    if voice_query_param and "voice_query" in st.query_params:
        del st.query_params["voice_query"]
