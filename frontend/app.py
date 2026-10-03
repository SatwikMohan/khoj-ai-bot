import base64
import json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualization import render_ai_visualization
config.configure_runtime_environment()


DEFAULT_API_URL = config.QA_API_URL
BACKEND_CALL_MODE = config.BACKEND_CALL_MODE.strip().lower()
DEFAULT_QA_TEMPERATURE = max(0.0, min(1.0, float(config.QA_TEMPERATURE)))
DEFAULT_TTS_ENGINE = config.TTS_ENGINE.strip().lower()
DEFAULT_TTS_VOICE = config.TTS_VOICE
if DEFAULT_TTS_ENGINE in {"auto", "piper", "espeak"}:
    DEFAULT_TTS_VOICE = "configured"
DEFAULT_AUDIO_MIME = "audio/mpeg" if DEFAULT_TTS_ENGINE in {"edge", "edge-tts"} else "audio/wav"
APP_DIR = Path(__file__).resolve().parent
voice_query_component = components.declare_component(
    "texmin_voice_query", path=str(APP_DIR / "voice_query_component")
)


st.set_page_config(page_title="AI", page_icon="\u2726", layout="wide", initial_sidebar_state="collapsed")
st.markdown(f"<style>{(APP_DIR / 'theme.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


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
        st.session_state.tts_enabled = config.TTS_ENABLED
    if "tts_voice_id" not in st.session_state:
        st.session_state.tts_voice_id = DEFAULT_TTS_VOICE
    if "tts_tone" not in st.session_state:
        st.session_state.tts_tone = config.TTS_TONE.strip().lower()
    if "tts_rate" not in st.session_state:
        st.session_state.tts_rate = config.TTS_RATE
    if "tts_pitch" not in st.session_state:
        st.session_state.tts_pitch = config.TTS_PITCH
    if "tts_max_words" not in st.session_state:
        st.session_state.tts_max_words = config.TTS_MAX_WORDS
    if "avatar_enabled" not in st.session_state:
        st.session_state.avatar_enabled = config.AVATAR_ENABLED
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


def render_message(message: dict) -> None:
    role = message.get("role", "assistant")
    label = "You" if role == "user" else "AI"
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
                <span class="role-label">AI</span>
                <div class="markdown-body">{content_html}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def publish_ai_activity(line_container, signal_container, phase: str, label: str) -> None:
    """Mirror backend lifecycle events into the visualization iframe."""
    line_container.markdown(
        f'<div class="activity-line">{escape(label)}</div>',
        unsafe_allow_html=True,
    )
    event = json.dumps({"phase": phase, "label": label}, ensure_ascii=False)
    with signal_container:
        components.html(
            f'<script>localStorage.setItem("texmin_ai_activity", '
            f'JSON.stringify({{...{event}, at: Date.now()}}));</script>',
            height=0,
        )


def render_chat_scroll_observer() -> None:
    components.html(
        """<script>
        try {
          const marker = window.parent.document.querySelector(".conversation-marker");
          const wrapper = marker?.closest('[data-testid="stVerticalBlockBorderWrapper"]');
          const scroller = wrapper && [wrapper, ...wrapper.querySelectorAll("*")].find(el =>
            /auto|scroll/.test(getComputedStyle(el).overflowY) && el.clientHeight > 0
          );
          if (scroller) {
            let nearBottom = true;
            const scroll = () => { if (nearBottom) scroller.scrollTop = scroller.scrollHeight; };
            scroller.addEventListener("scroll", () => {
              nearBottom = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 180;
            });
            new MutationObserver(scroll).observe(scroller, {childList: true, subtree: true, characterData: true});
            requestAnimationFrame(() => { scroller.scrollTop = scroller.scrollHeight; });
          }
        } catch (_) {}
        </script>""",
        height=0,
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
        render_ai_visualization(
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
    activity_line=None,
    activity_signal=None,
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

    if avatar_container is not None:
        with avatar_container:
            if st.session_state.avatar_enabled:
                render_ai_visualization(
                    "", "", autoplay=True, resume_listener_on_end=True,
                    queue_id=queue_id, request_id=payload["request_id"], thinking=True,
                )
            elif queue_id:
                render_audio_queue_player(queue_id, payload["request_id"])

    try:
        publish_ai_activity(activity_line, activity_signal, "retrieving", "Retrieving")
        render_streaming_message(container, "", status)
        last_phase = "retrieving"
        for event in _qa_events(url, payload):
            flush_tts()
            event_type = event.get("type")
            if event_type == "status":
                status = event.get("message") or status
                phase = "retrieving" if "search" in status.lower() else "generating"
                if phase != last_phase:
                    publish_ai_activity(activity_line, activity_signal, phase, phase.capitalize())
                    last_phase = phase
                if not answer:
                    render_streaming_message(container, "", status)
            elif event_type == "token":
                token = event.get("text", "")
                if last_phase != "generating":
                    publish_ai_activity(activity_line, activity_signal, "generating", "Generating")
                    last_phase = "generating"
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
                publish_ai_activity(activity_line, activity_signal, "error", "Something went wrong")
                return "", [], event.get("message", "Streaming failed."), None, audio_mime, audio_error
            elif event_type == "cancelled":
                if tts_executor is not None:
                    tts_executor.shutdown(wait=False, cancel_futures=True)
                if queue_id and queue_sender is not None:
                    _send_avatar_queue_event(queue_sender, queue_id, queue_sequence, final=True)
                publish_ai_activity(activity_line, activity_signal, "interrupted", "Interrupted")
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

    if not queue_id:
        publish_ai_activity(activity_line, activity_signal, "idle", "Ready")
    else:
        activity_line.empty()
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

left_col, right_col = st.columns([1, 1], gap="small", vertical_alignment="top")
with left_col:
    avatar_stream_container = st.empty()

with right_col:
    with st.container(height=570, border=False):
        st.markdown('<div class="conversation-marker"></div>', unsafe_allow_html=True)
        render_chat_scroll_observer()
        if not st.session_state.messages:
            st.markdown('<div class="empty-state">Ask a question or start speaking.</div>', unsafe_allow_html=True)
        else:
            for message in st.session_state.messages:
                render_message(message)
        live_response_container = st.container()
    activity_line = st.empty()
    activity_signal = st.empty()
    activity_line.markdown('<div class="activity-line">Ready</div>', unsafe_allow_html=True)
    voice_component_query = render_voice_query_component()
    typed_question = st.chat_input("Ask anything...")
question = typed_question or (voice_component_query or {}).get("text") or voice_query_param

latest_audio_b64, latest_audio_text, latest_audio_autoplay, latest_audio_mime = latest_assistant_audio()
if not question and st.session_state.avatar_enabled:
    with avatar_stream_container:
        render_ai_visualization(
            latest_audio_b64 or "",
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
        answer, sources, error, audio_bytes, audio_mime, audio_error = ask_api_stream(
            question,
            stream_container,
            avatar_stream_container,
            activity_line=activity_line,
            activity_signal=activity_signal,
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
