import base64
import io
import wave
import json
import sys
import threading
import uuid
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from html import escape

import markdown
import streamlit as st
import streamlit.components.v1 as components

# Import the same configuration used by the backend, also in local Streamlit runs.
BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
import config
from services.language_service import response_language
from services.speech_chunker import IncrementalSpeechSegments
from services.request_lifecycle import registry
from helpers.request_models import ChatMessage, TTSRequest
from services.qa_service import answer_question, stream_answer_events, warm_up_qa_engine
from services.stt_service import warm_up_stt
from services.tts_service import synthesize_speech, warm_up_tts
from services.voice_interaction import begin_voice_interaction, cancel_interaction, transcribe_interaction
from visualization import render_ai_visualization
from audio_manager import audio_manager_script
config.configure_runtime_environment()


@st.cache_resource(show_spinner=False)
def _start_model_warmup():
    def warm():
        for name, enabled, callback in (
            ("tts", config.TTS_ENABLED and config.TTS_PRELOAD_VOICES, warm_up_tts),
            ("stt", config.STT_WARMUP_ON_STARTUP, warm_up_stt),
            ("qa", config.QA_WARMUP_ON_STARTUP, warm_up_qa_engine),
        ):
            if not enabled:
                continue
            started = time.perf_counter()
            try:
                callback()
                print(json.dumps({"event": name + "_warmup_completed",
                                  "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}), flush=True)
            except Exception as exc:
                print(json.dumps({"event": name + "_warmup_failed",
                                  "error_type": type(exc).__name__}), flush=True)
    worker = threading.Thread(target=warm, name="texmin-model-warmup", daemon=True)
    worker.start()
    return worker


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
if (config.TTS_ENABLED and config.TTS_PRELOAD_VOICES) or config.STT_WARMUP_ON_STARTUP or config.QA_WARMUP_ON_STARTUP:
    _start_model_warmup()
st.markdown(f"<style>{(APP_DIR / 'theme.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_session_id" not in st.session_state:
        st.session_state.conversation_session_id = uuid.uuid4().hex
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
    if "processed_voice_events" not in st.session_state:
        st.session_state.processed_voice_events = []
    if "accepted_voice_request_id" not in st.session_state:
        st.session_state.accepted_voice_request_id = ""
    if "voice_error" not in st.session_state:
        st.session_state.voice_error = None


def markdown_to_html(content: str) -> str:
    safe_markdown = escape(content or "")
    return markdown.markdown(
        safe_markdown,
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )


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

    audio_error = message.get("audio_error")
    if role == "assistant" and audio_error:
        st.markdown(f'<div class="audio-note">{escape(audio_error)}</div>', unsafe_allow_html=True)

def render_voice_query_component() -> dict | None:
    value = voice_query_component(
        default=None,
        key="texmin_voice_query",
        height=120,
        # Leave room for recorder stop/event delivery before the decoder limit.
        max_recording_seconds=min(config.RECORDING_MAX_SECONDS, config.WHISPER_MAX_AUDIO_SECONDS - 1),
        no_speech_timeout_seconds=config.RECORDING_NO_SPEECH_TIMEOUT_SECONDS,
        silence_ms=config.RECORDING_SILENCE_MS,
        recording_buffer_ms=config.RECORDING_BUFFER_MS,
        speech_confirm_ms=config.RECORDING_SPEECH_CONFIRM_MS,
        voice_turn_watchdog_seconds=config.VOICE_TURN_WATCHDOG_SECONDS,
        session_id=st.session_state.conversation_session_id,
        accepted_request_id=st.session_state.accepted_voice_request_id,
        voice_error_request_id=(st.session_state.voice_error or {}).get("request_id", ""),
        voice_error=(st.session_state.voice_error or {}).get("message", ""),
    )
    if not isinstance(value, dict):
        return None
    kind = str(value.get("kind", ""))
    request_id = str(value.get("request_id", ""))
    event_key = f"{kind}:{request_id}"
    if (kind not in {"begin", "audio", "cancel"} or not request_id
            or len(request_id) > 128
            or event_key in st.session_state.processed_voice_events):
        return None
    st.session_state.processed_voice_events = (
        st.session_state.processed_voice_events + [event_key]
    )[-32:]
    session_id = st.session_state.conversation_session_id
    if kind == "begin":
        begin_voice_interaction(session_id, request_id)
        st.session_state.voice_error = None
        st.session_state.accepted_voice_request_id = request_id
        st.rerun()  # Deliver the acknowledgement before the browser records audio.
    if kind == "cancel":
        cancel_interaction(session_id, request_id)
        if st.session_state.accepted_voice_request_id == request_id:
            st.session_state.accepted_voice_request_id = ""
        return None
    if kind != "audio" or st.session_state.accepted_voice_request_id != request_id:
        return None
    worker_pool = None
    try:
        audio = base64.b64decode(str(value.get("audio_b64", "")), validate=True)
        worker_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="texmin-stt")
        pending = worker_pool.submit(
            transcribe_interaction, audio,
            str(value.get("audio_mime", "audio/webm")), session_id, request_id,
        )
        deadline = time.monotonic() + float(config.STT_REQUEST_TIMEOUT_SECONDS)
        progress = st.empty()
        while True:
            try:
                result = pending.result(timeout=0.1)
                break
            except TimeoutError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Speech recognition timed out.")
                progress.empty()  # Let Streamlit process a newer text or voice query.
        text = str(result.get("text", "")).strip()
        if text and registry.current(session_id, request_id) is not None:
            st.session_state.accepted_voice_request_id = ""
            st.session_state.voice_error = None
            return {
                "text": text, "request_id": request_id, "input_type": "audio",
                "language": result.get("language", ""),
            }
        raise ValueError("Speech recognition returned no usable transcript.")
    except Exception as exc:
        cancel_interaction(session_id, request_id)
        st.session_state.accepted_voice_request_id = ""
        st.session_state.voice_error = {"request_id": request_id, "message": str(exc)}
        st.rerun()  # Deliver the error to the component; no answer or audio is active.
    except BaseException:
        cancel_interaction(session_id, request_id)
        raise
    finally:
        if worker_pool is not None:
            worker_pool.shutdown(wait=False, cancel_futures=True)
    return None


def _qa_events(payload: dict):
    history = [ChatMessage(**item) for item in payload["chat_history"]]
    return stream_answer_events(
        question=payload["question"],
        top_k=payload["top_k"],
        temperature=payload["temperature"],
        chat_history=history,
        session_id=payload.get("session_id"),
        request_id=payload.get("request_id"),
        input_type=payload.get("input_type", "text"),
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
    try:
        history = [ChatMessage(**item) for item in _chat_history_payload(exclude_latest_user=True)]
        result = answer_question(
            question=question,
            top_k=st.session_state.top_k,
            temperature=st.session_state.temperature,
            chat_history=history,
            session_id=st.session_state.conversation_session_id,
        )
        data = result.model_dump() if hasattr(result, "model_dump") else result.dict()
        return data.get("answer", ""), data.get("sources", []), None
    except Exception as exc:
        return "", [], f"Question answering failed: {exc}"


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
        st.iframe(
            f'<script>localStorage.setItem("texmin_ai_activity", '
            f'JSON.stringify({{...{event}, at: Date.now()}}));</script>',
            height=1,
        )


def render_chat_scroll_observer() -> None:
    st.iframe(
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
        height=1,
    )


def _send_avatar_queue_event(
    sender_container,
    request_id: str,
    queue_id: str,
    sequence: int,
    audio_bytes: bytes | None = None,
    spoken_text: str = "",
    audio_mime: str = DEFAULT_AUDIO_MIME,
    final: bool = False,
    language: str = "",
    voice_id: str = "",
) -> None:
    event = {
        "type": "texmin:avatar-queue",
        "queueId": queue_id,
        "requestId": request_id,
        "sequence": sequence,
        "audio": base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else "",
        "skip": not audio_bytes and not final,
        "mime": audio_mime,
        "language": language,
        "voiceId": voice_id,
        "text": spoken_text,
        "final": final,
        "finalSequence": sequence if final else None,
    }
    event_b64 = base64.b64encode(json.dumps(event, ensure_ascii=False).encode("utf-8")).decode("ascii")
    with sender_container:
        st.iframe(
            f"""
            <script>
                const queueEvent = JSON.parse(new TextDecoder().decode(
                    Uint8Array.from(atob("{event_b64}"), character => character.charCodeAt(0))
                ));
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
            height=1,
        )


def render_audio_queue_player(queue_id: str, request_id: str, session_id: str) -> None:
    manager_script = audio_manager_script(session_id, request_id, queue_id)
    st.iframe(
        f"""
        <audio id="voice" controls playsinline style="width:100%;height:42px"></audio>
        <div id="speechStatus" role="status" style="font:13px sans-serif;color:#b4ffc8"></div>
        {manager_script}
        <script>
          const startedEpoch = Date.now();
          document.addEventListener("texmin:audio-stopped", event => {{
            const reason = event.detail.reason;
            if (reason === "queue_limit") document.getElementById("speechStatus").textContent = "Speech stopped: playback queue full. Text answer remains available.";
            if (reason === "queue_stalled") document.getElementById("speechStatus").textContent = "Speech stopped: an audio chunk did not arrive.";
            if (reason === "playback_failed") document.getElementById("speechStatus").textContent = "Speech playback failed. Check browser audio settings.";
          }});
          function interrupt() {{
            const at = Number(localStorage.getItem("texmin_voice_interrupt_at") || 0);
            if (at && at >= startedEpoch) window.texminAudioManager.stop("interrupted");
          }}
          window.addEventListener("storage", interrupt);
          window.setInterval(interrupt, 250);
        </script>
        """,
        height=48,
    )


def ask_api_stream(
    question: str,
    container,
    avatar_container=None,
    activity_line=None,
    activity_signal=None,
    request_id: str | None = None,
    input_type: str = "text",
    speak: bool = True,
) -> tuple[str, list[dict], str | None, bytes | None, str, str | None]:
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
    first_visible_logged = False
    first_audio = None
    speech_segments = IncrementalSpeechSegments()
    first_audio_started_at = None
    response_started_at = time.perf_counter()
    audio_mime = DEFAULT_AUDIO_MIME
    audio_error = None
    done_received = False
    queue_sequence = 0
    queue_id = uuid.uuid4().hex if avatar_container is not None and st.session_state.tts_enabled and speak else None
    st.session_state.audio_queue_active = bool(queue_id)
    queue_sender = st.container() if queue_id else None
    tts_executor: ThreadPoolExecutor | None = None
    tts_futures: dict[int, tuple[Future, str, float]] = {}
    next_audio_sequence = 0
    tts_wait_placeholder = st.empty() if queue_id else None
    tts_config = _current_tts_config(question)
    tts_config["session_id"] = payload["session_id"]
    tts_config["request_id"] = payload["request_id"]
    spoken_words = 0
    last_audio_delivered_at = None
    def submit_tts(segment: str) -> None:
        nonlocal queue_sequence, tts_executor, spoken_words
        current_request = registry.current(payload["session_id"], payload["request_id"])
        if (not queue_id or not tts_config["enabled"] or not segment.strip()
                or current_request is None or current_request.speech_cancelled.is_set()):
            return
        available_words = max(0, int(tts_config["max_words"]) - spoken_words)
        if not available_words:
            return
        words = segment.split()
        segment = " ".join(words[:available_words])
        spoken_words += min(len(words), available_words)
        if len(tts_futures) >= int(config.TTS_MAX_PENDING_JOBS):
            flush_tts(block=True)
            current_request = registry.current(payload["session_id"], payload["request_id"])
            if (current_request is None or current_request.speech_cancelled.is_set()
                    or not tts_config["enabled"]):
                return
        if tts_executor is None:
            tts_executor = ThreadPoolExecutor(max_workers=int(config.TTS_STREAM_CONCURRENCY), thread_name_prefix="texmin-tts")
        sequence = queue_sequence
        queue_sequence += 1
        if sequence == 0:
            print(json.dumps({"event": "qa_first_speech_segment",
                              "request_id": payload["request_id"],
                              "elapsed_ms": round((time.perf_counter() - response_started_at) * 1000, 1),
                              "text_chars": len(segment)}), flush=True)
        submitted_at = time.monotonic()
        tts_futures[sequence] = (
            tts_executor.submit(_request_tts, segment, tts_config, sequence, submitted_at),
            segment, submitted_at,
        )
        print(json.dumps({"event": "tts_chunk_queued", "request_id": payload["request_id"],
                          "sequence": sequence, "queue_depth": len(tts_futures),
                          "text_chars": len(segment)}), flush=True)

    def finish_audio_queue_after_abort() -> None:
        if queue_id and queue_sender is not None:
            for skipped_sequence in range(next_audio_sequence, queue_sequence):
                _send_avatar_queue_event(queue_sender, payload["request_id"], queue_id, skipped_sequence)
            _send_avatar_queue_event(queue_sender, payload["request_id"], queue_id, queue_sequence, final=True)

    def flush_tts(block: bool = False) -> None:
        nonlocal next_audio_sequence, audio_error, audio_mime, first_audio, first_audio_started_at, last_audio_delivered_at
        if queue_sender is None:
            return
        if registry.current(payload["session_id"], payload["request_id"]) is None:
            for future, _, _ in tts_futures.values():
                future.cancel()
            tts_futures.clear()
            return
        while next_audio_sequence in tts_futures:
            future, segment, submitted_at = tts_futures[next_audio_sequence]
            if not block and not future.done():
                break
            try:
                deadline = submitted_at + float(config.TTS_RESPONSE_TIMEOUT_SECONDS)
                while True:
                    try:
                        segment_audio, segment_mime, segment_error = future.result(timeout=0.1)
                        break
                    except TimeoutError:
                        if time.monotonic() >= deadline:
                            raise
                        if tts_wait_placeholder is not None:
                            tts_wait_placeholder.empty()
            except TimeoutError:
                audio_error = "Speech generation timed out. Your text answer is available above."
                tts_config["enabled"] = False
                current_request = registry.current(payload["session_id"], payload["request_id"])
                if current_request is not None:
                    current_request.speech_cancelled.set()
                print(json.dumps({"event": "tts_chunk_timeout", "request_id": payload["request_id"],
                                  "sequence": next_audio_sequence}), flush=True)
                future.cancel()
                for skipped_sequence in range(next_audio_sequence, queue_sequence):
                    _send_avatar_queue_event(queue_sender, payload["request_id"], queue_id, skipped_sequence)
                tts_futures.clear()
                next_audio_sequence = queue_sequence
                break
            except Exception as exc:
                segment_audio, segment_mime, segment_error = None, DEFAULT_AUDIO_MIME, str(exc)
            del tts_futures[next_audio_sequence]
            if segment_error:
                audio_error = segment_error
                _send_avatar_queue_event(queue_sender, payload["request_id"], queue_id, next_audio_sequence)
            elif segment_audio:
                if first_audio is None:
                    first_audio = segment_audio
                    first_audio_started_at = round((time.perf_counter() - response_started_at) * 1000, 1)
                    print(json.dumps({"event": "voice_first_audio_ready", "request_id": payload["request_id"], "first_audio_ms": first_audio_started_at}), flush=True)
                audio_mime = segment_mime
                delivered_at = time.monotonic()
                print(json.dumps({"event": "audio_chunk_delivered", "request_id": payload["request_id"],
                                  "sequence": next_audio_sequence,
                                  "gap_ms": round((delivered_at - last_audio_delivered_at) * 1000, 1)
                                  if last_audio_delivered_at is not None else None,
                                  "queue_depth": len(tts_futures)}), flush=True)
                last_audio_delivered_at = delivered_at
                _send_avatar_queue_event(
                    queue_sender,
                    payload["request_id"],
                    queue_id,
                    next_audio_sequence,
                    segment_audio,
                    segment,
                    segment_mime,
                    language=tts_config["language"],
                    voice_id=tts_config["voice_id"],
                )
            next_audio_sequence += 1

    if avatar_container is not None:
        with avatar_container:
            if st.session_state.avatar_enabled:
                render_ai_visualization(
                    queue_id=queue_id,
                    request_id=payload["request_id"], session_id=payload["session_id"], thinking=True,
                )
            elif queue_id:
                render_audio_queue_player(queue_id, payload["request_id"], payload["session_id"])

    try:
        render_streaming_message(container, "", "Preparing your request")
        last_phase = ""
        with closing(_qa_events(payload)) as events:
            for event in events:
                for segment in speech_segments.flush_due():
                    submit_tts(segment)
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
                    if token and not first_visible_logged:
                        first_visible_logged = True
                        print(json.dumps({"event": "qa_first_visible_text",
                                          "request_id": payload["request_id"],
                                          "elapsed_ms": round((time.perf_counter() - response_started_at) * 1000, 1)}),
                              flush=True)
                    for segment in speech_segments.push(token):
                        submit_tts(segment)
                    render_streaming_message(container, answer, status)
                elif event_type == "replace":
                    answer = event.get("text", "")
                    speech_segments = IncrementalSpeechSegments()
                    if queue_id:
                        # A late language switch invalidates previously queued
                        # speech. Keep the corrected text and stop that queue.
                        tts_config["enabled"] = False
                        current_request = registry.current(payload["session_id"], payload["request_id"])
                        if current_request is not None:
                            current_request.speech_cancelled.set()
                        for future, _, _ in tts_futures.values():
                            future.cancel()
                        tts_futures.clear()
                        st.iframe(
                            '<script>localStorage.setItem("texmin_voice_interrupt_at", String(Date.now()));</script>',
                            height=1,
                        )
                        audio_error = "Speech stopped after answer language correction. You can ask again with speech enabled."
                    render_streaming_message(container, answer, status)
                elif event_type == "done":
                    done_received = True
                    answer = event.get("answer") or answer
                    sources = event.get("sources", [])
                    status = ""
                    render_streaming_message(container, answer, status)
                elif event_type == "error":
                    # Keep text and already delivered clips. Skip pending speech
                    # so the browser can finish its ordered queue without waiting
                    # forever for a cancelled sequence.
                    cancel_interaction(payload["session_id"], payload["request_id"])
                    for future, _, _ in tts_futures.values():
                        future.cancel()
                    tts_futures.clear()
                    if tts_executor is not None:
                        tts_executor.shutdown(wait=False, cancel_futures=True)
                    finish_audio_queue_after_abort()
                    publish_ai_activity(activity_line, activity_signal, "error", "Generation stopped early")
                    return answer, [], event.get("message", "Streaming failed."), None, audio_mime, audio_error
                elif event_type == "cancelled":
                    if tts_executor is not None:
                        tts_executor.shutdown(wait=False, cancel_futures=True)
                    finish_audio_queue_after_abort()
                    publish_ai_activity(activity_line, activity_signal, "interrupted", "Interrupted")
                    return "", [], "__cancelled__", None, audio_mime, None
    except Exception as exc:
        cancel_interaction(payload["session_id"], payload["request_id"])
        if tts_executor is not None:
            tts_executor.shutdown(wait=False, cancel_futures=True)
        finish_audio_queue_after_abort()
        return answer, [], f"Question answering failed: {exc}", None, audio_mime, audio_error
    except BaseException:
        cancel_interaction(payload["session_id"], payload["request_id"])
        if tts_executor is not None:
            tts_executor.shutdown(wait=False, cancel_futures=True)
        raise

    if not done_received or not answer.strip():
        if tts_executor is not None:
            tts_executor.shutdown(wait=False, cancel_futures=True)
        finish_audio_queue_after_abort()
        return (
            answer,
            [],
            "The answer stream ended without a complete response. Check the app and Ollama logs.",
            None,
            audio_mime,
            audio_error,
        )

    if queue_id and queue_sender is not None:
        for segment in speech_segments.finish():
            submit_tts(segment)
        try:
            flush_tts(block=True)
        except BaseException:
            cancel_interaction(payload["session_id"], payload["request_id"])
            if tts_executor is not None:
                tts_executor.shutdown(wait=False, cancel_futures=True)
            raise
        if tts_executor is not None:
            tts_executor.shutdown(wait=False, cancel_futures=True)
        _send_avatar_queue_event(
            queue_sender,
            payload["request_id"],
            queue_id,
            next_audio_sequence,
            final=True,
        )
        audio_bytes = None
    else:
        audio_bytes = None

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
        "language": response_language(question),
    }


def _request_tts(text: str, tts_options: dict, sequence: int | None = None, submitted_at: float | None = None) -> tuple[bytes | None, str, str | None]:
    if not tts_options["enabled"]:
        return None, DEFAULT_AUDIO_MIME, None

    voice_id = tts_options["voice_id"]
    if not voice_id:
        return None, DEFAULT_AUDIO_MIME, "Voice is off: add a voice name in the sidebar to hear replies."

    payload = {
        "text": text,
        "session_id": tts_options.get("session_id"),
        "request_id": tts_options.get("request_id"),
        "voice_id": voice_id,
        "language": tts_options["language"],
        "tone": tts_options["tone"],
        "rate": tts_options["rate"],
        "pitch": tts_options["pitch"],
        "volume": "+0%",
        "response_format": "mp3" if DEFAULT_TTS_ENGINE in {"edge", "edge-tts"} else "wav",
        "max_words": tts_options["max_words"],
    }

    try:
        started = time.perf_counter()
        queue_wait_ms = round((time.monotonic() - submitted_at) * 1000, 1) if submitted_at is not None else None
        print(json.dumps({"event": "tts_chunk_started", "request_id": payload["request_id"],
                          "sequence": sequence, "language": payload["language"],
                          "voice_id": voice_id, "queue_wait_ms": queue_wait_ms}), flush=True)
        request = TTSRequest(**payload)
        audio, audio_mime = synthesize_speech(request)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        audio_duration_s = None
        if audio_mime == "audio/wav":
            try:
                with wave.open(io.BytesIO(audio), "rb") as clip:
                    audio_duration_s = round(clip.getnframes() / clip.getframerate(), 3)
            except (wave.Error, EOFError, ZeroDivisionError):
                pass
        print(json.dumps({"event": "tts_chunk_completed", "request_id": payload["request_id"],
                          "sequence": sequence, "bytes": len(audio),
                          "elapsed_ms": elapsed_ms, "audio_duration_s": audio_duration_s,
                          "rtf": round(elapsed_ms / 1000 / audio_duration_s, 3)
                          if audio_duration_s else None}), flush=True)
        return audio, audio_mime, None
    except Exception as exc:
        print(json.dumps({"event": "tts_chunk_failed", "request_id": payload["request_id"],
                          "sequence": sequence, "error": str(exc)}), flush=True)
        return None, DEFAULT_AUDIO_MIME, f"Voice is unavailable: {exc}"


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
    if st.session_state.voice_error:
        st.error("Microphone transcription failed: " + st.session_state.voice_error["message"]
                 + " Check the device, then start the conversation again.")
    st.session_state.tts_enabled = st.toggle(
        "Speak answers", value=st.session_state.tts_enabled, key="speech_toggle"
    )
    if st.button("Stop speech", key="stop_speech"):
        st.iframe(
            '<script>localStorage.setItem("texmin_voice_interrupt_at", String(Date.now()));</script>',
            height=1,
        )
    typed_question = st.chat_input("Ask anything...")
    retry_question = None
    if st.session_state.messages and st.session_state.messages[-1].get("incomplete"):
        if st.button("Retry incomplete answer (text only)", key="retry_incomplete_answer"):
            retry_question = next(
                (str(item.get("content", "")) for item in reversed(st.session_state.messages[:-1])
                 if item.get("role") == "user"), None
            )
question = typed_question or (voice_component_query or {}).get("text") or voice_query_param or retry_question

if not question and st.session_state.avatar_enabled:
    with avatar_stream_container:
        render_ai_visualization()

if question:
    if st.session_state.messages and st.session_state.messages[-1].get("role") == "user":
        # A rerun interrupted its previous answer; omit that unfinished turn.
        st.session_state.messages.pop()
    if typed_question:
        st.iframe(
            '<script>window.localStorage.setItem("texmin_voice_interrupt_at", String(Date.now()));</script>',
            height=1,
        )
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
            speak=not bool(retry_question),
        )

    if voice_component_query and not typed_question:
        # Signal completion without a rerun, which would replace the playing avatar.
        turn_state = json.dumps({
            "requestId": voice_component_query["request_id"],
            "state": "finished" if error != "__cancelled__" else "cancelled",
            "awaitAudio": bool(st.session_state.audio_queue_active),
            "at": int(time.time() * 1000),
        }).replace("</", "<\\/")
        turn_key = json.dumps("texmin_voice_turn:" + st.session_state.conversation_session_id)
        st.iframe(
            f"<script>localStorage.setItem({turn_key}, JSON.stringify({turn_state}));</script>",
            height=1,
        )

    if error == "__cancelled__":
        if st.session_state.messages and st.session_state.messages[-1].get("role") == "user":
            st.session_state.messages.pop()
        stream_container.empty()
    elif error:
        if answer.strip():
            render_streaming_message(stream_container, answer, "Incomplete: " + error)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": [],
                 "error": True, "incomplete": True, "audio_error": audio_error}
            )
            st.error(error)
        else:
            stream_container.error(error)
            st.session_state.messages.append(
                {"role": "assistant", "content": error, "sources": [], "error": True}
            )
    else:
        if audio_error:
            st.warning(audio_error)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "audio_error": audio_error,
            }
        )
    if voice_query_param and "voice_query" in st.query_params:
        del st.query_params["voice_query"]
