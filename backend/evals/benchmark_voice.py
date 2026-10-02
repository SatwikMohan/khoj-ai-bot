"""Measure token-to-first-playable-chunk overlap using installed offline models."""

import argparse
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import config
from helpers.request_models import TTSRequest
from services.language_service import detect_language
from services.qa_service import stream_answer_events
from services.speech_chunker import IncrementalSpeechSegments
from services.tts_service import synthesize_speech, warm_up_tts


def run(question: str) -> dict:
    warmup_started = time.perf_counter()
    if config.TTS_PRELOAD_VOICES:
        warm_up_tts()
    tts_warmup_ms = round((time.perf_counter() - warmup_started) * 1000, 1)
    session = "voice-benchmark-" + uuid.uuid4().hex
    request_id = uuid.uuid4().hex
    started = time.perf_counter()
    chunker = IncrementalSpeechSegments()
    first_token_ms = None
    first_audio_ms = None
    llm_done_ms = None
    future = None
    first_segment_submitted_ms = None
    answer = ""
    backend_timings = {}
    error = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        for event in stream_answer_events(question, top_k=3, session_id=session, request_id=request_id):
            if event["type"] == "token":
                first_token_ms = first_token_ms or round((time.perf_counter() - started) * 1000, 1)
                answer += event["text"]
                for segment in chunker.push(event["text"]):
                    if future is None:
                        first_segment_submitted_ms = round((time.perf_counter() - started) * 1000, 1)
                        future = executor.submit(synthesize_speech, TTSRequest(
                            text=segment, language=detect_language(segment), session_id=session,
                            request_id=request_id, response_format="wav",
                        ))
            elif event["type"] == "done":
                llm_done_ms = round((time.perf_counter() - started) * 1000, 1)
                backend_timings = event.get("timings_ms", {})
            elif event["type"] == "error":
                error = event["message"]
            if future and future.done() and first_audio_ms is None:
                try:
                    audio, _mime = future.result()
                    if audio:
                        first_audio_ms = round((time.perf_counter() - started) * 1000, 1)
                except Exception as exc:
                    error = str(exc)
        if future and first_audio_ms is None:
            try:
                audio, _mime = future.result(timeout=config.TTS_RESPONSE_TIMEOUT_SECONDS)
                if audio:
                    first_audio_ms = round((time.perf_counter() - started) * 1000, 1)
            except Exception as exc:
                error = str(exc)
    return {
        "question": question,
        "model": config.OLLAMA_CHAT_MODEL,
        "tts_engine": config.TTS_ENGINE,
        "tts_warmup_ms": tts_warmup_ms,
        "first_token_ms": first_token_ms,
        "first_segment_submitted_ms": first_segment_submitted_ms,
        "first_playable_audio_ms": first_audio_ms,
        "llm_done_ms": llm_done_ms,
        "audio_ready_before_llm_done": bool(first_audio_ms is not None and llm_done_ms is not None and first_audio_ms < llm_done_ms),
        "answer_chars": len(answer),
        "backend_timings_ms": backend_timings,
        "error": error,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default="Explain the main underground mining equipment and safety precautions in five detailed sentences.")
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.question), ensure_ascii=True, indent=2))
