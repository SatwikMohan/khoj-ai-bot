"""Repeatable offline text-to-speech streaming benchmark.

Run against an installed local index, Ollama chat model and voice. The simulated
playback gap assumes immediate gapless browser scheduling; compare it with the
browser's playback_completed console event to isolate playback overhead.
"""
import argparse
import io
import json
import math
import statistics
import sys
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from helpers.request_models import TTSRequest
from services.language_service import response_language
from services.qa_service import stream_answer_events
from services.request_lifecycle import registry
from services.speech_chunker import IncrementalSpeechSegments
from services.tts_service import synthesize_speech, warm_up_tts


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(fraction * len(ordered)) - 1)], 1)


def run(question: str, *, top_k: int = 3, warmup: bool = True) -> dict:
    warmup_started = time.perf_counter()
    if warmup and config.TTS_PRELOAD_VOICES:
        warm_up_tts()
    tts_warmup_ms = round((time.perf_counter() - warmup_started) * 1000, 1)
    session = "voice-benchmark-" + uuid.uuid4().hex
    request_id = uuid.uuid4().hex
    started = time.perf_counter()
    chunker = IncrementalSpeechSegments()
    language = response_language(question)
    answer = ""
    first_text_ms = None
    llm_done_ms = None
    backend_timings = {}
    model_metrics = {}
    context_chars = None
    error = None
    completed = False
    speech_disabled = False
    submitted = 0
    next_delivery = 0
    futures = {}
    clips = []
    tts_errors = []

    def synthesize(sequence, segment, queued_at):
        begin = time.perf_counter()
        audio, mime = synthesize_speech(TTSRequest(
            text=segment, language=language, session_id=session,
            request_id=request_id, response_format="wav",
        ))
        ready = time.perf_counter()
        duration_s = None
        if mime == "audio/wav":
            with wave.open(io.BytesIO(audio), "rb") as clip:
                duration_s = clip.getnframes() / clip.getframerate()
        return {
            "sequence": sequence, "text_chars": len(segment),
            "queue_wait_ms": round((begin - queued_at) * 1000, 1),
            "synthesis_ms": round((ready - begin) * 1000, 1),
            "ready_ms": round((ready - started) * 1000, 1),
            "audio_duration_s": round(duration_s, 3) if duration_s else None,
            "rtf": round((ready - begin) / duration_s, 3) if duration_s else None,
            "bytes": len(audio), "mime": mime, "language": language,
        }

    with ThreadPoolExecutor(max_workers=int(config.TTS_STREAM_CONCURRENCY)) as executor:
        def drain(block=False):
            nonlocal next_delivery
            while next_delivery in futures:
                future = futures[next_delivery]
                if not block and not future.done():
                    break
                try:
                    clips.append(future.result(timeout=float(config.TTS_RESPONSE_TIMEOUT_SECONDS)))
                except Exception as exc:
                    tts_errors.append({"sequence": next_delivery, "error": str(exc)})
                del futures[next_delivery]
                next_delivery += 1

        def submit(segment):
            nonlocal submitted
            if speech_disabled or not segment.strip():
                return
            if len(futures) >= int(config.TTS_MAX_PENDING_JOBS):
                drain(block=True)
            sequence = submitted
            submitted += 1
            futures[sequence] = executor.submit(synthesize, sequence, segment, time.perf_counter())

        try:
            for event in stream_answer_events(question, top_k=top_k, session_id=session, request_id=request_id):
                for segment in chunker.flush_due():
                    submit(segment)
                drain()
                kind = event.get("type")
                if kind == "token":
                    if first_text_ms is None:
                        first_text_ms = round((time.perf_counter() - started) * 1000, 1)
                    answer += event["text"]
                    for segment in chunker.push(event["text"]):
                        submit(segment)
                elif kind == "replace":
                    answer = event["text"]
                    speech_disabled = True
                    for future in futures.values():
                        future.cancel()
                    futures.clear()
                elif kind == "done":
                    completed = True
                    answer = event.get("answer") or answer
                    llm_done_ms = round((time.perf_counter() - started) * 1000, 1)
                    backend_timings = event.get("timings_ms", {})
                    model_metrics = event.get("model_metrics", {})
                    context_chars = event.get("context_chars")
                elif kind in {"error", "cancelled"}:
                    error = event.get("message", kind)
                    break
            if completed and not speech_disabled:
                for segment in chunker.finish():
                    submit(segment)
            while futures:
                drain(block=True)
        finally:
            registry.cancel(session, request_id)

    playback_cursor = None
    simulated_gaps = []
    for clip in clips:
        if clip["audio_duration_s"] is None:
            continue
        ready = clip["ready_ms"]
        if playback_cursor is None:
            playback_cursor = ready
        else:
            simulated_gaps.append(max(0.0, ready - playback_cursor))
            playback_cursor = max(playback_cursor, ready)
        playback_cursor += clip["audio_duration_s"] * 1000
    eval_count = model_metrics.get("eval_count")
    eval_duration = model_metrics.get("eval_duration")
    return {
        "question": question, "model": config.OLLAMA_CHAT_MODEL,
        "embed_model": config.OLLAMA_EMBED_MODEL, "tts_engine": config.TTS_ENGINE,
        "num_ctx": config.OLLAMA_NUM_CTX, "num_predict": config.OLLAMA_NUM_PREDICT,
        "context_chars": context_chars, "tts_warmup_ms": tts_warmup_ms,
        "first_visible_text_ms": first_text_ms,
        "first_generated_audio_ms": clips[0]["ready_ms"] if clips else None,
        "llm_done_ms": llm_done_ms,
        "total_ms": round((time.perf_counter() - started) * 1000, 1),
        "model_metrics": model_metrics,
        "llm_tokens_per_second": round(eval_count * 1e9 / eval_duration, 2)
        if eval_count and eval_duration else None,
        "tts_rtf": round(sum(c["synthesis_ms"] for c in clips) /
                         (1000 * sum(c["audio_duration_s"] for c in clips)), 3)
        if clips and all(c["audio_duration_s"] for c in clips) else None,
        "simulated_audio_gap_median_ms": percentile(simulated_gaps, 0.5),
        "simulated_audio_gap_p95_ms": percentile(simulated_gaps, 0.95),
        "answer_chars": len(answer), "completed": completed,
        "audio_chunks_generated": len(clips), "audio_chunks_failed": len(tts_errors),
        "clips": clips, "backend_timings_ms": backend_timings,
        "tts_errors": tts_errors, "error": error,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default=(
        "Explain the main underground mining equipment and safety precautions "
        "in five detailed sentences."))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.question, top_k=args.top_k, warmup=not args.no_warmup)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
