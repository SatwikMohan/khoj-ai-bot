# Streaming stabilization report — 2026-10-11

## Outcome and validation boundary

The application no longer applies one 180-second wall clock to retrieval, model generation, and a slow stream consumer. Text and audio still stream concurrently. Failures after partial delivery retain the text, label it incomplete, and offer a text-only retry. Speech stays ordered through one browser playback owner. Voice mode now detects an utterance, submits its finalized offline transcript, and resumes listening after playback without per-turn controls.

The DGX Spark model and browser playback latency have **not** been measured here. This Windows machine does not have the configured `llama3.1:8b-instruct-q4_K_M` or `bge-m3` models. Its installed `llama3.2:latest` returned HTTP 500 while allocating a 1,362,493,440-byte CPU buffer. The measured Piper results below are development-machine synthesis measurements, not a full answer or DGX result.

## Root cause matrix

| Finding | Evidence | Change |
| --- | --- | --- |
| The 180-second error was application generated. The old `bounded_events` clock began before the QA worker and never reset. It covered preprocessing, retrieval, model setup, generation, and time spent waiting for the frontend to pull events. | Previous `backend/services/response_stream.py` used `started = time.monotonic()`, subtracted that from `timeout_seconds` on every consumer iteration, and built the quoted error. `backend/services/qa_service.py` supplied `QA_RESPONSE_TIMEOUT_SECONDS = 180`. | Producer-stage clocks now distinguish worker admission, retrieval, model client setup, first model output, inter-output stall, and total generation. Queue backpressure is excluded from the model clock. The total generation deadline is finite (600 s), and a progressing stream is not declared stalled at 45 s. |
| “Preparing an answer from the retrieved passages” was only the last status label. It remained visible while model initialization and generation ran. It did not identify the failing stage. | `_stream_answer_events` emitted that status before building the prompt and calling `chain.stream`; `bounded_events` repeated it in the timeout message. | Timeout events now carry a category and accurate stage; logs record request ID and elapsed time. |
| The Streamlit process paid a cold voice-model load on its first chunk. The configured preload previously ran only in the separate API startup, while Streamlit imports the backend service directly. | `backend/main.py` calls `warm_up_tts`, but the old `frontend/app.py` did not. Same-text Piper measurements: cold 4.704 s, warm 0.324 s. | Streamlit starts one cached background voice warmup. An immediately submitted question can still overlap warmup; the voice worker lock prevents duplicate model loading. |
| Speech segment release could wait for a very long sentence or the model's final token. | The citation filter buffered up to 400 characters without punctuation; the speech chunker waited for punctuation or 320 characters. | The filter's bounded buffer is 200 characters with an 80-character suffix guard. The speech chunker releases a word-boundary phrase after 1.5 s once it has at least 60 characters. Hindi graphemes remain intact because the timed cut uses whitespace. |
| Two TTS executor threads did not imply two parallel syntheses for one voice. | `PersistentPiperWorker.synthesize` serializes access with a lock; Veena uses the same worker pattern. | Default synthesis concurrency is one, with at most four pending jobs. The browser still plays clips sequentially and synthesis can run while earlier clips play. |
| Ordinary voice turns required Record, Stop, transcript review, and Send. The browser meter only displayed volume; Streamlit stored the transcript for a manual button. | Previous `frontend/voice_query_component/index.html` had no silence-triggered stop; previous `frontend/app.py` held `pending_voice`. | The existing meter confirms speech, applies the configured 1100 ms silence endpoint, and sends one recording to offline STT. A valid transcript goes directly into the existing QA stream. One Start/End control owns the session. |
| Actual audible gap on DGX is unresolved. | Local warm Piper chunks took 0.410 s / 4.795 s audio (RTF 0.085) and 0.529 s / 4.923 s audio (RTF 0.107). That does not explain a long warm gap by itself. The configured LLM could not run locally, and the prior browser had no gap metric. | Request logs now separate model, queue wait, synthesis, delivery, and browser playback times. The browser logs median and p95 inter-clip gaps on completion. |

Ollama's HTTP client timeout remains 120 seconds per blocking operation. The old 180-second message was not produced by Ollama. A cancelled synchronous `chain.stream` read can occupy one bounded QA worker slot until the underlying Ollama read returns or times out; its output is invalidated and cannot reach a newer request.

## Architecture and configuration

- One backend QA worker per active request, at most four worker slots globally, and a 64-event bounded bridge.
- Retrieval: 120 s. Model client setup: 30 s. First model output including Ollama load and prefill: 120 s. Generation inactivity: 45 s. Total generation: 600 s. Worker admission: 30 s.
- One TTS executor worker per Streamlit request, at most four pending jobs. The existing persistent voice model is reused. A timed-out speech request sets a speech-only cancellation event, leaving text generation intact. The producer rechecks cancellation after waiting for a full queue, so a timed-out clip cannot trigger another synthesis job.
- One browser audio owner per session, ordered sequence numbers, a 30-second watchdog for missing clips after the final marker, and a 900-second voice-turn recovery watchdog for a missing completion signal.
- The response and audio completion states remain separate. A partial failed response is not marked `answer_ready`; pending speech sequences are skipped explicitly so playback can finish.
- No chat model, embedding model, quantization, retrieval depth, answer-token limit, response language rule, or voice model was changed.

Structured logs contain IDs, counts, durations and categories. Raw prompts and audio remain absent by default. Existing `QA_DIAGNOSTICS_ENABLED` still controls detailed prompt logging.

## Model lifecycle evidence

The chat wrapper is created for a request, but Ollama owns the chat weights. The request passes the configured 1800-second keep-alive; constructing the wrapper is not evidence of a model reload. The local Ollama CLI is version 0.40.0, and the installed ChatOllama accepts keep_alive as int, str, or None. The Windows ollama list shows only llama3.2:latest and nomic-embed-text:latest; ollama ps showed no resident model. The configured DGX chat and embedding tags are absent here, so chat model residency across real turns remains unverified.

A fresh Windows session-aware Piper request took 3.629 s for its first chunk and 0.303/0.314 s for the next two chunks; this confirms reuse within the UI request path. An isolated TTS call without session and request IDs intentionally uses a new subprocess for each call and measured 5.014/3.967/3.519 s. That isolated path is not the Streamlit speech path.

STT uses a locked cached model loader. Piper and Veena use the existing persistent worker map; Streamlit now starts their configured warmup once per process. The tts_worker_started event records every persistent worker process launch so repeated loads can be counted. In a fresh two-turn Windows check, one Piper worker-start event was emitted and both turns used PID 17792; that verifies process reuse across those two turns.

## Hands-free voice lifecycle

One initial Start conversation click grants the browser microphone permission. The existing microphone meter samples at 100 ms, confirms speech using `RECORDING_SPEECH_CONFIRM_MS`, and finalizes a recording after `RECORDING_SILENCE_MS` of quiet or the configured maximum speech duration. Idle recorder chunks retain only a short pre-roll, so waiting for the user does not grow memory indefinitely. The microphone track closes before offline STT starts, avoiding self-transcription of the answer.

The Streamlit component bridge accepts one `begin` and one `audio` event per request ID. A bounded history of processed event keys rejects repeated or delayed callbacks. STT returns a finalized nonempty transcript, which immediately enters the existing direct Python QA path under the same request ID; there is no transcript review form. The browser holds the voice session in local storage across ordinary Streamlit reruns. A bounded watchdog stops a turn that never receives its completion signal and shows a restart action. The server signals answer completion through a same-origin iframe without rerunning the page. The audio owner marks playback completion in the same session, after which the component starts the next microphone capture. This keeps the avatar and player alive until the last clip ends.

The mode is intentionally half duplex. Listening pauses during STT, answer generation, and playback. End conversation, typed input, or a newer query invalidates the old microphone and audio work. A microphone or STT failure stops voice mode and displays a recovery message. The browser smoke covers three synthetic utterances, natural pauses, automatic submission, a text-only answer, no listening before playback completion, an external typed-query interruption without a duplicate cancel event, and recovery from a stalled turn; it does not validate real microphone acoustics or DGX model latency.

## Measurements

| Check | Before | After / observed | Scope |
| --- | ---: | ---: | --- |
| Backend tests | 125 passed | 138 passed | Deterministic Windows tests |
| Frontend tests | 9 passed | 11 passed | Streamlit tests, including automatic voice submission and STT failure recovery |
| Piper same-text first chunk | 4.704 s, RTF 1.141 | 0.324 s, RTF 0.086 after model load | Local cold versus warm synthesis proxy for the new Streamlit preload |
| Piper subsequent chunks | Not recorded | 0.410 s (RTF 0.085), 0.529 s (RTF 0.107) | Three-chunk local run; voice was warm after chunk 0 |
| Piper first UI chunk, same text | 3.629 s cold, RTF 0.802 | 0.323 s after 7.410 s startup warmup, RTF 0.071 | Separate fresh Windows processes; session-aware service path |
| Piper subsequent UI chunks | Not recorded cold | 0.303 s and 0.314 s warm, RTF 0.069 and 0.070 | Same persistent worker and text; 4.4-4.5 s audio clips |
| Isolated TTS calls without request IDs | 5.014 s first call | 3.967 s and 3.519 s later calls | Separate per-call subprocess; not the UI speech path |
| Configured LLM first token / tokens per second | Unavailable | Unavailable | Configured DGX models absent on Windows |
| First generated audio, browser gap median / p95 | Unavailable | Unavailable | Requires configured model and real browser playback |
| Browser queue behavior | Existing test | Headless Edge smoke: `data-pass="true"` | Synthetic clips; confirms order, overflow, failure, and completion signaling |
| Browser voice conversation | Manual workflow | Headless Edge smoke: 3 synthetic turns (one text-only), 4 listening starts, no early submission/resumption or duplicate cancel | Synthetic microphone and server bridge; browser state machine only |

The additional session-aware and isolated TTS checks used the same new phrase in separate fresh processes. They demonstrate lifecycle behavior, not an end-to-end latency comparison. The 4.704 s versus 0.324 s pair uses the same phrase and process. It measures the effect of a loaded Piper worker; it is not a before/after end-to-end request benchmark. Audio durations differed between runs (4.122 s versus 3.750 s). No GPU acceleration claim is made.

## Files changed

| File | Change |
| --- | --- |
| backend/config.py | Stage deadlines, bounded worker settings, and speech buffering defaults. |
| backend/evals/README.md | Benchmark instructions and scope. |
| backend/evals/benchmark_voice.py | Per-chunk synthesis, queue, RTF, and simulated gap measurements. |
| backend/services/qa_service.py | Stage markers, model timing metadata, and failure categories. |
| backend/services/request_lifecycle.py | Separate speech cancellation token. |
| backend/services/response_stream.py | Producer-stage deadlines, cancellation, and structured timeout events. |
| backend/services/speech_chunker.py | Timed word-boundary phrase release. |
| backend/services/tts_service.py | Speech cancellation checks and worker-start diagnostics. |
| backend/tests/test_stream_stage_policy.py | Deterministic timeout and chunking regressions. |
| frontend/README.md | One-click voice operation and half-duplex limit. |
| frontend/app.py | Cached voice warmup, automatic transcript submission, bounded TTS delivery, and turn completion signaling. |
| frontend/assets/audio_manager.js | Ordered queue watchdog, playback gap logging, and completion state for listening resumption. |
| frontend/audio_manager.py | Audio queue deadline passed to the browser. |
| frontend/tests/audio_queue_smoke.html | Ordered playback and completion-state checks. |
| frontend/tests/test_interface.py | Direct voice submission, STT failure, warmup, partial answer, and speech timeout tests. |
| frontend/tests/voice_browser_smoke.py | Three synthetic hands-free turns, natural pauses, typed interruption, and stall recovery. |
| frontend/visualization.py | Audio queue stall status. |
| frontend/voice_query_component/index.html | Speech confirmation, silence endpoint, automatic capture, session state, and resumption. |
| STREAMING_STABILIZATION_REPORT.md | Diagnosis, measurements, limits, and DGX validation procedure. |

## Reproduction and DGX Spark validation

1. On DGX, verify the configured tags and actual placement: `docker compose exec ollama ollama list`, `docker compose exec ollama ollama ps`, and `docker compose exec app nvidia-smi`.
2. Run answer quality against the promoted index: `docker compose exec -w /app app python backend/evals/evaluate_answer_quality.py --output /tmp/dgx_answer_quality.json`. Review English, Hindi, Hinglish, grounding, and version-comparison cases before changing any model.
3. Run model-only cold and warm comparisons with the configured model: `docker compose exec -w /app app python backend/evals/benchmark_chat_runtime.py llama3.1:8b-instruct-q4_K_M --repeats 3 --max-tokens 160`. Record context size, prompt/eval token counts and rates, first token, and total time.
4. Run the full direct-service voice path at least three times with the same representative knowledge-base question: `docker compose exec -w /app app python backend/evals/benchmark_voice.py "What is a bucket wheel excavator, and where is it used?" --top-k 3`. The report includes first visible text, first generated audio, each queue wait and TTS RTF, and median/p95 **simulated** playback starvation gaps. It does not include browser startup overhead.
5. In the actual Streamlit browser, ask the same questions with speech enabled. Match request IDs in the structured server logs and browser console. Use `playback_completed.gap_median_ms` and `gap_p95_ms` for audible inter-clip scheduling, plus `qa_first_visible_text`, `qa_first_speech_segment`, `voice_first_audio_ready`, and `tts_chunk_completed` for stage attribution. Test a new query during generation and during playback; verify stale audio stops.
6. Compare these outputs with the same procedure on the prior deployed revision, with identical Ollama weights, index, context size, output limit, warmup state, voice, and question set. This repository's Windows host cannot supply that DGX baseline.

## Remaining risks

The synchronous Ollama iterator cannot be forcibly interrupted inside a blocked socket read. Worker slots bound accumulation, and cancellation invalidates late output, but a slot may remain occupied for up to the configured Ollama read timeout. Browser playback gap measurements and Veena's actual RTF on DGX are still required before declaring the audio-gap objective met. Background warmup may not finish before an immediate first question. A blocked noninterruptible STT model call can also finish after the UI timeout; its cancelled token rejects the late result, and the STT semaphore limits active inference to one.

## Final local verification

- Backend unit suite: 138 tests passed.
- Frontend Streamlit suite: 11 tests passed, including direct automatic voice submission, STT failure recovery, and speech timeout recovery.
- Headless Edge audio queue smoke: data-pass=true; PASS.
- Headless Edge voice conversation smoke: two scenarios passed, with three synthetic turns, automatic fourth listening start, typed interruption, and stalled-turn recovery.
- git diff --check: passed.