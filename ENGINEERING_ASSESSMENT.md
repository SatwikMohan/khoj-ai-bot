# ASTRA engineering assessment — 2026-10-11

## Architecture and baseline

The working tree was clean. No AGENTS.md was found. This checkout runs on
Windows AMD64, not the target DGX; Spark hardware and memory cannot be measured here.
Streamlit (`frontend/app.py`) calls shared Python services directly. FastAPI
(`backend/main.py`, `backend/routes`) is an optional second entry point.
Ollama owns chat and embedding weights (Llama 3.1 / BGE-M3 configured).
Whisper uses a locked, cached local model. Piper/Veena use persistent subprocesses.
Browser MediaRecorder and an adaptive energy detector implement half-duplex
automatic turn-taking. Streaming answers feed bounded sentence synthesis and an
ordered browser audio queue. Stop/cancellation is supported; acoustic barge-in is not.
Chroma plus SQLite lexical search supports hybrid retrieval, version metadata,
candidate index promotion, and incremental ingestion (`train_engine.py`).
Recent conversation context is bounded, but history lives only in Streamlit state.

The existing frontend baseline passes 11 tests. System Python cannot run the
backend suite reliably (missing PyAV and multipart); use the project virtualenv.
Existing implementation and tests are evidence of functionality, not proof of
underground speech quality, complete offline operation, or target performance.

## Confirmed defects and first implementation phase

1. `frontend/voice_query_component/index.html` drops initial MediaRecorder chunks
   during silence. This can discard container headers. Retain complete recordings,
   restart silent capture on a bounded interval, and bound total capture duration.
   Test delayed speech, complete header delivery, short pauses, and turn resumption.
2. `frontend/app.py` only warms TTS; configured STT/QA warm-up is never called by
   the deployed entry point. Run enabled warm-ups once per process in a background
   worker, with independent failure handling. Test reruns and failures using mocks.
3. `backend/services/qa_service.py` discards ordinary answer source metadata;
   `frontend/app.py` never displays sources. Return actual selected evidence and
   render a collapsed source panel, preserving natural spoken text. Test page
   numbering, unknown pages, and streamed answer source retention.

## Model loading and performance findings

No evidence yet establishes repeated chat weight loading on every request.
Ollama receives a 1,800-second keep-alive; Compose permits two resident models.
Chat client construction occurs per request, but is distinct from loading weights.
Speech workers deliberately restart after cancellation, timeout, or worker failure.
STT already guards concurrent initialization. Do not remove those safeguards.
Existing telemetry measures retrieval, generation, first output, TTS and playback.
Add Ollama load-duration telemetry to distinguish loading from token generation;
collect cold/warm measurements on Spark before changing residency policy.

## Remaining phases, tests, and risks

- Persistence: add a local conversation store and explicit resume controls, with
  per-browser ownership and atomic writes. Test restart recovery and session
  isolation before exposing history; a shared global conversation picker is unsafe.
- Retrieval: retain the active index. Evaluate representative approved documents
  with `evaluate_retrieval.py`; inspect authority/conflicting-source metadata.
  Retrieved sources indicate evidence supplied to the model, not verified
  sentence-by-sentence entailment or approval status.
- Speech: test real microphone recordings, noise, pronunciation, prolonged pauses,
  repeated interruptions and recovery. Energy detection cannot reliably classify
  machinery as non-speech. Explicit retry remains preferable to unlimited retries.
- Offline: local assets and local model flags are present; provisioning is separate.
  Check Chroma/Streamlit telemetry, optional engines, container image availability,
  and browser traffic under external-network denial. Flags alone are insufficient.
  Compose runs incremental ingestion during startup; measure its cost before
  changing deployment ordering. Do not rebuild or migrate the existing index.
- DGX: inspect architecture, installed CUDA/PyTorch, model availability, active
  embedding fingerprint and combined memory. Benchmark cold and repeated warm
  requests and record actual network activity. No target certification is claimed.

Regression risks: total-duration limits can end unusually long utterances;
warm-up uses memory earlier; source panels must escape document content and avoid
absolute paths. Keep UI framework, APIs, models, and index format intact.

## First-phase verification

- Baseline: 139 backend tests and 11 frontend tests passed using the project
  virtualenv. The system-Python run failed because its dependencies differ.
- After changes: 139 backend tests and 13 frontend tests passed. Added frontend
  coverage checks all configured warm-ups, failure isolation, single invocation
  across reruns, and source visibility/page handling immediately and on rerun.
- Three Edge browser scenarios passed: ordinary turn-taking/typed interruption,
  stalled-turn recovery, and repeated silent capture restart. The recording fake
  checks retention of the first container chunk; it does not establish real codec
  compatibility or microphone quality.
- A parallel regression attempt exhausted memory while loading test assets and
  the language classifier. Subsequent runs passed; this is not a DGX benchmark.
- Ollama load duration is now included in completion telemetry when supplied by
  the runtime. No cold/warm inference benchmark or external-network-denial test
  was performed. No models, system packages, or indexes were modified.
- The frontend README no longer recommends a routine destructive force rebuild.

## User-directed presentation update

The user requested answer-only conversation without document/page panels.
Removed automatic source rendering for both streamed and historical answers.
Retrieval and internal source metadata remain intact; explicit source-location
questions retain the existing supported-location response behavior.
The frontend regression now verifies that normal answers hide source metadata.
This supersedes the source-panel presentation described above.
