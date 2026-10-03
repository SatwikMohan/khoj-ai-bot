# Voice stabilization implementation report

## Root causes

- The installed Piper English (en_US-lessac-medium) and Hindi (hi_IN-pratham-medium) models are different single-speaker voices. Neither yields one Indian-accent speaker across languages. The eSpeak fallback could change voice again. Roman Hinglish previously entered the Hindi phonemizer.
- The Hinglish prompt requested Devanagari even for Roman-script questions. TTS text preparation was separate from displayed text, but the speech chunker could split a long word or Devanagari cluster.
- Browser speech recognition was deliberately disabled to keep audio local, yet the recording loop had a high amplitude threshold and no no-speech or maximum-duration timeout. It handed base64 audio to a blocking Streamlit rerun. The async STT route ran synchronous inference on the event loop.
- Voice input did not take request ownership until after transcription. Old STT output could therefore become an active query. The browser could not reach the backend through the remote gateway.

## Changes

- Microphone activation begins an audio request in the same session registry as text QA. The ID and cancellation event persist through recording, STT, QA, and TTS. New requests cancel the old owner; stale uploads, results, and answer streams are discarded. Existing browser interruption signaling stops playback at activation.
- The browser records with MediaRecorder, detects speech and silence, supports manual stop and Escape cancellation, and has bounded no-speech, recording, and transcription timeouts. It uploads supported audio directly to the local API, displays the transcription, and submits it as a query. Permission, recording, and upload callbacks validate their generation.
- The gateway proxies /api/* to FastAPI. Local port 8000 access has restricted CORS origins. Streamlit uses backend HTTP so frontend and backend processes share the registry.
- STT decodes browser audio through PyAV to 16 kHz mono PCM, validates signal and duration before inference, serializes cached model loading, and runs inference outside the FastAPI event loop.
- Hinglish prompting requests Roman-script Hindi with standard English technical terms. English and Hindi prompts specify spelling and script. No blind spell checker or transliterator changes displayed text. Speech chunks wait for word boundaries.
- Roman Hinglish uses the Piper English phonemizer; Devanagari uses Hindi. The automatic eSpeak fallback is disabled. Piper WAV output must be 22.05 kHz, mono, 16-bit.

## TTS assessment and remaining limitation

The two installed Piper models each have one speaker at 22.05 kHz. Piper remains configured because it is local and warm synthesis is fast. It **does not meet the requested single Indian-accent speaker across languages**. The English voice is US English, and Hindi words in Roman Hinglish or English terms embedded in Devanagari may still sound unnatural. Four generated samples were checked as WAV data, but no human listening test was possible.

[Coqui XTTS-v2's model card](https://huggingface.co/coqui/XTTS-v2) lists English, Hindi, and cross-language voice cloning. It was not selected because its weights and an approved Indian speaker reference are absent, Hinglish quality has not been verified by listening here, and its [model license](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt) limits use. The [Piper voice catalog](https://github.com/rhasspy/piper/blob/master/VOICES.md) lists Hindi voices; no matching Indian English speaker is installed.

## Measured performance

These measurements are from this Windows workspace, not deployment or browser measurements.

| Operation | Result |
| --- | ---: |
| Piper English short clip, cold worker | 5,382 ms |
| Piper Hindi short clip, cold worker | 3,949 ms |
| Piper Roman Hinglish clip, warm English worker | 239 ms |
| Piper Hindi technical clip, warm Hindi worker | 340 ms |
| Local API voice replacement, 20 calls | 2.72 ms median, 3.64 ms p95 |
| STT transcription latency | Not measured: configured Whisper weights absent |
| End-to-end first-audio latency | Not measured: configured chat model absent |
| Browser audio playback latency | Not measured: no browser/microphone run |
| End-to-end cancellation latency | Not measured: local API replacement only |

Generated WAV clips were mono, 16-bit, 22.05 kHz. A generated English clip decoded through the STT input path to 16 kHz mono, 2.38 seconds.

## Validation

- Passed: 93 backend tests, including five new voice lifecycle tests for stale transcription, recording replacement, session isolation, MIME rejection, and word boundaries.
- Passed: Python compilation and git diff whitespace checks.
- Passed: Piper synthesis of English, Hindi, Roman Hinglish, and Hindi with technical terms as valid WAV.
- Not executed: live multilingual microphone transcription, browser permission/playback tests, human listening, and full configured RAG/LLM text quality. The configured openai/whisper-large-v3-turbo files are absent from the offline cache, and the configured llama3.1:8b-instruct-q4_K_M Ollama tag is not installed on this host.

## Files changed

Modified: Caddyfile; backend/config.py; backend/helpers/request_models.py; backend/main.py; backend/routes/qa_routes.py; backend/routes/stt_routes.py; backend/services/language_service.py; backend/services/qa_service.py; backend/services/request_lifecycle.py; backend/services/speech_chunker.py; backend/services/stt_service.py; backend/services/tts_service.py; frontend/app.py; frontend/voice_query_component/index.html.

Added: backend/tests/test_voice_lifecycle.py; VOICE_STABILIZATION_REPORT.md. Deleted: none.

