# Model configuration and offline operation

`backend/.env` is loaded by both local Streamlit and the combined application.
Restart the application after editing it. Docker injects this file when a
container is **created**; `docker compose restart` does not refresh those values.
Use `docker compose up -d --no-deps --force-recreate app` after configuration edits.
Copy your updated source and `.env` to the DGX before rebuilding there.

The supported chat/embedding provider is Ollama. Select any **Ollama chat model**
for `OLLAMA_CHAT_MODEL` and an **Ollama embedding model** for `OLLAMA_EMBED_MODEL`.
These roles are different; arbitrary Hugging Face repository names are not Ollama
model names. The optional reranker uses Qwen3's yes/no scoring protocol and is
configured separately. Set `RERANK_ENABLED=false` for a lightweight local setup.

| Setting | Local example | DGX example |
| --- | --- | --- |
| `OLLAMA_CHAT_MODEL` | `qwen3:4b` | `qwen3.5:35b` |
| `OLLAMA_EMBED_MODEL` | `qwen3-embedding:0.6b` | `qwen3-embedding:8b-q8_0` |
| `OLLAMA_NUM_CTX` | `4096` | `8192` |
| `CONTEXT_MAX_CHARS` | `6000` | `16000` |
| `EMBEDDING_BATCH_SIZE` | `32` | `256` |
| `STT_ENGINE` | `faster-whisper` | `transformers` |
| `WHISPER_MODEL` | `small` | unused by transformers |
| `WHISPER_TRANSFORMERS_MODEL` | unused by faster-whisper | `openai/whisper-large-v3-turbo` |
| `WHISPER_DEVICE` | `cpu` | `auto` |
| `WHISPER_COMPUTE_TYPE` | `int8` | `float16` |

These are example profiles, not performance guarantees. Actual speed and language
quality depend on the model and available RAM/GPU memory. A chat-model change
does not require indexing. An embedding-model or embedding-prefix change does:
run `python train_engine.py` from `backend`. This builds and promotes a compatible
collection while retaining the previous collection. Do not delete the vector DB.

`OLLAMA_THINK=auto` discovers model capabilities and disables optional reasoning
unless `QA_REASONING_ENABLED=true`. Models with mandatory reasoning may require
an explicit supported effort level in `OLLAMA_THINK` and a larger
`OLLAMA_NUM_PREDICT`. `OLLAMA_THINK=default` leaves the option to the model.
`OLLAMA_REQUEST_TIMEOUT_SECONDS` bounds network inactivity;
`QA_RESPONSE_TIMEOUT_SECONDS` bounds the whole answer, including retrieval.
A timed-out native model operation may still be finishing in the background;
new requests receive a busy error until it releases its slot, preventing a buildup
of abandoned inference workers.

## Offline speech

The default is Piper CPU speech, using local `.onnx` and `.onnx.json` files:

```dotenv
TTS_ENGINE=piper
PIPER_HINDI_VOICE=hi_IN-pratham-medium
PIPER_ENGLISH_VOICE=en_US-lessac-medium
TTS_FALLBACK_ENGINES=espeak
TTS_TIMEOUT_SECONDS=45
RESPONSE_LANGUAGE=hinglish
HINGLISH_SCRIPT=mixed
```

Hindi words should be written in Devanagari. English terms may remain in Latin
script; code-switching pronunciation is model-dependent and should be listened
to on representative answers. The Hindi voice is a preset speaker, not a voice
clone. The espeak-ng fallback is less natural but works offline when installed.
Set `TTS_FALLBACK_ENGINES=` to disable that fallback. Selected engines appear in
the `tts_completed` log event. Reference recordings, transcripts and gated HF
access are no longer used by the default speech stack.

Locally, voices default to `backend/models/piper`. Compose sets
`PIPER_MODEL_DIR=/models/piper` in the persistent `ai_models` volume.
Set `PIPER_MODEL_DIR` explicitly only if you use a different local directory.
Voice values may also be absolute paths to compatible Piper ONNX models.
Both Hindi and English models are provisioned and checked.

Provision once with internet access, or copy the model files from a connected
machine. Runtime synthesis never downloads anything:

```bash
cd backend
python scripts/provision_models.py --only-tts
python scripts/provision_models.py --only-tts --offline
```

The offline command actually synthesizes Hindi and English WAVs and checks their
audio content. Full provisioning (without `--only-tts`) also downloads the selected
Whisper and optional reranker assets. Piper worker processes are terminated if
they exceed the speech timeout. An audio failure leaves the text answer visible.

Piper API: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md

Hindi voice: https://huggingface.co/rhasspy/piper-voices/tree/main/hi/hi_IN/pratham/medium

## DGX deployment

The DGX image uses `nvcr.io/nvidia/pytorch:26.08-py3`. If registry authentication is
required, run `docker login nvcr.io` using the literal username `$oauthtoken` and
your NGC API key as the password. Place source documents in `backend/raw_data_files`.

After transferring the updated files and editing `backend/.env` on the DGX:

```bash
docker compose build app backend-model-init index-init
docker compose up -d --force-recreate
docker compose logs --tail=100 -f backend-model-init index-init app ollama
```

Initial provisioning needs internet for uncached dependencies and model assets.
Existing documents, vectors, and model volumes are retained. Compose's index init
handles embedding-model changes; unchanged indexed files are skipped.
After successful provisioning, offline restarts can skip the init jobs:

```bash
docker compose up -d --no-deps ollama app gateway
```

For a chat-model-only change, first ensure that the new model is downloaded:

```bash
docker compose run --rm --no-deps ollama-model-init
docker compose up -d --no-deps --force-recreate app
```

For a voice change, provision the new voice before recreating the app:

```bash
docker compose run --rm --no-deps backend-model-init python scripts/provision_models.py --only-tts
docker compose up -d --no-deps --force-recreate app
```

## Local development without NVIDIA Docker

Use a Python 3.12 virtual environment, install backend and frontend requirements,
and install `espeak-ng` if you want the fallback. On Linux, also install FFmpeg and
the OCR utilities required by your documents. Piper bundles its own phonemizer.
Copy the example environment only for a new setup; preserve an existing `.env`.
Ollama must be running at the configured URL.

```bash
python -m pip install -r backend/requirements.txt -r frontend/requirements.txt
ollama pull qwen3:4b
ollama pull qwen3-embedding:0.6b
cd backend
python scripts/provision_models.py
python train_engine.py
cd ..
python frontend/run_combined.py
```

Replace those two pull arguments with your configured model names. The combined
runner starts both Streamlit and the retained FastAPI endpoints. Streamlit calls
backend functions directly by default; Ollama is a local service and does not
require internet for inference.

If an answer fails, the page shows the error immediately. Logs include the active
model, completion reason, retrieval time and generation time. `/ready` checks
document QA; `/health` provides deeper speech and reranker diagnostics.

## HTTPS, microphone access and retrieval evaluation

Set `TEXMIN_HOST` to the DNS or `.local` hostname clients use, then open
`https://<TEXMIN_HOST>`. Remote browser microphones need HTTPS. The gateway uses
a private local CA; export and trust its root on your client machines:

```bash
docker compose cp gateway:/data/caddy/pki/authorities/local/root.crt ./texmin-root.crt
```

An SSH tunnel to localhost:8501 is another option. For an intentional fresh index
and corpus-specific retrieval evaluation:

```bash
docker compose run --rm --no-deps index-init python train_engine.py --force-rebuild
docker compose run --rm --no-deps app python backend/evaluate_retrieval.py backend/evals/golden.jsonl --top-k 5
```

Populate `backend/evals/golden.jsonl` with questions and expected sources first.
Local scanned-document ingestion requires Tesseract on PATH; Docker includes
English and Hindi OCR language data. The browser records speech locally and sends
audio to the configured Whisper backend, rather than using a browser vendor's
online transcription service.
