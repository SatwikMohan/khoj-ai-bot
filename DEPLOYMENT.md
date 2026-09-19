# Texmin AI Deployment

This repository contains a FastAPI backend and a Streamlit frontend. In deployment they run
inside one `app` container and share the same imported Python service layer and model caches:

- `backend/`: API, text-to-speech, document ingestion, and vector search.
- `frontend/`: Streamlit chat UI and avatar assets.
- `backend/.env`: runtime settings for the application and Compose services.
- `backend/raw_data_files/`: source documents mounted into the backend container.
- `backend/vector_db/`: persistent Chroma database mounted into the backend container.

## DGX Spark / NVIDIA GPU Deployment

DGX Spark ships with Docker and the NVIDIA Container Runtime. The backend image uses the
DGX-compatible `nvcr.io/nvidia/pytorch:26.08-py3` base image, so authenticate to NGC first:

```bash
docker login nvcr.io
```

Use `$oauthtoken` as the username and an NGC personal API key as the password.

1. Copy `backend/.env.example` to `backend/.env` and review the model and storage settings.
2. Put source documents in `backend/raw_data_files/`.
3. Run the first-time online provisioning pass:

```bash
docker compose up --build
```

Compose performs these readiness-gated steps automatically:

- pulls the configured Ollama response and embedding models;
- provisions Faster-Whisper and Kokoro into the persistent `ai_models` volume;
- builds a versioned vector and lexical index;
- atomically promotes the new index only after ingestion finishes;
- starts the combined application after all required models and the index are ready.

Streamlit uses `BACKEND_CALL_MODE=inprocess`, so chat, STT, and TTS do not make HTTP calls.
FastAPI still runs on port 8000 in the same application process for integrations and health
checks. Neither an in-process call nor a call to localhost/Compose DNS requires internet.

Subsequent starts can operate without internet access because model and index data are persistent.
Start the already-provisioned stack with:

```bash
docker compose up -d
```

Set `TEXMIN_HOST` to the DNS or `.local` hostname clients will use. Open
`https://<TEXMIN_HOST>`. Microphones require HTTPS when the UI is opened from another machine.
The gateway uses a private local CA so the system remains offline. Export its root certificate:

```bash
docker compose cp gateway:/data/caddy/pki/authorities/local/root.crt ./texmin-root.crt
```

Trust `texmin-root.crt` once on each authorized Windows, macOS, or Linux client. Alternatively,
use an SSH tunnel to `localhost:8501`. Check API readiness with
`docker compose exec app curl -f http://localhost:8000/ready`.

### Offline Hinglish voice

Kokoro is English-only in this project. For Hinglish, use IndicF5 with mixed-script output:

1. Accept the gated `ai4bharat/IndicF5` model terms while the DGX still has internet access.
2. Put a clean 5-15 second Hindi/Hinglish WAV in `backend/voice_samples/reference.wav`.
3. Add its exact transcript and these settings to `backend/.env`:

```dotenv
PROVISION_INDICF5=true
HF_TOKEN=your_hugging_face_token_for_first_provisioning_only
INDICF5_REFERENCE_AUDIO=/voices/reference.wav
INDICF5_REFERENCE_TEXT=reference audio ka exact transcript
RESPONSE_LANGUAGE=hinglish
HINGLISH_SCRIPT=mixed
TTS_ENGINE=auto
```

4. Run `docker compose up --build` once online. After provisioning succeeds, remove `HF_TOKEN`
   from the runtime environment and use `docker compose up -d` offline.

Mixed script means Hindi words are emitted in Devanagari while English words remain Latin;
this is still spoken Hinglish and gives IndicF5 substantially better pronunciation than feeding
Roman Hindi to an English Kokoro voice. The FastAPI `/tts/speech` endpoint remains available.

### Rebuilding and evaluation

Changing the embedding model automatically creates a new collection. Do not delete the current
collection during a rebuild. To intentionally create a fresh candidate:

```bash
docker compose run --rm index-init python train_engine.py --force-rebuild
```

Populate `backend/evals/golden.jsonl`, then measure corpus-specific recall and latency:

```bash
docker compose run --rm app python backend/evaluate_retrieval.py backend/evals/golden.jsonl --top-k 5
```

## Local Development

For the same in-process behavior used by deployment, install both requirement files and run:

```bash
python -m pip install -r backend/requirements.txt -r frontend/requirements.txt
python frontend/run_combined.py
```

For local non-Docker runs, set `OLLAMA_BASE_URL=http://localhost:11434`. To test the old
HTTP transport, set `BACKEND_CALL_MODE=http` and `QA_API_URL=http://127.0.0.1:8000`.
The QA API sends recent chat history with each question, but the backend trims it before prompting so document context still gets most of the token budget.
Document ingestion OCRs scanned PDF pages and standalone image files when `OCR_ENABLED=true`.
The backend Docker image includes Tesseract with English and Hindi language data; for local non-Docker runs, install Tesseract separately and keep it on `PATH`.
Speech recognition is performed by Faster-Whisper on the server. The browser uses only local
volume analysis for endpointing and never invokes vendor speech recognition. `TTS_ENGINE=auto`
selects Kokoro for English, IndicF5 for Hindi
when provisioned, and validates/falls back from broken audio automatically. Edge TTS remains an
explicit online-only option.
