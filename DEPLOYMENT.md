# Model configuration and offline operation

`backend/config.py` is the single source of non-secret application settings for
the backend, frontend, indexing, and provisioning. Use Python values (`True`,
`False`, numbers, and quoted strings), not dotenv syntax. Docker bind-mounts this
same file. After initial deployment, ordinary settings changes need only
`docker compose restart app`; locally, restart the Python process. New model
selections also require provisioning, and embedding changes require indexing.
The legacy `.env` is ignored and is not copied into images or migrated into config.
Do not put credentials in the tracked Python file.

Paths and the Ollama URL are selected for local versus Docker use in `config.py`.
The application identifies Ollama models by **name** through `OLLAMA_BASE_URL`;
it never needs a filesystem path to their weights. In Compose, the `ollama`
service owns its persistent `ollama_models` volume at `/root/.ollama`.
`ollama-model-init` pulls configured names into that volume once. Rebuilding the
app image does not erase the models. For a first offline boot, populate that
volume on a connected machine before transferring it to the DGX; building the
app image alone does not include model weights.
Docker infrastructure (GPU reservations, published ports, DNS, and Ollama server
process settings) remains in Compose. Keep its ports in sync if changing API_PORT.

The supported chat/embedding provider is Ollama. Select any **Ollama chat model**
for `OLLAMA_CHAT_MODEL` and an **Ollama embedding model** for `OLLAMA_EMBED_MODEL`.
These roles are different; arbitrary Hugging Face repository names are not Ollama
model names. The optional reranker is a non-reasoning sequence classifier and is
configured separately. Set `RERANK_ENABLED = False` for a lightweight local setup.

| Setting | Local example | DGX example |
| --- | --- | --- |
| `OLLAMA_CHAT_MODEL` | `llama3.2:latest` | `llama3.1:8b-instruct-q4_K_M` (configured) |
| `OLLAMA_CHAT_QUANTIZATION` | `Q4_K_M` | `Q4_K_M` (the model tag selects actual weights) |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | `bge-m3` (configured candidate) |
| `OLLAMA_EMBED_DIMENSIONS` | `768` | `1024` for BGE-M3 |
| `OLLAMA_NUM_CTX` | `4096` | `8192` |
| `CONTEXT_MAX_CHARS` | `6000` | `16000` |
| `EMBEDDING_BATCH_SIZE` | `32` | `256` |
| `STT_ENGINE` | `faster-whisper` | `transformers` |
| `WHISPER_MODEL` | `small` | unused by transformers |
| `WHISPER_TRANSFORMERS_MODEL` | unused by faster-whisper | `openai/whisper-large-v3-turbo` |
| `WHISPER_DEVICE` | `cpu` | `auto` |
| `WHISPER_COMPUTE_TYPE` | `int8` | `float16` |

The DGX entries are now active in `backend/config.py`; their performance and
retrieval quality still need validation on Spark. A chat-model change does not require indexing. An
embedding-model or embedding-prefix change does: run `python train_engine.py`
to build a candidate, evaluate it, then run
`python train_engine.py --promote-candidate`. The active collection stays in use
until promotion. Plan an embedding change in a maintenance window: the app may
reload the bind-mounted config before promotion and become unready. Restart it
after promotion. Do not delete the vector DB. See
`DGX_SPARK_REEVALUATION.md` for measurements and validation gates.

`OLLAMA_THINK = "false"` and `QA_REASONING_ENABLED = False` are required. Inference
rejects Ollama models that advertise a thinking capability. Use a standard chat
model; reasoning-capable models are not accepted even when thinking can be disabled.
`OLLAMA_REQUEST_TIMEOUT_SECONDS` bounds network inactivity;
`QA_RESPONSE_TIMEOUT_SECONDS` bounds the whole answer, including retrieval.
A timed-out native model operation may still be finishing in the background;
request cancellation discards its result and closes its stream where supported.

## Offline speech

The default is Piper CPU speech, using local `.onnx` and `.onnx.json` files:

```python
TTS_ENGINE = "piper"
PIPER_HINDI_VOICE = "hi_IN-pratham-medium"
PIPER_ENGLISH_VOICE = "en_IN-spicor-english"
PIPER_DEVICE = "cpu"
TTS_FALLBACK_ENGINES = ""
TTS_TIMEOUT_SECONDS = 45
RESPONSE_LANGUAGE = "auto"
HINGLISH_SCRIPT = "roman"
```

Speech-only phrase routing transliterates a curated set of Romanized Hindi
words and sends English technical terms to the English phonemizer. Displayed
chat text remains unchanged. The Hindi and English voices are separate speakers.
The selected English default is the Indian English `en_IN-spicor-english`.
Its repository labels the checkpoint AGPL-3.0; review the terms for deployment.
`scripts/provision_models.py --only-tts` downloads this third-party model
and JSON with pinned SHA256 verification. Stage the files under
`backend/models/piper` locally or `/models/piper` in Docker for offline use.
Comparison clips are under `backend/evals/voice_samples`.
The espeak-ng fallback is less natural but works offline when installed.
Set `TTS_FALLBACK_ENGINES = "espeak"` to enable that optional fallback. Selected engines appear in
the `tts_completed` log event. Reference recordings, transcripts and gated HF
access are no longer used by the default speech stack.

Locally, voices default to `backend/models/piper`. In Docker, `config.py` selects
`PIPER_MODEL_DIR = "/models/piper"` in the persistent `ai_models` volume.
Set `PIPER_MODEL_DIR` explicitly only if you use a different local directory.
Voice values may also be absolute paths to compatible Piper ONNX models.
PIPER_DEVICE defaults to CPU; CUDA requires an ARM64-compatible ONNX Runtime GPU
installation in the Spark container. Both active Hindi and English models are
provisioned and checked. The query text selects
the answer language automatically. `langid` handles longer Latin-script queries
offline; script and Hinglish rules handle shorter ones. Ambiguous short queries
fall back to English. `PIPER_ADDITIONAL_VOICES` maps other ISO language codes to
locally installed Piper voices. Other languages require a configured Piper voice, or an installed `espeak-ng` voice when the fallback is enabled; `espeak-ng --voices` lists them. If no voice supports the language,
the text answer remains visible and speech returns a clear error.

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

After transferring the updated files, build images on the ARM64 DGX and
provision the selected models. `backend/config.py` now names the DGX trial
models. Prepare their index before starting the app:

```bash
docker compose build app backend-model-init index-init
docker compose up -d ollama
docker compose run --rm --no-deps ollama-model-init
docker compose run --rm --no-deps backend-model-init
docker compose run --rm --no-deps index-init python train_engine.py
```

Wait until Ollama is healthy before running model init. On a fresh vector store,
the indexer activates its initial complete collection. If an older index was
transferred, the indexer stages a candidate instead. In that case run:

```bash
docker compose run --rm --no-deps index-init python evaluate_retrieval.py evals/golden_local.jsonl --top-k 5 --candidate
docker compose run --rm --no-deps index-init python train_engine.py --promote-candidate
```

Inspect the evaluation and expand its labels before promotion. Then start the
application and gateway:

```bash
docker compose up -d --force-recreate
docker compose logs --tail=100 -f backend-model-init index-init app ollama
```

Initial provisioning needs internet for uncached dependencies and model assets.
Existing documents, vectors, and model volumes are retained. Compose's index init
stages an embedding-model change; it does **not** activate the candidate.
Changing the embedding name in config and recreating the app before promotion
will make `/ready` fail because the active index still uses the old model.
After successful provisioning, offline restarts can skip the init jobs:

```bash
docker compose up -d --no-deps ollama app gateway
```

For a chat-model-only change, edit `backend/config.py`, then ensure the new model
is downloaded before restarting the app:

```bash
docker compose run --rm --no-deps ollama-model-init
docker compose restart app
```

For an embedding-model change, record baseline retrieval first and schedule a
maintenance window. Edit the embedding name and dimension in `backend/config.py`,
then stage and inspect the candidate before restarting the app:

```bash
docker compose run --rm --no-deps ollama-model-init
docker compose run --rm --no-deps index-init python train_engine.py --force-rebuild
docker compose run --rm --no-deps index-init python evaluate_retrieval.py evals/golden_local.jsonl --top-k 5 --candidate
docker compose run --rm --no-deps index-init python train_engine.py --promote-candidate
docker compose restart app
docker compose exec ollama ollama list
docker compose exec ollama ollama ps
```

Promotion checks OCR readiness, vector count, embedding dimension, and labelled
recall. If it rejects the candidate, restore the old embedding settings before
restarting the app. For a changed embedding model, save the old evaluation
output before editing config and compare it with the candidate result; the
automatic gate has an absolute minimum but only compares to the active index
when both use the same embedding profile.

For a voice change, provision the new voice before recreating the app:

```bash
docker compose run --rm --no-deps backend-model-init python scripts/provision_models.py --only-tts
docker compose restart app
```

## Local development without NVIDIA Docker

Use a Python 3.12 virtual environment, install backend and frontend requirements,
and install `espeak-ng` if you want the fallback. On Linux, also install FFmpeg and
the OCR utilities required by your documents. Piper bundles its own phonemizer.
Edit `backend/config.py` for your hardware; no `.env` is needed.
Ollama must be running at the configured URL.

```bash
python -m pip install -r backend/requirements.txt -r frontend/requirements.txt
cd backend
python scripts/provision_ollama.py
python scripts/provision_models.py
python train_engine.py
cd ..
cd frontend && streamlit run app.py
```

The provisioning script reads both model names from config.py.
Streamlit imports the backend services directly. Ollama is a local model service
and does not require internet for inference.

Ingestion version 8 records years and versions found in source paths and includes
them in chunk IDs. Run `python train_engine.py` to build a candidate, then
promote it after validation. Retrieval combines parallel dense and lexical search,
diversifies before reranking, and keeps separate source/version headers within the
context budget. Dates in an unversioned document are treated as content evidence,
not asserted as its version.

If an answer fails, the page shows the error immediately. Logs include the active
model, completion reason, retrieval time and generation time. The optional FastAPI
server retains /ready and /health for external clients.

## HTTPS, microphone access and retrieval evaluation

Set `TEXMIN_HOST` in `backend/config.py` to the DNS or `.local` hostname clients use, then open
`https://<TEXMIN_HOST>`. Remote browser microphones need HTTPS. The gateway uses
a private local CA; export and trust its root on your client machines:

```bash
docker compose cp gateway:/data/caddy/pki/authorities/local/root.crt ./texmin-root.crt
```

Compose's `config-init` generates the Caddyfile from the shared Python config.
After a hostname change, regenerate it and restart the gateway:

```bash
docker compose run --rm --no-deps config-init
docker compose restart gateway
```

An SSH tunnel to localhost:8501 is another option. For an intentional fresh index
and corpus-specific retrieval evaluation:

```bash
docker compose run --rm --no-deps index-init python train_engine.py --force-rebuild
docker compose run --rm --no-deps app python backend/evaluate_retrieval.py backend/evals/golden_local.jsonl --top-k 5
```

Expand `backend/evals/golden_local.jsonl` with Hindi documents, scans, tables,
code, and versioned sources before selecting an embedding model.
Local scanned-document ingestion requires Tesseract on PATH; Docker includes
English and Hindi OCR language data. The browser records speech locally and sends
audio to the configured Whisper backend, rather than using a browser vendor's
online transcription service.

## Piper download fails with “No address associated with hostname”

This is DNS resolution failure, not a missing HF token. The configured Piper
voices are public, and their downloader does not use HF_TOKEN. A redirect to
Hugging Face's file-storage host can fail even when `huggingface.co` resolves.
The configured public Whisper and reranker models also do not require gated
access. Revoke any token previously pasted into chat; never move it into config.

On the DGX, compare the host and container download paths (HEAD only, no model
download). The failing hostname in curl's error is the one to investigate:

```bash
curl -fIL --connect-timeout 10 --max-time 30 'https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx?download=true'
docker compose run --rm --no-deps backend-model-init curl -fIL --connect-timeout 10 --max-time 30 'https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx?download=true'
```

If only the container fails, check Docker's DNS/proxy configuration. Use a DNS
server reachable and approved on your network; do not blindly replace corporate
DNS or disable TLS checks. If both fail, correct the host/network configuration
or stage the voices from an internet-connected machine. The workspace cannot
repair the DGX network automatically.

On a connected machine with the project dependencies installed:

```bash
python backend/scripts/provision_models.py --only-tts
```

Copy the resulting `backend/models/piper` directory (the selected Hindi and
Indian English `.onnx` and adjacent `.onnx.json` files) into the project
directory on the DGX as `piper-voices`. Import and verify them without internet:

```bash
docker compose run --rm --no-deps -v "$PWD/piper-voices:/voice-import:ro" backend-model-init sh -c 'mkdir -p /models/piper && cp /voice-import/*.onnx /voice-import/*.onnx.json /models/piper/'
docker compose run --rm --no-deps backend-model-init python scripts/provision_models.py --only-tts --offline
```

This intentionally replaces matching voice files in the model cache; it leaves
Ollama models and document vectors alone. Whisper/reranker must also be cached
before fully offline operation. Once DNS works, resume:

```bash
docker compose up -d --force-recreate backend-model-init index-init app gateway
docker compose logs --tail=100 -f backend-model-init index-init app ollama
```

If all models and the index are already prepared, use the offline startup command
above instead; full provisioning can still contact Hugging Face for metadata.

Use Compose service name `ollama`, not container name `texmin-ollama`, for logs.
See [Docker DNS documentation](https://docs.docker.com/engine/network/#dns-services)
and the [public Piper voice files](https://huggingface.co/rhasspy/piper-voices/tree/main/hi/hi_IN/pratham/medium).
