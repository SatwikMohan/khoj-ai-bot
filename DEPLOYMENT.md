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
`ollama-model-init` verifies configured names in that volume at startup.
Model downloads require an explicit provisioning command. Rebuilding the app
image does not erase the models. For a first offline boot, populate that
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

The checked-in working tree may use the Windows local model trio. Before DGX
provisioning, set `OLLAMA_CHAT_MODEL`, `OLLAMA_EMBED_MODEL`, and
`OLLAMA_EMBED_DIMENSIONS` together in `backend/config.py` for the DGX profile.
Performance and retrieval quality still need validation on Spark. A chat-model change does not require indexing. An
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

## Fast offline speech profile (2026-10-12)

The current default is `TTS_ENGINE = "piper"` on DGX and Windows.
The Veena instructions below describe the optional older voice profile.
Piper uses persistent CPU workers and preloads English and Hindi at app startup.
It does not require internet for inference. English and Hindi use different
speakers; the selected English checkpoint uses US phonemization. Roman-script
Hinglish remains unsupported by this profile; its written answer remains available.

After transferring the updated code/config, provision the smaller Piper voices
once while connected (or transfer their ONNX and JSON files into
`/models/piper` in the existing `ai_models` volume):

```bash
docker compose build app backend-model-init
docker compose run --rm --no-deps backend-model-init python scripts/provision_models.py --only-tts --engine piper
docker compose run --rm --no-deps backend-model-init python scripts/provision_models.py --only-tts --engine piper --offline
docker compose up -d --no-deps --force-recreate app
docker compose logs --tail=100 -f app
```

After warm-up, measure on the DGX without loading QA or STT in the benchmark:

```bash
docker compose exec app python /app/backend/evals/benchmark_tts_only.py --runs 10
docker compose cp app:/app/backend/evals/voice_samples/tts_speed ./dgx-tts-speed
```

The benchmark saves cold/warm timings, WAV files and any error before exiting.
The separate benchmark process has its own cold worker; it does not share the
app's warmed models. Do not compare its cold first call with warm chat latency.

## Optional Veena speech

The Docker deployment on the NVIDIA DGX uses [Veena](https://huggingface.co/maya-research/Veena)
with its built-in `kavya` speaker for English, Hindi, and Hinglish. One model
speaks every chunk, so language changes cannot switch speaker identity.
Veena generates 24 kHz WAV through the local
[SNAC 24 kHz codec](https://huggingface.co/hubertsiuzdak/snac_24khz).
Both model revisions are pinned in `backend/config.py`; the weights are cached
in the persistent `/models` volume by an explicit provisioning command.
`backend-model-init` only verifies cached assets. Runtime speech
uses local files and requires the DGX CUDA GPU. The Veena checkpoint is about
7.6 GB; allow space for its cache and the rest of the assistant's models.

Windows development keeps the small Piper voices. Its `en_IN-spicor-english`
checkpoint uses **US English phonemization**, despite the `en_IN` name. The
Windows voice therefore cannot be used to judge the DGX Indian accent. Piper
also uses separate Hindi and English speakers; the previous single-speaker
routing keeps one speaker per answer chunk, but its accent and expression
remain limited. Piper refuses Roman-script Hinglish because its English
phonemizer cannot reliably pronounce Hindi words.

The active selection is in `backend/config.py`:

```python
TTS_ENGINE = "veena" if RUNNING_IN_DOCKER else "piper"
VEENA_SPEAKER = "kavya"
TTS_FALLBACK_ENGINES = ""
```

The empty fallback list prevents a failed generation from changing to a
second speaker. A voice error leaves the written answer visible. The service
loads the Veena model at startup and terminates its worker on cancellation or
timeout. `backend/services/veena_worker.py` keeps model weights in memory after
warmup. Model audio quality, latency, and Roman-script Hinglish pronunciation
still require a listening test on the DGX; local backend tests validate routing
and WAV handling without loading the large model.

Provision the configured voice and verify its cached files:

```bash
docker compose run --rm --no-deps backend-model-init python scripts/provision_models.py --only-tts
docker compose run --rm --no-deps backend-model-init python scripts/provision_models.py --only-tts --offline
```

After the app starts on the DGX, generate four clips for an accent and
pronunciation listening check. The Roman Hinglish clip matches the app's current
reply script; the mixed-script clip shows whether Devanagari improves Hindi
pronunciation without changing speaker identity:

```bash
docker compose exec app python /app/backend/evals/veena_smoke.py
docker cp texmin-app:/app/backend/evals/voice_samples/veena_smoke ./veena-smoke
```

For local Piper testing, run `python backend/scripts/provision_models.py
--only-tts` from the project root. To stage Veena on another connected machine,
pass `--engine veena`; copy both `backend/models/veena` and
`backend/models/snac_24khz` to the DGX model volume. The offline command checks
that model files are present; the application startup performs CUDA synthesis.

## DGX deployment

The DGX image uses `nvcr.io/nvidia/pytorch:26.08-py3`. If registry authentication is
required, run `docker login nvcr.io` using the literal username `$oauthtoken` and
your NGC API key as the password. Place source documents in `backend/raw_data_files`.

After transferring the updated files and setting the DGX model trio in
`backend/config.py`, check free disk and memory before downloading the selected
models. Build images on the ARM64 DGX, then explicitly provision weights and
prepare the index:

```bash
df -h .
free -h
nvidia-smi
docker compose build app backend-model-init index-init
docker compose up -d ollama
docker compose run --rm --no-deps ollama-model-init python scripts/provision_ollama.py
docker compose run --rm --no-deps backend-model-init python scripts/provision_models.py
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
docker compose run --rm --no-deps ollama-model-init python scripts/provision_ollama.py
docker compose restart app
```

For an embedding-model change, record baseline retrieval first and schedule a
maintenance window. Edit the embedding name and dimension in `backend/config.py`,
then stage and inspect the candidate before restarting the app:

```bash
docker compose run --rm --no-deps ollama-model-init python scripts/provision_ollama.py
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

## Voice model download fails

Veena and SNAC are public Hugging Face models and need no gated-model token.
If downloading on the DGX fails, compare host and container DNS/proxy access to
`huggingface.co` and the redirect host named in the error. Do not disable TLS
checks to work around network errors.

To stage the weights from a connected machine with the backend dependencies
installed:

```bash
python backend/scripts/provision_models.py --only-tts --engine veena
```

Copy `backend/models/veena` and `backend/models/snac_24khz` into a directory
named `voice-stage` on the DGX. Import them into the persistent model volume and
verify without network access:

```bash
docker compose run --rm --no-deps -v "$PWD/voice-stage:/voice-stage:ro" backend-model-init sh -c 'mkdir -p /models/veena /models/snac_24khz && cp -a /voice-stage/veena/. /models/veena/ && cp -a /voice-stage/snac_24khz/. /models/snac_24khz/'
docker compose run --rm --no-deps backend-model-init python scripts/provision_models.py --only-tts --offline
```

Whisper and reranker assets must also be cached for fully offline startup. Use
Compose service name `ollama` for logs. See the
[Docker DNS documentation](https://docs.docker.com/engine/network/#dns-services).


## Production microphone and language validation

The Streamlit component records the **browser host's** microphone with
`getUserMedia` and `MediaRecorder`. The app container receives encoded audio
through the component protocol and decodes it with PyAV. No `/dev/snd` mount is
needed for a browser running on a laptop. A USB microphone plugged into the DGX
will not appear in that laptop browser. To use that device, run a browser on the
DGX desktop, forward its input through a remote desktop session that supports
microphone redirection, or attach the microphone to the client computer.
The app does not implement server-side ALSA capture.

On the DGX host, check a locally attached USB microphone:

```bash
lsusb
arecord -l
wpctl status
pactl list short sources
ls -l /dev/snd
arecord -D default -f S16_LE -r 16000 -c 1 -d 5 /tmp/texmin-mic.wav
```

`pactl` or `wpctl` may be absent if PulseAudio or PipeWire is not installed.
The last command is a host capture check only; it is not required when the
browser is on another machine. For a real offline STT inference check, transfer
a recorded WAV/WebM fixture into the app container and run:

```bash
docker compose exec app python backend/scripts/diagnose_voice.py
docker compose exec app python backend/scripts/transcribe_sample.py /tmp/texmin-mic.wav
docker compose exec app python backend/evals/veena_smoke.py
```

The fixture path must exist inside the container. For a host WAV, use
`docker cp /tmp/texmin-mic.wav texmin-app:/tmp/texmin-mic.wav` first.
The diagnostic checks cached model files, packages, CUDA visibility, storage,
and relevant audio nodes without recording or downloading anything. Validate
actual CUDA inference with the two sample commands; `torch.cuda.is_available()`
alone is insufficient.

In the browser, open the configured HTTPS hostname and trust the Caddy local CA,
or use an SSH tunnel to localhost. Then verify:

1. Browser site settings show microphone permission granted and the expected input device selected.
2. The microphone button shows a running duration, waveform, and input device name. Stop it explicitly.
3. The transcript appears for editing. Submit English, Hindi, then Hinglish questions and inspect the answer language.
4. Start a long answer, then ask a second question. Confirm old text and audio stop.
5. Deny permission and unplug the microphone in separate trials; text chat must remain usable.
6. Turn speech off, then on. Stop speech during playback and retry after a TTS failure.
7. Inspect `docker compose logs --tail=200 app` for request IDs, STT stage, answer language, voice selection, and error category. Raw audio is not logged.

The configured offline speech support is:

| Input | DGX Veena | Windows Piper |
| --- | --- | --- |
| English | Model card supports; DGX listening pending | Valid WAV generated locally |
| Hindi in Devanagari | Model card supports; DGX listening pending | Valid WAV generated locally |
| Roman-script Hindi / Hinglish | Model card describes code mixing; DGX listening pending | Explicit unsupported error |
| Other languages | Unsupported by selected Veena checkpoint | Requires separately configured and validated voice |

Veena is a 3B parameter Hindi/English model with code-mixed claims and an Apache
2.0 model license. XTTS-v2 was considered, but its public model license permits
noncommercial use only; it was not added. Model cards are evidence of stated
support, not a listening result. The Windows sample files and timings are in
`backend/evals/voice_samples/production_check/results.json` after running
`python backend/evals/evaluate_language_aware_tts.py --output-dir
backend/evals/voice_samples/production_check`. The benchmark does not rate pronunciation or prosody. The first answer
sentence is held until a conservative language check passes; speech can then
start before the answer ends. A later language switch stops queued speech and
replaces the text after one bounded correction attempt.

## Reproducible release checks

On Windows with the project virtual environment installed:

```powershell
cd backend
..\.venv\Scripts\python.exe -m unittest discover -s tests -p test_*.py
cd ..\frontend
..\.venv\Scripts\python.exe -m unittest discover -s tests -p test_*.py
cd ..
.\.venv\Scripts\python.exe backend/scripts/diagnose_voice.py
.\.venv\Scripts\python.exe backend/evals/evaluate_language_aware_tts.py --output-dir backend/evals/voice_samples/production_check
```

Open `frontend/tests/audio_queue_smoke.html` in a current Chromium browser. It
must display `PASS` after checking ordering, stale messages, queue limits, and
playback failure recovery. The voice capture component uses Streamlit's declared
component protocol. Playback and visualization helpers use `st.iframe` from the
pinned Streamlit 1.62 dependency. Browser buffers are capped by
`TTS_MAX_QUEUED_SEGMENTS` in `backend/config.py`; if playback falls behind that
limit, speech stops with a visible error while the text answer remains available.

The Windows run on 2026-10-11 passed 125 backend and 7 frontend tests; headless
Edge passed the audio queue smoke test. The local Whisper cache and CUDA device
were missing, so the real STT inference check could not run there. Ollama also
failed KV allocation during a real QA smoke test on that host, even after a
smaller context was tried. Provision and run the real fixture and browser checks
on the DGX before calling the deployment validated.
