# AI interface

Install Ollama and pull the local models first:

```powershell
irm https://ollama.com/install.ps1 | iex
ollama pull llama3.2:latest
ollama pull nomic-embed-text
ollama list
```

Install Python dependencies and rebuild the local vector database after changing embedding models:

```powershell
cd ../backend
python -m pip install -r requirements.txt
python train_engine.py --force-rebuild
```

Run the combined application from the repository root:

```powershell
python -m pip install -r backend/requirements.txt -r frontend/requirements.txt
python frontend/run_combined.py
```

The interface uses two columns: the original offline GLTF avatar on the left and a
scrollable conversation on the right. Text and voice input remain available beneath
the conversation. Listening, retrieval, generation, playback, and interruption drive
the avatar's available rig motion; there is no settings panel. Configure the avatar
model, TTS, and API transport in backend/config.py.

The default BACKEND_CALL_MODE value of http calls FastAPI using QA_API_URL.
The combined deployment runs both services locally. For a separate frontend container, set
QA_API_URL to a backend address reachable from that container and route the configured
BROWSER_API_PATH to FastAPI for browser microphone requests. The frontend image includes
only small shared voice helpers; model inference stays in the backend.

All application settings live in `backend/config.py`. For example:

```python
OLLAMA_KEEP_ALIVE = 1800
QA_WARMUP_ON_STARTUP = False
RETRIEVAL_FETCH_K = 40
RETRIEVAL_MMR_ENABLED = True
MMR_LAMBDA_MULT = 0.35
CONTEXT_MAX_CHARS = 16000
CHAT_HISTORY_MAX_TURNS = 4
CHAT_HISTORY_MAX_CHARS = 1200
CHAT_HISTORY_MESSAGE_CHARS = 320
```

The broader retrieval defaults help the bot compare repeated topics across different
facts, years, rules, figures, and sources instead of answering from the first matching chunk only.
Recent chat history is included only to resolve follow-up questions and is trimmed before prompting.

Text-to-speech routes Hindi and English phrases through separate offline Piper
voices, including known Romanized Hindi and English technical terms in a Hinglish
response. Displayed chat text remains unchanged. Other languages require a
configured local Piper voice or an installed espeak-ng fallback. The default
English Piper voice is the selected Indian English `en_IN-spicor-english`.
Its repository labels the checkpoint AGPL-3.0; review the terms for deployment.
See [the avatar and speech report](../AVATAR_TTS_REPORT.md) for test results and limits.
Set `TTS_ENGINE = "piper"` and run `python scripts/provision_models.py --only-tts` from
`backend` once to download and verify the voices. No reference WAV is required.
See [deployment and model configuration](../DEPLOYMENT.md) for local/DGX settings,
offline setup and switching models. The pull commands above are examples; use
the names selected in `backend/config.py`. Restart the process after edits; no
`.env` is loaded. The combined image is the default deployment. To build the
optional HTTP-only frontend image, use the repository root as build context:
`docker build -f frontend/Dockerfile .`, and configure HTTP mode and QA_API_URL
in the shared config before building it.
