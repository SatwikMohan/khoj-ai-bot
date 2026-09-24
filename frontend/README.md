# Texmin AI Streamlit Chat

Install Ollama and pull the local models first:

```powershell
irm https://ollama.com/install.ps1 | iex
ollama pull qwen3:4b
ollama pull qwen3-embedding:0.6b
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

The UI imports backend service functions by default, while the FastAPI endpoints remain live on
port 8000. Set `BACKEND_CALL_MODE = "http"` in `backend/config.py` to test the HTTP transport instead.

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

Text-to-speech uses Piper CPU voices for Hindi/mixed-script Hinglish and English.
Set `TTS_ENGINE = "piper"` and run `python scripts/provision_models.py --only-tts` from
`backend` once to download and verify the voices. No reference WAV is required.
See [deployment and model configuration](../DEPLOYMENT.md) for local/DGX settings,
offline setup and switching models. The pull commands above are examples; use
the names selected in `backend/config.py`. Restart the process after edits; no
`.env` is loaded. The combined image is the default deployment. To build the
optional HTTP-only frontend image, use the repository root as build context:
`docker build -f frontend/Dockerfile .`, and configure HTTP mode and QA_API_URL
in the shared config before building it.
