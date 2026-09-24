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
port 8000. Set `BACKEND_CALL_MODE=http` to test the HTTP transport instead.

Latency tuning defaults are set for conversational use:

```powershell
OLLAMA_KEEP_ALIVE=1800
QA_WARMUP_ON_STARTUP=false
RETRIEVAL_FETCH_K=60
RETRIEVAL_MMR_ENABLED=true
MMR_LAMBDA_MULT=0.25
CONTEXT_MAX_CHARS=24000
CHAT_HISTORY_MAX_TURNS=8
CHAT_HISTORY_MAX_CHARS=2400
CHAT_HISTORY_MESSAGE_CHARS=520
```

The broader retrieval defaults help the bot compare repeated topics across different
facts, years, rules, figures, and sources instead of answering from the first matching chunk only.
Recent chat history is included only to resolve follow-up questions and is trimmed before prompting.

Text-to-speech uses Piper CPU voices for Hindi/mixed-script Hinglish and English.
Set `TTS_ENGINE=piper` and run `python scripts/provision_models.py --only-tts` from
`backend` once to download and verify the voices. No reference WAV is required.
See [deployment and model configuration](../DEPLOYMENT.md) for local/DGX settings,
offline setup and switching models. The pull commands above are examples; use
the names selected in `backend/.env`.
