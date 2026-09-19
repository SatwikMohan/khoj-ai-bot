# Texmin AI Streamlit Chat

Install Ollama and pull the local models first:

```powershell
irm https://ollama.com/install.ps1 | iex
ollama pull qwen3.5:122b
ollama pull qwen3-embedding:8b-q8_0
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
QA_WARMUP_ON_STARTUP=true
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

Text-to-speech uses local Kokoro for English and provisioned IndicF5 for Hindi/mixed-script
Hinglish when `TTS_ENGINE=auto`. Edge TTS is online-only.
