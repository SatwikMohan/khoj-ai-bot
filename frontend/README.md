# Texmin AI Streamlit Chat

Install Ollama and pull the local models first:

```powershell
irm https://ollama.com/install.ps1 | iex
ollama pull qwen3:30b
ollama pull embeddinggemma
ollama list
```

Install Python dependencies and rebuild the local vector database after changing embedding models:

```powershell
cd ../backend
python -m pip install -r requirements.txt
python train_engine.py --force-rebuild
```

Run the FastAPI backend first:

```powershell
python -m uvicorn main:app --reload
```

Then run the Streamlit UI:

```powershell
cd ../frontend
python -m streamlit run app.py
```

The UI calls `http://127.0.0.1:8000/qa/ask` by default. You can change the API URL from the sidebar or set `QA_API_URL`.

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

Text-to-speech defaults to local offline system voices with `TTS_ENGINE=local`.
Set `TTS_ENGINE=edge` only when you want Edge TTS and have internet access.
