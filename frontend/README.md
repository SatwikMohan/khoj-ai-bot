# Texmin AI Streamlit Chat

Install Ollama and pull the local models first:

```powershell
irm https://ollama.com/install.ps1 | iex
ollama pull llama3.2
ollama pull nomic-embed-text
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
RETRIEVAL_FETCH_K=16
RETRIEVAL_MMR_ENABLED=false
```

Set `RETRIEVAL_MMR_ENABLED=true` if you prefer broader multi-document diversity over fastest first-token latency for direct Q&A.
