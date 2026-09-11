# Texmin AI Deployment

This repository is split into a FastAPI backend and a Streamlit frontend:

- `backend/`: API, text-to-speech, document ingestion, and vector search.
- `frontend/`: Streamlit chat UI and avatar assets.
- `backend/.env`: runtime settings for the backend and Compose services.
- `backend/raw_data_files/`: source documents mounted into the backend container.
- `backend/vector_db/`: persistent Chroma database mounted into the backend container.

## DGX Spark / NVIDIA GPU Deployment

1. Install Docker, Docker Compose, and the NVIDIA Container Toolkit on the DGX host.
2. Copy `backend/.env.example` to `backend/.env` and adjust model names or runtime settings if needed.
3. Start the stack:

```bash
docker compose up --build -d
```

4. Pull the Ollama models inside the Ollama container:

```bash
docker compose exec ollama ollama pull qwen3:30b
docker compose exec ollama ollama pull embeddinggemma
```

5. Build or refresh the vector database. Do this whenever `OLLAMA_EMBED_MODEL`,
   `CHUNK_SIZE`, or `CHUNK_OVERLAP` changes:

```bash
docker compose run --rm backend python train_engine.py --force-rebuild
```

6. Open the frontend at `http://<dgx-host>:8501`. The backend and Ollama services stay private inside the Docker network.

## Local Development

Run the backend:

```bash
cd backend
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Run the frontend in another terminal:

```bash
cd frontend
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

For local non-Docker runs, set `OLLAMA_BASE_URL=http://localhost:11434` and `QA_API_URL=http://127.0.0.1:8000`.
The QA API sends recent chat history with each question, but the backend trims it before prompting so document context still gets most of the token budget.
Text-to-speech defaults to `TTS_ENGINE=local`, which uses the operating system's offline voices.
Use `TTS_ENGINE=edge` only if you explicitly want Edge TTS and have internet access.
