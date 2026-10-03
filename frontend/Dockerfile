FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/frontend

COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend /app/frontend
COPY backend/config.py /app/backend/config.py
# HTTP-only frontend still uses these small shared voice/text helpers.
COPY backend/services/__init__.py backend/services/audio_chunks.py backend/services/language_service.py backend/services/speech_chunker.py /app/backend/services/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
