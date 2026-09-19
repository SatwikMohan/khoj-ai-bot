ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.08-py3
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/backend \
    BACKEND_SOURCE_DIR=/app/backend \
    BACKEND_CALL_MODE=inprocess

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl espeak-ng ffmpeg libsndfile1 python3-venv tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv --system-site-packages /opt/texmin-venv
ENV PATH="/opt/texmin-venv/bin:${PATH}"

COPY backend/requirements.txt /tmp/backend-requirements.txt
COPY frontend/requirements.txt /tmp/frontend-requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r /tmp/backend-requirements.txt -r /tmp/frontend-requirements.txt \
    && python -m pip uninstall -y torchaudio \
    && python -c "import torch, torchaudio; print('Verified NVIDIA Torch/TorchAudio:', torch.__version__, torchaudio.__version__)"

COPY backend /app/backend
COPY frontend /app/frontend

EXPOSE 8000 8501

CMD ["python", "/app/frontend/run_combined.py"]
