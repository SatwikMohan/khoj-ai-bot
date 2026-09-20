ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.08-py3
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake curl espeak-ng ffmpeg git libsndfile1 ninja-build python3-venv tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv --system-site-packages /opt/texmin-venv
ENV PATH="/opt/texmin-venv/bin:${PATH}"

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir --no-deps --force-reinstall "git+https://github.com/AI4Bharat/IndicF5.git@main" \
    && python -m pip uninstall -y torchaudio \
    && CMAKE_BUILD_PARALLEL_LEVEL=8 USE_CUDA=1 BUILD_SOX=0 BUILD_KALDI=0 BUILD_RNNT=0 python -m pip install --no-cache-dir --no-build-isolation --no-deps "git+https://github.com/pytorch/audio.git@v2.11.0"

COPY scripts/patch_f5_tts.py /tmp/patch_f5_tts.py
RUN python /tmp/patch_f5_tts.py \
    && python -c "import inspect, torch, torchaudio; from f5_tts.infer.utils_infer import load_model; assert 'ckpt_path' not in inspect.signature(load_model).parameters; print('Verified IndicF5/Torch/TorchAudio:', torch.__version__, torchaudio.__version__)"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
