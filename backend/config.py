"""Shared, non-secret application configuration.

Edit this file and restart the application. Docker mounts this exact file too;
no .env is loaded. Never put tokens/passwords here. If an optional dependency
needs credentials, supply them through its own credential store/environment.
"""

import os
from pathlib import Path


# Model selection. Examples for a smaller local setup:
# OLLAMA_CHAT_MODEL = "qwen3:4b"
# OLLAMA_EMBED_MODEL = "qwen3-embedding:0.6b"
# STT_ENGINE = "faster-whisper"; WHISPER_MODEL = "small"
# WHISPER_DEVICE = "cpu"; WHISPER_COMPUTE_TYPE = "int8"
# RERANK_ENABLED = False; EMBEDDING_BATCH_SIZE = 32
OLLAMA_CHAT_MODEL = "llama3.2:latest"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_KEEP_ALIVE = 1800
OLLAMA_NUM_CTX = 8192
OLLAMA_NUM_PREDICT = 768
OLLAMA_REQUEST_TIMEOUT_SECONDS = 120
OLLAMA_THINK = "auto"
QA_REASONING_ENABLED = False
QA_RESPONSE_TIMEOUT_SECONDS = 180
QA_WARMUP_ON_STARTUP = False
QA_TEMPERATURE = 0.3
QA_TOP_K = 5
# None selects model-specific embedding prefixes automatically; "" disables them.
EMBED_QUERY_PREFIX = None
EMBED_DOCUMENT_PREFIX = None

# Resolve storage from this file, not the directory a command is launched from.
BACKEND_ROOT = Path(__file__).resolve().parent
RUNNING_IN_DOCKER = Path("/.dockerenv").exists()
DATA_ROOT = Path("/app") if RUNNING_IN_DOCKER else BACKEND_ROOT
MODEL_ROOT = Path("/models") if RUNNING_IN_DOCKER else BACKEND_ROOT / "models"
OLLAMA_BASE_URL = "http://ollama:11434" if RUNNING_IN_DOCKER else "http://localhost:11434"
RAW_DATA_DIR = str(DATA_ROOT / "raw_data_files")
VECTOR_DB_DIR = str(DATA_ROOT / "vector_db")
HF_HOME = str(MODEL_ROOT / "huggingface")
WHISPER_MODEL_DIR = str(MODEL_ROOT / "whisper")
PIPER_MODEL_DIR = str(MODEL_ROOT / "piper")
OFFLINE_MODE = True

# Assistant and UI. Service ports below must match Docker's port mappings.
ASSISTANT_NAME = "Khoj"
ASSISTANT_PERSONA = "a calm, perceptive colleague who explains difficult material in plain language"
RESPONSE_LANGUAGE = "hinglish"
HINGLISH_SCRIPT = "mixed"
BACKEND_CALL_MODE = "inprocess"
BACKEND_SOURCE_DIR = str(BACKEND_ROOT)
QA_API_URL = "http://127.0.0.1:8000"
API_HOST = "0.0.0.0"
API_PORT = 8000
API_LOG_LEVEL = "info"
TEXMIN_HOST = "localhost"
AVATAR_MODEL_FILE = "assets/scene.gltf"

# Indexing and retrieval. Changing embedding models requires running ingestion.
CHROMA_COLLECTION_NAME = "texmin_qa"
CHUNK_SIZE = 850
CHUNK_OVERLAP = 150
EMBEDDING_BATCH_SIZE = 256
INGESTION_WORKERS = 8
FAST_FILE_CHECK = True
MIN_EXTRACTED_TEXT_CHARS = 80
OCR_ENABLED = True
OCR_LANG = "eng+hin"
OCR_PDF_DPI = 200
OCR_TESSERACT_CONFIG = "--psm 6"
RETRIEVAL_FETCH_K = 40
RETRIEVAL_MMR_ENABLED = True
HYBRID_SEARCH_ENABLED = True
HYBRID_RRF_K = 60
ALLOW_LOW_RELEVANCE_FALLBACK = True
RELEVANCE_SCORE_THRESHOLD = 0.15
MIN_RETRIEVED_TEXT_CHARS = 40
MMR_LAMBDA_MULT = 0.35
SUMMARY_CONTEXT_CHUNKS = 8
CONTEXT_MAX_CHARS = 16000
CHAT_HISTORY_MAX_TURNS = 4
CHAT_HISTORY_MAX_CHARS = 1200
CHAT_HISTORY_MESSAGE_CHARS = 320
RERANK_ENABLED = True
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
RERANK_CANDIDATES = 12
RERANK_MAX_LENGTH = 1024

# Speech recognition. "auto" uses CUDA when available for Transformers.
STT_ENGINE = "transformers"
WHISPER_MODEL = "large-v3-turbo"
WHISPER_TRANSFORMERS_MODEL = "openai/whisper-large-v3-turbo"
WHISPER_DEVICE = "auto"
WHISPER_COMPUTE_TYPE = "float16"
WHISPER_CPU_THREADS = 4
WHISPER_WORKERS = 1
WHISPER_BEAM_SIZE = 1
WHISPER_BEST_OF = 1
WHISPER_VAD_SILENCE_MS = 350
WHISPER_VAD_SPEECH_PAD_MS = 180
WHISPER_MAX_AUDIO_SECONDS = 30
WHISPER_MAX_NEW_TOKENS = 160
STT_WARMUP_ON_STARTUP = True
STT_MAX_CONCURRENT = 1
STT_MAX_AUDIO_BYTES = 25 * 1024 * 1024
STT_MIN_SPEECH_SECONDS = 0.35
STT_MIN_AUDIO_PEAK = 0.012
STT_MIN_AUDIO_RMS = 0.0025
STT_MIN_SNR_DB = 4.0
STT_MIN_FRAME_RMS = 0.004
STT_MIN_ACTIVE_FRAME_RATIO = 0.02

# Reference-free offline speech.
TTS_ENGINE = "piper"
PIPER_HINDI_VOICE = "hi_IN-pratham-medium"
PIPER_ENGLISH_VOICE = "en_US-lessac-medium"
TTS_FALLBACK_ENGINES = "espeak"
TTS_TIMEOUT_SECONDS = 45
TTS_RESPONSE_TIMEOUT_SECONDS = 120
TTS_FAILURE_THRESHOLD = 2
TTS_FAILURE_COOLDOWN_SECONDS = 60
TTS_MIN_AUDIO_BYTES = 1024
TTS_STREAM_MIN_CHARS = 160
TTS_TONE = "warm"
TTS_RATE = "+0%"
TTS_PITCH = "+0Hz"
TTS_VOICE = "configured"
ESPEAK_RATE = 155

# Optional legacy/online engines. Edge is rejected while OFFLINE_MODE is True.
KOKORO_VOICE = "af_heart"
KOKORO_LANGUAGE = "a"
KOKORO_FALLBACK_VOICES = "af_bella,bf_emma"
EDGE_TTS_VOICE = "en-IN-NeerjaNeural"
EDGE_TTS_HINDI_VOICE = "hi-IN-SwaraNeural"
EDGE_TTS_TONE = "neutral"
EDGE_TTS_RATE = "+0%"
EDGE_TTS_PITCH = "+0Hz"


def configure_runtime_environment(*, offline: bool | None = None) -> None:
    """Translate configuration for libraries that only accept environment flags.

    Application settings are read directly from this module. Provisioning opts
    into online downloads explicitly; credentials are never read or copied here.
    """
    offline = OFFLINE_MODE if offline is None else offline
    os.environ["HF_HOME"] = HF_HOME
    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1" if offline else "0"
