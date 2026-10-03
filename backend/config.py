"""Shared, non-secret application configuration.

Edit this file and restart the application. Docker mounts this exact file too;
no .env is loaded. Never put tokens/passwords here. If an optional dependency
needs credentials, supply them through its own credential store/environment.
"""

import os
from pathlib import Path


# DGX Spark deployment models. The Windows test system only has the commented
# smaller models installed; switch all three values together for local tests.
# OLLAMA_CHAT_MODEL = "llama3.2:latest"
# OLLAMA_EMBED_MODEL = "nomic-embed-text"
# OLLAMA_EMBED_DIMENSIONS = 768
# STT_ENGINE = "faster-whisper"; WHISPER_MODEL = "small"
# WHISPER_DEVICE = "cpu"; WHISPER_COMPUTE_TYPE = "int8"
# RERANK_ENABLED = False; EMBEDDING_BATCH_SIZE = 32
OLLAMA_CHAT_MODEL = "llama3.1:8b-instruct-q4_K_M"
OLLAMA_EMBED_MODEL = "bge-m3"
# Provision these tags in the Ollama Docker volume and build/promote a new index
# before starting the app; the Windows nomic index uses a different vector space.
# The application sends these model names to Ollama; it never opens model files.
# Docker's Ollama service stores weights in its persistent /root/.ollama volume.
OLLAMA_CHAT_QUANTIZATION = "Q4_K_M"  # Documentation only; select actual weights with the Ollama model tag.
OLLAMA_EMBED_DIMENSIONS = 1024
OLLAMA_KEEP_ALIVE = 1800
OLLAMA_NUM_CTX = 8192
OLLAMA_NUM_PREDICT = 768
OLLAMA_REQUEST_TIMEOUT_SECONDS = 120
OLLAMA_THINK = "false"
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
OLLAMA_BASE_URL = "http://ollama:11434" if RUNNING_IN_DOCKER else "http://localhost:11434"  # Compose service URL, never a weight path.
RAW_DATA_DIR = str(DATA_ROOT / "raw_data_files")
VECTOR_DB_DIR = str(DATA_ROOT / "vector_db")
HF_HOME = str(MODEL_ROOT / "huggingface")
WHISPER_MODEL_DIR = str(MODEL_ROOT / "whisper")
PIPER_MODEL_DIR = str(MODEL_ROOT / "piper")
OFFLINE_MODE = True

# Assistant and UI. Service ports below must match Docker's port mappings.
ASSISTANT_NAME = "Khoj"
ASSISTANT_PERSONA = "a calm, perceptive colleague who explains difficult material in plain language"
RESPONSE_LANGUAGE = "auto"
HINGLISH_SCRIPT = "roman"
BACKEND_CALL_MODE = "http"
BACKEND_SOURCE_DIR = str(BACKEND_ROOT)
QA_API_URL = "http://127.0.0.1:8000"
BROWSER_API_PATH = "/api"
BROWSER_ALLOWED_ORIGINS = ("http://localhost:8501", "http://127.0.0.1:8501")
API_HOST = "0.0.0.0"
API_PORT = 8000
API_LOG_LEVEL = "info"
TEXMIN_HOST = "localhost"
AVATAR_MODEL_FILE = "assets/scene.gltf"  # Retained for older deployments.
AVATAR_ENABLED = True
TTS_ENABLED = True
TTS_MAX_WORDS = 140

# Indexing and retrieval. Changing embedding models requires running ingestion.
CHROMA_COLLECTION_NAME = "texmin_qa"
INGESTION_VERSION = 8
INDEX_PROMOTION_EVAL_FILE = str(BACKEND_ROOT / "evals" / "golden_local.jsonl")
INDEX_MIN_RECALL_AT_5 = 0.8
CHUNK_SIZE = 850
CHUNK_OVERLAP = 150
EMBEDDING_BATCH_SIZE = 32
INGESTION_WORKERS = 2
FAST_FILE_CHECK = False
MIN_EXTRACTED_TEXT_CHARS = 80
OCR_ENABLED = True
OCR_LANG = "eng+hin"
OCR_PDF_DPI = 200
OCR_TESSERACT_CONFIG = "--psm 6"
OCR_TESSERACT_EXECUTABLE = "tesseract"
OCR_EMBEDDED_IMAGES = True
OCR_MAX_IMAGES_PER_PAGE = 12
OCR_MIN_IMAGE_PIXELS = 10000
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
RERANK_ENABLED = False  # Enable only after provisioning and measuring reranker weights on DGX.
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_CANDIDATES = 12
RERANK_MAX_LENGTH = 1024
VERSION_CANDIDATES = 16

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
STT_AUDIO_SAMPLE_RATE = 16000
RECORDING_BUFFER_MS = 250
REQUEST_CANCELLATION_POLL_SECONDS = 0.1
RECORDING_NO_SPEECH_TIMEOUT_SECONDS = 10
RECORDING_MAX_SECONDS = 30
RECORDING_SILENCE_MS = 1100
RECORDING_SPEECH_CONFIRM_MS = 150
STT_REQUEST_TIMEOUT_SECONDS = 90

# Reference-free offline speech.
TTS_ENGINE = "piper"
PIPER_HINDI_VOICE = "hi_IN-pratham-medium"
PIPER_ENGLISH_VOICE = "en_US-lessac-medium"
# Optional ISO 639-1 code to installed Piper voice name or absolute ONNX path.
PIPER_ADDITIONAL_VOICES: dict[str, str] = {}
TTS_PRELOAD_VOICES = True
TTS_FALLBACK_ENGINES = ""  # Avoid a different speaker after a Piper failure.
TTS_TIMEOUT_SECONDS = 45
TTS_RESPONSE_TIMEOUT_SECONDS = 120
TTS_FAILURE_THRESHOLD = 2
TTS_FAILURE_COOLDOWN_SECONDS = 60
TTS_MIN_AUDIO_BYTES = 1024
TTS_SAMPLE_RATE = 22050
TTS_CHANNELS = 1
TTS_SAMPLE_WIDTH_BYTES = 2
TTS_STREAM_MIN_CHARS = 60
TTS_STREAM_MAX_CHARS = 320
TTS_STREAM_CONCURRENCY = 2
QA_STREAM_HEARTBEAT_SECONDS = 0.35
TTS_TONE = "warm"
TTS_RATE = "+0%"
TTS_PITCH = "+0Hz"
TTS_VOICE = "configured"
TTS_HINGLISH_VOICE_LANGUAGE = "en"  # Roman text must not enter the Hindi phonemizer.
ESPEAK_RATE = 155


def validate_config() -> None:
    """Reject incompatible application settings before work begins."""
    if not OLLAMA_CHAT_MODEL.strip() or not OLLAMA_EMBED_MODEL.strip():
        raise ValueError("Both Ollama model names must be configured.")
    if QA_REASONING_ENABLED or OLLAMA_THINK.strip().lower() not in {"false", "auto"}:
        raise ValueError("This application requires a non-reasoning chat model and disabled thinking.")
    if CHUNK_SIZE <= 0 or not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
        raise ValueError("CHUNK_OVERLAP must be smaller than positive CHUNK_SIZE.")
    if OLLAMA_EMBED_DIMENSIONS <= 0 or EMBEDDING_BATCH_SIZE <= 0 or INGESTION_WORKERS <= 0:
        raise ValueError("Embedding dimensions, batch size and ingestion workers must be positive.")
    if STT_AUDIO_SAMPLE_RATE != 16000 or RECORDING_BUFFER_MS <= 0 or REQUEST_CANCELLATION_POLL_SECONDS <= 0:
        raise ValueError("Whisper input must be 16 kHz and recording/cancellation intervals positive.")
    if min(TTS_SAMPLE_RATE, TTS_CHANNELS, TTS_SAMPLE_WIDTH_BYTES) <= 0:
        raise ValueError("TTS audio format settings must be positive.")
    if not 0 < TTS_STREAM_MIN_CHARS < TTS_STREAM_MAX_CHARS or TTS_STREAM_CONCURRENCY <= 0:
        raise ValueError("Streaming speech chunk sizes and concurrency must be positive and ordered.")
    if not 0 <= INDEX_MIN_RECALL_AT_5 <= 1:
        raise ValueError("INDEX_MIN_RECALL_AT_5 must be between zero and one.")
    if min(QA_TOP_K, RETRIEVAL_FETCH_K, CONTEXT_MAX_CHARS, VERSION_CANDIDATES) <= 0:
        raise ValueError("Retrieval limits must be positive.")
    if TTS_ENGINE.strip().lower() in {"piper", "auto"} and TTS_HINGLISH_VOICE_LANGUAGE != "en":
        raise ValueError("Roman Hinglish requires the English Piper phonemizer.")
    if OFFLINE_MODE and TTS_ENGINE.strip().lower() in {"edge", "edge-tts"}:
        raise ValueError("Edge TTS requires internet and cannot be the offline speech engine.")

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
    validate_config()
    offline = OFFLINE_MODE if offline is None else offline
    os.environ["HF_HOME"] = HF_HOME
    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1" if offline else "0"
