import argparse
import hashlib
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen


BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))
import config


def provision_whisper() -> None:
    engine = config.STT_ENGINE.strip().lower()
    if engine in {"transformers", "pytorch", "torch"}:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        model = config.WHISPER_TRANSFORMERS_MODEL
        print(f"Provisioning PyTorch Whisper model: {model}")
        AutoProcessor.from_pretrained(model, local_files_only=False)
        AutoModelForSpeechSeq2Seq.from_pretrained(
            model,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            local_files_only=False,
        )
        return

    from faster_whisper import WhisperModel

    model = config.WHISPER_MODEL
    model_dir = Path(config.WHISPER_MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Provisioning Whisper model: {model}")
    WhisperModel(
        model,
        device="cpu",
        compute_type="int8",
        download_root=str(model_dir),
        local_files_only=False,
    )


def _matches_sha256(path: Path, expected: str) -> bool:
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == expected.lower()


def _download_pinned_file(destination: Path, url: str, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            digest = hashlib.sha256()
            with urlopen(url, timeout=120) as response:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    temporary.write(block)
                    digest.update(block)
        if digest.hexdigest() != expected_sha256.lower():
            raise RuntimeError(f"SHA256 mismatch for {destination.name}")
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _provision_custom_voice(path: Path, source: dict[str, str], offline: bool) -> None:
    files = (
        (path, source["model_url"], source["model_sha256"]),
        (Path(str(path) + ".json"), source["config_url"], source["config_sha256"]),
    )
    for destination, url, expected in files:
        if _matches_sha256(destination, expected):
            continue
        if offline:
            raise RuntimeError(f"Missing or mismatched offline voice: {destination}")
        print(f"Downloading and verifying {destination.name}", flush=True)
        _download_pinned_file(destination, url, expected)


def provision_tts(offline: bool = False, engine_override: str | None = None) -> None:
    from helpers.request_models import TTSRequest
    from services.tts_service import _piper_path, _synthesize_with_engine, _validate_audio

    engine = (engine_override or config.TTS_ENGINE).strip().lower()
    if engine == "auto":
        engine = "piper"
    if engine == "piper":
        from piper.download_voices import download_voice

        for language in ("hindi", "english", *config.PIPER_ADDITIONAL_VOICES):
            path = _piper_path(language)
            custom_source = config.PIPER_CUSTOM_VOICE_SOURCES.get(path.stem)
            if custom_source:
                _provision_custom_voice(path, custom_source, offline)
            elif not path.is_file() or not Path(str(path) + ".json").is_file():
                if offline:
                    raise RuntimeError(f"Missing offline voice: {path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                download_voice(path.stem, path.parent)
        # Check all configured files and synthesize the default voices.
        from services import tts_service
        for language, text in (("hi", "नमस्ते। आपका स्वागत है।"), ("en", "Hello. Your voice is ready.")):
            audio, media = tts_service._synthesize_piper_speech(TTSRequest(text=text, language=language, response_format="wav"))
            _validate_audio(audio, media)
            print(f"Piper {language} offline synthesis passed.", flush=True)
    elif engine == "veena":
        from huggingface_hub import snapshot_download

        assets = (
            (config.VEENA_MODEL_ID, config.VEENA_MODEL_REVISION, config.VEENA_MODEL_DIR),
            (config.VEENA_CODEC_ID, config.VEENA_CODEC_REVISION, config.VEENA_CODEC_DIR),
        )
        for model_id, revision, destination in assets:
            print(f"Provisioning offline speech asset: {model_id}@{revision}", flush=True)
            model_path = Path(destination)
            if not offline:
                snapshot_download(
                    repo_id=model_id, revision=revision, local_dir=destination,
                    allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.bin"],
                )
            required = (
                ("config.json", "tokenizer.json", "model.safetensors.index.json",
                 "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
                if model_id == config.VEENA_MODEL_ID else ("config.json", "pytorch_model.bin")
            )
            missing = [name for name in required if not (model_path / name).is_file()]
            if missing:
                raise RuntimeError(f"Missing speech model files in {destination}: {', '.join(missing)}")
        print("Veena and SNAC assets are cached; CUDA synthesis is verified when the app starts.", flush=True)
    elif engine == "espeak":
        audio, media = _synthesize_with_engine(engine, TTSRequest(text="नमस्ते। आपका स्वागत है।"))
        _validate_audio(audio, media)
    elif engine in {"edge", "edge-tts"}:
        raise RuntimeError("Edge TTS cannot be provisioned for offline use. Choose piper or espeak.")
    else:
        raise RuntimeError(f"Unsupported provisioning engine: {engine}. Choose piper or espeak.")


def provision_reranker() -> None:
    if not config.RERANK_ENABLED:
        return
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = config.RERANK_MODEL
    print(f"Provisioning reranker: {model}")
    AutoTokenizer.from_pretrained(model)
    AutoModelForSequenceClassification.from_pretrained(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download configured offline assets and verify speech.")
    parser.add_argument("--only-tts", action="store_true")
    parser.add_argument("--engine", choices=("piper", "veena"), help="Override the configured speech engine for provisioning.")
    parser.add_argument("--offline", action="store_true", help="Verify cached TTS assets without downloading.")
    args = parser.parse_args()
    if args.offline and not args.only_tts:
        parser.error("--offline currently requires --only-tts")
    config.configure_runtime_environment(offline=args.offline)
    provision_tts(offline=args.offline, engine_override=args.engine)
    if not args.only_tts:
        provision_whisper()
        provision_reranker()
    print("Configured offline assets are ready.", flush=True)
