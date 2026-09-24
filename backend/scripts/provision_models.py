import argparse
import sys
import os
import json
from pathlib import Path


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


def provision_tts(offline: bool = False) -> None:
    from helpers.request_models import TTSRequest
    from services.tts_service import _piper_path, _synthesize_with_engine, _validate_audio

    engine = config.TTS_ENGINE.strip().lower()
    if engine == "auto":
        engine = "piper"
    if engine == "piper":
        from piper.download_voices import download_voice

        for language in ("hindi", "english"):
            path = _piper_path(language)
            if not path.is_file() or not Path(str(path) + ".json").is_file():
                if offline:
                    raise RuntimeError(f"Missing offline voice: {path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                download_voice(path.stem, path.parent)
        # Exercise each model even if RESPONSE_LANGUAGE forces Hinglish.
        from services import tts_service
        previous = config.RESPONSE_LANGUAGE
        try:
            for language, text in (("hindi", "नमस्ते। आपका स्वागत है।"), ("english", "Hello. Your voice is ready.")):
                config.RESPONSE_LANGUAGE = language
                audio, media = tts_service._synthesize_piper_speech(TTSRequest(text=text, response_format="wav"))
                _validate_audio(audio, media)
                print(f"Piper {language} offline synthesis passed.", flush=True)
        finally:
            config.RESPONSE_LANGUAGE = previous
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = config.RERANK_MODEL
    print(f"Provisioning reranker: {model}")
    AutoTokenizer.from_pretrained(model)
    AutoModelForCausalLM.from_pretrained(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download configured offline assets and verify speech.")
    parser.add_argument("--only-tts", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Verify cached TTS assets without downloading.")
    args = parser.parse_args()
    if args.offline and not args.only_tts:
        parser.error("--offline currently requires --only-tts")
    config.configure_runtime_environment(offline=args.offline)
    provision_tts(offline=args.offline)
    if not args.only_tts:
        provision_whisper()
        provision_reranker()
    print("Configured offline assets are ready.", flush=True)
