import os
import json
from pathlib import Path


def env_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def provision_whisper() -> None:
    engine = os.getenv("STT_ENGINE", "faster-whisper").strip().lower()
    if engine in {"transformers", "pytorch", "torch"}:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        model = os.getenv(
            "WHISPER_TRANSFORMERS_MODEL", "openai/whisper-large-v3-turbo"
        )
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

    model = os.getenv("WHISPER_MODEL", "large-v3-turbo")
    model_dir = Path(os.getenv("WHISPER_MODEL_DIR", "/models/whisper"))
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Provisioning Whisper model: {model}")
    WhisperModel(
        model,
        device="cpu",
        compute_type="int8",
        download_root=str(model_dir),
        local_files_only=False,
    )


def kokoro_voices() -> list[str]:
    configured = os.getenv(
        "KOKORO_PROVISION_VOICES",
        "af_heart,af_bella,af_nicole,am_adam,bf_emma,bm_george",
    )
    return list(dict.fromkeys(voice.strip() for voice in configured.split(",") if voice.strip()))


def provision_kokoro(voices: list[str]) -> None:
    from kokoro import KPipeline

    print(f"Provisioning Kokoro model, phonemizer assets, and voices: {', '.join(voices)}")
    pipeline = KPipeline(lang_code=os.getenv("KOKORO_LANGUAGE", "a"))
    for voice in voices:
        generated = pipeline("Voice ready.", voice=voice, speed=1.0)
        first_segment = next(iter(generated), None)
        if first_segment is None:
            raise RuntimeError(f"Kokoro voice '{voice}' returned no audio during provisioning.")
        print(f"Provisioned Kokoro voice: {voice}")


def provision_indicf5() -> None:
    if not env_flag("PROVISION_INDICF5", False):
        print("IndicF5 provisioning skipped. Set PROVISION_INDICF5=true after accepting its terms.")
        return
    from transformers import AutoModel

    location = os.getenv("INDICF5_MODEL_DIR", "ai4bharat/IndicF5")
    print(f"Provisioning IndicF5 model: {location}")
    AutoModel.from_pretrained(
        location,
        trust_remote_code=True,
        local_files_only=False,
        low_cpu_mem_usage=False,
    )


def provision_reranker() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")
    print(f"Provisioning reranker: {model}")
    AutoTokenizer.from_pretrained(model)
    AutoModelForCausalLM.from_pretrained(model)


if __name__ == "__main__":
    model_root = Path(os.getenv("MODEL_ROOT", "/models"))
    marker = model_root / "provisioned.json"
    configured_kokoro_voices = kokoro_voices()
    desired = {
        "stt_engine": os.getenv("STT_ENGINE", "faster-whisper"),
        "whisper": (
            os.getenv("WHISPER_TRANSFORMERS_MODEL", "openai/whisper-large-v3-turbo")
            if os.getenv("STT_ENGINE", "faster-whisper").strip().lower()
            in {"transformers", "pytorch", "torch"}
            else os.getenv("WHISPER_MODEL", "large-v3-turbo")
        ),
        "reranker": os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B"),
        "kokoro_language": os.getenv("KOKORO_LANGUAGE", "a"),
        "kokoro_voices": configured_kokoro_voices,
        "indicf5": env_flag("PROVISION_INDICF5", False),
        "indicf5_model": os.getenv("INDICF5_MODEL_DIR", "ai4bharat/IndicF5"),
    }
    if marker.exists():
        try:
            if json.loads(marker.read_text(encoding="utf-8")) == desired:
                print("Neural speech and reranker models are already provisioned.")
                raise SystemExit(0)
        except json.JSONDecodeError:
            pass

    provision_whisper()
    provision_kokoro(configured_kokoro_voices)
    provision_reranker()
    provision_indicf5()
    model_root.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(desired, indent=2, sort_keys=True), encoding="utf-8")
