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


def provision_indicf5() -> bool:
    if not env_flag("PROVISION_INDICF5", False):
        print("IndicF5 provisioning skipped. Set PROVISION_INDICF5=true after accepting its terms.")
        return False
    location = os.getenv("INDICF5_MODEL_DIR", "ai4bharat/IndicF5")
    print(f"Provisioning IndicF5 model: {location}")
    from services.indicf5_runtime import load_indicf5_runtime

    runtime = load_indicf5_runtime(location, local_files_only=False)
    reference_audio = os.getenv("INDICF5_REFERENCE_AUDIO", "").strip()
    reference_text = os.getenv("INDICF5_REFERENCE_TEXT", "").strip()
    if not reference_audio or not Path(reference_audio).is_file():
        raise RuntimeError(
            "INDICF5_REFERENCE_AUDIO must point to a mounted reference WAV during provisioning."
        )
    if not reference_text or reference_text == "reference audio ka exact transcript":
        raise RuntimeError("INDICF5_REFERENCE_TEXT must be the exact spoken transcript, not the placeholder.")

    audio, sample_rate = runtime.synthesize("नमस्ते।", reference_audio, reference_text)
    if audio is None or len(audio) < sample_rate // 4:
        raise RuntimeError("IndicF5 smoke test returned empty or abnormally short audio.")
    print(f"IndicF5 synthesis smoke test passed at {sample_rate} Hz.")
    return True


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
            existing = json.loads(marker.read_text(encoding="utf-8"))
            settings_match = all(existing.get(key) == value for key, value in desired.items())
            indic_ready = not desired["indicf5"] or existing.get("indicf5_ready", False)
            if settings_match and indic_ready:
                print("Neural speech and reranker models are already provisioned.")
                raise SystemExit(0)
        except json.JSONDecodeError:
            pass

    provision_whisper()
    provision_kokoro(configured_kokoro_voices)
    provision_reranker()
    indicf5_ready = False
    try:
        indicf5_ready = provision_indicf5()
    except Exception as exc:
        if env_flag("INDICF5_REQUIRED", False):
            raise
        print(f"WARNING: IndicF5 provisioning failed; continuing without Hinglish TTS: {exc}")
    model_root.mkdir(parents=True, exist_ok=True)
    marker_payload = {**desired, "indicf5_ready": indicf5_ready}
    marker.write_text(json.dumps(marker_payload, indent=2, sort_keys=True), encoding="utf-8")
