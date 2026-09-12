import os
import json
from pathlib import Path


def env_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def provision_whisper() -> None:
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


def provision_kokoro() -> None:
    from kokoro import KPipeline

    print("Provisioning Kokoro model and English phonemizer assets")
    KPipeline(lang_code=os.getenv("KOKORO_LANGUAGE", "a"))


def provision_indicf5() -> None:
    if not env_flag("PROVISION_INDICF5", False):
        print("IndicF5 provisioning skipped. Set PROVISION_INDICF5=true after accepting its terms.")
        return
    from transformers import AutoModel

    location = os.getenv("INDICF5_MODEL_DIR", "ai4bharat/IndicF5")
    print(f"Provisioning IndicF5 model: {location}")
    AutoModel.from_pretrained(location, trust_remote_code=True, local_files_only=False)


def provision_reranker() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")
    print(f"Provisioning reranker: {model}")
    AutoTokenizer.from_pretrained(model)
    AutoModelForCausalLM.from_pretrained(model)


if __name__ == "__main__":
    model_root = Path(os.getenv("MODEL_ROOT", "/models"))
    marker = model_root / "provisioned.json"
    desired = {
        "whisper": os.getenv("WHISPER_MODEL", "large-v3-turbo"),
        "reranker": os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B"),
        "kokoro_language": os.getenv("KOKORO_LANGUAGE", "a"),
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
    provision_kokoro()
    provision_reranker()
    provision_indicf5()
    model_root.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(desired, indent=2, sort_keys=True), encoding="utf-8")
