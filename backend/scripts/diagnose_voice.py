"""Read-only offline speech and browser deployment diagnostics.

Run from any working directory: python backend/scripts/diagnose_voice.py
This checks prerequisites and cached files. It does not claim microphone,
CUDA inference, or listening quality without a live test.
"""
import importlib.util
import json
import platform
import os
import shutil
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import config


def package_status(name):
    return importlib.util.find_spec(name) is not None


def report():
    config.validate_config()
    packages = {name: package_status(name) for name in (
        "av", "numpy", "soundfile", "torch", "transformers",
        "faster_whisper", "piper", "snac", "streamlit", "langid",
    )}
    cuda = False
    if packages["torch"]:
        import torch
        cuda = bool(torch.cuda.is_available())
    whisper = {"engine": config.STT_ENGINE, "model": config.WHISPER_TRANSFORMERS_MODEL}
    if config.STT_ENGINE.strip().lower() in {"transformers", "torch", "pytorch"}:
        try:
            from huggingface_hub import try_to_load_from_cache
            model_files = ("config.json", "preprocessor_config.json", "model.safetensors")
            whisper["cached_files"] = {
                name: Path(found).is_file() if isinstance(
                    found := try_to_load_from_cache(config.WHISPER_TRANSFORMERS_MODEL, name, cache_dir=config.HF_HOME), str
                ) else False
                for name in model_files
            }
            whisper["ready_for_loading"] = all(whisper["cached_files"].values())
        except Exception as exc:
            whisper["cache_check_error"] = str(exc)
    else:
        root = Path(config.WHISPER_MODEL_DIR)
        whisper["model_dir_exists"] = root.is_dir()
    tts = {"engine": config.TTS_ENGINE}
    if config.TTS_ENGINE == "veena":
        model = Path(config.VEENA_MODEL_DIR)
        codec = Path(config.VEENA_CODEC_DIR)
        tts["model_files_present"] = all((model / name).is_file() for name in (
            "config.json", "tokenizer.json", "model.safetensors.index.json",
            "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
        ))
        tts["codec_files_present"] = all((codec / name).is_file() for name in (
            "config.json", "pytorch_model.bin",
        ))
        tts["declared_languages"] = ["en", "hi", "hinglish"]
    elif config.TTS_ENGINE in {"piper", "auto"}:
        root = Path(config.PIPER_MODEL_DIR)
        tts["english_files_present"] = all((root / (config.PIPER_ENGLISH_VOICE + suffix)).is_file()
                                           for suffix in (".onnx", ".onnx.json"))
        tts["hindi_files_present"] = all((root / (config.PIPER_HINDI_VOICE + suffix)).is_file()
                                         for suffix in (".onnx", ".onnx.json"))
        tts["declared_languages"] = ["en", "hi"]
    model_root = Path(config.MODEL_ROOT)
    storage_root = model_root if model_root.exists() else model_root.parent
    storage = shutil.disk_usage(storage_root)
    audio_nodes = sorted(str(path) for path in Path("/dev/snd").glob("*")) if sys.platform.startswith("linux") else []
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config_valid": True,
        "offline_mode": config.OFFLINE_MODE,
        "model_root": str(model_root),
        "model_root_readable": os.access(model_root, os.R_OK) if model_root.exists() else False,
        "model_root_writable": os.access(model_root, os.W_OK) if model_root.exists() else False,
        "free_model_storage_gb": round(storage.free / 1024 ** 3, 2),
        "container_audio_nodes": audio_nodes,
        "browser_mic_requires_container_audio_nodes": False,
        "packages": packages,
        "cuda_available": cuda,
        "stt": whisper,
        "tts": tts,
        "browser_prerequisites": [
            "Open the app over HTTPS or localhost.",
            "Grant microphone permission in the browser for this origin.",
            "Choose the browser host's input device; a USB device on a remote DGX is not a browser microphone.",
            "Allow browser playback when prompted.",
        ],
        "linux_host_checks": [
            "lsusb", "arecord -l", "pactl list short sources",
            "wpctl status", "ls -l /dev/snd",
        ],
        "hardware_validation_required": True,
    }


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, default=str))
