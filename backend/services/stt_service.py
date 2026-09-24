import config
import os
import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIBE_SEMAPHORE = threading.BoundedSemaphore(
    max(1, int(config.STT_MAX_CONCURRENT))
)


class STTEngineError(RuntimeError):
    pass


def _load_environment() -> None:
    config.configure_runtime_environment()


@lru_cache(maxsize=1)
def _model():
    _load_environment()
    engine = config.STT_ENGINE.strip().lower()
    if engine in {"transformers", "pytorch", "torch"}:
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError as exc:
            raise STTEngineError(
                "PyTorch Whisper requires torch and transformers."
            ) from exc

        model_name = config.WHISPER_TRANSFORMERS_MODEL
        device = config.WHISPER_DEVICE.strip().lower()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise STTEngineError("CUDA was requested for ASR but is unavailable to the backend container.")
        dtype = torch.float16 if device == "cuda" else torch.float32
        if device == "cuda" and config.WHISPER_COMPUTE_TYPE == "bfloat16":
            dtype = torch.bfloat16
        offline = config.OFFLINE_MODE
        try:
            processor = AutoProcessor.from_pretrained(model_name, local_files_only=offline)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                local_files_only=offline,
            ).to(device)
            model.eval()
            return {
                "engine": "transformers",
                "model": model,
                "processor": processor,
                "device": device,
                "dtype": dtype,
                "model_name": model_name,
            }
        except Exception as exc:
            raise STTEngineError(
                f"Could not load offline PyTorch ASR model '{model_name}': {exc}"
            ) from exc

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise STTEngineError(
            "Offline speech recognition is not installed. Install faster-whisper."
        ) from exc

    model_name = config.WHISPER_MODEL
    device = config.WHISPER_DEVICE
    compute_type = config.WHISPER_COMPUTE_TYPE
    try:
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=int(config.WHISPER_CPU_THREADS),
            num_workers=int(config.WHISPER_WORKERS),
            download_root=config.WHISPER_MODEL_DIR or None,
            local_files_only=config.OFFLINE_MODE,
        )
        return {
            "engine": "faster-whisper",
            "model": model,
            "device": device,
            "model_name": model_name,
        }
    except Exception as exc:
        raise STTEngineError(f"Could not load offline ASR model '{model_name}': {exc}") from exc


def warm_up_stt() -> None:
    _load_environment()
    if not config.STT_WARMUP_ON_STARTUP:
        return
    _model()


def stt_runtime_status() -> dict:
    try:
        model_state = _model()
        return {
            "status": "ready",
            "engine": model_state["engine"],
            "model": model_state["model_name"],
            "device": model_state["device"],
        }
    except STTEngineError as exc:
        return {"status": "unavailable", "error": str(exc)}


def transcribe_audio(
    audio: bytes,
    suffix: str = ".webm",
    language: str | None = None,
) -> dict:
    _load_environment()
    max_bytes = int(config.STT_MAX_AUDIO_BYTES)
    if not audio:
        raise STTEngineError("No audio was received.")
    if len(audio) > max_bytes:
        raise STTEngineError(f"Audio exceeds the {max_bytes // (1024 * 1024)} MB limit.")

    safe_suffix = suffix if suffix in {".webm", ".wav", ".mp3", ".ogg", ".m4a"} else ".webm"
    path = ""
    started_at = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(suffix=safe_suffix, delete=False) as audio_file:
            audio_file.write(audio)
            path = audio_file.name

        with TRANSCRIBE_SEMAPHORE:
            model_state = _model()
            if model_state["engine"] == "transformers":
                text, detected_language, language_probability, duration = _transcribe_transformers(
                    model_state, path, language
                )
            else:
                segments, info = model_state["model"].transcribe(
                    path,
                    language=language or None,
                    beam_size=int(config.WHISPER_BEAM_SIZE),
                    best_of=int(config.WHISPER_BEST_OF),
                    vad_filter=True,
                    vad_parameters={
                        "min_silence_duration_ms": int(config.WHISPER_VAD_SILENCE_MS),
                        "speech_pad_ms": int(config.WHISPER_VAD_SPEECH_PAD_MS),
                    },
                    condition_on_previous_text=False,
                    word_timestamps=False,
                )
                text = " ".join(
                    segment.text.strip() for segment in segments if segment.text.strip()
                ).strip()
                detected_language = getattr(info, "language", language or "unknown")
                language_probability = float(getattr(info, "language_probability", 0.0))
                duration = float(getattr(info, "duration", 0.0))
    except STTEngineError:
        raise
    except Exception as exc:
        raise STTEngineError(f"Offline transcription failed: {exc}") from exc
    finally:
        if path:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass

    if not text:
        raise STTEngineError("Speech was not detected in the recording.")

    return {
        "text": text,
        "language": detected_language,
        "language_probability": round(language_probability, 4),
        "duration_seconds": round(duration, 3),
        "latency_ms": round((time.perf_counter() - started_at) * 1000, 1),
    }


def _decode_audio(path: str):
    try:
        import av
        import numpy as np
    except ImportError as exc:
        raise STTEngineError("Offline audio decoding requires PyAV and NumPy.") from exc

    samples = []
    try:
        with av.open(path) as container:
            resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
            for frame in container.decode(audio=0):
                for converted in resampler.resample(frame):
                    samples.append(converted.to_ndarray().reshape(-1))
            for converted in resampler.resample(None):
                samples.append(converted.to_ndarray().reshape(-1))
    except Exception as exc:
        raise STTEngineError(f"Could not decode the microphone recording: {exc}") from exc

    if not samples:
        raise STTEngineError("Speech was not detected in the recording.")
    audio = np.concatenate(samples).astype(np.float32) / 32768.0
    max_seconds = float(config.WHISPER_MAX_AUDIO_SECONDS)
    if len(audio) > int(16000 * max_seconds):
        audio = audio[: int(16000 * max_seconds)]
    _validate_speech_signal(audio, np)
    return audio


def _validate_speech_signal(audio, np_module=None) -> None:
    if np_module is None:
        try:
            import numpy as np_module
        except ImportError as exc:
            raise STTEngineError("Offline speech validation requires NumPy.") from exc

    minimum_duration = float(config.STT_MIN_SPEECH_SECONDS)
    if len(audio) < int(16000 * minimum_duration):
        raise STTEngineError("Speech was not detected in the recording.")

    peak = float(np_module.max(np_module.abs(audio)))
    rms = float(np_module.sqrt(np_module.mean(np_module.square(audio))))
    if peak < float(config.STT_MIN_AUDIO_PEAK) or rms < float(
        config.STT_MIN_AUDIO_RMS
    ):
        raise STTEngineError("Speech was not detected in the recording.")

    frame_size = 320
    usable = len(audio) - (len(audio) % frame_size)
    if usable < frame_size * 4:
        return
    frames = audio[:usable].reshape(-1, frame_size)
    frame_rms = np_module.sqrt(np_module.mean(np_module.square(frames), axis=1))
    noise_floor = float(np_module.percentile(frame_rms, 20))
    speech_level = float(np_module.percentile(frame_rms, 90))
    minimum_snr_db = float(config.STT_MIN_SNR_DB)
    snr_db = 20.0 * float(np_module.log10((speech_level + 1e-6) / (noise_floor + 1e-6)))
    active_threshold = max(noise_floor * 2.0, float(config.STT_MIN_FRAME_RMS))
    active_ratio = float(np_module.mean(frame_rms >= active_threshold))
    if snr_db < minimum_snr_db or active_ratio < float(
        config.STT_MIN_ACTIVE_FRAME_RATIO
    ):
        raise STTEngineError("Only background noise was detected. Please speak closer to the microphone.")


def _transcribe_transformers(model_state: dict, path: str, language: str | None):
    import torch

    audio = _decode_audio(path)
    processor = model_state["processor"]
    model = model_state["model"]
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_features = inputs.input_features.to(
        device=model_state["device"], dtype=model_state["dtype"]
    )
    generate_kwargs = {
        "max_new_tokens": int(config.WHISPER_MAX_NEW_TOKENS),
        "task": "transcribe",
    }
    if language:
        generate_kwargs["language"] = language
    attention_mask = getattr(inputs, "attention_mask", None)
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask.to(model_state["device"])
    with torch.inference_mode():
        generated_ids = model.generate(input_features, **generate_kwargs)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return text, language or "auto", 0.0, len(audio) / 16000.0
