import config
import json
import os
import re
import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_SPEECH_LANGUAGES = frozenset({"en", "hi"})
UNSUPPORTED_SCRIPT = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff\u0400-\u04ff\u0370-\u03ff\u4e00-\u9fff]")
TRANSCRIBE_SEMAPHORE = threading.BoundedSemaphore(
    max(1, int(config.STT_MAX_CONCURRENT))
)


class STTEngineError(RuntimeError):
    pass


class STTRecoverableError(STTEngineError):
    """A clip with no usable question; continuous listening can resume."""


def _load_environment() -> None:
    config.configure_runtime_environment()


_MODEL_LOAD_LOCK = threading.Lock()


def _model():
    # lru_cache alone may run its wrapped function twice on simultaneous misses.
    with _MODEL_LOAD_LOCK:
        return _model_cached()


@lru_cache(maxsize=1)
def _model_cached():
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
    cancelled: threading.Event | None = None,
) -> dict:
    _load_environment()
    max_bytes = int(config.STT_MAX_AUDIO_BYTES)
    if cancelled is not None and cancelled.is_set():
        raise STTEngineError("This voice interaction was replaced.")
    if not audio:
        raise STTEngineError("No audio was received.")
    if len(audio) > max_bytes:
        raise STTEngineError(f"Audio exceeds the {max_bytes // (1024 * 1024)} MB limit.")
    if language and language.lower() not in SUPPORTED_SPEECH_LANGUAGES:
        raise STTEngineError("The microphone supports English, Hindi and Hinglish speech.")

    safe_suffix = suffix if suffix in {".webm", ".wav", ".mp3", ".ogg", ".m4a"} else ".webm"
    path = ""
    started_at = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(suffix=safe_suffix, delete=False) as audio_file:
            audio_file.write(audio)
            path = audio_file.name

        samples = _decode_audio(path)
        if cancelled is not None and cancelled.is_set():
            raise STTEngineError("This voice interaction was replaced.")
        while not TRANSCRIBE_SEMAPHORE.acquire(timeout=float(config.REQUEST_CANCELLATION_POLL_SECONDS)):
            if cancelled is not None and cancelled.is_set():
                raise STTEngineError("This voice interaction was replaced.")
        try:
            if cancelled is not None and cancelled.is_set():
                raise STTEngineError("This voice interaction was replaced.")
            model_state = _model()
            spoken_language = None
            if model_state["engine"] == "transformers":
                text, detected_language, language_probability, duration = _transcribe_transformers(
                    model_state, samples, language, cancelled
                )
                if not language and detected_language == "en":
                    # Whisper can emit an English translation even with the
                    # transcribe task. Check the audio language independently
                    # before accepting English as the person's spoken style.
                    try:
                        acoustic_language, acoustic_probability = _preferred_supported_language(
                            model_state, samples
                        )
                    except STTEngineError:
                        acoustic_language, acoustic_probability = None, 0.0
                    if (acoustic_language == "hi" and acoustic_probability >=
                            float(config.STT_HINDI_AUDIO_OVERRIDE_MIN_PROB)):
                        retry_error = None
                        try:
                            hindi_text, _, _, _ = _transcribe_transformers(
                                model_state, samples, "hi", cancelled
                            )
                        except Exception as exc:
                            if cancelled is not None and cancelled.is_set():
                                raise STTEngineError("This voice interaction was replaced.")
                            retry_error = type(exc).__name__
                            hindi_text = ""
                        anchors = re.findall(r"\b[A-Z][A-Z0-9]{1,}\b|\b\d+(?:[./-]\d+)*\b", text)
                        anchors_preserved = all(re.search(
                            r"(?<![A-Za-z0-9])" + re.escape(anchor) + r"(?![A-Za-z0-9])",
                            hindi_text, re.IGNORECASE,
                        ) for anchor in anchors)
                        if (re.search(r"[\u0900-\u097f]", hindi_text)
                                and len(hindi_text.split()) >= 2 and anchors_preserved):
                            text = hindi_text
                            detected_language = "hi"
                            spoken_language = "hinglish" if re.search(r"[A-Za-z]", hindi_text) else "hi"
                            language_probability = acoustic_probability
                            print(json.dumps({"event": "stt_spoken_language_recovered",
                                              "spoken_language": spoken_language,
                                              "acoustic_probability": round(acoustic_probability, 3)}),
                                  flush=True)
                        else:
                            print(json.dumps({"event": "stt_hindi_retry_rejected",
                                              "reason": retry_error or "transcript_or_identifier_mismatch",
                                              "acoustic_probability": round(acoustic_probability, 3)}),
                                  flush=True)
                if not language and (detected_language not in SUPPORTED_SPEECH_LANGUAGES
                                     or UNSUPPORTED_SCRIPT.search(text)):
                    recovered_language, language_probability = _preferred_supported_language(
                        model_state, samples
                    )
                    text, detected_language, _, duration = _transcribe_transformers(
                        model_state, samples, recovered_language, cancelled
                    )
            else:
                segments, info = model_state["model"].transcribe(
                    samples,
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
        finally:
            TRANSCRIBE_SEMAPHORE.release()
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

    if cancelled is not None and cancelled.is_set():
        raise STTEngineError("This voice interaction was replaced.")
    if not text or not re.search(r"[A-Za-z\u0900-\u097f]", text):
        raise STTRecoverableError("Speech was not detected in the recording.")
    if detected_language not in SUPPORTED_SPEECH_LANGUAGES or UNSUPPORTED_SCRIPT.search(text):
        raise STTRecoverableError(
            "The microphone could not recognize English, Hindi or Hinglish. Please speak again."
        )

    return {
        "text": text,
        "language": detected_language,
        "spoken_language": spoken_language or detected_language,
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
        raise STTRecoverableError("Speech was not detected in the recording.")
    audio = np.concatenate(samples).astype(np.float32) / 32768.0
    max_seconds = float(config.WHISPER_MAX_AUDIO_SECONDS)
    if len(audio) > int(16000 * max_seconds):
        raise STTEngineError(f"Recording exceeds the {max_seconds:g} second limit.")
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
        raise STTRecoverableError("Speech was not detected in the recording.")

    peak = float(np_module.max(np_module.abs(audio)))
    rms = float(np_module.sqrt(np_module.mean(np_module.square(audio))))
    if peak < float(config.STT_MIN_AUDIO_PEAK) or rms < float(
        config.STT_MIN_AUDIO_RMS
    ):
        raise STTRecoverableError("Speech was not detected in the recording.")

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
        raise STTRecoverableError("Only background noise was detected. Please speak closer to the microphone.")
    active_frames = frames[frame_rms >= active_threshold]
    if len(active_frames) * frame_size / 16000 < float(config.STT_MIN_VOICED_SECONDS):
        raise STTRecoverableError("Only a brief sound was detected. Please speak again.")
    # Broadband bumps and rustling can have high energy without being speech.
    window = np_module.hanning(frame_size)
    power = np_module.abs(np_module.fft.rfft(active_frames * window, axis=1)) ** 2
    speech_band = power[:, 3:70] + 1e-12  # 150-3450 Hz at 16 kHz.
    flatness = np_module.exp(np_module.mean(np_module.log(speech_band), axis=1)) / (
        np_module.mean(speech_band, axis=1)
    )
    if float(np_module.median(flatness)) > float(config.STT_MAX_SPECTRAL_FLATNESS):
        raise STTRecoverableError("Only background noise was detected. Please speak closer to the microphone.")


def _preferred_supported_language(model_state: dict, audio) -> tuple[str, float]:
    """Use Whisper language logits to recover a dialect misheard as another language."""
    import torch

    model = model_state["model"]
    config_map = getattr(model.generation_config, "lang_to_id", {}) or {}
    allowed = [(code, config_map.get(f"<|{code}|>")) for code in ("en", "hi")]
    if any(token_id is None for _, token_id in allowed):
        raise STTEngineError("Microphone language recovery is unavailable.")
    features = model_state["processor"](
        audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
    ).input_features.to(device=model_state["device"], dtype=model_state["dtype"])
    start_id = model.generation_config.decoder_start_token_id
    start = torch.tensor([[start_id]], device=model_state["device"])
    with torch.inference_mode():
        logits = model(input_features=features, decoder_input_ids=start,
                       use_cache=False).logits[0, -1]
    ids = list(config_map.values())
    probabilities = torch.softmax(logits[ids].float(), dim=0)
    scores = {code: float(probabilities[ids.index(token_id)].item())
              for code, token_id in allowed}
    chosen = max(scores, key=scores.get)
    probability = scores[chosen]
    if probability < float(config.WHISPER_LANGUAGE_RETRY_MIN_PROB):
        raise STTRecoverableError(
            "The microphone could not recognize English, Hindi or Hinglish. Please speak again."
        )
    return chosen, probability

def _transcribe_transformers(model_state: dict, audio, language: str | None, cancelled: threading.Event | None = None):
    import torch

    if cancelled is not None and cancelled.is_set():
        raise STTEngineError("This voice interaction was replaced.")
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
    if cancelled is not None and cancelled.is_set():
        raise STTEngineError("This voice interaction was replaced.")
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    raw = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    match = re.search(r"<\|([a-z]{2})\|>", raw)
    # Transformers does not expose a calibrated language probability here.
    detected = language or (match.group(1) if match else "unknown")
    return text, detected, 0.0, len(audio) / 16000.0
