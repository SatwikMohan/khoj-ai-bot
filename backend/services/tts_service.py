import config
import asyncio
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from functools import lru_cache
from pathlib import Path


from helpers.request_models import TTSRequest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_FORMATS = {"mp3", "wav"}
DEFAULT_ENGLISH_VOICE = "en-IN-NeerjaNeural"
DEFAULT_HINDI_VOICE = "hi-IN-SwaraNeural"
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
SENTENCE_RE = re.compile(r"[^.!?।]+[.!?।]*")
TONE_PRESETS = {
    "neutral": {"rate": "+0%", "pitch": "+0Hz", "volume": "+0%"},
    "warm": {"rate": "-4%", "pitch": "-2Hz", "volume": "+0%"},
    "cheerful": {"rate": "+8%", "pitch": "+8Hz", "volume": "+6%"},
    "calm": {"rate": "-12%", "pitch": "-5Hz", "volume": "-2%"},
    "serious": {"rate": "-7%", "pitch": "-7Hz", "volume": "+0%"},
    "energetic": {"rate": "+14%", "pitch": "+6Hz", "volume": "+8%"},
}
PYTTSX3_LOCK = threading.Lock()
NEURAL_TTS_LOCK = threading.Lock()
VOICE_HEALTH_LOCK = threading.Lock()
VOICE_FAILURES: dict[str, tuple[int, float]] = {}


class TTSEngineError(RuntimeError):
    pass


def _handle_tts_loop_exception(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    error = context.get("exception")
    if (
        os.name == "nt"
        and isinstance(error, ConnectionResetError)
        and getattr(error, "winerror", None) == 10054
    ):
        return
    loop.default_exception_handler(context)


def _tts_loop_factory() -> asyncio.AbstractEventLoop:
    loop = asyncio.SelectorEventLoop() if os.name == "nt" else asyncio.new_event_loop()
    loop.set_exception_handler(_handle_tts_loop_exception)
    return loop


def _run_tts_coroutine(coroutine):
    with asyncio.Runner(loop_factory=_tts_loop_factory) as runner:
        return runner.run(coroutine)


def _load_environment() -> None:
    config.configure_runtime_environment()


def _strip_markdown_table(text: str) -> str:
    spoken_rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            spoken_rows.append(line)
            continue

        cells = [cell.strip("* `") for cell in stripped.strip("|").split("|")]
        if all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        if cells:
            spoken_rows.append("; ".join(cell for cell in cells if cell))

    return "\n".join(spoken_rows)


def _join_spoken_lines(text: str) -> str:
    lines = [line.strip(" .") for line in text.splitlines() if line.strip(" .")]
    normalized = []
    for line in lines:
        if line.endswith((".", "!", "?", "।")):
            normalized.append(line)
        else:
            normalized.append(f"{line}.")
    return " ".join(normalized)


def _markdown_to_spoken_text(text: str, max_words: int) -> str:
    spoken = _strip_markdown_table(text)
    spoken = re.sub(r"```.*?```", " ", spoken, flags=re.DOTALL)
    spoken = re.sub(r"`([^`]+)`", r"\1", spoken)
    spoken = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", spoken)
    spoken = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", spoken)
    spoken = re.sub(r"^#{1,6}\s*", "", spoken, flags=re.MULTILINE)
    spoken = re.sub(r"^\s*[-*+]\s+", "", spoken, flags=re.MULTILINE)
    spoken = re.sub(r"^\s*\d+\.\s+", "", spoken, flags=re.MULTILINE)
    spoken = re.sub(r"[*_~>#]", "", spoken)
    spoken = _join_spoken_lines(spoken)
    spoken = re.sub(r"\s+", " ", spoken).strip()
    spoken = re.sub(r"(\.\s*){2,}", ". ", spoken)

    words = spoken.split()
    if len(words) > max_words:
        spoken = " ".join(words[:max_words]).rstrip(" ,;:")
        spoken += ". I will pause here, so the spoken version stays clear and comfortable."

    return spoken


def _contains_devanagari(text: str) -> bool:
    return bool(DEVANAGARI_RE.search(text))


def _speech_segments(text: str, default_voice: str, hindi_voice: str) -> list[tuple[str, str]]:
    segments = []
    for match in SENTENCE_RE.finditer(text):
        segment = match.group(0).strip()
        if not segment:
            continue

        voice = hindi_voice if _contains_devanagari(segment) else default_voice
        if segments and segments[-1][0] == voice:
            segments[-1] = (voice, f"{segments[-1][1]} {segment}")
        else:
            segments.append((voice, segment))

    if not segments and text.strip():
        voice = hindi_voice if _contains_devanagari(text) else default_voice
        segments.append((voice, text.strip()))

    return segments


def _prosody_settings(payload: TTSRequest) -> tuple[str, str, str]:
    tone = (payload.tone or "neutral").strip().lower()
    if tone == "custom":
        return payload.rate, payload.pitch, payload.volume
    if tone not in TONE_PRESETS:
        options = ", ".join([*TONE_PRESETS.keys(), "custom"])
        raise TTSEngineError(f"Unknown tone '{payload.tone}'. Choose one of: {options}.")

    preset = TONE_PRESETS[tone]
    return preset["rate"], preset["pitch"], preset["volume"]


async def _generate_audio(text: str, voice: str, rate: str, pitch: str, volume: str) -> bytes:
    import edge_tts

    communicate = edge_tts.Communicate(
        text,
        voice,
        rate=rate,
        pitch=pitch,
        volume=volume,
    )
    audio_chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])
    return b"".join(audio_chunks)


async def _generate_segmented_audio(
    segments: list[tuple[str, str]], rate: str, pitch: str, volume: str
) -> bytes:
    audio_chunks = []
    for voice, text in segments:
        audio_chunks.append(await _generate_audio(text, voice, rate, pitch, volume))
    return b"".join(audio_chunks)


def synthesize_speech(payload: TTSRequest) -> tuple[bytes, str]:
    _load_environment()
    response_format = payload.response_format.lower()
    if response_format not in SUPPORTED_FORMATS:
        raise TTSEngineError(
            f"Unsupported audio format '{payload.response_format}'. Use 'wav' for local TTS or 'mp3' for Edge TTS."
        )

    engine_name = config.TTS_ENGINE.strip().lower()
    fallback_names = [
        name.strip().lower()
        for name in config.TTS_FALLBACK_ENGINES.split(",")
        if name.strip()
    ]
    if engine_name == "auto":
        engines = ["piper", *fallback_names]
    else:
        engines = [engine_name, *fallback_names]

    errors = []
    for candidate in dict.fromkeys(engines):
        if _voice_circuit_open(candidate):
            errors.append(f"{candidate}: temporarily disabled after repeated failures")
            continue
        try:
            if candidate in {"edge", "edge-tts"} and config.OFFLINE_MODE:
                raise TTSEngineError("Edge TTS requires internet; use piper or espeak offline.")
            audio, media_type = _synthesize_with_engine(candidate, payload)
            _validate_audio(audio, media_type)
            _record_voice_success(candidate)
            print(json.dumps({"event": "tts_completed", "engine": candidate}), flush=True)
            return audio, media_type
        except Exception as exc:
            _record_voice_failure(candidate)
            errors.append(f"{candidate}: {exc}")

    raise TTSEngineError("All configured offline voices failed. " + " | ".join(errors))


def _synthesize_with_engine(engine_name: str, payload: TTSRequest) -> tuple[bytes, str]:
    if engine_name == "piper":
        return _synthesize_piper_speech(payload)
    if engine_name == "espeak":
        return _synthesize_espeak_speech(payload)
    if engine_name in {"local", "offline", "pyttsx3"}:
        return _synthesize_local_speech(payload)
    if engine_name in {"edge", "edge-tts"}:
        return _synthesize_edge_speech(payload)
    if engine_name == "kokoro":
        return _synthesize_kokoro_speech(payload)
    if engine_name in {"indicf5", "indic-f5"}:
        raise TTSEngineError("IndicF5 has been retired. Set TTS_ENGINE=piper and provision the Piper voices.")
    raise TTSEngineError(f"Unknown TTS engine '{engine_name}'.")


def _voice_circuit_open(engine_name: str) -> bool:
    threshold = int(config.TTS_FAILURE_THRESHOLD)
    cooldown = int(config.TTS_FAILURE_COOLDOWN_SECONDS)
    with VOICE_HEALTH_LOCK:
        failures, failed_at = VOICE_FAILURES.get(engine_name, (0, 0.0))
    return failures >= threshold and time.monotonic() - failed_at < cooldown


def _record_voice_failure(engine_name: str) -> None:
    with VOICE_HEALTH_LOCK:
        failures, _ = VOICE_FAILURES.get(engine_name, (0, 0.0))
        VOICE_FAILURES[engine_name] = (failures + 1, time.monotonic())


def _record_voice_success(engine_name: str) -> None:
    with VOICE_HEALTH_LOCK:
        VOICE_FAILURES.pop(engine_name, None)


def _validate_audio(audio: bytes, media_type: str) -> None:
    if len(audio) < int(config.TTS_MIN_AUDIO_BYTES):
        raise TTSEngineError("voice returned an empty or truncated audio segment")
    if media_type != "audio/wav":
        return
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav_file:
            duration = wav_file.getnframes() / max(1, wav_file.getframerate())
            sample = wav_file.readframes(min(wav_file.getnframes(), wav_file.getframerate()))
    except (wave.Error, EOFError) as exc:
        raise TTSEngineError(f"voice returned an invalid WAV file: {exc}") from exc
    if duration < 0.08 or not sample or not any(sample):
        raise TTSEngineError("voice returned silent or abnormally short audio")


def tts_runtime_status() -> dict:
    _load_environment()
    with VOICE_HEALTH_LOCK:
        failures = {name: count for name, (count, _failed_at) in VOICE_FAILURES.items()}
    status = {
        "status": "ready",
        "engine": config.TTS_ENGINE,
        "fallback_engines": config.TTS_FALLBACK_ENGINES,
        "failures": failures,
    }
    engine = status["engine"].strip().lower()
    if engine == "auto":
        engine = "piper"
        status["selected_engine"] = engine
    if engine == "piper":
        paths = [_piper_path("hindi"), _piper_path("english")]
        missing = [str(path) for path in paths if not path.is_file() or not Path(str(path) + ".json").is_file()]
        if missing:
            status.update(status="unavailable", error="Missing Piper voice files: " + ", ".join(missing))
        status["voices"] = [str(path) for path in paths]
    elif engine == "espeak":
        if not shutil.which("espeak-ng"):
            status.update(status="unavailable", error="Install espeak-ng for the offline fallback.")
    elif engine == "kokoro":
        try:
            voice = config.KOKORO_VOICE
            _warm_kokoro_voice(config.KOKORO_LANGUAGE, voice)
            status["voice"] = voice
        except TTSEngineError as exc:
            status["status"] = "unavailable"
            status["error"] = str(exc)
    elif engine in {"indicf5", "indic-f5"}:
        status.update(status="unavailable", error="IndicF5 is retired; set TTS_ENGINE=piper.")
    return status


def warm_up_tts() -> None:
    status = tts_runtime_status()
    if status["status"] != "ready":
        raise TTSEngineError(status.get("error", "TTS warm-up failed."))


def _synthesize_edge_speech(payload: TTSRequest) -> tuple[bytes, str]:
    voice = payload.voice_id or config.EDGE_TTS_VOICE
    hindi_voice = config.EDGE_TTS_HINDI_VOICE
    rate, pitch, volume = _prosody_settings(payload)

    spoken_text = _markdown_to_spoken_text(payload.text, payload.max_words)
    if not spoken_text:
        raise TTSEngineError("There is no speakable text after formatting was removed.")

    segments = _speech_segments(spoken_text, voice, hindi_voice)

    try:
        audio = _run_tts_coroutine(_generate_segmented_audio(segments, rate, pitch, volume))
    except Exception as exc:
        voices = ", ".join(sorted({segment_voice for segment_voice, _ in segments}))
        raise TTSEngineError(f"Edge TTS could not synthesize voice '{voices}': {exc}") from exc

    if not audio:
        raise TTSEngineError("Edge TTS did not return audio data.")
    return audio, "audio/mpeg"


def _parse_percent(value: str, default: int = 0) -> int:
    match = re.fullmatch(r"\s*([+-]?\d+)\s*%\s*", value or "")
    if not match:
        return default
    return int(match.group(1))


def _voice_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, (list, tuple, set)):
        return " ".join(_voice_text(item) for item in value)
    return str(value)


def _voice_matches(voice, query: str) -> bool:
    query = (query or "").strip().lower()
    if not query:
        return False

    haystack = " ".join(
        [
            _voice_text(getattr(voice, "id", "")),
            _voice_text(getattr(voice, "name", "")),
            _voice_text(getattr(voice, "languages", "")),
        ]
    ).lower()
    return query in haystack


def _voice_has_language(voice, language_terms: tuple[str, ...]) -> bool:
    haystack = " ".join(
        [
            _voice_text(getattr(voice, "id", "")),
            _voice_text(getattr(voice, "name", "")),
            _voice_text(getattr(voice, "languages", "")),
        ]
    ).lower()
    return any(term in haystack for term in language_terms)


def _select_local_voice(engine, requested_voice: str | None, text: str) -> str | None:
    voices = engine.getProperty("voices") or []
    if not voices:
        return None

    if requested_voice:
        for voice in voices:
            if _voice_matches(voice, requested_voice):
                return voice.id

    language_terms = ("hi", "hindi", "india") if _contains_devanagari(text) else ("en", "english")
    for voice in voices:
        if _voice_has_language(voice, language_terms):
            return voice.id

    return getattr(voices[0], "id", None)


def _configure_local_engine(engine, payload: TTSRequest, spoken_text: str) -> None:
    voice_id = _select_local_voice(engine, payload.voice_id, spoken_text)
    if voice_id:
        try:
            engine.setProperty("voice", voice_id)
        except Exception:
            pass

    rate, _pitch, volume = _prosody_settings(payload)
    base_rate = int(engine.getProperty("rate") or 200)
    rate_multiplier = max(0.45, min(1.8, 1 + (_parse_percent(rate) / 100)))
    engine.setProperty("rate", int(base_rate * rate_multiplier))

    base_volume = float(engine.getProperty("volume") or 1.0)
    adjusted_volume = max(0.0, min(1.0, base_volume + (_parse_percent(volume) / 100)))
    engine.setProperty("volume", adjusted_volume)


def _audio_array_to_wav(audio, sample_rate: int = 24000) -> bytes:
    try:
        import numpy as np
    except ImportError as exc:
        raise TTSEngineError("Neural TTS requires numpy.") from exc

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        raise TTSEngineError("Neural TTS returned no samples.")
    peak = float(np.max(np.abs(samples)))
    if peak > 1.0:
        samples = samples / peak
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return output.getvalue()


@lru_cache(maxsize=2)
def _kokoro_pipeline(language_code: str):
    try:
        from kokoro import KPipeline
    except ImportError as exc:
        raise TTSEngineError("Kokoro is not installed.") from exc
    try:
        return KPipeline(lang_code=language_code)
    except Exception as exc:
        raise TTSEngineError(f"Kokoro could not load its local model: {exc}") from exc


@lru_cache(maxsize=16)
def _warm_kokoro_voice(language_code: str, voice: str) -> None:
    try:
        with NEURAL_TTS_LOCK:
            generated = _kokoro_pipeline(language_code)("Voice ready.", voice=voice, speed=1.0)
            if next(iter(generated), None) is None:
                raise TTSEngineError(f"Kokoro voice '{voice}' returned no audio.")
    except TTSEngineError:
        raise
    except Exception as exc:
        raise TTSEngineError(f"Kokoro voice '{voice}' is unavailable offline: {exc}") from exc


def _kokoro_voice_candidates(requested_voice: str) -> list[str]:
    default_voice = config.KOKORO_VOICE.strip() or "af_heart"
    primary_voice = (
        requested_voice
        if re.fullmatch(r"[ab][fm]_[a-z0-9_]+", requested_voice)
        else default_voice
    )
    fallback_voices = [
        voice.strip()
        for voice in config.KOKORO_FALLBACK_VOICES.split(",")
        if re.fullmatch(r"[ab][fm]_[a-z0-9_]+", voice.strip())
    ]
    return list(dict.fromkeys([primary_voice, default_voice, *fallback_voices]))


def _synthesize_kokoro_speech(payload: TTSRequest) -> tuple[bytes, str]:
    if _contains_devanagari(payload.text):
        raise TTSEngineError("Kokoro is not the configured Hindi voice; trying the Indic fallback.")

    spoken_text = _markdown_to_spoken_text(payload.text, payload.max_words)
    if not spoken_text:
        raise TTSEngineError("There is no speakable text after formatting was removed.")

    requested_voice = (payload.voice_id or "").strip()
    voices = _kokoro_voice_candidates(requested_voice)
    language_code = config.KOKORO_LANGUAGE
    rate, _pitch, _volume = _prosody_settings(payload)
    speed = max(0.65, min(1.35, 1 + (_parse_percent(rate) / 100)))

    errors = []
    for voice in voices:
        try:
            with NEURAL_TTS_LOCK:
                generated = _kokoro_pipeline(language_code)(spoken_text, voice=voice, speed=speed)
                parts = [audio for _graphemes, _phonemes, audio in generated]
            if not parts:
                raise TTSEngineError("returned no audio segments")
            import numpy as np

            audio = np.concatenate([np.asarray(part, dtype=np.float32).reshape(-1) for part in parts])
            return _audio_array_to_wav(audio), "audio/wav"
        except Exception as exc:
            errors.append(f"{voice}: {exc}")

    raise TTSEngineError("Kokoro voices failed. " + " | ".join(errors))


def _piper_path(language: str) -> Path:
    directory = Path(config.PIPER_MODEL_DIR)
    name = (config.PIPER_HINDI_VOICE if language == "hindi" else config.PIPER_ENGLISH_VOICE).strip()
    path = Path(name if name.endswith(".onnx") else name + ".onnx")
    return path if path.is_absolute() else directory / path


def _synthesize_piper_speech(payload: TTSRequest) -> tuple[bytes, str]:
    language = "hindi" if (
        _contains_devanagari(payload.text)
        or config.RESPONSE_LANGUAGE in {"hindi", "hinglish"}
    ) else "english"
    path = _piper_path(language)
    if not path.is_file() or not Path(str(path) + ".json").is_file():
        raise TTSEngineError(
            f"Piper needs {path} and its .onnx.json config. "
            "Run python scripts/provision_models.py --only-tts first."
        )
    spoken_text = _markdown_to_spoken_text(payload.text, payload.max_words)
    if not spoken_text:
        raise TTSEngineError("There is no speakable text.")
    rate, _, _ = _prosody_settings(payload)
    speed = max(0.65, min(1.5, 1 + _parse_percent(rate) / 100))
    job = {"model": str(path.resolve()), "text": spoken_text, "length_scale": 1 / speed}
    try:
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "services" / "piper_worker.py")],
            input=json.dumps(job).encode("utf-8"), capture_output=True,
            timeout=float(config.TTS_TIMEOUT_SECONDS),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except subprocess.TimeoutExpired as exc:
        raise TTSEngineError("Piper speech generation timed out; its worker was stopped.") from exc
    if result.returncode:
        raise TTSEngineError(result.stderr.decode("utf-8", errors="replace")[-1500:])
    return result.stdout, "audio/wav"


def _synthesize_espeak_speech(payload: TTSRequest) -> tuple[bytes, str]:
    executable = shutil.which("espeak-ng")
    if not executable:
        raise TTSEngineError("Install espeak-ng to use the offline fallback voice.")
    language = "hi" if (
        _contains_devanagari(payload.text)
        or config.RESPONSE_LANGUAGE in {"hindi", "hinglish"}
    ) else "en"
    # A real file gives WAV a complete header (unlike espeak's --stdout stream).
    with tempfile.TemporaryDirectory(prefix="khoj-voice-") as directory:
        path = Path(directory) / "speech.wav"
        subprocess.run(
            [executable, "-v", language, "-s", str(config.ESPEAK_RATE), "-w", str(path), "--stdin"],
            input=_markdown_to_spoken_text(payload.text, payload.max_words).encode("utf-8"),
            capture_output=True, check=True,
            timeout=float(config.TTS_TIMEOUT_SECONDS),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return path.read_bytes(), "audio/wav"


def _synthesize_local_speech(payload: TTSRequest) -> tuple[bytes, str]:
    import pyttsx3

    spoken_text = _markdown_to_spoken_text(payload.text, payload.max_words)
    if not spoken_text:
        raise TTSEngineError("There is no speakable text after formatting was removed.")

    output_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
            output_path = output_file.name

        with PYTTSX3_LOCK:
            engine = pyttsx3.init()
            _configure_local_engine(engine, payload, spoken_text)
            engine.save_to_file(spoken_text, output_path)
            engine.runAndWait()
            engine.stop()

        audio = Path(output_path).read_bytes()
    except Exception as exc:
        raise TTSEngineError(f"Local offline TTS could not synthesize speech: {exc}") from exc
    finally:
        if output_path:
            try:
                Path(output_path).unlink(missing_ok=True)
            except OSError:
                pass

    if not audio:
        raise TTSEngineError("Local offline TTS did not return audio data.")
    return audio, "audio/wav"
