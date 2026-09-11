import asyncio
import os
import re
import tempfile
import threading
from pathlib import Path

from dotenv import load_dotenv

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


class TTSEngineError(RuntimeError):
    pass


def _load_environment() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise TTSEngineError(f"{name} is missing. Add it to .env before using text to speech.")
    return value


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

    engine_name = os.getenv("TTS_ENGINE", "local").strip().lower()
    if engine_name in {"local", "offline", "pyttsx3"}:
        return _synthesize_local_speech(payload)
    if engine_name not in {"edge", "edge-tts"}:
        raise TTSEngineError("Unknown TTS_ENGINE. Use 'local' for offline speech or 'edge' for Edge TTS.")

    return _synthesize_edge_speech(payload)


def _synthesize_edge_speech(payload: TTSRequest) -> tuple[bytes, str]:
    voice = payload.voice_id or os.getenv("EDGE_TTS_VOICE", DEFAULT_ENGLISH_VOICE)
    hindi_voice = os.getenv("EDGE_TTS_HINDI_VOICE", DEFAULT_HINDI_VOICE)
    rate, pitch, volume = _prosody_settings(payload)

    spoken_text = _markdown_to_spoken_text(payload.text, payload.max_words)
    if not spoken_text:
        raise TTSEngineError("There is no speakable text after formatting was removed.")

    segments = _speech_segments(spoken_text, voice, hindi_voice)

    try:
        audio = asyncio.run(_generate_segmented_audio(segments, rate, pitch, volume))
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
