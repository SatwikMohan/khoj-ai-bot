"""Join already played, compatible WAV chunks for optional history replay."""

import io
import wave


def join_audio_chunks(chunks: list[bytes], mime: str) -> bytes | None:
    if not chunks:
        return None
    if len(chunks) == 1:
        return chunks[0]
    if mime != "audio/wav":
        return None
    output = io.BytesIO()
    try:
        with wave.open(output, "wb") as combined:
            expected = None
            for clip in chunks:
                with wave.open(io.BytesIO(clip), "rb") as source:
                    audio_format = (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype())
                    if expected is None:
                        expected = audio_format
                        combined.setnchannels(audio_format[0])
                        combined.setsampwidth(audio_format[1])
                        combined.setframerate(audio_format[2])
                        combined.setcomptype(audio_format[3], "not compressed")
                    if audio_format != expected:
                        return None
                    combined.writeframes(source.readframes(source.getnframes()))
    except (wave.Error, EOFError):
        return None
    return output.getvalue()


def join_speech_segments(chunks: list[bytes], sample_rate: int) -> bytes | None:
    """Normalize Piper WAV clips and crossfade phrase boundaries without clicks."""
    from array import array

    def pcm(clip: bytes) -> list[int]:
        with wave.open(io.BytesIO(clip), "rb") as source:
            if source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getcomptype() != "NONE":
                raise ValueError("Piper phrase must be uncompressed mono 16-bit WAV")
            original_rate = source.getframerate()
            values = array("h")
            values.frombytes(source.readframes(source.getnframes()))
        if original_rate != sample_rate and values:
            length = max(1, round(len(values) * sample_rate / original_rate))
            result = []
            for index in range(length):
                position = index * original_rate / sample_rate
                left = min(int(position), len(values) - 1)
                right = min(left + 1, len(values) - 1)
                fraction = position - int(position)
                result.append(round(values[left] * (1 - fraction) + values[right] * fraction))
            return result
        return list(values)

    if not chunks:
        return None
    combined: list[int] = []
    fade = max(1, round(sample_rate * 0.012))
    trim_limit = max(1, round(sample_rate * 0.06))
    for clip in chunks:
        values = pcm(clip)
        if not values:
            continue
        start = 0
        while start < min(trim_limit, len(values) - 1) and abs(values[start]) < 180:
            start += 1
        end = len(values)
        while end > max(start + 1, len(values) - trim_limit) and abs(values[end - 1]) < 180:
            end -= 1
        values = values[start:end]
        if combined:
            overlap = min(fade, len(combined), len(values))
            for index in range(overlap):
                fraction = (index + 1) / (overlap + 1)
                combined[-overlap + index] = round(
                    combined[-overlap + index] * (1 - fraction) + values[index] * fraction
                )
            combined.extend(values[overlap:])
        else:
            combined.extend(values)
    if not combined:
        return None
    data = array("h", (max(-32768, min(32767, value)) for value in combined))
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(data.tobytes())
    return output.getvalue()
