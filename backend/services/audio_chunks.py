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
