"""Incremental sentence-sized speech units for chunk-based offline synthesis."""

import re
import time

import config


SENTENCE_END = re.compile(r"[.!?\u0964](?=\s|$)")


def split_speakable_prefix(buffer: str) -> tuple[list[str], str]:
    minimum = int(config.TTS_STREAM_MIN_CHARS)
    maximum = int(config.TTS_STREAM_MAX_CHARS)
    if not buffer.strip():
        return [], buffer
    segments = []
    cursor = 0
    for match in SENTENCE_END.finditer(buffer):
        candidate = buffer[cursor:match.end()].strip()
        if len(candidate) >= minimum:
            segments.append(candidate)
            cursor = match.end()
    remainder = buffer[cursor:]
    while len(remainder) >= maximum:
        cut = max(remainder.rfind(mark, minimum, maximum) for mark in (" ", ",", ";", ":", "\n"))
        if cut < minimum:
            next_space = re.search(r"\s", remainder[maximum:])
            if next_space is None:
                break  # Wait for a word boundary; never split a word or Hindi grapheme.
            cut = maximum + next_space.start()
        candidate = remainder[:cut].strip(" ,;:\n")
        if candidate:
            segments.append(candidate)
        remainder = remainder[cut:]
    return segments, remainder


class IncrementalSpeechSegments:
    def __init__(self):
        self.buffer = ""
        self.buffered_at = None

    def push(self, token: str) -> list[str]:
        if token and not self.buffer:
            self.buffered_at = time.monotonic()
        self.buffer += token
        segments, self.buffer = split_speakable_prefix(self.buffer)
        if segments:
            self.buffered_at = time.monotonic() if self.buffer.strip() else None
        return segments

    def flush_due(self, now: float | None = None) -> list[str]:
        """Release a natural phrase when punctuation is slow to arrive."""
        now = time.monotonic() if now is None else now
        if (self.buffered_at is None or
                now - self.buffered_at < float(config.TTS_STREAM_MAX_BUFFER_SECONDS) or
                len(self.buffer.strip()) < int(config.TTS_STREAM_MIN_CHARS)):
            return []
        maximum = int(config.TTS_STREAM_MAX_CHARS)
        prefix = self.buffer[:maximum]
        boundaries = list(re.finditer(r"\s+", prefix))
        if not boundaries:
            return []
        cut = boundaries[-1].end()
        if cut < int(config.TTS_STREAM_MIN_CHARS):
            return []
        segment = self.buffer[:cut].strip()
        self.buffer = self.buffer[cut:]
        self.buffered_at = now if self.buffer.strip() else None
        return [segment] if segment else []

    def finish(self) -> list[str]:
        final = self.buffer.strip()
        self.buffer = ""
        self.buffered_at = None
        return [final] if final else []
