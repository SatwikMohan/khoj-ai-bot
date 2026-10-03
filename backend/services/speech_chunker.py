"""Incremental sentence-sized speech units for chunk-based offline synthesis."""

import re

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

    def push(self, token: str) -> list[str]:
        self.buffer += token
        segments, self.buffer = split_speakable_prefix(self.buffer)
        return segments

    def finish(self) -> list[str]:
        final = self.buffer.strip()
        self.buffer = ""
        return [final] if final else []
