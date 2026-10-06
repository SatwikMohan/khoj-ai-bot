"""Bounded response delivery and incremental removal of explicit reasoning tags."""

import queue
import re
import threading
import time
from contextlib import closing

from services.request_lifecycle import ActiveRequest, registry


class AnswerTextFilter:
    """Handle tags split across chunks without censoring ordinary answer sentences."""

    TAGS = ("<think>", "</think>", "<analysis>", "</analysis>", "<answer>", "</answer>")

    def __init__(self):
        self.pending = ""
        self.hidden = False

    def feed(self, text: str, final: bool = False) -> str:
        self.pending += text
        output = []
        while self.pending:
            lowered = self.pending.lower()
            tag = next((tag for tag in self.TAGS if lowered.startswith(tag)), None)
            if tag:
                if tag in {"<think>", "<analysis>"}:
                    self.hidden = True
                elif tag in {"</think>", "</analysis>", "<answer>"}:
                    self.hidden = False
                self.pending = self.pending[len(tag):]
                continue
            if not final and any(tag.startswith(lowered) for tag in self.TAGS):
                break
            if not self.hidden:
                output.append(self.pending[0])
            self.pending = self.pending[1:]
        return "".join(output)


class CitationTextFilter:
    """Remove only generated retrieval labels before tokens reach chat or TTS.

    The filter buffers one sentence or Markdown line so split model chunks
    cannot expose half of a citation before it can be recognized.
    """

    _PREFIX = re.compile(
        r"(?i)\b(?:from|according to|as (?:stated|mentioned) in)\s+"
        r"(?:the\s+)?(?:reference|source|document|evidence)\s*#?\d+"
        r"(?:\s*,?\s*page\s*\d+)?\s*[,;:]?\s*"
    )
    _CONTEXT = re.compile(
        r"(?i)\b(?:as (?:mentioned|shown|stated) in|according to|from)\s+"
        r"(?:the\s+)?(?:retrieved|provided|supplied|private)\s+"
        r"(?:context|passages?|material)\s*[,;:]?\s*"
    )
    _PAGE = re.compile(
        r"(?i)(?:,?\s*(?:on|at|see|refer to)\s+page\s+\d+"
        r"(?:\s+of\s+(?:the\s+)?document)?)"
    )
    _NUMBERED = re.compile(
        r"(?i)[ \t]+(?:\[\s*\d+(?:\s*,\s*\d+)*\s*\]|\[\s*evidence\s+\d+\s*\]|\(reference\s+\d+\))"
        r"(?=\s|[.,;:!?\u0964]|$)"
    )
    _REFERENCE_SUBJECT = re.compile(
        r"(?i)\b(?:reference|document|source|chunk|evidence)\s*#?\d+\s*"
        r"(?:[,;:]|(?:states?|says?|shows?|mentions?|contains?|indicates?)\s+(?:that\s+)?)\s*"
    )
    _PAGE_SUBJECT = re.compile(
        r"(?i)\bpage\s+\d+\s+(?:of\s+(?:the\s+)?document\s+)?"
        r"(?:states?|says?|shows?|mentions?|contains?|indicates?)\s+(?:that\s+)?"
    )
    _SOURCE_LINE = re.compile(r"(?im)^\s*source\s*:\s*(?:document|reference)\s*\d+\s*$")

    def __init__(self, *, enabled: bool = True):
        self.enabled = enabled
        self.pending = ""
        self.in_code = False

    @classmethod
    def _clean(cls, text: str) -> str:
        leading_label = bool(
            cls._PREFIX.match(text.lstrip()) or cls._REFERENCE_SUBJECT.match(text.lstrip())
            or cls._PAGE_SUBJECT.match(text.lstrip()) or cls._CONTEXT.match(text.lstrip())
            or cls._PAGE.match(text.lstrip())
        )
        text = cls._SOURCE_LINE.sub("", text)
        text = cls._PREFIX.sub("", text)
        text = cls._REFERENCE_SUBJECT.sub("", text)
        text = cls._PAGE_SUBJECT.sub("", text)
        text = cls._CONTEXT.sub("", text)
        text = cls._PAGE.sub("", text)
        text = cls._NUMBERED.sub("", text)
        text = re.sub(r"\s+([.,;:!?\u0964])", r"\1", text)
        text = re.sub(r"(?m)^\s*[,;:]\s*", "", text)
        if leading_label:
            text = re.sub(r"^([a-z])", lambda m: m.group(1).upper(), text, count=1)
        return text

    def feed(self, text: str, final: bool = False) -> str:
        if not self.enabled:
            return text
        self.pending += text
        output = []
        while self.pending:
            match = re.search(r"[.!?\u0964](?:\s|$)|\n", self.pending)
            if match:
                end = match.end()
            elif len(self.pending) > 400:
                # Keep enough tail to recognize a label split across chunks.
                end = len(self.pending) - 120
            elif final:
                end = len(self.pending)
            else:
                break
            segment = self.pending[:end]
            if segment.lstrip().startswith(chr(96) * 3):
                output.append(segment)
                self.in_code = not self.in_code
            else:
                output.append(segment if self.in_code else self._clean(segment))
            self.pending = self.pending[end:]
        return "".join(output)


def bounded_events(factory, timeout_seconds: float, heartbeat_seconds: float = 2.0,
                   request: ActiveRequest | None = None):
    """Bridge a blocking model iterator to bounded events with cancellation."""
    events = queue.Queue(maxsize=64)
    cancelled = request.cancelled if request else threading.Event()
    finished = threading.Event()

    def send(event):
        while not cancelled.is_set():
            try:
                events.put(event, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def produce():
        try:
            with closing(factory()) as iterator:
                for event in iterator:
                    if cancelled.is_set():
                        break
                    if not send(event) or event.get("type") in {"done", "error"}:
                        break
        except Exception as exc:
            if not cancelled.is_set():
                if request:
                    request.status = "failed"
                send({"type": "error", "message": str(exc) or type(exc).__name__})
        finally:
            finished.set()
            if request:
                registry.finish(request)

    started = time.monotonic()
    worker = threading.Thread(target=produce, daemon=True, name="qa-response")
    if request:
        request.worker = worker
    worker.start()
    stage = "Preparing your answer"
    completed = False
    try:
        while True:
            if cancelled.is_set():
                if request:
                    yield {"type": "cancelled", "request_id": request.request_id}
                return
            remaining = timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                yield {"type": "error", "message": f"Answer timed out after {timeout_seconds:g}s during: {stage}. Check Ollama or choose a smaller chat model in backend/config.py."}
                return
            try:
                event = events.get(timeout=min(heartbeat_seconds, remaining))
            except queue.Empty:
                if cancelled.is_set():
                    if request:
                        yield {"type": "cancelled", "request_id": request.request_id}
                    return
                if finished.is_set():
                    yield {"type": "error", "message": "The answer stream ended without a final response. Please retry."}
                    return
                yield {"type": "status", "message": f"{stage} ({time.monotonic() - started:.0f}s)"}
                continue
            if event.get("type") == "status":
                stage = event.get("message", stage)
            if cancelled.is_set():
                if request:
                    yield {"type": "cancelled", "request_id": request.request_id}
                return
            yield event
            if event.get("type") in {"done", "error"}:
                completed = event.get("type") == "done"
                return
    finally:
        if not completed:
            cancelled.set()
