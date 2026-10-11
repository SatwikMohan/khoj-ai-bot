"""Bounded response delivery and incremental removal of explicit reasoning tags."""

import json
import queue
import re
import threading
import time
from contextlib import closing

import config
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
            elif len(self.pending) > 200:
                # Keep enough tail to recognize a label split across chunks.
                end = len(self.pending) - 80
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


class StreamProgress:
    """Producer clock; browser and TTS backpressure never count as model work."""

    def __init__(self):
        self.lock = threading.Lock()
        self.stage = "admission"
        self.stage_started = self.last_advance = time.monotonic()
        self.generation_started = None
        self.generation_finished = None
        self.blocked_total = 0.0
        self.blocked_since = None
        self.stage_blocked = self.advance_blocked = self.generation_blocked = 0.0

    def _blocked(self, now):
        return self.blocked_total + (now - self.blocked_since if self.blocked_since else 0.0)

    def mark(self, stage: str):
        with self.lock:
            now = time.monotonic()
            blocked = self._blocked(now)
            self.stage = stage
            self.stage_started = self.last_advance = now
            self.stage_blocked = self.advance_blocked = blocked
            if stage == "first_token":
                if self.generation_started is None:
                    self.generation_started = now
                    self.generation_blocked = blocked
                self.generation_finished = None
            elif stage == "finalizing" and self.generation_started is not None:
                self.generation_finished = max(
                    0.0, now - self.generation_started - blocked + self.generation_blocked
                )

    def advance(self):
        with self.lock:
            now = time.monotonic()
            blocked = self._blocked(now)
            if self.stage == "first_token":
                self.stage = "generation"
                self.stage_started = now
                self.stage_blocked = blocked
            self.last_advance = now
            self.advance_blocked = blocked

    def block(self):
        with self.lock:
            if self.blocked_since is None:
                self.blocked_since = time.monotonic()

    def unblock(self):
        with self.lock:
            if self.blocked_since is not None:
                self.blocked_total += time.monotonic() - self.blocked_since
                self.blocked_since = None

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            blocked = self._blocked(now)
            generation_s = (
                self.generation_finished if self.generation_finished is not None
                else max(0.0, now - self.generation_started - blocked + self.generation_blocked)
                if self.generation_started is not None else 0.0
            )
            return {
                "stage": self.stage,
                "stage_s": max(0.0, now - self.stage_started - blocked + self.stage_blocked),
                "idle_s": max(0.0, now - self.last_advance - blocked + self.advance_blocked),
                "generation_s": generation_s,
            }


_WORKER_SLOTS = threading.BoundedSemaphore(max(1, int(config.QA_MAX_ACTIVE_WORKERS)))


def _failure_category(stage: str, exc: Exception) -> str:
    detail = str(exc).lower()
    if any(word in detail for word in ("out of memory", "allocate buffer", "resource exhausted")):
        return "resource_exhaustion"
    if stage in {"model_init", "first_token"} and any(
        word in detail for word in ("connection refused", "not found", "unavailable", "connect")
    ):
        return "model_unavailable"
    return "retrieval_failed" if stage == "retrieval" else "generation_failed"


def bounded_events(factory, timeout_seconds: float, heartbeat_seconds: float = 2.0,
                   request: ActiveRequest | None = None, *,
                   stage_timeouts: dict[str, float] | None = None,
                   progress: StreamProgress | None = None):
    """Bridge blocking inference to bounded events with producer-stage deadlines.

    The legacy timeout_seconds remains a wall-clock limit only when no stage
    policy is supplied. QA supplies separate retrieval, model and generation
    limits. Queue backpressure is excluded from model clocks.
    """
    events = queue.Queue(maxsize=64)
    cancelled = request.cancelled if request else threading.Event()
    finished = threading.Event()
    progress = progress or StreamProgress()
    stage_label = "Preparing your answer"
    started = time.monotonic()
    token_count = 0

    def send(event):
        while not cancelled.is_set():
            try:
                events.put(event, timeout=0.1)
                progress.unblock()
                return True
            except queue.Full:
                progress.block()
        progress.unblock()
        return False

    def produce():
        acquired = False
        terminal_kind = None
        try:
            while not cancelled.is_set():
                acquired = _WORKER_SLOTS.acquire(timeout=0.1)
                if acquired:
                    break
            if not acquired:
                return
            with closing(factory()) as iterator:
                for event in iterator:
                    if cancelled.is_set():
                        break
                    if event.get("type") in {"done", "error"}:
                        terminal_kind = event["type"]
                    if not send(event) or terminal_kind is not None:
                        break
        except Exception as exc:
            if not cancelled.is_set():
                terminal_kind = "error"
                snapshot = progress.snapshot()
                category = _failure_category(snapshot["stage"], exc)
                if request:
                    registry.fail(request, snapshot["stage"], exc)
                print(json.dumps({
                    "event": "qa_stream_failed", "request_id": request.request_id if request else None,
                    "category": category, "stage": snapshot["stage"],
                    "error": str(exc)[:400],
                }), flush=True)
                send({"type": "error", "category": category,
                      "message": f"{category.replace('_', ' ').capitalize()}: {exc}"})
        finally:
            if acquired:
                _WORKER_SLOTS.release()
            if request:
                if terminal_kind == "done" and not cancelled.is_set():
                    registry.finish(request)
                elif not cancelled.is_set() and request.status != "failed":
                    registry.fail(request, progress.snapshot()["stage"],
                                  RuntimeError("The answer stream ended without completion."))
            finished.set()

    worker = threading.Thread(target=produce, daemon=True, name="qa-response")
    if request:
        request.worker = worker
    worker.start()
    completed = False
    try:
        while True:
            if cancelled.is_set():
                if request:
                    yield {"type": "cancelled", "request_id": request.request_id}
                return
            try:
                if (stage_timeouts is not None and
                        progress.snapshot()["generation_s"] >= stage_timeouts["total_generation"]):
                    raise queue.Empty
                event = events.get_nowait()
            except queue.Empty:
                snapshot = progress.snapshot()
                category = None
                limit = None
                if stage_timeouts is None:
                    if time.monotonic() - started >= timeout_seconds:
                        category, limit = "response_timeout", timeout_seconds
                else:
                    if snapshot["generation_s"] >= stage_timeouts["total_generation"]:
                        category, limit = "generation_deadline", stage_timeouts["total_generation"]
                    elif snapshot["stage"] == "generation" and snapshot["idle_s"] >= stage_timeouts["inter_token"]:
                        category, limit = "generation_stalled", stage_timeouts["inter_token"]
                    elif snapshot["stage_s"] >= stage_timeouts.get(snapshot["stage"], float("inf")):
                        category, limit = {
                            "admission": "worker_queue_timeout",
                            "retrieval": "retrieval_timeout",
                            "model_init": "model_init_timeout",
                            "first_token": "first_token_timeout",
                            "finalizing": "generation_stalled",
                        }.get(snapshot["stage"], "stage_timeout"), stage_timeouts[snapshot["stage"]]
                if category:
                    message = {
                        "worker_queue_timeout": "Answer worker queue was full.",
                        "retrieval_timeout": "Document search timed out.",
                        "model_init_timeout": "Model initialization timed out.",
                        "first_token_timeout": "The model did not produce a first token in time.",
                        "generation_stalled": "Answer generation stalled.",
                        "generation_deadline": "Answer generation exceeded its total deadline.",
                        "response_timeout": "Answer request timed out.",
                    }.get(category, "Answer stage timed out.")
                    if request:
                        registry.fail(request, snapshot["stage"], TimeoutError(message))
                    print(json.dumps({
                        "event": "qa_stream_timeout", "request_id": request.request_id if request else None,
                        "category": category, "stage": snapshot["stage"],
                        "limit_s": limit, "stage_elapsed_s": round(snapshot["stage_s"], 2),
                        "generation_elapsed_s": round(snapshot["generation_s"], 2),
                        "partial": token_count > 0,
                    }), flush=True)
                    yield {"type": "error", "category": category, "partial": token_count > 0,
                           "message": message + (" Generation stopped early; the answer is incomplete." if token_count else "")}
                    return
                if finished.is_set():
                    yield {"type": "error", "category": "stream_ended",
                           "partial": token_count > 0,
                           "message": "The answer stream ended without a final response. Please retry."}
                    return
                try:
                    event = events.get(timeout=max(0.01, min(heartbeat_seconds, 0.25)))
                except queue.Empty:
                    yield {"type": "status", "message": f"{stage_label} ({time.monotonic() - started:.0f}s)"}
                    continue
            # A blocking queue read can return just after the deadline.
            # The next loop emits the stage-specific error instead of a late done.
            if (stage_timeouts is not None and
                    progress.snapshot()["generation_s"] >= stage_timeouts["total_generation"]):
                continue
            if event.get("type") == "status":
                stage_label = event.get("message", stage_label)
            if cancelled.is_set():
                if request:
                    yield {"type": "cancelled", "request_id": request.request_id}
                return
            if event.get("type") == "token":
                token_count += 1
            if event.get("type") == "error":
                event.setdefault("partial", token_count > 0)
            yield event
            if event.get("type") in {"done", "error"}:
                completed = event.get("type") == "done"
                return
    finally:
        if not completed:
            cancelled.set()
