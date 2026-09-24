"""Bounded response delivery and incremental removal of explicit reasoning tags."""

import queue
import threading
import time
from contextlib import closing


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


# Hold this permit until the actual worker finishes, even after a UI timeout.
# Repeated requests must not accumulate blocked inference workers.
RESPONSE_SLOT = threading.BoundedSemaphore(1)


def bounded_events(factory, timeout_seconds: float, heartbeat_seconds: float = 2.0):
    if not RESPONSE_SLOT.acquire(blocking=False):
        yield {"type": "error", "message": "A previous answer is still running. Please wait before retrying."}
        return
    events = queue.Queue(maxsize=64)
    cancelled = threading.Event()
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
                    if not send(event) or event.get("type") in {"done", "error"}:
                        break
        except Exception as exc:
            send({"type": "error", "message": str(exc) or type(exc).__name__})
        finally:
            finished.set()
            RESPONSE_SLOT.release()

    started = time.monotonic()
    threading.Thread(target=produce, daemon=True, name="qa-response").start()
    stage = "Preparing your answer"
    try:
        while True:
            remaining = timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                yield {"type": "error", "message": f"Answer timed out after {timeout_seconds:g}s during: {stage}. Check Ollama or choose a smaller chat model in backend/config.py."}
                return
            try:
                event = events.get(timeout=min(heartbeat_seconds, remaining))
            except queue.Empty:
                if finished.is_set():
                    yield {"type": "error", "message": "The answer stream ended without a final response. Please retry."}
                    return
                yield {"type": "status", "message": f"{stage} ({time.monotonic() - started:.0f}s)"}
                continue
            if event.get("type") == "status":
                stage = event.get("message", stage)
            yield event
            if event.get("type") in {"done", "error"}:
                return
    finally:
        cancelled.set()
