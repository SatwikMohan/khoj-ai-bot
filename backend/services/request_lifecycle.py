"""Latest request ownership within one conversation session."""

import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class ActiveRequest:
    session_id: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    input_type: str = "text"
    created_at: float = field(default_factory=time.time)
    status: str = "processing"
    cancelled: threading.Event = field(default_factory=threading.Event)
    worker: threading.Thread | None = None

    def stop(self) -> None:
        self.cancelled.set()
        self.status = "cancelled"


class RequestRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._active: dict[str, ActiveRequest] = {}

    def begin(self, session_id: str, request_id: str | None = None, *, input_type: str = "text", status: str = "processing") -> ActiveRequest:
        request = ActiveRequest(session_id=session_id, request_id=request_id or uuid.uuid4().hex, input_type=input_type, status=status)
        with self._lock:
            self._active = {key: value for key, value in self._active.items()
                            if value.status not in {"completed", "cancelled"} or time.time() - value.created_at < 3600}
            previous = self._active.get(session_id)
            if previous:
                previous.stop()
            self._active[session_id] = request
        return request

    def promote_voice(self, session_id: str, request_id: str) -> ActiveRequest | None:
        """Keep the voice interaction's cancellation token through STT, QA and TTS."""
        with self._lock:
            request = self._active.get(session_id)
            if request is None or request.request_id != request_id or request.input_type != "audio" or request.cancelled.is_set() or request.status != "transcribed":
                return None
            request.status = "processing"
            return request

    def set_status(self, request: ActiveRequest, status: str) -> bool:
        with self._lock:
            if self._active.get(request.session_id) is not request or request.cancelled.is_set():
                return False
            request.status = status
            return True

    def is_current(self, request: ActiveRequest) -> bool:
        with self._lock:
            return self._active.get(request.session_id) is request and not request.cancelled.is_set()

    def cancel(self, session_id: str, request_id: str | None = None) -> bool:
        with self._lock:
            request = self._active.get(session_id)
            if request is None or request_id and request.request_id != request_id:
                return False
            request.stop()
            self._active.pop(session_id, None)
            return True

    def current(self, session_id: str, request_id: str) -> ActiveRequest | None:
        with self._lock:
            request = self._active.get(session_id)
            return request if request and request.request_id == request_id and not request.cancelled.is_set() else None

    def finish(self, request: ActiveRequest) -> None:
        with self._lock:
            if self._active.get(request.session_id) is request and request.session_id.startswith("anonymous-"):
                self._active.pop(request.session_id, None)
            if not request.cancelled.is_set() and request.status != "failed":
                request.status = "completed"


registry = RequestRegistry()
