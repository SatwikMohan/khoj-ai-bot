"""Ensure old cancellation watchers cannot outlive a completed speech job."""
import io
import struct
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from services.tts_service import PersistentPiperWorker


class WorkerReuseTests(unittest.TestCase):
    def test_watchdog_is_joined_before_successful_worker_reuse(self):
        payload = b"\x01audio"
        process = Mock(pid=123)
        process.poll.return_value = None
        process.stdout = io.BytesIO((struct.pack("<Q", len(payload)) + payload) * 2)
        process.stdin = io.BytesIO()
        watchers = []

        class Watcher:
            def __init__(self, **kwargs):
                self.joined = False
                watchers.append(self)
            def start(self):
                if len(watchers) > 1:
                    assert watchers[-2].joined
            def join(self):
                self.joined = True

        worker = PersistentPiperWorker(Path("voice.onnx"))
        cancelled = threading.Event()
        with patch("services.tts_service.subprocess.Popen", return_value=process) as spawn, patch(
            "services.tts_service.threading.Thread", Watcher
        ):
            self.assertEqual(worker.synthesize("First", 1.0, cancelled), b"audio")
            self.assertTrue(watchers[0].joined)
            cancelled.set()
            self.assertEqual(worker.synthesize("Second", 1.0, threading.Event()), b"audio")
            self.assertTrue(watchers[1].joined)
            spawn.assert_called_once()
            process.terminate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
