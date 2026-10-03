"""Voice and text share one session owner from recording through the answer."""
import unittest
import uuid
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app
from services.request_lifecycle import registry
from services.speech_chunker import split_speakable_prefix


class VoiceLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.session = "voice-test-" + uuid.uuid4().hex

    def begin(self, request_id):
        return self.client.post(
            f"/qa/sessions/{self.session}/interactions",
            json={"request_id": request_id},
        )

    def upload(self, request_id):
        return self.client.post(
            "/stt/transcribe",
            data={"session_id": self.session, "request_id": request_id},
            files={"audio": ("recording.webm", b"valid-placeholder", "audio/webm")},
        )

    def test_new_recording_replaces_old_one_and_isolates_sessions(self):
        self.assertEqual(self.begin("first").status_code, 200)
        other = registry.begin("other-" + self.session, "other", input_type="audio")
        old = registry.current(self.session, "first")
        self.assertEqual(self.begin("second").status_code, 200)
        self.assertTrue(old.cancelled.is_set())
        self.assertIsNone(registry.current(self.session, "first"))
        self.assertTrue(registry.is_current(other))
        self.assertEqual(self.upload("first").status_code, 409)

    def test_stale_transcription_cannot_become_a_query(self):
        self.begin("voice")
        with patch("routes.stt_routes.transcribe_audio") as transcribe:
            def replace(*args):
                registry.begin(self.session, "typed")
                return {"text": "stale"}
            transcribe.side_effect = replace
            self.assertEqual(self.upload("voice").status_code, 409)
        self.assertEqual(registry.current(self.session, "typed").input_type, "text")
        self.assertIsNone(registry.promote_voice(self.session, "voice"))

    def test_live_transcription_preserves_request_for_qa(self):
        self.begin("voice")
        self.assertIsNone(registry.promote_voice(self.session, "voice"))
        with patch("routes.stt_routes.transcribe_audio", return_value={"text": "Mujhe Python samjhao"}) as transcribe:
            response = self.upload("voice")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "Mujhe Python samjhao")
        self.assertEqual(transcribe.call_args.args[1:3], (".webm", None))
        self.assertIs(registry.promote_voice(self.session, "voice"), registry.current(self.session, "voice"))
        self.assertIsNone(registry.promote_voice(self.session, "voice"))

    def test_mismatched_audio_format_fails_before_model(self):
        self.begin("voice")
        response = self.client.post(
            "/stt/transcribe",
            data={"session_id": self.session, "request_id": "voice"},
            files={"audio": ("recording.wav", b"anything", "audio/webm")},
        )
        self.assertEqual(response.status_code, 415)

    def test_long_speech_chunk_waits_for_word_boundary(self):
        word = "\u0905\u0902\u0924\u0930\u0930\u093e\u0937\u094d\u091f\u094d\u0930\u0940\u092f\u0915\u0930\u0923" * 30
        segments, remainder = split_speakable_prefix(word)
        self.assertEqual(segments, [])
        self.assertEqual(remainder, word)


if __name__ == "__main__":
    unittest.main()
