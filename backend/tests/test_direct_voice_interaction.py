"""Direct voice service lifecycle tests without loading a speech model."""
import unittest
from unittest.mock import patch

from services.request_lifecycle import registry
from services.voice_interaction import (
    begin_voice_interaction, cancel_interaction, transcribe_interaction,
)


class VoiceInteractionTests(unittest.TestCase):
    def tearDown(self):
        registry.cancel("voice-session-a")
        registry.cancel("voice-session-b")

    def test_transcription_uses_the_request_token_and_preserves_session_isolation(self):
        begin_voice_interaction("voice-session-a", "voice-a")
        begin_voice_interaction("voice-session-b", "voice-b")
        with patch("services.voice_interaction.transcribe_audio", return_value={"text": "hello"}) as transcribe:
            result = transcribe_interaction(b"audio", "audio/webm;codecs=opus", "voice-session-a", "voice-a")
        self.assertEqual(result["text"], "hello")
        self.assertEqual(transcribe.call_args.args[1:3], (".webm", None))
        self.assertIs(transcribe.call_args.args[3], registry.current("voice-session-a", "voice-a").cancelled)
        self.assertIsNotNone(registry.current("voice-session-b", "voice-b"))

    def test_new_request_cancels_previous_without_touching_another_session(self):
        previous = registry.begin("voice-session-a", "old")
        begin_voice_interaction("voice-session-b", "other")
        begin_voice_interaction("voice-session-a", "new")
        self.assertTrue(previous.cancelled.is_set())
        self.assertIsNone(registry.current("voice-session-a", "old"))
        self.assertIsNotNone(registry.current("voice-session-b", "other"))
        self.assertFalse(cancel_interaction("voice-session-a", "old"))
        self.assertTrue(cancel_interaction("voice-session-a", "new"))

    def test_stale_audio_is_rejected(self):
        begin_voice_interaction("voice-session-a", "old")
        begin_voice_interaction("voice-session-a", "new")
        with self.assertRaisesRegex(RuntimeError, "replaced"):
            transcribe_interaction(b"audio", "audio/webm", "voice-session-a", "old")


if __name__ == "__main__":
    unittest.main()
