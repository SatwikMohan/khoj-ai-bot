"""Frontend structure and streamed chat smoke tests."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "frontend"))
import config
from visualization import render_ai_visualization
from avatar_renderer import render_script


class InterfaceTests(unittest.TestCase):
    def test_two_column_chat_without_settings_and_streamed_reply(self):
        def fake_stream(**kwargs):
            self.assertEqual(kwargs["question"], "What is the limit?")
            yield {"type": "status", "message": "Searching your documents"}
            yield {"type": "token", "text": "The limit is "}
            yield {"type": "token", "text": "100 kg."}
            yield {"type": "done", "answer": "The limit is 100 kg.", "sources": []}

        with patch.object(config, "TTS_ENABLED", False), patch(
            "services.qa_service.stream_answer_events", side_effect=fake_stream
        ) as stream:
            app = AppTest.from_file(str(ROOT / "frontend" / "app.py"), default_timeout=10).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.columns), 2)
            self.assertEqual(len(app.chat_input), 1)
            self.assertEqual(len(app.slider), 0)
            self.assertEqual(len(app.selectbox), 0)
            app.chat_input[0].set_value("What is the limit?").run()
        self.assertFalse(app.exception)
        self.assertEqual(stream.call_count, 1)
        self.assertTrue(any("The limit is 100 kg." in str(item.value) for item in app.markdown))

    def test_microphone_audio_reaches_direct_stt_and_qa_services(self):
        from services.request_lifecycle import registry

        seen = []
        request_id = "voice-direct-test"
        session_id = None

        def voice_component(**kwargs):
            nonlocal session_id
            session_id = kwargs["session_id"]
            seen.append(kwargs["accepted_request_id"])
            if len(seen) == 1:
                return {"kind": "begin", "request_id": request_id}
            return {
                "kind": "audio", "request_id": request_id,
                "audio_mime": "audio/webm", "audio_b64": "YXVkaW8=",
            }

        def fake_stream(**kwargs):
            self.assertEqual(kwargs["input_type"], "audio")
            self.assertEqual(kwargs["request_id"], request_id)
            yield {"type": "token", "text": "Voice answer."}
            yield {"type": "done", "answer": "Voice answer.", "sources": []}

        try:
            with patch.object(config, "TTS_ENABLED", False), patch(
                "streamlit.components.v1.declare_component", return_value=voice_component
            ), patch(
                "services.voice_interaction.transcribe_audio", return_value={"text": "Voice question"}
            ) as transcribe, patch(
                "services.qa_service.stream_answer_events", side_effect=fake_stream
            ) as stream:
                app = AppTest.from_file(str(ROOT / "frontend" / "app.py"), default_timeout=10).run()
            self.assertFalse(app.exception)
            self.assertEqual(seen[:2], ["", request_id])
            self.assertEqual(transcribe.call_count, 1)
            self.assertEqual(stream.call_count, 1)
            self.assertTrue(any("Voice answer." in str(item.value) for item in app.markdown))
        finally:
            if session_id:
                registry.cancel(session_id)

    def test_missing_avatar_asset_shows_an_error(self):
        with patch.object(config, "AVATAR_MODEL_FILE", "assets/missing-avatar.gltf"):
            import_map, script = render_script()
        self.assertEqual(import_map, "{}")
        self.assertIn("Avatar renderer unavailable", script)
        self.assertIn("modelStatus", script)

    def test_visualization_is_local_and_supports_audio_queue(self):
        with patch("visualization.components.html") as render:
            render_ai_visualization(queue_id="test-queue", thinking=True)
        html = render.call_args.args[0]
        self.assertIn("texmin:avatar-queue", html)
        self.assertIn("texmin_ai_activity", html)
        self.assertIn("texmin_voice_interrupt_at", html)
        self.assertNotIn("https://unpkg.com/three", html)
        self.assertIn("male04 face rigged", html)
        self.assertIn('id="avatarScene"', html)
        self.assertIn("data:model/gltf+json;base64,", html)
        self.assertIn("data:text/javascript;base64,", html)
        self.assertEqual(render.call_args.kwargs["height"], 620)


if __name__ == "__main__":
    unittest.main()

