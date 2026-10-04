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


class FakeResponse:
    ok = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def iter_lines(self, **_):
        yield 'data: {"type":"status","message":"Searching your documents"}'
        yield 'data: {"type":"token","text":"The limit is 100 kg."}'
        yield 'data: {"type":"done","answer":"The limit is 100 kg.","sources":[]}'


class InterfaceTests(unittest.TestCase):
    def test_two_column_chat_without_settings_and_streamed_reply(self):
        with patch.object(config, "TTS_ENABLED", False), patch("requests.post", return_value=FakeResponse()):
            app = AppTest.from_file(str(ROOT / "frontend" / "app.py"), default_timeout=10).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.columns), 2)
            self.assertEqual(len(app.chat_input), 1)
            self.assertEqual(len(app.slider), 0)
            self.assertEqual(len(app.selectbox), 0)
            app.chat_input[0].set_value("What is the limit?").run()
        self.assertFalse(app.exception)
        self.assertTrue(any("The limit is 100 kg." in str(item.value) for item in app.markdown))

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

