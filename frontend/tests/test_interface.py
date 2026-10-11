"""Frontend structure and streamed chat smoke tests."""
import sys
import threading
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "frontend"))
import config
from visualization import render_ai_visualization
from avatar_renderer import render_script, avatar_asset_manifest


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

    def test_streamlit_preloads_voice_once_for_direct_service_calls(self):
        warmed = threading.Event()
        import streamlit as st
        st.cache_resource.clear()
        with patch.object(config, "TTS_ENABLED", True), patch.object(
            config, "TTS_PRELOAD_VOICES", True
        ), patch.object(config, "AVATAR_ENABLED", False), patch(
            "services.tts_service.warm_up_tts", side_effect=warmed.set
        ) as warm:
            app = AppTest.from_file(str(ROOT / "frontend" / "app.py"), default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertTrue(warmed.wait(2))
            app.run()
        self.assertFalse(app.exception)
        self.assertEqual(warm.call_count, 1)

    def test_partial_answer_stays_visible_and_retry_does_not_duplicate_query(self):
        calls = []

        def fake_stream(**kwargs):
            calls.append(kwargs["question"])
            yield {"type": "token", "text": "The first verified detail is visible."}
            if len(calls) == 1:
                yield {"type": "error", "category": "generation_stalled",
                       "partial": True, "message": "Answer generation stalled."}
            else:
                yield {"type": "done", "answer": "The first verified detail is visible.",
                       "sources": []}

        with patch.object(config, "TTS_ENABLED", False), patch(
            "services.qa_service.stream_answer_events", side_effect=fake_stream
        ):
            app = AppTest.from_file(str(ROOT / "frontend" / "app.py"), default_timeout=10).run()
            app.chat_input[0].set_value("Explain ventilation").run()
            self.assertFalse(app.exception)
            self.assertTrue(any("The first verified detail is visible." in str(item.value)
                                for item in app.markdown))
            app.run()
            self.assertEqual(calls, ["Explain ventilation"])
            app.button(key="retry_incomplete_answer").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(calls, ["Explain ventilation", "Explain ventilation"])

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
            self.assertEqual(kwargs["question"], "Voice question")
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
            self.assertEqual(len(app.text_area), 0)
            self.assertTrue(any("Voice answer." in str(item.value) for item in app.markdown))
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(stream.call_count, 1)
        finally:
            if session_id:
                registry.cancel(session_id)

    def test_failed_transcription_does_not_submit_or_repeat_audio(self):
        from services.request_lifecycle import registry
        from services.stt_service import STTEngineError

        request_id = "voice-stt-failure"
        session = []
        calls = 0

        def voice_component(**kwargs):
            nonlocal calls
            calls += 1
            session.append(kwargs["session_id"])
            if calls == 1:
                return {"kind": "begin", "request_id": request_id}
            return {"kind": "audio", "request_id": request_id,
                    "audio_mime": "audio/webm", "audio_b64": "YXVkaW8="}

        try:
            with patch.object(config, "TTS_ENABLED", False), patch(
                "streamlit.components.v1.declare_component", return_value=voice_component
            ), patch(
                "services.voice_interaction.transcribe_audio",
                side_effect=STTEngineError("Recognition unavailable")
            ) as transcribe, patch(
                "services.qa_service.stream_answer_events"
            ) as stream:
                app = AppTest.from_file(str(ROOT / "frontend" / "app.py"), default_timeout=10).run()
                app.run()
            self.assertFalse(app.exception)
            self.assertEqual(transcribe.call_count, 1)
            self.assertEqual(stream.call_count, 0)
            self.assertTrue(any("Recognition unavailable" in str(item.value)
                                for item in app.error))
        finally:
            for session_id in set(session):
                registry.cancel(session_id)

    def test_missing_avatar_asset_shows_an_error(self):
        with patch.object(config, "AVATAR_MODEL_FILE", "assets/missing-avatar.gltf"):
            import_map, script = render_script()
        self.assertEqual(import_map, "{}")
        self.assertIn("3D avatar unavailable", script)
        self.assertIn("modelStatus", script)

    def test_avatar_resource_filename_case_is_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "Scene.gltf"
            model.write_text('{"asset":{"version":"2.0"}}', encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                avatar_asset_manifest(str(Path(folder) / "scene.gltf"))

    def test_avatar_rejects_git_lfs_pointer(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "scene.gltf"
            model.write_text("version https://git-lfs.github.com/spec/v1\\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Git LFS pointer"):
                avatar_asset_manifest(str(model))

    def test_avatar_model_and_every_external_resource_exist(self):
        path, model, resources = avatar_asset_manifest(config.AVATAR_MODEL_FILE)
        self.assertEqual(path.name, "scene.gltf")
        self.assertTrue(model.get("meshes"))
        self.assertGreater(len(resources), 1)
        self.assertTrue(all(item.stat().st_size > 0 for item in resources))

    def test_visualization_is_local_and_supports_audio_queue(self):
        with patch("visualization.st.iframe") as render:
            render_ai_visualization(queue_id="test-queue", request_id="test-request", session_id="test-session", thinking=True)
        html = render.call_args.args[0]
        self.assertIn("texmin:avatar-queue", html)
        self.assertIn("texmin_audio_owner:", html)
        self.assertIn("playback_ended", html)
        self.assertIn("item.requestId !== settings.requestId", html)
        self.assertIn("texmin_ai_activity", html)
        self.assertIn("texmin_voice_interrupt_at", html)
        self.assertNotIn("https://unpkg.com/three", html)
        self.assertIn("Invincible - Debbie Grayson", html)
        self.assertIn('id="avatarScene"', html)
        self.assertIn("data:model/gltf+json;base64,", html)
        self.assertIn("data:text/javascript;base64,", html)
        self.assertEqual(render.call_args.kwargs["height"], 620)


    def test_tts_timeout_stops_queued_synthesis_but_preserves_text(self):
        from services.request_lifecycle import registry

        started = threading.Event()
        release = threading.Event()
        session = []

        def fake_stream(**kwargs):
            session.append((kwargs["session_id"], kwargs["request_id"]))
            registry.begin(kwargs["session_id"], kwargs["request_id"])
            first = "The mine ventilation system moves clean air through every working area safely."
            second = "Workers inspect the fans and document hazards before the next shift begins."
            yield {"type": "token", "text": first}
            self.assertTrue(started.wait(1))
            yield {"type": "token", "text": " " + second}
            yield {"type": "done", "answer": first + " " + second, "sources": []}

        def slow_speech(_request):
            started.set()
            release.wait(1)
            return b"", "audio/wav"

        try:
            with patch.object(config, "TTS_ENABLED", True), patch.object(
                config, "TTS_PRELOAD_VOICES", False
            ), patch.object(config, "AVATAR_ENABLED", False), patch.object(
                config, "TTS_MAX_PENDING_JOBS", 1
            ), patch.object(config, "TTS_RESPONSE_TIMEOUT_SECONDS", 0.03), patch(
                "services.qa_service.stream_answer_events", side_effect=fake_stream
            ), patch("services.tts_service.synthesize_speech", side_effect=slow_speech) as speech:
                app = AppTest.from_file(str(ROOT / "frontend" / "app.py"), default_timeout=10).run()
                app.chat_input[0].set_value("Explain ventilation").run()
                self.assertFalse(app.exception)
                self.assertTrue(any("Workers inspect the fans" in str(item.value)
                                    for item in app.markdown))
                self.assertEqual(speech.call_count, 1)
                active = registry.current(*session[0])
                self.assertTrue(active.speech_cancelled.is_set())
        finally:
            release.set()
            for session_id, _ in session:
                registry.cancel(session_id)

if __name__ == "__main__":
    unittest.main()
