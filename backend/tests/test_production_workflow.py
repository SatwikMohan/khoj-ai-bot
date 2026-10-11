"""Regression checks for real audio decoding and per-request language control."""
import io
import math
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableGenerator

import config
from helpers.request_models import TTSRequest
from services.language_service import (
    answer_language_matches, resolve_language_context,
)
from services.qa_service import stream_answer_events, _general_response, _not_enough_context_response
from services.request_lifecycle import registry, DuplicateRequestError
from services.stt_service import STTEngineError, _decode_audio
from services.tts_service import TTSEngineError, synthesize_speech


class LanguageContractTests(unittest.TestCase):
    def test_current_query_and_override_own_language(self):
        english = resolve_language_context("What is ventilation?")
        hindi = resolve_language_context("old English", transcript="\u092e\u0941\u091d\u0947 \u092c\u0924\u093e\u0913")
        mixed = resolve_language_context("CMR ka structure kya hai?")
        selected = resolve_language_context("What is ventilation?", selected_language="hi")
        self.assertEqual([english.target_language, hindi.target_language,
                          mixed.target_language, selected.target_language],
                         ["en", "hi", "hinglish", "hi"])
        self.assertEqual(hindi.script, "Devanagari")

    def test_obvious_wrong_language_is_rejected(self):
        self.assertFalse(answer_language_matches("The storage limit is 100 kg.", "hi"))
        self.assertFalse(answer_language_matches(
            "La estructura del cat?logo contiene los detalles de seguridad para todos los trabajadores.",
            "en",
        ))
        self.assertTrue(answer_language_matches(
            "\u092d\u0902\u0921\u093e\u0930\u0923 \u0915\u0940 \u0938\u0940\u092e\u093e 100 kg \u0939\u0948\u0964", "hi"))
        self.assertFalse(answer_language_matches("The storage limit is 100 kg.", "hinglish"))
        self.assertTrue(answer_language_matches("Storage limit kya hai? Mujhe batao.", "hinglish"))

    def test_hinglish_shortcuts_preserve_roman_script(self):
        self.assertTrue(answer_language_matches(
            _general_response("aap kaise ho").answer, "hinglish"))
        self.assertTrue(answer_language_matches(
            _not_enough_context_response("CMR ka rule kya hai?").answer, "hinglish"))

    def test_wrong_language_draft_is_replaced_once(self):
        calls = []
        def generate(inputs):
            list(inputs)
            calls.append(1)
            if len(calls) == 1:
                yield AIMessageChunk(content="The storage limit is 100 kg.")
            else:
                yield AIMessageChunk(content="\u092d\u0902\u0921\u093e\u0930\u0923 \u0915\u0940 \u0938\u0940\u092e\u093e 100 kg \u0939\u0948\u0964")
        model = RunnableGenerator(generate)
        with patch.multiple(config, RESPONSE_LANGUAGE="auto", QA_RESPONSE_TIMEOUT_SECONDS=5), patch(
            "services.qa_service._load_environment"
        ), patch("services.qa_service._vector_store"), patch(
            "services.qa_service._retrieve_context",
            return_value=[Document(page_content="The storage limit is 100 kg.",
                                   metadata={"source": "rules.pdf"})]
        ), patch("services.qa_service._llm", return_value=model):
            events = list(stream_answer_events(
                "\u092d\u0902\u0921\u093e\u0930\u0923 \u0915\u0940 \u0938\u0940\u092e\u093e \u0915\u094d\u092f\u093e \u0939\u0948?"
            ))
        self.assertEqual(len(calls), 2)
        self.assertEqual([event["type"] for event in events].count("replace"), 0)
        self.assertNotIn("The storage limit", "".join(
            event.get("text", "") for event in events if event["type"] == "token"
        ))
        self.assertEqual(events[-1]["language"], "hi")
        self.assertTrue(answer_language_matches(events[-1]["answer"], "hi"))


class RequestOwnershipTests(unittest.TestCase):
    def test_duplicate_request_id_cannot_start_twice(self):
        session = "duplicate-regression"
        try:
            original = registry.begin(session, "same-id")
            with self.assertRaises(DuplicateRequestError):
                registry.begin(session, "same-id")
            self.assertIs(registry.current(session, "same-id"), original)
            replacement = registry.begin(session, "new-id")
            self.assertTrue(original.cancelled.is_set())
            self.assertIs(registry.current(session, "new-id"), replacement)
        finally:
            registry.cancel(session)


class AudioBoundaryTests(unittest.TestCase):
    def test_stereo_48khz_wav_is_decoded_as_mono_16khz(self):
        rate = 48000
        t = np.arange(rate, dtype=np.float64) / rate
        signal = np.where((t > .2) & (t < .7), np.sin(2 * math.pi * 440 * t) * .35, 0)
        stereo = np.column_stack((signal, signal * .8))
        data = (stereo * 32767).astype("<i2").tobytes()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "speech.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(rate)
                wav.writeframes(data)
            audio = _decode_audio(str(path))
        self.assertGreater(len(audio), 15500)
        self.assertLess(len(audio), 16500)
        self.assertGreater(float(np.max(np.abs(audio))), .1)

    def test_invalid_audio_is_not_sent_to_stt(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "broken.wav"
            path.write_bytes(b"not a wave")
            with self.assertRaisesRegex(STTEngineError, "decode"):
                _decode_audio(str(path))


class SpeechSupportTests(unittest.TestCase):
    def test_piper_declines_hinglish_without_fake_english_speech(self):
        with patch.object(config, "TTS_ENGINE", "piper"):
            with self.assertRaisesRegex(TTSEngineError, "no validated voice"):
                synthesize_speech(TTSRequest(
                    text="Mujhe safety rules samjhao.", language="hinglish",
                    response_format="wav",
                ))

    def test_veena_declines_unsupported_language_before_loading_model(self):
        with patch.object(config, "TTS_ENGINE", "veena"):
            with self.assertRaisesRegex(TTSEngineError, "no validated voice"):
                synthesize_speech(TTSRequest(
                    text="Bonjour tout le monde.", language="fr", response_format="wav",
                ))


if __name__ == "__main__":
    unittest.main()
