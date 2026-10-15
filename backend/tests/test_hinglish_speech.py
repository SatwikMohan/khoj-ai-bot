"""Speech conversion keeps written answers intact and never rewrites quantities."""
import io
import json
import unittest
from unittest.mock import patch
import config
from helpers.request_models import TTSRequest
from services.hinglish_speech import devanagari_speech
from services.language_service import resolve_language_context
from services.tts_service import synthesize_speech, TTSEngineError


MUJHE = "".join(map(chr, [0x92e, 0x941, 0x91d, 0x947]))
BATAO = "".join(map(chr, [0x92c, 0x924, 0x93e, 0x913]))
AUR = "".join(map(chr, [0x914, 0x930]))


class HinglishSpeechTests(unittest.TestCase):
    def setUp(self):
        flags = patch.multiple(config, OFFLINE_MODE=True, TTS_ENGINE="piper",
                               TTS_HINGLISH_TRANSLITERATION=True, RESPONSE_LANGUAGE="auto")
        flags.start()
        self.addCleanup(flags.stop)
        metadata = patch("services.hinglish_speech.thinking_setting", return_value=False)
        metadata.start()
        self.addCleanup(metadata.stop)

    def convert(self, text, words, **extra):
        body = {"done": True, "message": {"content": json.dumps({"words": words})}, **extra}
        with patch("services.hinglish_speech.urllib.request.urlopen",
                   return_value=io.BytesIO(json.dumps(body).encode())) as call:
            result = devanagari_speech(text)
        sent = json.loads(call.call_args.args[0].data)
        self.assertFalse(sent["stream"])
        self.assertEqual(sent["model"], config.OLLAMA_CHAT_MODEL)
        self.assertEqual(call.call_args.kwargs["timeout"], config.TTS_TRANSLITERATION_TIMEOUT_SECONDS)
        return result

    def test_preserves_quantities_units_acronyms_and_punctuation(self):
        output = self.convert("Mujhe 100.5 kg, GPU aur B12 batao!",
                              [MUJHE, BATAO, BATAO, AUR, BATAO, BATAO])
        self.assertEqual(output, f"{MUJHE} 100.5 kg, GPU {AUR} B12 {BATAO}!")

    def test_invalid_conversions_are_rejected(self):
        for words in ([], ["mujhe", "batao"], [MUJHE, "do something else"],
                      [MUJHE, chr(0x967)], [MUJHE, BATAO + " 100"]):
            with self.subTest(words=words), self.assertRaises(ValueError):
                self.convert("mujhe batao", words)

    def test_truncated_conversion_is_rejected(self):
        with self.assertRaises(ValueError):
            self.convert("mujhe", [MUJHE], done_reason="length")

    def test_english_segment_can_stay_unchanged(self):
        self.assertEqual(self.convert("Check the safety valve.", ["Check", "the", "safety", "valve"]),
                         "Check the safety valve.")

    def test_timeout_reports_conversion_failure(self):
        with patch("services.hinglish_speech.urllib.request.urlopen", side_effect=TimeoutError):
            with self.assertRaisesRegex(ValueError, "written answer"):
                devanagari_speech("mujhe batao")

    def test_hinglish_answer_stays_roman_but_piper_receives_hindi(self):
        original = TTSRequest(text="Mujhe batao.", language="hinglish")
        hindi = MUJHE + " " + BATAO + "."
        self.assertEqual(resolve_language_context(original.text).target_language, "hinglish")
        for engine in ("piper", "goonj"):
            with self.subTest(engine=engine), patch.object(config, "TTS_ENGINE", engine), patch(
                "services.hinglish_speech.devanagari_speech", return_value=hindi
            ), patch("services.tts_service._synthesize_with_engine", return_value=(b"audio", "audio/wav")) as synth, patch(
                "services.tts_service._validate_audio"
            ), patch("services.tts_service._voice_circuit_open", return_value=False):
                synthesize_speech(original)
                self.assertEqual(synth.call_args.args[0], engine)
                self.assertEqual(synth.call_args.args[1].language, "hi")
                self.assertEqual(synth.call_args.args[1].text, hindi)
        self.assertEqual(original.text, "Mujhe batao.")
        self.assertEqual(original.language, "hinglish")

    def test_conversion_error_never_reaches_piper(self):
        with patch("services.hinglish_speech.devanagari_speech", side_effect=ValueError("Conversion failed")), patch(
            "services.tts_service._synthesize_with_engine"
        ) as synth:
            with self.assertRaises(TTSEngineError):
                synthesize_speech(TTSRequest(text="Mujhe batao.", language="hinglish"))
        synth.assert_not_called()

    def test_cancelled_conversion_never_reaches_synthesis(self):
        from services.request_lifecycle import registry
        active = registry.begin("conversion-test")
        request = TTSRequest(text="Mujhe batao.", language="hinglish",
                             session_id=active.session_id, request_id=active.request_id)
        def convert(_text):
            active.speech_cancelled.set()
            return MUJHE + " " + BATAO
        try:
            with patch("services.hinglish_speech.devanagari_speech", side_effect=convert), patch(
                "services.tts_service._synthesize_with_engine"
            ) as synth:
                with self.assertRaisesRegex(TTSEngineError, "cancelled"):
                    synthesize_speech(request)
            synth.assert_not_called()
        finally:
            registry.cancel(active.session_id)
