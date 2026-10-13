"""Language-aware offline speech routing and WAV continuity."""

import io
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from helpers.request_models import TTSRequest
from services.audio_chunks import join_speech_segments
from services.speech_language import speech_phrases, normalize_speech_text
from services.language_service import detect_language
from services.tts_service import _markdown_to_spoken_text, _synthesize_piper_speech
from services.request_lifecycle import registry


def wav_sample(value: int, rate: int = 22050) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(value.to_bytes(2, "little", signed=True) * (rate // 5))
    return output.getvalue()


class LanguageAwareSpeechTests(unittest.TestCase):
    def test_shared_speaker_preserves_english_words(self):
        from services.tts_service import _piper_path
        with patch.object(config, "PIPER_SHARED_VOICE", "hi_IN-pratham-medium"):
            self.assertEqual(_piper_path("en"), _piper_path("hi"))
            self.assertEqual(_piper_path("en").stem, "hi_IN-pratham-medium")
        self.assertEqual(speech_phrases("Check the safety valve.", "en"),
                         [("en", "Check the safety valve.")])

    def test_separate_speakers_remain_configurable(self):
        from services.tts_service import _piper_path
        with patch.object(config, "PIPER_SHARED_VOICE", ""):
            self.assertNotEqual(_piper_path("en"), _piper_path("hi"))

    def test_hinglish_keeps_one_speaker_and_original_words(self):
        text = "Yaar, mujhe kal office jaana hai. Can you remind me at 9 AM?"
        self.assertEqual(speech_phrases(text, "hinglish"), [("en", text)])
        self.assertEqual(speech_phrases("The system works hai and here is the result.", "hinglish"),
                         [("en", "The system works hai and here is the result.")])

    def test_short_roman_query_is_identified_as_hinglish(self):
        self.assertEqual(detect_language("Yaar, kal office jaana hai"), "hinglish")

    def test_hindi_with_english_terms_keeps_one_hindi_speaker(self):
        text = "\u092e\u0941\u091d\u0947 \u0906\u091c Python project \u0915\u0930\u0928\u093e \u0939\u0948\u0964"
        self.assertEqual(speech_phrases(text, "hi"), [("hi", text)])

    def test_truncated_hindi_chunk_ends_with_readable_hindi(self):
        spoken = _markdown_to_spoken_text("\u0928\u092e\u0938\u094d\u0924\u0947 \u0906\u092a \u0915\u0948\u0938\u0947 \u0939\u0948\u0902", 1)
        self.assertIn("\u092e\u0948\u0902 \u092f\u0939\u093e\u0901", spoken)
        self.assertNotIn("?", spoken)

    def test_speech_only_numbers_dates_and_abbreviations(self):
        spoken = normalize_speech_text("GPU API \u0915\u0940 fee \u20b9500 \u0939\u0948 on 2026-10-04 at 9 AM", "hi")
        self.assertIn("G P U A P I", spoken)
        self.assertIn("500 \u0930\u0941\u092a\u092f\u0947", spoken)
        self.assertIn("4 \u0905\u0915\u094d\u091f\u0942\u092c\u0930 2026", spoken)
        self.assertIn("9 A M", spoken)

    def test_plain_english_and_other_language_do_not_get_hindi_voice(self):
        self.assertEqual(speech_phrases("Can you remind me at 9 AM?", "en"),
                         [("en", "Can you remind me at 9 AM?")])
        self.assertEqual(speech_phrases("Bonjour tout le monde.", "fr"),
                         [("fr", "Bonjour tout le monde.")])

    def test_wav_join_normalizes_rate_and_smooths_boundary(self):
        audio = join_speech_segments([wav_sample(10000), wav_sample(-10000, 24000)], 22050)
        with wave.open(io.BytesIO(audio), "rb") as wav:
            self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getsampwidth()),
                             (22050, 1, 2))
            frames = wav.readframes(wav.getnframes())
        from array import array
        values = array("h")
        values.frombytes(frames)
        self.assertLess(max(abs(b - a) for a, b in zip(values, values[1:])), 20000)

    def test_piper_refuses_roman_hindi_instead_of_silent_english_voice(self):
        from services.tts_service import TTSEngineError
        with self.assertRaisesRegex(TTSEngineError, "cannot reliably speak Roman-script Hindi"):
            _synthesize_piper_speech(TTSRequest(
                text="Yaar, mujhe kal office jaana hai.",
                language="hinglish", response_format="wav",
            ))



if __name__ == "__main__":
    unittest.main()

