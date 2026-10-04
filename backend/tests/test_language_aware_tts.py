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


def wav_sample(value: int, rate: int = 22050) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(value.to_bytes(2, "little", signed=True) * (rate // 5))
    return output.getvalue()


class LanguageAwareSpeechTests(unittest.TestCase):
    def test_hinglish_preserves_english_phrase_and_transliterates_known_hindi(self):
        phrases = speech_phrases(
            "Yaar, mujhe kal office jaana hai. Can you remind me at 9 AM?", "hinglish"
        )
        self.assertEqual([voice for voice, _ in phrases], ["hi", "en", "hi", "en"])
        self.assertEqual(phrases[-1][1], "Can you remind me at 9 AM?")
        self.assertIn("मुझे कल", phrases[0][1])

    def test_short_roman_query_is_identified_as_hinglish(self):
        self.assertEqual(detect_language("Yaar, kal office jaana hai"), "hinglish")

    def test_mixed_script_keeps_technical_english_together(self):
        phrases = speech_phrases(
            "मुझे आज एक Python project complete करना है। Can you help me?", "hi"
        )
        self.assertEqual(phrases[1], ("en", "Python project complete"))
        self.assertEqual(phrases[-1], ("en", "Can you help me?"))

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

    def test_mixed_piper_request_uses_both_configured_voices(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in (config.PIPER_HINDI_VOICE, config.PIPER_ENGLISH_VOICE):
                (Path(directory) / (name + ".onnx")).touch()
                (Path(directory) / (name + ".onnx.json")).touch()
            calls = []
            class Worker:
                def __init__(self, path):
                    self.path = path
                def synthesize(self, text, length_scale, cancelled):
                    calls.append((self.path.name, text))
                    return wav_sample(2000)
            with patch.object(config, "PIPER_MODEL_DIR", directory), patch(
                "services.tts_service._piper_worker", side_effect=Worker
            ):
                audio, mime = _synthesize_piper_speech(TTSRequest(
                    text="मुझे आज Python project करना है।", language="hi", response_format="wav"
                ))
            self.assertEqual(mime, "audio/wav")
            self.assertTrue(audio.startswith(b"RIFF"))
            self.assertEqual([name for name, _ in calls], [
                config.PIPER_HINDI_VOICE + ".onnx",
                config.PIPER_ENGLISH_VOICE + ".onnx",
                config.PIPER_HINDI_VOICE + ".onnx",
            ])


if __name__ == "__main__":
    unittest.main()

