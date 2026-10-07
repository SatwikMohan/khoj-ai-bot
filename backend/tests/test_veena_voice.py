"""Veena frame decoding and single-speaker backend routing without model weights."""

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
from scripts.provision_models import provision_tts
from services.tts_service import _synthesize_veena_speech
from services.veena_worker import (
    AUDIO_OFFSET, CODEBOOK_SIZE, END_OF_SPEECH, START_OF_SPEECH,
    split_audio_tokens,
)


def wav_sample() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes((1000).to_bytes(2, "little", signed=True) * 4800)
    return buffer.getvalue()


class VeenaVoiceTests(unittest.TestCase):
    def test_interleaved_audio_codes_are_validated_and_grouped(self):
        frame = [AUDIO_OFFSET + i * CODEBOOK_SIZE + i for i in range(7)]
        self.assertEqual(
            split_audio_tokens([START_OF_SPEECH, *frame, END_OF_SPEECH]),
            [[0], [1, 4], [2, 3, 5, 6]],
        )
        with self.assertRaisesRegex(ValueError, "invalid audio token"):
            split_audio_tokens([START_OF_SPEECH, *frame[:3], AUDIO_OFFSET + CODEBOOK_SIZE + 3, *frame[4:], END_OF_SPEECH])

    def test_pinned_assets_can_be_verified_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory) / "veena"
            codec_dir = Path(directory) / "codec"
            calls = []
            def download(*, repo_id, revision, local_dir, allow_patterns):
                calls.append((repo_id, revision))
                destination = Path(local_dir)
                destination.mkdir(parents=True)
                (destination / "config.json").write_text("{}", encoding="utf-8")
                if destination == codec_dir:
                    (destination / "pytorch_model.bin").write_bytes(b"weights")
                else:
                    for name in ("tokenizer.json", "model.safetensors.index.json"):
                        (destination / name).write_text("{}", encoding="utf-8")
                    for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
                        (destination / name).write_bytes(b"weights")
            with patch.object(config, "VEENA_MODEL_DIR", str(model_dir)), patch.object(
                config, "VEENA_CODEC_DIR", str(codec_dir)
            ), patch("huggingface_hub.snapshot_download", side_effect=download) as snapshot:
                provision_tts(engine_override="veena")
                self.assertEqual(len(calls), 2)
                self.assertEqual(calls[0][1], config.VEENA_MODEL_REVISION)
                self.assertEqual(calls[1][1], config.VEENA_CODEC_REVISION)
                snapshot.reset_mock()
                provision_tts(offline=True, engine_override="veena")
                snapshot.assert_not_called()

    def test_english_hindi_and_hinglish_use_one_worker(self):
        calls = []
        class Worker:
            def synthesize(self, text, length_scale, cancelled):
                calls.append(text)
                return wav_sample()
        with patch("services.tts_service._veena_worker", return_value=Worker()), patch.object(config, "TTS_SAMPLE_RATE", 24000):
            for language, text in (
                ("en", "Please explain the report."),
                ("hi", "\u092e\u0941\u091d\u0947 \u0930\u093f\u092a\u094b\u0930\u094d\u091f \u0938\u092e\u091d\u093e\u090f\u0902\u0964"),
                ("hinglish", "Yaar, mujhe report explain karo."),
            ):
                audio, media_type = _synthesize_veena_speech(
                    TTSRequest(text=text, language=language, response_format="wav")
                )
                self.assertEqual(media_type, "audio/wav")
                with wave.open(io.BytesIO(audio), "rb") as wav:
                    self.assertEqual(wav.getframerate(), 24000)
            self.assertEqual(len(calls), 3)
            self.assertIn("mujhe", calls[-1])


if __name__ == "__main__":
    unittest.main()
