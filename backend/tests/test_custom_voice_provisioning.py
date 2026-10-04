"""Pinned third-party Piper voice provisioning remains usable offline."""

import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.provision_models import _provision_custom_voice


class CustomVoiceProvisioningTests(unittest.TestCase):
    def test_download_verifies_both_files_then_accepts_offline(self):
        model = b"small test ONNX payload"
        metadata = b'{"sample_rate": 22050}'
        source = {
            "model_url": "https://example.test/model",
            "model_sha256": hashlib.sha256(model).hexdigest(),
            "config_url": "https://example.test/config",
            "config_sha256": hashlib.sha256(metadata).hexdigest(),
        }
        def fake_urlopen(url, timeout):
            return io.BytesIO(model if url == source["model_url"] else metadata)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.onnx"
            with patch("scripts.provision_models.urlopen", side_effect=fake_urlopen) as opened:
                _provision_custom_voice(path, source, offline=False)
                self.assertEqual(opened.call_count, 2)
            self.assertEqual(path.read_bytes(), model)
            self.assertEqual(Path(str(path) + ".json").read_bytes(), metadata)
            _provision_custom_voice(path, source, offline=True)

    def test_offline_rejects_hash_mismatch(self):
        payload = b"expected"
        source = {
            "model_url": "https://example.test/model",
            "model_sha256": hashlib.sha256(payload).hexdigest(),
            "config_url": "https://example.test/config",
            "config_sha256": hashlib.sha256(b"{}").hexdigest(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.onnx"
            path.write_bytes(b"wrong")
            with self.assertRaisesRegex(RuntimeError, "mismatched offline voice"):
                _provision_custom_voice(path, source, offline=True)


if __name__ == "__main__":
    unittest.main()
