import ast
import io
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import config
from scripts.configure_gateway import render_gateway
from scripts.provision_ollama import provision_models
from services.embedding_service import embedding_profile


class SharedConfigTests(unittest.TestCase):
    def test_all_module_setting_references_exist(self):
        root = Path(__file__).resolve().parents[2]
        for directory in (root / "backend", root / "frontend"):
            for path in directory.rglob("*.py"):
                if any(part in {"models", "__pycache__", "tests"} for part in path.parts):
                    continue
                tree = ast.parse(path.read_text(encoding="utf-8-sig"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "config":
                        with self.subTest(path=str(path), setting=node.attr):
                            self.assertTrue(hasattr(config, node.attr), node.attr)

    def test_model_settings_do_not_come_from_environment(self):
        with patch.dict(os.environ, {"OLLAMA_EMBED_MODEL": "ignored-legacy-setting"}):
            self.assertEqual(embedding_profile().model, config.OLLAMA_EMBED_MODEL)

    def test_explicit_empty_embedding_prefix_disables_auto_prefix(self):
        with patch.multiple(config, EMBED_QUERY_PREFIX="", EMBED_DOCUMENT_PREFIX=""):
            profile = embedding_profile("nomic-embed-text")
            self.assertEqual((profile.query_prefix, profile.document_prefix), ("", ""))

    def test_library_flags_use_config_and_allow_online_provisioning(self):
        with patch.dict(os.environ), patch.multiple(config, HF_HOME="test-cache", OFFLINE_MODE=True):
            config.configure_runtime_environment()
            self.assertEqual(os.environ["HF_HOME"], "test-cache")
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            config.configure_runtime_environment(offline=False)
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "0")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "0")
            self.assertIs(config.OFFLINE_MODE, True)

    def test_storage_paths_are_absolute(self):
        for setting in (config.RAW_DATA_DIR, config.VECTOR_DB_DIR, config.PIPER_MODEL_DIR, config.HF_HOME):
            self.assertTrue(Path(setting).is_absolute())

    def test_ocr_uses_config_without_shadowing_module(self):
        import train_engine
        from PIL import Image

        recognize = Mock(return_value=" Sample document ")
        with patch.multiple(config, OCR_ENABLED=True, OCR_LANG="eng", OCR_TESSERACT_CONFIG="--psm 6"), patch.dict(
            "sys.modules", {"pytesseract": SimpleNamespace(image_to_string=recognize)}
        ):
            self.assertEqual(train_engine.ocr_image(Image.new("RGB", (4, 4))), "Sample document")
            self.assertEqual(recognize.call_args.kwargs, {"lang": "eng", "config": "--psm 6"})


class GatewayConfigTests(unittest.TestCase):
    def test_template_uses_shared_hostname(self):
        with patch.object(config, "TEXMIN_HOST", "spark-fa50.local"):
            self.assertEqual(render_gateway("__TEXMIN_HOST__ { tls internal }"), "spark-fa50.local { tls internal }")

    def test_rejects_malformed_addresses_and_directive_injection(self):
        for hostname in ("", "http://localhost", "localhost:8080", "localhost\n{", "host/path", "host name"):
            with self.subTest(hostname=hostname), patch.object(config, "TEXMIN_HOST", hostname):
                with self.assertRaises(ValueError):
                    render_gateway("__TEXMIN_HOST__ { tls internal }")


class OllamaProvisioningTests(unittest.TestCase):
    def test_cached_configured_models_do_not_get_downloaded(self):
        response = io.BytesIO(json.dumps({"models": [{"name": "chat:latest"}, {"name": "embed:latest"}]}).encode())
        with patch.multiple(config, OLLAMA_CHAT_MODEL="chat", OLLAMA_EMBED_MODEL="embed"), patch(
            "scripts.provision_ollama.urllib.request.urlopen", return_value=response
        ) as request:
            provision_models()
        self.assertEqual(request.call_count, 1)

    def test_pull_reads_configured_models_and_reports_success(self):
        responses = [io.BytesIO(b'{"models": []}'), io.BytesIO(b'{"status":"success"}\n')]
        with patch.multiple(config, OLLAMA_CHAT_MODEL="test-model", OLLAMA_EMBED_MODEL="test-model"), patch(
            "scripts.provision_ollama.urllib.request.urlopen", side_effect=responses
        ) as request:
            provision_models()
        self.assertEqual(request.call_count, 2)
        self.assertEqual(json.loads(request.call_args.args[0].data)["model"], "test-model")

    def test_error_or_incomplete_pull_does_not_report_success(self):
        for event in (b'{"error":"model not found"}\n', b'{"status":"pulling manifest"}\n'):
            with self.subTest(event=event), patch.multiple(config, OLLAMA_CHAT_MODEL="missing", OLLAMA_EMBED_MODEL="missing"), patch(
                "scripts.provision_ollama.urllib.request.urlopen",
                side_effect=[io.BytesIO(b'{"models": []}'), io.BytesIO(event)],
            ):
                with self.assertRaises(RuntimeError):
                    provision_models()


if __name__ == "__main__":
    unittest.main()
