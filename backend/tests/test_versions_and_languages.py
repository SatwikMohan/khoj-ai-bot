import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
import chromadb

import config
from helpers.request_models import TTSRequest
from services.language_service import detect_language, language_instruction
from services.lexical_service import add_lexical_chunks
from services.qa_service import _format_context, _retrieve_context
from services.tts_service import TTSEngineError, _espeak_supports, _piper_path, _synthesize_espeak_speech, _synthesize_piper_speech
from services.versioning import document_identity, select_documents
from train_engine import base_metadata, chunk_id


def record(year=None, index=0, source=None, text=None):
    source = source or (f"CMR-{year}.pdf" if year else "CMR.pdf")
    metadata = {"source": source, "chunk_index": index}
    if year:
        metadata["year"] = str(year)
    return Document(page_content=text or f"CMR structure and columns for {year or 'the catalog'} include mineral, grade and status.", metadata=metadata)


class VersionSelectionTests(unittest.TestCase):
    def setUp(self):
        self.docs = [record(2025), record(2024)]

    def test_general_query_covers_two_versions(self):
        self.assertEqual({d.metadata["year"] for d in select_documents(self.docs, "CMR structure", 2)}, {"2024", "2025"})

    def test_three_versions_are_retained(self):
        docs = [record(y) for y in (2026, 2025, 2024)]
        self.assertEqual(len(select_documents(docs, "CMR structure", 3)), 3)

    def test_random_file_names_can_share_query_entity(self):
        docs = [record(2025, source="a91.pdf"), record(2024, source="b27.pdf")]
        self.assertEqual({d.metadata["year"] for d in select_documents(docs, "CMR structure", 2)}, {"2024", "2025"})
        self.assertEqual([d.metadata["year"] for d in select_documents(docs, "latest CMR structure", 2)], ["2025"])

    def test_unrelated_sources_are_not_forced_into_final_context(self):
        leading = record(source="Dragline.pdf", text="A dragline excavator moves material with a suspended bucket.")
        second = record(index=1, source="Dragline.pdf", text="A dragline has a boom and a large bucket for overburden removal.")
        unrelated = record(source="BWE.pdf", text="A bucket wheel excavator has continuous digging machinery.")
        self.assertEqual(select_documents([leading, second, unrelated], "What is a dragline?", 2), [leading, second])

    def test_latest_and_historical_scope(self):
        self.assertEqual(select_documents(self.docs, "latest CMR structure", 2)[0].metadata["year"], "2025")
        self.assertEqual(len(select_documents(self.docs, "latest CMR structure", 2)), 1)
        self.assertEqual(select_documents(self.docs, "historical CMR structure", 2)[0].metadata["year"], "2024")
        self.assertEqual(len(select_documents(self.docs, "historical CMR structure", 2)), 2)

    def test_explicit_year_and_comparison(self):
        self.assertEqual([d.metadata["year"] for d in select_documents(self.docs, "CMR in 2024", 2)], ["2024"])
        self.assertEqual({d.metadata["year"] for d in select_documents(self.docs, "CMR changes 2024 versus 2025", 2)}, {"2024", "2025"})

    def test_explicit_version_without_year(self):
        docs = [record(source="CMR-v1.pdf"), record(source="CMR-v2.pdf")]
        self.assertEqual(select_documents(docs, "CMR version 1", 2)[0].metadata["source"], "CMR-v1.pdf")
        self.assertEqual(len(select_documents(docs, "latest CMR", 2)), 1)
        self.assertEqual(select_documents(docs, "latest CMR", 2)[0].metadata["source"], "CMR-v2.pdf")

    def test_year_and_version_constraints_both_apply(self):
        docs = [record(2024, source="CMR-2024-v1.pdf"), record(2024, source="CMR-2024-v2.pdf")]
        self.assertEqual([doc.metadata["source"] for doc in select_documents(docs, "CMR 2024 version 1", 2)], ["CMR-2024-v1.pdf"])

    def test_no_version_metadata_is_allowed(self):
        doc = record()
        self.assertEqual(select_documents([doc], "CMR structure", 2), [doc])
        self.assertEqual(select_documents([doc], "CMR structure in 2024", 2), [])

    def test_identical_chunks_in_distinct_versions_survive(self):
        text = "The catalog has the same description of grade and status in both releases."
        docs = [record(2024, source="CMR.pdf", text=text), record(2025, source="CMR.pdf", text=text)]
        self.assertNotEqual(document_identity(docs[0]), document_identity(docs[1]))
        self.assertEqual(len(select_documents(docs, "CMR structure", 2)), 2)
        self.assertNotEqual(chunk_id(docs[0]), chunk_id(docs[1]))

    def test_partly_overlapping_chunks_keep_both_versions(self):
        docs = self.docs + [record(2025, index=1, text="CMR status changed in the newer catalog and some columns remain the same.")]
        self.assertEqual({d.metadata["year"] for d in select_documents(docs, "CMR structure", 2)}, {"2024", "2025"})

    def test_context_budget_retains_each_version_header(self):
        with patch.object(config, "CONTEXT_MAX_CHARS", 240):
            context = _format_context(self.docs)
        self.assertLessEqual(len(context), 240)
        self.assertIn("year 2024", context)
        self.assertIn("year 2025", context)

    def test_year_from_directory_is_indexed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "2024" / "CMR.pdf"
            path.parent.mkdir()
            path.touch()
            with patch("train_engine.REPO_ROOT", root), patch("train_engine.PROJECT_ROOT", root):
                metadata = base_metadata(path, root)
        self.assertEqual(metadata["year"], "2024")


class RetrievalIntegrationTests(unittest.TestCase):
    def test_followup_history_year_does_not_change_requested_scope(self):
        old, new = record(2024), record(2025)

        class DenseIndex:
            def similarity_search_with_relevance_scores(self, question, k):
                return [(new, 0.9), (old, 0.8)]

        rewritten = "What about 2024? Recent conversation topic: CMR in 2025"
        with patch.multiple(config, HYBRID_SEARCH_ENABLED=False, RETRIEVAL_MMR_ENABLED=False):
            result = _retrieve_context(DenseIndex(), rewritten, 2, scope_question="What about 2024?")
        self.assertEqual([doc.metadata["year"] for doc in result], ["2024"])

    def test_local_chroma_and_lexical_index_cover_two_versions(self):
        class LocalEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [[0.0, 0.5] if "2024" in text else [0.5, 0.0] for text in texts]

            def embed_query(self, text):
                return [0.5, 0.0]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = record(2024), record(2025)
            store = Chroma(collection_name="versions_local_test", client=chromadb.EphemeralClient(), embedding_function=LocalEmbeddings())
            store.add_documents([old, new], ids=["old", "new"])
            add_lexical_chunks(root, "versions_local_test", [old, new], ["old", "new"])
            with patch.multiple(config, HYBRID_SEARCH_ENABLED=True, RERANK_ENABLED=False, RETRIEVAL_FETCH_K=1), patch(
                "services.qa_service._active_index", return_value=("versions_local_test", "test")
            ), patch("services.qa_service._vector_db_dir", return_value=root):
                results = _retrieve_context(store, "CMR structure", 2)
            self.assertEqual({d.metadata["year"] for d in results}, {"2024", "2025"})

    def test_lexical_version_is_kept_when_dense_search_favors_one_year(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = record(2024), record(2025)
            add_lexical_chunks(root, "versions", [old, new], ["old", "new"])

            class DenseIndex:
                def similarity_search_with_relevance_scores(self, question, k):
                    return [(new, 0.9)]

            with patch.multiple(config, HYBRID_SEARCH_ENABLED=True, RERANK_ENABLED=False), patch(
                "services.qa_service._active_index", return_value=("versions", "test")
            ), patch("services.qa_service._vector_db_dir", return_value=root):
                results = _retrieve_context(DenseIndex(), "CMR structure", 2)
            self.assertEqual({d.metadata["year"] for d in results}, {"2024", "2025"})

    def test_dense_and_lexical_search_run_concurrently(self):
        barrier = threading.Barrier(2)
        doc = record(2025)

        class DenseIndex:
            def similarity_search_with_relevance_scores(self, question, k):
                barrier.wait(timeout=2)
                return [(doc, 0.9)]

        def lexical(*args):
            barrier.wait(timeout=2)
            return []

        with patch.object(config, "HYBRID_SEARCH_ENABLED", True), patch(
            "services.qa_service._active_index", return_value=("versions", "test")
        ), patch("services.qa_service.lexical_search", side_effect=lexical):
            self.assertEqual(len(_retrieve_context(DenseIndex(), "CMR structure", 1)), 1)

    def test_lexical_failure_keeps_dense_answer_available(self):
        doc = record(2025)

        class DenseIndex:
            def similarity_search_with_relevance_scores(self, question, k):
                return [(doc, 0.9)]

        with patch.object(config, "HYBRID_SEARCH_ENABLED", True), patch(
            "services.qa_service._active_index", return_value=("versions", "test")
        ), patch("services.qa_service.lexical_search", side_effect=OSError("index unavailable")):
            self.assertEqual(_retrieve_context(DenseIndex(), "CMR structure", 1), [doc])


class LanguageAndVoiceTests(unittest.TestCase):
    def test_query_language_is_automatic_and_identifiers_do_not_dominate(self):
        self.assertEqual(detect_language("What is CMR structure?"), "en")
        self.assertEqual(detect_language("CMR ka structure kya hai?"), "hinglish")
        self.assertEqual(detect_language("सीएमआर की संरचना क्या है?"), "hi")
        self.assertEqual(detect_language("CMR?"), "en")
        self.assertEqual(detect_language("Explique la estructura del catalogo"), "es")
        self.assertEqual(detect_language("Qual é a estrutura do catálogo?"), "pt")
        with patch("services.language_service._classifier", return_value=SimpleNamespace(classify=lambda text: ("es", 0.95))):
            self.assertEqual(detect_language("Explique la estructura del catálogo"), "es")
        self.assertIn("(es)", language_instruction("es"))

    def test_request_language_selects_piper_voice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (config.PIPER_ENGLISH_VOICE, config.PIPER_HINDI_VOICE):
                (root / (name + ".onnx")).touch()
                (root / (name + ".onnx.json")).touch()
            calls = []

            def run(*args, **kwargs):
                calls.append(json.loads(kwargs["input"])["model"])
                return SimpleNamespace(returncode=0, stdout=b"audio")

            with patch.object(config, "PIPER_MODEL_DIR", directory), patch("services.tts_service.subprocess.run", side_effect=run):
                _synthesize_piper_speech(TTSRequest(text="English answer", language="en"))
                _synthesize_piper_speech(TTSRequest(text="हिंदी उत्तर", language="hi"))
            self.assertIn(config.PIPER_ENGLISH_VOICE, calls[0])
            self.assertIn(config.PIPER_HINDI_VOICE, calls[1])

    def test_additional_voice_is_selected_by_language_code(self):
        with patch.object(config, "PIPER_ADDITIONAL_VOICES", {"ta": "ta_voice"}):
            self.assertEqual(_piper_path("ta").name, "ta_voice.onnx")
        with self.assertRaises(TTSEngineError):
            _piper_path("zz")

    def test_multilingual_espeak_fallback_selects_requested_language(self):
        calls = []

        def run(args, **kwargs):
            calls.append(args)
            if args[1].startswith("--voices="):
                return SimpleNamespace(returncode=0, stdout="header\nes Spanish\n")
            Path(args[args.index("-w") + 1]).write_bytes(b"wav")
            return SimpleNamespace(returncode=0)

        _espeak_supports.cache_clear()
        with patch("services.tts_service.shutil.which", return_value="espeak-ng"), patch(
            "services.tts_service.subprocess.run", side_effect=run
        ):
            audio, media = _synthesize_espeak_speech(TTSRequest(text="Hola a todos", language="es"))
        self.assertEqual((audio, media), (b"wav", "audio/wav"))
        self.assertIn("es", calls[-1])


if __name__ == "__main__":
    unittest.main()
