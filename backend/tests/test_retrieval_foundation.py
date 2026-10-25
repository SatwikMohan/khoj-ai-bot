import tempfile
import unittest
import io
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import config

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from services.embedding_service import embedding_profile
from services.lexical_service import add_lexical_chunks, lexical_search
from services.qa_service import (
    _chat_llm,
    _clean_model_answer,
    _direct_answer_content,
    _detect_language_style,
    _general_response,
    _format_chat_history,
    _history_aware_query,
    _query_identifiers,
    _is_general_query,
    _is_summary_query,
    _looks_like_internal_analysis,
    _mentions_identifiers,
    _retrieve_context,
    _topic_opener_response,
)
from services.stt_service import (
    STTEngineError, STTRecoverableError, _preferred_supported_language,
    _validate_speech_signal, transcribe_audio,
)
from services.tts_service import TTSEngineError, _validate_audio, tts_runtime_status


class EmbeddingProfileTests(unittest.TestCase):
    def test_embeddinggemma_uses_asymmetric_prompts(self):
        profile = embedding_profile("embeddinggemma")
        self.assertIn("query", profile.query_prefix)
        self.assertIn("text", profile.document_prefix)

    def test_nomic_uses_search_prefixes(self):
        profile = embedding_profile("nomic-embed-text")
        self.assertEqual(profile.query_prefix, "search_query: ")
        self.assertEqual(profile.document_prefix, "search_document: ")


class ChatModelConfigurationTests(unittest.TestCase):
    def test_reasoning_disabled_is_sent_to_ollama_as_think_false(self):
        model = _chat_llm("qwen3.5:35b", "http://ollama:11434", 0.2, False)
        params = model._chat_params([HumanMessage(content="Answer directly.")])
        self.assertIs(params["think"], False)


class RetrievalQueryTests(unittest.TestCase):
    def test_hybrid_retrieval_keeps_best_dense_candidate_below_threshold(self):
        document = Document(
            page_content="The permitted explosive storage limit is 100 kilograms.",
            metadata={"source": "rule.pdf", "chunk_index": 1},
        )

        class VectorStore:
            @staticmethod
            def similarity_search_with_relevance_scores(_question, k):
                return [(document, 0.11)]

        environment = {
            "HYBRID_SEARCH_ENABLED": True,
            "RELEVANCE_SCORE_THRESHOLD": 0.15,
            "RERANK_ENABLED": False,
            "ALLOW_LOW_RELEVANCE_FALLBACK": True,
        }
        with (
            patch.multiple("config", **environment),
            patch("services.qa_service._active_index", return_value=("test", "fingerprint")),
            patch("services.qa_service.lexical_search", return_value=[]),
        ):
            results = _retrieve_context(VectorStore(), "storage limit", 1)

        self.assertEqual(results, [document])

    def test_weak_hybrid_hit_is_not_passed_to_the_answer_model(self):
        unrelated = Document(
            page_content="The handbook lists 30 critical minerals for clean energy.",
            metadata={"source": "minerals.pdf", "chunk_index": 1},
        )

        class VectorStore:
            @staticmethod
            def similarity_search_with_relevance_scores(_question, k):
                return [(unrelated, 0.11)]

        with patch.multiple(config, HYBRID_SEARCH_ENABLED=True,
                            ALLOW_LOW_RELEVANCE_FALLBACK=False,
                            RELEVANCE_SCORE_THRESHOLD=0.25,
                            RERANK_ENABLED=False), patch(
            "services.qa_service._active_index", return_value=("test", "fingerprint")
        ), patch("services.qa_service.lexical_search", return_value=[]):
            self.assertEqual(_retrieve_context(VectorStore(), "DGMS safety rules", 1), [])

    def test_identifier_keeps_matching_evidence_and_rejects_unrelated_evidence(self):
        unrelated = Document(page_content="Mining safety information is important.",
                             metadata={"source": "general.pdf", "chunk_index": 1})
        relevant = Document(page_content="DGMS issues safety circulars for active mines and their operators.",
                            metadata={"source": "circular.pdf", "chunk_index": 2})

        class VectorStore:
            @staticmethod
            def similarity_search_with_relevance_scores(_question, k):
                return [(unrelated, 0.9), (relevant, 0.52)]

        with patch.multiple(config, HYBRID_SEARCH_ENABLED=True,
                            ALLOW_LOW_RELEVANCE_FALLBACK=False,
                            RELEVANCE_SCORE_THRESHOLD=0.25,
                            RERANK_ENABLED=False), patch(
            "services.qa_service._active_index", return_value=("test", "fingerprint")
        ), patch("services.qa_service.lexical_search", return_value=[]):
            self.assertEqual(_retrieve_context(VectorStore(), "What is DGMS?", 2), [relevant])

    def test_lowercase_acronym_is_grounded_without_treating_language_as_entity(self):
        self.assertEqual(_query_identifiers("dgms ko elaborate kare"), ("dgms",))
        self.assertEqual(_query_identifiers("Explain the rules in HINDI"), ())

    def test_acronym_retrieval_does_not_confuse_mixed_case_plural(self):
        plural = Document(page_content="State DGMs receive mineral exploration training.",
                          metadata={"source": "handbook.pdf"})
        acronym = Document(page_content="DGMS is defined in the indexed mining safety rules.",
                           metadata={"source": "rules.pdf"})
        self.assertFalse(_mentions_identifiers(plural, _query_identifiers("what is dgms")))
        self.assertTrue(_mentions_identifiers(acronym, _query_identifiers("what is dgms")))

    def test_lexical_or_hit_needs_substantive_overlap(self):
        unrelated = Document(page_content="Mining safety information is important.",
                             metadata={"source": "general.pdf", "chunk_index": 1})
        relevant = Document(page_content="Critical minerals include lithium and cobalt.",
                            metadata={"source": "minerals.pdf", "chunk_index": 2})

        class VectorStore:
            @staticmethod
            def similarity_search_with_relevance_scores(_question, k):
                return []

        with patch.multiple(config, HYBRID_SEARCH_ENABLED=True,
                            ALLOW_LOW_RELEVANCE_FALLBACK=False,
                            RERANK_ENABLED=False), patch(
            "services.qa_service._active_index", return_value=("test", "fingerprint")
        ), patch("services.qa_service.lexical_search", return_value=[unrelated, relevant]):
            self.assertEqual(_retrieve_context(VectorStore(), "Explain critical minerals", 2), [relevant])

    def test_assistant_claims_are_not_reused_as_evidence(self):
        history = [
            {"role": "user", "content": "Explain DGMS mine safety rules"},
            {"role": "assistant", "content": "DGMS is a fictional mineral council."},
        ]
        formatted = _format_chat_history(history)
        self.assertIn("DGMS mine safety rules", formatted)
        self.assertNotIn("fictional mineral council", formatted)
        self.assertIn("DGMS", _history_aware_query("When did it apply?", history))

    def test_standalone_question_is_not_polluted_by_history(self):
        history = [{"role": "user", "content": "unrelated annual report"}]
        question = "What is the permitted explosive storage limit?"
        self.assertEqual(_history_aware_query(question, history), question)

    def test_short_pronoun_followup_uses_history(self):
        history = [{"role": "user", "content": "What changed in the rule amended in 2025?"}]
        result = _history_aware_query("When did it apply?", history)
        self.assertIn("Recent conversation", result)

    def test_hinglish_pronoun_followup_keeps_previous_topic(self):
        history = [{"role": "user", "content": "DGMS ke bare mein jankari de"}]
        result = _history_aware_query("isko detail me samjhao", history)
        self.assertIn("DGMS", result)
        self.assertIn("isko detail me samjhao", result)

    def test_latest_version_followup_keeps_previous_entity(self):
        history = [{"role": "user", "content": "What is the structure of CMR?"}]
        result = _history_aware_query("latest version", history)
        self.assertIn("CMR", result)

    def test_short_about_followup_keeps_previous_user_topic(self):
        history = [
            {"role": "user", "content": "Let's discuss the DGMS rules"},
            {"role": "assistant", "content": "What would you like to explore?"},
        ]
        result = _history_aware_query("I want to know about the circulars", history)
        self.assertIn("DGMS rules", result)
        self.assertIn("circulars", result)

    def test_standalone_metric_question_does_not_inherit_old_topic(self):
        history = [{"role": "user", "content": "Tell me about DGMS circulars"}]
        question = "number of critical mines"
        self.assertEqual(_history_aware_query(question, history), question)

    def test_polluted_assistant_history_is_not_used_for_followup_retrieval(self):
        history = [
            {
                "role": "assistant",
                "content": "First, I need to identify the most relevant references.",
            }
        ]
        self.assertEqual(
            _history_aware_query("What did it require?", history),
            "What did it require?",
        )


class ConversationStyleTests(unittest.TestCase):
    def setUp(self):
        language = patch("config.RESPONSE_LANGUAGE", "auto")
        language.start()
        self.addCleanup(language.stop)

    def test_response_language_can_force_hinglish_for_voice_queries(self):
        with patch("config.RESPONSE_LANGUAGE", "hinglish"):
            self.assertEqual(_detect_language_style("What are the rules?"), "hinglish")

    def test_auto_response_language_still_detects_english(self):
        with patch("config.RESPONSE_LANGUAGE", "auto"):
            self.assertEqual(_detect_language_style("What are the rules?"), "english")

    def test_wellbeing_gets_a_specific_conversational_reply(self):
        answer = _general_response("How are you?").answer
        self.assertIn("doing well", answer)
        self.assertNotIn("indexed documents", answer)

    def test_hindi_greeting_is_recognized_without_retrieval(self):
        self.assertTrue(_is_general_query("नमस्ते।"))
        self.assertIn("नमस्ते", _general_response("नमस्ते।").answer)

    def test_private_reasoning_is_removed_from_tagged_answer(self):
        raw = "<think>The user asked about DGMS.</think><answer>DGMS rules govern mine safety.</answer>"
        self.assertEqual(_clean_model_answer(raw), "DGMS rules govern mine safety.")

    def test_broad_explanation_uses_summary_retrieval(self):
        self.assertTrue(_is_summary_query("Explain the DGMS rules"))

    def test_topic_opener_addresses_the_person_directly(self):
        response = _topic_opener_response("lets talk about dgms")
        self.assertIsNotNone(response)
        self.assertIn("DGMS", response.answer)
        self.assertNotIn("the user", response.answer.lower())

    def test_narrated_planning_is_detected_before_streaming(self):
        leaked = "Okay, let me break this down. The user just said let's talk about DGMS."
        self.assertTrue(_looks_like_internal_analysis(leaked))

    def test_third_person_and_context_narration_are_rejected(self):
        leaks = (
            "They asked for a general discussion about DGMS.",
            "Based on the provided context, DGMS is a regulator.",
            "I should give a concise overview of mine safety.",
            'Reference 1 mentions "30 critical minerals" but not mines.',
            "Looking at the MCQs, several values appear.",
        )
        for leaked in leaks:
            with self.subTest(leaked=leaked):
                self.assertTrue(_looks_like_internal_analysis(leaked))

    def test_valid_answer_is_salvaged_after_narrated_planning(self):
        raw = (
            "The user asked for an overview. I should keep it concise. "
            "DGMS regulates occupational safety in Indian mines."
        )
        self.assertEqual(
            _direct_answer_content(raw),
            "DGMS regulates occupational safety in Indian mines.",
        )

    def test_context_framing_is_removed_without_losing_answer(self):
        raw = "Based on the provided context, DGMS issues mine-safety circulars."
        self.assertEqual(
            _direct_answer_content(raw),
            "DGMS issues mine-safety circulars.",
        )

    def test_more_topic_openers_are_answered_directly(self):
        for question in ("Can we talk about DGMS?", "I want to talk about DGMS"):
            with self.subTest(question=question):
                response = _topic_opener_response(question)
                self.assertIsNotNone(response)
                self.assertNotIn("the user", response.answer.lower())


class LexicalIndexTests(unittest.TestCase):
    def test_exact_regulation_identifier_is_retrievable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = Document(
                page_content="DGMS Technical Circular 05/2025 sets the inspection interval.",
                metadata={"source": "circular.pdf", "chunk_index": 1},
            )
            add_lexical_chunks(root, "test_collection", [document], ["chunk-1"])
            results = lexical_search(root, "test_collection", "Circular 05 2025", 3)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].metadata["source"], "circular.pdf")


class SpeechReliabilityTests(unittest.TestCase):
    @staticmethod
    def wav_bytes(frames: bytes) -> bytes:
        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(frames)
        return output.getvalue()

    def test_silent_voice_segment_is_rejected(self):
        audio = self.wav_bytes(b"\x00\x00" * 2000)
        with self.assertRaises(TTSEngineError):
            _validate_audio(audio, "audio/wav")

    def test_hinglish_auto_tts_uses_piper_without_reference_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            for voice in (config.PIPER_HINDI_VOICE, config.PIPER_ENGLISH_VOICE):
                (Path(directory) / f"{voice}.onnx").write_bytes(b"test")
                (Path(directory) / f"{voice}.onnx.json").write_text("{}")
            environment = {
                "TTS_ENGINE": "auto",
                "RESPONSE_LANGUAGE": "hinglish",
                "PIPER_MODEL_DIR": directory,
            }
            with patch.multiple("config", **environment):
                status = tts_runtime_status()

        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["selected_engine"], "piper")

    def test_empty_microphone_recording_is_rejected_before_model_load(self):
        with self.assertRaises(STTEngineError):
            transcribe_audio(b"")

    def test_short_noise_burst_is_not_sent_to_whisper(self):
        rng = np.random.default_rng(11)
        audio = rng.normal(0.0, 0.002, 16000).astype(np.float32)
        audio[5000:6600] += rng.normal(0.0, 0.04, 1600)
        with self.assertRaises(STTEngineError):
            _validate_speech_signal(audio)

    def test_sustained_broadband_noise_is_not_sent_to_whisper(self):
        rng = np.random.default_rng(13)
        audio = rng.normal(0.0, 0.002, 16000).astype(np.float32)
        audio[4800:11200] += rng.normal(0.0, 0.04, 6400)
        with self.assertRaises(STTEngineError):
            _validate_speech_signal(audio)

    def test_modulated_voiced_signal_is_allowed(self):
        rng = np.random.default_rng(17)
        audio = rng.normal(0.0, 0.002, 16000).astype(np.float32)
        timeline = np.arange(6400) / 16000.0
        audio[4800:11200] += (
            0.04 * np.sin(2 * np.pi * 180 * timeline)
            + 0.016 * np.sin(2 * np.pi * 360 * timeline)
        ).astype(np.float32)
        _validate_speech_signal(audio)

    def test_arabic_auto_detection_retries_supported_language(self):
        with (
            patch("services.stt_service._decode_audio", return_value=np.zeros(16000)),
            patch("services.stt_service._model", return_value={"engine": "transformers"}),
            patch("services.stt_service._preferred_supported_language", return_value=("hi", 0.18)) as preference,
            patch("services.stt_service._transcribe_transformers", side_effect=[
                ("?????", "ar", 0.0, 1.0),
                ("DGMS \u0915\u0947 \u092c\u093e\u0930\u0947 \u092e\u0947\u0902 \u092c\u0924\u093e\u0913", "hi", 0.0, 1.0),
            ]) as transcribe,
        ):
            result = transcribe_audio(b"recording")
        self.assertEqual(result["language"], "hi")
        self.assertIn("DGMS", result["text"])
        preference.assert_called_once()
        self.assertEqual(transcribe.call_args_list[1].args[2], "hi")

    def test_english_translation_of_hinglish_audio_is_retranscribed(self):
        with (
            patch("services.stt_service._decode_audio", return_value=np.zeros(16000)),
            patch("services.stt_service._model", return_value={"engine": "transformers"}),
            patch("services.stt_service._preferred_supported_language", return_value=("hi", 0.82)),
            patch("services.stt_service._transcribe_transformers", side_effect=[
                ("Tell me about DGMS", "en", 0.0, 1.0),
                ("DGMS के बारे में बताओ", "hi", 0.0, 1.0),
            ]) as transcribe,
        ):
            result = transcribe_audio(b"recording")
        self.assertEqual(result["text"], "DGMS के बारे में बताओ")
        self.assertEqual(result["spoken_language"], "hi")
        self.assertEqual(transcribe.call_args_list[1].args[2], "hi")

    def test_english_audio_is_not_forced_into_hindi(self):
        with (
            patch("services.stt_service._decode_audio", return_value=np.zeros(16000)),
            patch("services.stt_service._model", return_value={"engine": "transformers"}),
            patch("services.stt_service._preferred_supported_language", return_value=("en", 0.93)),
            patch("services.stt_service._transcribe_transformers", return_value=(
                "Tell me about DGMS", "en", 0.0, 1.0,
            )) as transcribe,
        ):
            result = transcribe_audio(b"recording")
        self.assertEqual(result["spoken_language"], "en")
        self.assertEqual(transcribe.call_count, 1)

    def test_hindi_retry_cannot_drop_spoken_identifier(self):
        with (
            patch("services.stt_service._decode_audio", return_value=np.zeros(16000)),
            patch("services.stt_service._model", return_value={"engine": "transformers"}),
            patch("services.stt_service._preferred_supported_language", return_value=("hi", 0.82)),
            patch("services.stt_service._transcribe_transformers", side_effect=[
                ("Tell me about DGMS", "en", 0.0, 1.0),
                ("मुझे सुरक्षा के बारे में बताओ", "hi", 0.0, 1.0),
            ]),
        ):
            with self.assertRaises(STTRecoverableError):
                transcribe_audio(b"recording")

    def test_hindi_retry_accepts_identifier_spoken_as_hindi_letters(self):
        with (
            patch("services.stt_service._decode_audio", return_value=np.zeros(16000)),
            patch("services.stt_service._model", return_value={"engine": "transformers"}),
            patch("services.stt_service._preferred_supported_language", return_value=("hi", 0.82)),
            patch("services.stt_service._transcribe_transformers", side_effect=[
                ("Tell me about DGMS", "en", 0.0, 1.0),
                ("\u0921\u0940\u091c\u0940\u090f\u092e\u090f\u0938 \u0915\u0947 \u092c\u093e\u0930\u0947 \u092e\u0947\u0902 \u092c\u0924\u093e\u0913", "hi", 0.0, 1.0),
            ]),
        ):
            result = transcribe_audio(b"recording")
        self.assertEqual(result["spoken_language"], "hi")
        self.assertIn("\u0921\u0940\u091c\u0940\u090f\u092e\u090f\u0938", result["text"])

    def test_language_recovery_uses_english_hindi_logits(self):
        import torch

        class Model:
            generation_config = SimpleNamespace(
                lang_to_id={"<|en|>": 1, "<|hi|>": 2, "<|ar|>": 3},
                decoder_start_token_id=0,
            )

            def __call__(self, **kwargs):
                self.last_kwargs = kwargs
                logits = torch.tensor([[[0.0, 2.0, 1.0, 3.0]]])
                return SimpleNamespace(logits=logits)

        model = Model()
        processor = lambda *_args, **_kwargs: SimpleNamespace(
            input_features=torch.zeros(1, 80, 3000)
        )
        state = {
            "model": model, "processor": processor,
            "device": "cpu", "dtype": torch.float32,
        }
        language, probability = _preferred_supported_language(
            state, np.zeros(16000, dtype=np.float32)
        )
        self.assertEqual(language, "en")
        self.assertGreater(probability, 0.02)
        self.assertIn("decoder_input_ids", model.last_kwargs)

    def test_unsupported_transcript_never_reaches_question_answering(self):
        with (
            patch("services.stt_service._decode_audio", return_value=np.zeros(16000)),
            patch("services.stt_service._model", return_value={"engine": "transformers"}),
            patch("services.stt_service._preferred_supported_language", return_value=("en", 0.14)),
            patch("services.stt_service._transcribe_transformers", side_effect=[
                ("?????", "ar", 0.0, 1.0),
                ("?????", "en", 0.0, 1.0),
            ]),
        ):
            with self.assertRaises(STTRecoverableError):
                transcribe_audio(b"recording")

    def test_stationary_background_noise_is_rejected(self):
        rng = np.random.default_rng(7)
        noise = rng.normal(0.0, 0.006, 16000).astype(np.float32)
        with self.assertRaises(STTEngineError):
            _validate_speech_signal(noise)


if __name__ == "__main__":
    unittest.main()
