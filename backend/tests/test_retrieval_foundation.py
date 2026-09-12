import tempfile
import unittest
import io
import wave
from pathlib import Path

from langchain_core.documents import Document

from services.embedding_service import embedding_profile
from services.lexical_service import add_lexical_chunks, lexical_search
from services.qa_service import (
    _clean_model_answer,
    _general_response,
    _history_aware_query,
    _is_general_query,
    _is_summary_query,
    _looks_like_internal_analysis,
    _topic_opener_response,
)
from services.stt_service import STTEngineError, transcribe_audio
from services.tts_service import TTSEngineError, _validate_audio


class EmbeddingProfileTests(unittest.TestCase):
    def test_embeddinggemma_uses_asymmetric_prompts(self):
        profile = embedding_profile("embeddinggemma")
        self.assertIn("query", profile.query_prefix)
        self.assertIn("text", profile.document_prefix)

    def test_nomic_uses_search_prefixes(self):
        profile = embedding_profile("nomic-embed-text")
        self.assertEqual(profile.query_prefix, "search_query: ")
        self.assertEqual(profile.document_prefix, "search_document: ")


class RetrievalQueryTests(unittest.TestCase):
    def test_standalone_question_is_not_polluted_by_history(self):
        history = [{"role": "user", "content": "unrelated annual report"}]
        question = "What is the permitted explosive storage limit?"
        self.assertEqual(_history_aware_query(question, history), question)

    def test_short_pronoun_followup_uses_history(self):
        history = [{"role": "assistant", "content": "The rule was amended in 2025."}]
        result = _history_aware_query("When did it apply?", history)
        self.assertIn("Recent conversation", result)


class ConversationStyleTests(unittest.TestCase):
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

    def test_empty_microphone_recording_is_rejected_before_model_load(self):
        with self.assertRaises(STTEngineError):
            transcribe_audio(b"")


if __name__ == "__main__":
    unittest.main()
