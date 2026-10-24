import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableGenerator

from services.evidence_gate import EvidenceDecision, EvidenceStatus
from services.semantic_evidence import _safe_search_query, assess_semantic_evidence, suggest_local_search_query
from services.qa_service import stream_answer_events


def response(decision, evidence=None, missing_parts=None, english_search_query=""):
    return SimpleNamespace(message=SimpleNamespace(content=json.dumps({
        "decision": decision, "evidence": evidence or [], "missing_parts": missing_parts or [],
        "english_search_query": english_search_query,
    })))


class SemanticEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.passages = [Document(
            page_content="State DGMs receive modern tools and training for mineral exploration in the region.",
            metadata={"source": "handbook.pdf", "year": "2024"},
        )]

    def test_hindi_question_can_receive_partial_english_evidence(self):
        decision = assess_semantic_evidence(
            "क्या आप DGMS के बारे में हिंदी में समझा सकते हैं?", self.passages,
            chat=lambda *_: response("PARTIAL_EVIDENCE", [{
                "passage_id": 0,
                "quote": "State DGMs receive modern tools and training for mineral exploration in the region.",
            }], ["The indexed passage does not define the abbreviation or full remit."]),
        )
        self.assertEqual(decision.status, EvidenceStatus.PARTIAL_EVIDENCE)
        self.assertEqual(decision.documents, tuple(self.passages))
        self.assertEqual(len(decision.verified_quotes), 1)

    def test_unrelated_question_can_be_rejected_despite_retrieved_passage(self):
        decision = assess_semantic_evidence(
            "What is the DGMS annual budget?", self.passages,
            chat=lambda *_: response("INSUFFICIENT_EVIDENCE"),
        )
        self.assertEqual(decision.status, EvidenceStatus.INSUFFICIENT_EVIDENCE)
        self.assertEqual(decision.documents, ())

    def test_model_cannot_introduce_fabricated_evidence(self):
        decision = assess_semantic_evidence(
            "Explain DGMS", self.passages,
            chat=lambda *_: response("ANSWERABLE", [{
                "passage_id": 0,
                "quote": "DGMS is the national safety authority for all mining operations.",
            }]),
        )
        self.assertEqual(decision.status, EvidenceStatus.INSUFFICIENT_EVIDENCE)

    def test_model_failure_is_distinct_from_missing_information(self):
        def fails(*_):
            raise TimeoutError("local model timed out")
        decision = assess_semantic_evidence("Explain DGMS", self.passages, chat=fails)
        self.assertEqual(decision.status, EvidenceStatus.VERIFICATION_UNAVAILABLE)

    def test_followup_topic_is_for_reference_resolution_only(self):
        captured = []
        def examine(payload, _model):
            captured.append(payload)
            return response("INSUFFICIENT_EVIDENCE")
        assess_semantic_evidence("Can you explain it again in Hindi?", self.passages,
                                 topic_context="Current question: Can you explain it again in Hindi? "
                                               "Recent conversation topic: Explain DGMS",
                                 chat=examine)
        self.assertIn("Earlier user topic for resolving words like 'it'", captured[0])
        self.assertIn("Recent conversation topic: Explain DGMS", captured[0])

    def test_rewrite_preserves_identifiers_numbers_and_negation(self):
        self.assertEqual(_safe_search_query("DGMS 2024 में क्या नहीं करता?",
                                            "What does DGMS not do in 2024?"),
                         "What does DGMS not do in 2024?")
        self.assertEqual(_safe_search_query("DGMS 2024 में क्या नहीं करता?",
                                            "What does mine safety do in 2025?"), "")

    def test_missing_hindi_retrieval_can_request_one_english_search(self):
        query = suggest_local_search_query(
            "खनन सुरक्षा में विस्फोटकों की क्या भूमिका है?",
            chat=lambda *_: response("INSUFFICIENT_EVIDENCE",
                                     english_search_query="What is the role of explosives in mine safety?"),
        )
        self.assertIn("explosives", query)

    def test_verified_passage_and_missing_details_reach_final_model(self):
        prompts = []

        def generate(inputs):
            prompts.extend(str(item) for item in inputs)
            yield AIMessageChunk(content="दस्तावेज़ में राज्य DGMs के प्रशिक्षण का उल्लेख है, लेकिन पूरा नाम नहीं दिया गया है।")

        with patch("services.qa_service._load_environment"), patch(
            "services.qa_service._vector_store"
        ), patch("services.qa_service._retrieve_context", return_value=self.passages), patch(
            "services.qa_service.assess_semantic_evidence",
            return_value=assess_semantic_evidence(
                "DGMS के बारे में बताइए", self.passages,
                chat=lambda *_: response("PARTIAL_EVIDENCE", [{
                    "passage_id": 0,
                    "quote": "State DGMs receive modern tools and training for mineral exploration in the region.",
                }], ["The full form is not established."]),
            ),
        ), patch("services.qa_service._llm", return_value=RunnableGenerator(generate)):
            events = list(stream_answer_events("DGMS के बारे में बताइए"))

        self.assertEqual(events[-1]["type"], "done")
        self.assertIn("पूरा नाम नहीं", events[-1]["answer"])
        self.assertIn("The full form is not established.", " ".join(prompts))
        self.assertIn("State DGMs receive modern tools", " ".join(prompts))

    def test_cross_language_retry_uses_new_relevant_passage(self):
        unrelated = Document(page_content="Critical minerals support mineral exploration and development.",
                             metadata={"source": "minerals.pdf"})
        relevant = Document(page_content="Mine safety procedures require keeping workers clear of explosives.",
                            metadata={"source": "safety.pdf"})
        decisions = [
            EvidenceDecision(EvidenceStatus.INSUFFICIENT_EVIDENCE, (), "not_supported",
                             search_query="Why is mine safety important around explosives?"),
            EvidenceDecision(EvidenceStatus.ANSWERABLE, (relevant,), "semantic_quote_verified",
                             verified_quotes=(relevant.page_content,)),
        ]
        def generate(inputs):
            list(inputs)
            yield AIMessageChunk(content="विस्फोटकों के पास कामगारों को सुरक्षित दूरी पर रखना आवश्यक है।")
        model = RunnableGenerator(generate)
        with patch("services.qa_service._load_environment"), patch(
            "services.qa_service._vector_store"
        ), patch("services.qa_service._retrieve_context", side_effect=[[unrelated], [relevant]]) as search, patch(
            "services.qa_service.assess_semantic_evidence", side_effect=decisions
        ), patch("services.qa_service._llm", return_value=model):
            events = list(stream_answer_events("खनन सुरक्षा में विस्फोटकों की क्या भूमिका है?"))
        self.assertEqual(search.call_count, 2)
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["sources"][0]["source"], "safety.pdf")


if __name__ == "__main__":
    unittest.main()
