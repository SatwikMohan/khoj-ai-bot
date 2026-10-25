import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableGenerator

from services.evidence_gate import EvidenceDecision, EvidenceStatus
from services.semantic_evidence import _parse_verdict, _safe_search_query, assess_semantic_evidence, suggest_local_search_query
from services.qa_service import stream_answer_events


def response(decision, evidence=None, missing_parts=None, english_search_query=""):
    return SimpleNamespace(message=SimpleNamespace(content=json.dumps({
        "decision": decision, "evidence": evidence or [], "missing_parts": missing_parts or [],
        "english_search_query": english_search_query,
    })))


class SemanticEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.passages = [Document(
            page_content="DGMS personnel receive modern tools and training for mineral exploration in the region.",
            metadata={"source": "handbook.pdf", "year": "2024"},
        )]

    def test_hindi_question_can_receive_partial_english_evidence(self):
        decision = assess_semantic_evidence(
            "क्या आप DGMS के बारे में हिंदी में समझा सकते हैं?", self.passages,
            chat=lambda *_: response("PARTIAL_EVIDENCE", [{
                "passage_id": 0,
                "quote": "DGMS personnel receive modern tools and training for mineral exploration in the region.",
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

    def test_paraphrased_question_can_use_direct_indexed_premise(self):
        passage = Document(
            page_content="Ventilation removes hazardous gases from mine workings.",
            metadata={"source": "safety.pdf"},
        )
        decision = assess_semantic_evidence(
            "What keeps harmful air from building up underground?", [passage],
            chat=lambda *_: response("ANSWERABLE", [{
                "passage_id": 0, "quote": passage.page_content,
            }]),
        )
        self.assertEqual(decision.status, EvidenceStatus.ANSWERABLE)
        self.assertEqual(decision.documents, (passage,))

    def test_model_cannot_introduce_fabricated_evidence(self):
        decision = assess_semantic_evidence(
            "Explain DGMS", self.passages,
            chat=lambda *_: response("ANSWERABLE", [{
                "passage_id": 0,
                "quote": "DGMS is the national safety authority for all mining operations.",
            }]),
        )
        self.assertEqual(decision.status, EvidenceStatus.VERIFICATION_UNAVAILABLE)
        self.assertEqual(decision.documents, ())

    def test_mixed_case_plural_does_not_establish_acronym(self):
        passage = Document(page_content="State DGMs receive training to develop auctionable mineral blocks.",
                           metadata={"source": "handbook.pdf"})
        def must_not_run(*_):
            self.fail("An unrelated mixed-case plural must be rejected before model verification")
        decision = assess_semantic_evidence("what is dgms", [passage], chat=must_not_run)
        self.assertEqual(decision.status, EvidenceStatus.INSUFFICIENT_EVIDENCE)
        self.assertEqual(decision.reason, "requested_identifier_not_present")

    def test_invalid_model_quote_can_be_repaired_with_literal_source_sentence(self):
        passage = Document(page_content=(
            "DGMS means Directorate General of Mines Safety. "
            "It is responsible for mine safety oversight in this document."
        ), metadata={"source": "definition.pdf"})
        replies = iter([
            response("ANSWERABLE", [{"passage_id": 0,
                                      "quote": "DGMS is the Directorate General of Mines Safety."}]),
            SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "decision": "ANSWERABLE", "sentence_ids": ["p0s0"], "missing_parts": [],
            }))),
        ])
        decision = assess_semantic_evidence("what is DGMS", [passage],
                                            chat=lambda *_: next(replies))
        self.assertEqual(decision.status, EvidenceStatus.ANSWERABLE)
        self.assertEqual(decision.reason, "semantic_sentence_id_verified")
        self.assertEqual(decision.documents, (passage,))
        self.assertEqual(decision.verified_quotes,
                         ("DGMS means Directorate General of Mines Safety.",))

    def test_repair_cannot_select_a_sentence_without_requested_acronym(self):
        passage = Document(page_content=(
            "DGMS is discussed in the report. "
            "An unrelated agency controls all mines everywhere."
        ), metadata={"source": "report.pdf"})
        replies = iter([
            response("ANSWERABLE", [{"passage_id": 0,
                                      "quote": "DGMS controls all mines everywhere."}]),
            SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "decision": "ANSWERABLE", "sentence_ids": ["p0s1"], "missing_parts": [],
            }))),
        ])
        decision = assess_semantic_evidence("what is DGMS", [passage],
                                            chat=lambda *_: next(replies))
        self.assertEqual(decision.status, EvidenceStatus.VERIFICATION_UNAVAILABLE)
        self.assertEqual(decision.documents, ())

    def test_model_failure_is_distinct_from_missing_information(self):
        def fails(*_):
            raise TimeoutError("local model timed out")
        decision = assess_semantic_evidence("Explain DGMS", self.passages, chat=fails)
        self.assertEqual(decision.status, EvidenceStatus.VERIFICATION_UNAVAILABLE)

    def test_wrapped_json_is_parsed_without_accepting_prose_as_evidence(self):
        wrapped = SimpleNamespace(message=SimpleNamespace(content=(
            '```json\n{"decision":"INSUFFICIENT_EVIDENCE","evidence":[],"missing_parts":[],"english_search_query":""}\n```'
        )))
        self.assertEqual(_parse_verdict(wrapped)["decision"], "INSUFFICIENT_EVIDENCE")

    def test_truncated_json_gets_one_compact_retry(self):
        replies = iter([
            SimpleNamespace(message=SimpleNamespace(content='{"decision":"PARTIAL_EVIDENCE","evidence":['),
                            done_reason="length", eval_count=256),
            response("PARTIAL_EVIDENCE", [{
                "passage_id": 0,
                "quote": "DGMS personnel receive modern tools and training for mineral exploration in the region.",
            }], ["The full definition is absent."]),
        ])
        calls = []
        def answer(*_):
            calls.append(1)
            return next(replies)
        decision = assess_semantic_evidence("Explain DGMS", self.passages, chat=answer)
        self.assertEqual(len(calls), 2)
        self.assertEqual(decision.status, EvidenceStatus.PARTIAL_EVIDENCE)

    def test_two_malformed_verdicts_still_fail_closed(self):
        decision = assess_semantic_evidence(
            "Explain DGMS", self.passages,
            chat=lambda *_: SimpleNamespace(message=SimpleNamespace(content='{"decision":')),
        )
        self.assertEqual(decision.status, EvidenceStatus.VERIFICATION_UNAVAILABLE)

    def test_complete_quote_in_truncated_json_is_only_partial_evidence(self):
        truncated = ('{"decision":"ANSWERABLE","evidence":[{"passage_id":0,'
                     '"quote":"DGMS personnel receive modern tools and training for '
                     'mineral exploration in the region."}')
        decision = assess_semantic_evidence(
            "Explain DGMS", self.passages,
            chat=lambda *_: SimpleNamespace(message=SimpleNamespace(content=truncated)),
        )
        self.assertEqual(decision.status, EvidenceStatus.PARTIAL_EVIDENCE)
        self.assertEqual(decision.documents, tuple(self.passages))

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
        self.assertEqual(_safe_search_query("What improves mine safety?",
                                            "What did DGMS change in 2025?"), "")

    def test_two_passages_can_jointly_support_different_requested_identifiers(self):
        passages = [
            Document(page_content="DGMS oversees safety inspections in mines.",
                     metadata={"source": "safety.pdf"}),
            Document(page_content="IBM maintains the national mineral inventory.",
                     metadata={"source": "inventory.pdf"}),
        ]
        decision = assess_semantic_evidence(
            "What do DGMS and IBM each do?", passages,
            chat=lambda *_: response("ANSWERABLE", [
                {"passage_id": 0, "quote": passages[0].page_content},
                {"passage_id": 1, "quote": passages[1].page_content},
            ]),
        )
        self.assertEqual(decision.status, EvidenceStatus.ANSWERABLE)
        self.assertEqual(decision.documents, tuple(passages))

    def test_english_indirect_question_retries_same_index_before_abstaining(self):
        weak = Document(page_content="The annual report discusses mine operations.",
                        metadata={"source": "annual.pdf"})
        useful = Document(page_content="Ventilation removes hazardous gases from mine workings.",
                          metadata={"source": "safety.pdf"})
        decisions = [
            EvidenceDecision(EvidenceStatus.INSUFFICIENT_EVIDENCE, (), "no_direct_support"),
            EvidenceDecision(EvidenceStatus.ANSWERABLE, (useful,), "semantic_quote_verified",
                             verified_quotes=(useful.page_content,)),
        ]

        def generate(inputs):
            list(inputs)
            yield AIMessageChunk(content="Ventilation removes hazardous gases from mine workings.")

        with patch("config.QA_ALLOW_MODEL_KNOWLEDGE", False), patch(
            "services.qa_service._load_environment"
        ), patch("services.qa_service._vector_store"), patch(
            "services.qa_service._retrieve_context", side_effect=[[weak], [useful]]
        ) as search, patch(
            "services.qa_service.suggest_local_search_query",
            return_value="mine ventilation hazardous gas removal"
        ) as rewrite, patch(
            "services.qa_service.assess_semantic_evidence", side_effect=decisions
        ) as verify, patch(
            "services.qa_service._llm", return_value=RunnableGenerator(generate)
        ):
            events = list(stream_answer_events("What keeps bad air from building up underground?"))
        self.assertEqual(events[-1]["type"], "done")
        self.assertIn("Ventilation removes", events[-1]["answer"])
        self.assertEqual(search.call_count, 2)
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(verify.call_args.args[1], [useful, weak])
        rewrite.assert_called_once()

    def test_second_search_without_support_still_abstains(self):
        weak = Document(page_content="The report lists equipment purchases.",
                        metadata={"source": "annual.pdf"})
        with patch("config.QA_ALLOW_MODEL_KNOWLEDGE", False), patch(
            "services.qa_service._load_environment"
        ), patch("services.qa_service._vector_store"), patch(
            "services.qa_service._retrieve_context", side_effect=[[weak], []]
        ), patch(
            "services.qa_service.suggest_local_search_query",
            return_value="mine ventilation hazardous gas removal"
        ), patch(
            "services.qa_service.assess_semantic_evidence",
            return_value=EvidenceDecision(EvidenceStatus.INSUFFICIENT_EVIDENCE, (),
                                          "no_direct_support")
        ), patch("services.qa_service._llm") as model:
            events = list(stream_answer_events("What keeps bad air from building up underground?"))
        model.assert_not_called()
        self.assertEqual(events[-1]["type"], "done")
        self.assertIn("don't know", events[-1]["answer"])

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

        with patch("config.QA_ALLOW_MODEL_KNOWLEDGE", False), patch("services.qa_service._load_environment"), patch(
            "services.qa_service._vector_store"
        ), patch("services.qa_service._retrieve_context", return_value=self.passages), patch(
            "services.qa_service.assess_semantic_evidence",
            return_value=assess_semantic_evidence(
                "DGMS के बारे में बताइए", self.passages,
                chat=lambda *_: response("PARTIAL_EVIDENCE", [{
                    "passage_id": 0,
                    "quote": "DGMS personnel receive modern tools and training for mineral exploration in the region.",
                }], ["The full form is not established."]),
            ),
        ), patch("services.qa_service._llm", return_value=RunnableGenerator(generate)):
            events = list(stream_answer_events("DGMS के बारे में बताइए"))

        self.assertEqual(events[-1]["type"], "done")
        self.assertIn("पूरा नाम नहीं", events[-1]["answer"])
        self.assertIn("The full form is not established.", " ".join(prompts))
        self.assertIn("DGMS personnel receive modern tools", " ".join(prompts))

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
        with patch("config.QA_ALLOW_MODEL_KNOWLEDGE", False), patch("services.qa_service._load_environment"), patch(
            "services.qa_service._vector_store"
        ), patch("services.qa_service._retrieve_context", side_effect=[[unrelated], [relevant]]) as search, patch(
            "services.qa_service.assess_semantic_evidence", side_effect=decisions
        ), patch("services.qa_service._llm", return_value=model):
            events = list(stream_answer_events("खनन सुरक्षा में विस्फोटकों की क्या भूमिका है?"))
        self.assertEqual(search.call_count, 2)
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["sources"], [])


if __name__ == "__main__":
    unittest.main()
