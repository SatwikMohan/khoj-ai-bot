"""Conversation references affect question understanding, never factual evidence."""

import unittest
from unittest.mock import patch

import config
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableGenerator

from services.evidence_gate import EvidenceDecision, EvidenceStatus
from services.qa_service import _contextual_question, _is_summary_query, stream_answer_events


class FollowupGroundingTests(unittest.TestCase):
    def test_restatement_in_another_language_keeps_previous_user_topic(self):
        history = [
            {"role": "user", "content": "What do you know about mining?"},
            {"role": "assistant", "content": "A fabricated mining law was passed."},
        ]
        self.assertEqual(
            _contextual_question("Could you explain it again in Hindi?", history),
            "What do you know about mining?",
        )
        for request in (
            "Could you explain again in English?",
            "English please",
            "Please say the answer once more in English",
            "Please make your reply English",
            "Repeat the information in English",
            "Summarize again",
        ):
            self.assertEqual(_contextual_question(request, history),
                             "What do you know about mining?", request)
        self.assertTrue(_is_summary_query("What do you know about mining?"))
        self.assertEqual(
            _contextual_question("aap isko english mein dubara samjha sakti hai", [
                {"role": "user", "content": "aap mining ke bare mein kya janti hai"},
            ]),
            "aap mining ke bare mein kya janti hai",
        )

    def test_hindi_script_followup_resolves_previous_subject(self):
        history = [{"role": "user", "content": "DGMS mine safety rules"}]
        self.assertIn("DGMS", _contextual_question("उसमें क्या बदलाव हुआ?", history))

    def test_new_year_replaces_old_year_without_losing_subject(self):
        history = [{"role": "user", "content": "What was the mine output in 2024?"}]
        resolved = _contextual_question("What about 2025?", history)
        self.assertIn("mine output", resolved)
        self.assertIn("2025", resolved)
        self.assertNotIn("2024", resolved)

    def test_independent_question_does_not_inherit_old_topic(self):
        history = [{"role": "user", "content": "Explain mine safety rules"}]
        for question in ("What are rare earth metals?", "Explain rare earth metals again"):
            self.assertEqual(_contextual_question(question, history), question)

    def test_restatement_uses_indexed_evidence_and_current_language(self):
        doc = Document(page_content="Mining is the extraction of minerals from the earth.",
                       metadata={"source": "manual.pdf"})
        searches = []
        prompts = []

        def retrieve(_store, query, _top_k, **_kwargs):
            searches.append(query)
            return [doc]

        def verify(question, documents, **_kwargs):
            self.assertEqual(question, "What do you know about mining?")
            self.assertEqual(documents, [doc])
            return EvidenceDecision(EvidenceStatus.ANSWERABLE, (doc,), "source_quote_verified")

        def generate(inputs):
            prompts.extend(str(item) for item in inputs)
            yield AIMessageChunk(content="Mining is the extraction of minerals from the earth.")

        history = [
            {"role": "user", "content": "What do you know about mining?"},
            {"role": "assistant", "content": "A fabricated mining law was passed."},
        ]
        with patch.multiple(config, RESPONSE_LANGUAGE="auto", QA_ALLOW_MODEL_KNOWLEDGE=False,
                            QA_SEMANTIC_EVIDENCE_ENABLED=True), patch(
            "services.qa_service._load_environment"
        ), patch("services.qa_service._vector_store"), patch(
            "services.qa_service._retrieve_summary_context", side_effect=retrieve
        ), patch("services.qa_service.assess_semantic_evidence", side_effect=verify), patch(
            "services.qa_service._llm", return_value=RunnableGenerator(generate)
        ):
            events = list(stream_answer_events(
                "Could you explain again in English?", chat_history=history,
            ))

        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["language"], "en")
        self.assertIn("extraction of minerals", events[-1]["answer"])
        self.assertEqual(searches, ["What do you know about mining?"])
        self.assertIn("Current request: Could you explain again in English?", " ".join(prompts))
        self.assertNotIn("fabricated mining law", " ".join(prompts))


if __name__ == "__main__":
    unittest.main()
