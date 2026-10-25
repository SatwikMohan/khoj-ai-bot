import unittest

from langchain_core.documents import Document

from services.evidence_gate import EvidenceStatus, NumberEvidenceStream, assess_evidence, unsupported_answer_numbers


class EvidenceGateTests(unittest.TestCase):
    def test_direct_fact_can_be_answered(self):
        docs = [Document(page_content="The storage limit is 100 kg.", metadata={"source": "rules.pdf"})]
        self.assertEqual(assess_evidence("What is the storage limit?", docs).status,
                         EvidenceStatus.ANSWERABLE)

    def test_related_passage_does_not_support_missing_relation(self):
        docs = [Document(page_content="The dragline removes overburden.",
                         metadata={"source": "Dragline.docx"})]
        self.assertEqual(assess_evidence("What is the dragline warranty period?", docs).status,
                         EvidenceStatus.INSUFFICIENT_EVIDENCE)

    def test_mines_are_not_minerals(self):
        docs = [Document(page_content="India listed 30 critical minerals in 2023.",
                         metadata={"source": "minerals.pdf"})]
        self.assertEqual(assess_evidence("How many critical mines were listed in 2023?", docs).status,
                         EvidenceStatus.INSUFFICIENT_EVIDENCE)

    def test_missing_edition_cannot_use_other_year(self):
        docs = [Document(page_content="Mine safety rules require training.",
                         metadata={"source": "safety-2024.pdf", "year": "2024"})]
        self.assertEqual(assess_evidence("What does the 2099 edition of the mine safety rules require?", docs).status,
                         EvidenceStatus.REQUIRED_VERSION_NOT_FOUND)

    def test_supported_part_and_missing_part(self):
        docs = [Document(page_content="A continuous miner cuts coal underground.",
                         metadata={"source": "UG Mines.docx"})]
        result = assess_evidence("What is a continuous miner, and what is its serial number?", docs)
        self.assertEqual(result.status, EvidenceStatus.PARTIAL_EVIDENCE)
        self.assertEqual(len(result.missing_parts), 1)

    def test_numeric_claim_validation(self):
        docs = [Document(page_content="The storage limit is 100 kg.", metadata={})]
        self.assertFalse(unsupported_answer_numbers("The limit is 100 kg.", "What is the limit?", docs))
        self.assertFalse(unsupported_answer_numbers("सीमा १०० किलोग्राम है।", "What is the limit?", docs))
        self.assertTrue(unsupported_answer_numbers("The limit is 500 kg.", "What is the limit?", docs))
        self.assertTrue(unsupported_answer_numbers("The limit is 100 days.", "What is the limit?", docs))

    def test_numeric_claim_words_and_list_markers_are_not_facts(self):
        docs = [Document(page_content="India identified 30 critical minerals.", metadata={})]
        answer = "India identified 30 minerals.\n1. They are important.\n6. This is a list item."
        self.assertFalse(unsupported_answer_numbers(answer, "Explain critical minerals", docs))
        self.assertFalse(unsupported_answer_numbers("भारत ने ३० खनिज पहचाने।", "Explain critical minerals", docs))
        self.assertTrue(unsupported_answer_numbers("India identified 31 minerals.", "Explain critical minerals", docs))

    def test_number_stream_keeps_approved_sentence_when_later_number_is_wrong(self):
        docs = [Document(page_content="The storage limit is 100 kg.", metadata={})]
        stream = NumberEvidenceStream("What is the storage limit?", docs)
        self.assertEqual(stream.push("The storage limit is 100 kg. "), ["The storage limit is 100 kg."])
        self.assertEqual(stream.push("The warranty is 500 days."), [])
        self.assertEqual(stream.accepted, "The storage limit is 100 kg.")
        self.assertTrue(stream.rejected)

    def test_same_scope_measurements_conflict(self):
        docs = [
            Document(page_content="The storage limit is 100 kg.", metadata={"source": "a.pdf"}),
            Document(page_content="The storage limit is 200 kg.", metadata={"source": "b.pdf"}),
        ]
        self.assertEqual(assess_evidence("What is the storage limit?", docs).status,
                         EvidenceStatus.CONFLICTING_EVIDENCE)

    def test_different_years_are_not_silently_called_same_scope_conflict(self):
        docs = [
            Document(page_content="The storage limit is 100 kg.",
                     metadata={"source": "a.pdf", "year": "2023"}),
            Document(page_content="The storage limit is 200 kg.",
                     metadata={"source": "a.pdf", "year": "2024"}),
        ]
        self.assertEqual(assess_evidence("Compare the storage limits for 2023 and 2024", docs).status,
                         EvidenceStatus.ANSWERABLE)


if __name__ == "__main__":
    unittest.main()
