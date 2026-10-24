import unittest

from langchain_core.documents import Document

from services.evidence_gate import EvidenceStatus, assess_evidence, unsupported_answer_numbers


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
        self.assertTrue(unsupported_answer_numbers("The limit is 500 kg.", "What is the limit?", docs))

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
