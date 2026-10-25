import unittest

from langchain_core.documents import Document

from services.query_correction import correct_query_from_passages


class QueryCorrectionTests(unittest.TestCase):
    def test_contextual_typo_uses_indexed_phrase(self):
        docs = [Document(page_content="Rare earth metals include neodymium and dysprosium.")]
        self.assertEqual(correct_query_from_passages("Explain Rear Earth Metals", docs),
                         "Explain Rare Earth Metals")

    def test_does_not_change_identifiers_numbers_or_negation(self):
        docs = [Document(page_content="Rare earth metals appear in the 2024 DGMS report.")]
        question = "Did DGMS not list Rear Earth Metals in 2024?"
        self.assertEqual(correct_query_from_passages(question, docs),
                         "Did DGMS not list Rare Earth Metals in 2024?")

    def test_unrelated_or_ambiguous_passages_do_not_rewrite_question(self):
        docs = [Document(page_content="The rear axle needs repair. Rare metals occur elsewhere.")]
        question = "Explain rear axle metals"
        self.assertEqual(correct_query_from_passages(question, docs), question)

    def test_single_term_needs_unique_repeated_near_match(self):
        docs = [Document(page_content="Minerals are useful. Critical minerals are important.")]
        self.assertEqual(correct_query_from_passages("minrals", docs), "minerals")
        self.assertEqual(correct_query_from_passages("minrals", []), "minrals")


if __name__ == "__main__":
    unittest.main()
