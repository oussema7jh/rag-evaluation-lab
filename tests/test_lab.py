import unittest

from lab import evaluate, rank


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            {"id": "a", "text": "Database backups and restoration"},
            {"id": "b", "text": "Security keys and authentication"},
        ]

    def test_retrieves_expected_document(self):
        for method in ("bm25", "hash", "hybrid"):
            self.assertEqual(rank(self.corpus, "database backups", method)[0], "a")

    def test_metrics_separate_answerable_and_unanswerable(self):
        questions = [
            {"id": "q1", "question": "database backups", "source_id": "a"},
            {"id": "q2", "question": "office lunch menu", "source_id": None},
        ]
        metrics, rows = evaluate(self.corpus, questions, "bm25", 1)
        self.assertEqual(metrics["recall_at_k"], 1)
        self.assertEqual(metrics["mrr"], 1)
        self.assertEqual(metrics["abstention_rate"], 1)
        self.assertEqual(rows[1]["abstained"], 1)


if __name__ == "__main__":
    unittest.main()

