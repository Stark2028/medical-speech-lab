import unittest

from medspeech.metrics import score


class MetricTests(unittest.TestCase):
    def test_term_recall_counts_clip_term_hits_not_repeated_mentions(self):
        result = score(
            ["Headache, headache and sore throat", "No headache"],
            ["Headache and sore throat", "No pain"],
            ["headache", "sore throat"],
        )
        self.assertEqual(result["term_occurrences"], 3)
        self.assertEqual(result["term_retained"], 2)
        self.assertAlmostEqual(result["term_recall"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
