import unittest

from medspeech.compare import paired_report


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.base = [
            {"source_split": "train", "index": 1, "reference": "knee pain",
             "prediction": "knee pain"},
            {"source_split": "test", "index": 4, "reference": "knee pain",
             "prediction": "knee pain"},
            {"source_split": "train", "index": 2, "reference": "fever",
             "prediction": "never"},
        ]

    def test_identical_predictions_have_zero_delta_and_two_groups(self):
        result = paired_report(self.base, self.base, ["fever"], samples=20)
        self.assertEqual(result["unique_transcript_groups"], 2)
        self.assertEqual(result["candidate_minus_baseline_wer"], 0)
        self.assertEqual(result["paired_wer_delta_95pct_interval"], [0.0, 0.0])

    def test_misaligned_clips_fail(self):
        with self.assertRaisesRegex(ValueError, "same clips"):
            paired_report(self.base, list(reversed(self.base)), [], samples=2)

    def test_improvement_has_negative_delta(self):
        candidate = [dict(row) for row in self.base]
        candidate[-1]["prediction"] = "fever"
        result = paired_report(self.base, candidate, ["fever"], samples=20)
        self.assertLess(result["candidate_minus_baseline_wer"], 0)


if __name__ == "__main__":
    unittest.main()
