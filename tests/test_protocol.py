import unittest

from medspeech.protocol import normalize_text, partition, verify_no_text_overlap


class ProtocolTests(unittest.TestCase):
    def test_same_phrase_stays_together(self):
        self.assertEqual(partition("My knee hurts!"), partition("my  knee hurts"))

    def test_normalization(self):
        self.assertEqual(normalize_text("  Sore-throat, AGAIN! "), "sore throat again")

    def test_overlap_is_rejected(self):
        rows = [{"sentence": "Knee pain!", "partition": "train"},
                {"sentence": "knee pain", "partition": "test"}]
        with self.assertRaises(ValueError):
            verify_no_text_overlap(rows)


if __name__ == "__main__":
    unittest.main()
