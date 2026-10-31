import sys
import types
import unittest
from unittest.mock import patch

from medspeech.prepare import build_manifest


class FakeSubset:
    def __init__(self, sentences, fingerprint):
        self.sentences = sentences
        self._fingerprint = fingerprint

    def __getitem__(self, key):
        if key != "sentence":
            raise KeyError(key)
        return self.sentences


class PrepareTests(unittest.TestCase):
    def test_manifest_pins_revision_and_keeps_duplicate_phrases_together(self):
        data = {
            "train": FakeSubset(["Knee pain!", "Headache", "Sore throat"], "train-fp"),
            "test": FakeSubset(["knee pain", "Fever", "Back pain"], "test-fp"),
        }
        # A larger variety ensures each hash bucket has an example.
        data["train"].sentences.extend(f"symptom phrase {n}" for n in range(100))
        calls = []

        def load_dataset(dataset_id, revision):
            calls.append((dataset_id, revision))
            return data

        datasets = types.ModuleType("datasets")
        datasets.load_dataset = load_dataset
        hub = types.ModuleType("huggingface_hub")
        hub.HfApi = lambda: types.SimpleNamespace(
            dataset_info=lambda _dataset_id: types.SimpleNamespace(sha="fixed-sha"))
        with patch.dict(sys.modules, {"datasets": datasets, "huggingface_hub": hub}):
            manifest = build_manifest("owner/data")

        self.assertEqual(calls, [("owner/data", "fixed-sha")])
        self.assertEqual(manifest["dataset_revision"], "fixed-sha")
        self.assertEqual(manifest["source_fingerprints"],
                         {"train": "train-fp", "test": "test-fp"})
        self.assertEqual(manifest["rows"][0]["partition"],
                         manifest["rows"][103]["partition"])
        self.assertEqual(sum(manifest["counts"].values()), 106)


if __name__ == "__main__":
    unittest.main()
