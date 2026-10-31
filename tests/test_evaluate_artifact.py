import os
import unittest
from pathlib import Path

from medspeech.evaluate import hash_model_artifact


class ArtifactTests(unittest.TestCase):
    def test_hash_changes_with_weights_or_configuration(self):
        root = Path(__file__).resolve().parents[1] / ".tmp" / f"model-hash-{os.getpid()}"
        root.mkdir(parents=True, exist_ok=True)
        weights = root / "model.safetensors"
        config = root / "config.json"
        try:
            weights.write_bytes(b"weights")
            config.write_text('{"a":1}', encoding="utf-8")
            original = hash_model_artifact(root)
            self.assertEqual(original, hash_model_artifact(root))
            config.write_text('{"a":2}', encoding="utf-8")
            self.assertNotEqual(original, hash_model_artifact(root))
        finally:
            config.unlink(missing_ok=True)
            weights.unlink(missing_ok=True)
            root.rmdir()


if __name__ == "__main__":
    unittest.main()
