import io
import unittest

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from medspeech.service import RateLimiter, create_app, decode_wav


def wav_bytes(rate=16000, seconds=1, amplitude=0.1):
    samples = np.full(rate * seconds, amplitude, dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, samples, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


class FakeEngine:
    model_id = "test-model"
    revision = "test-revision"

    def __init__(self):
        self.received = None

    def transcribe(self, samples):
        self.received = samples
        return "I have a headache"


class ServiceTests(unittest.TestCase):
    def test_rate_limit_window_and_memory_bound(self):
        limiter = RateLimiter(limit=2, window=60)
        self.assertTrue(limiter.allow("visitor", 0))
        self.assertTrue(limiter.allow("visitor", 1))
        self.assertFalse(limiter.allow("visitor", 2))
        self.assertTrue(limiter.allow("visitor", 61))
        for index in range(2050):
            self.assertTrue(limiter.allow(str(index), 100))
        self.assertLessEqual(len(limiter.hits), 2048)

    def test_decode_resamples_to_16k(self):
        samples, duration = decode_wav(wav_bytes(rate=48000))
        self.assertEqual(len(samples), 16000)
        self.assertEqual(duration, 1)

    def test_rejects_silence_and_long_clip(self):
        with self.assertRaisesRegex(ValueError, "silent"):
            decode_wav(wav_bytes(amplitude=0))
        with self.assertRaisesRegex(ValueError, "30 seconds"):
            decode_wav(wav_bytes(seconds=31))

    def test_api_transcribes_without_model_download(self):
        engine = FakeEngine()
        with TestClient(create_app(engine)) as client:
            self.assertEqual(client.get("/health/ready").status_code, 200)
            response = client.post(
                "/api/v1/transcribe", content=wav_bytes(),
                headers={"Content-Type": "audio/wav"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["text"], "I have a headache")
            self.assertEqual(len(engine.received), 16000)
            self.assertFalse(client.get("/api/v1/model").json()["clinical_use"])

    def test_api_rejects_wrong_type_and_large_body(self):
        with TestClient(create_app(FakeEngine())) as client:
            wrong = client.post(
                "/api/v1/transcribe", content=b"bad",
                headers={"Content-Type": "text/plain"})
            self.assertEqual(wrong.status_code, 415)
            large = client.post(
                "/api/v1/transcribe", content=b"0" * (8 * 1024 * 1024 + 1),
                headers={"Content-Type": "audio/wav"})
            self.assertEqual(large.status_code, 413)


if __name__ == "__main__":
    unittest.main()
