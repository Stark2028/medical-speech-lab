import unittest

import numpy as np

from medspeech.audio import add_gaussian_noise, mono_16k


class AudioTests(unittest.TestCase):
    def test_channel_first_stereo_is_averaged(self):
        record = {"sampling_rate": 16000, "array": [[0.0, 0.5], [1.0, -0.5]]}
        np.testing.assert_allclose(mono_16k(record), [0.5, 0.0])

    def test_mono_is_preserved(self):
        record = {"sampling_rate": 16000, "array": [0.25, -0.25]}
        np.testing.assert_allclose(mono_16k(record), [0.25, -0.25])

    def test_unexpected_rate_fails(self):
        with self.assertRaisesRegex(ValueError, "16000 Hz"):
            mono_16k({"sampling_rate": 8000, "array": [0.0]})

    def test_invalid_audio_fails(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            mono_16k({"sampling_rate": 16000, "array": [float("nan")]})

    def test_seeded_noise_is_reproducible_and_at_target_snr(self):
        samples = np.ones(16000, dtype=np.float32) * 0.1
        mixed = add_gaussian_noise(samples, 10.0, 123)
        np.testing.assert_array_equal(mixed, add_gaussian_noise(samples, 10.0, 123))
        noise = mixed.astype(np.float64) - samples
        actual_snr = 20 * np.log10(np.sqrt(np.mean(samples ** 2)) /
                                   np.sqrt(np.mean(noise ** 2)))
        self.assertAlmostEqual(float(actual_snr), 10.0, places=3)


if __name__ == "__main__":
    unittest.main()
