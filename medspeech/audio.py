"""Validate and convert the source dataset's channel-first audio to mono."""

from __future__ import annotations

import numpy as np


def mono_16k(record: dict) -> np.ndarray:
    """Return finite 16 kHz float32 mono samples, or fail with a useful error.

    This dataset stores arrays as [channel][sample], despite the card's 1D claim.
    Two-channel recordings are averaged so model input is deterministic.
    """
    rate = record.get("sampling_rate")
    if rate != 16000:
        raise ValueError(f"Expected 16000 Hz audio, got {rate!r}")
    samples = np.asarray(record.get("array"), dtype=np.float32)
    if samples.ndim == 2 and samples.shape[0] in (1, 2):
        samples = samples.mean(axis=0)
    elif samples.ndim != 1:
        raise ValueError(f"Expected mono or channel-first stereo audio, got {samples.shape}")
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("Audio is empty or contains non-finite values")
    return samples


def add_gaussian_noise(samples: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """Add seeded white noise at a measured RMS signal-to-noise ratio.

    The float waveform is not clipped, so the requested SNR remains defined.
    This is a synthetic stress test, not a simulation of a clinic.
    """
    if not np.isfinite(snr_db):
        raise ValueError("SNR must be finite")
    signal = np.asarray(samples, dtype=np.float32)
    if signal.ndim != 1 or signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError("Expected finite nonempty mono audio")
    rms = float(np.sqrt(np.mean(signal.astype(np.float64) ** 2)))
    if rms == 0:
        raise ValueError("Cannot define SNR for silent audio")
    noise = np.random.default_rng(seed).standard_normal(signal.size)
    noise = noise - noise.mean()
    noise_rms = float(np.sqrt(np.mean(noise ** 2)))
    target_noise_rms = rms / (10 ** (snr_db / 20))
    return (signal + noise * (target_noise_rms / noise_rms)).astype(np.float32)
