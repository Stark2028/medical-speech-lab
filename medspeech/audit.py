"""Summarize audio quality without exporting waveforms or transcript text."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .audio import mono_16k


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"],
                        default="train")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output file already exists; choose a new path")

    from datasets import load_dataset

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source = load_dataset(manifest["dataset_id"],
                          revision=manifest["dataset_revision"])
    actual = {name: subset._fingerprint for name, subset in source.items()}
    if actual != manifest["source_fingerprints"]:
        raise RuntimeError("Dataset fingerprint differs from manifest")

    channels = Counter()
    rates = Counter()
    durations = []
    invalid = []
    silent = 0
    clipped = 0
    for row in manifest["rows"]:
        if row["partition"] != args.split:
            continue
        record = source[row["source_split"]][row["index"]]["audio"]
        rates[str(record.get("sampling_rate"))] += 1
        raw = np.asarray(record.get("array"))
        channels[str(raw.shape[0] if raw.ndim == 2 else 1)] += 1
        try:
            samples = mono_16k(record)
        except (TypeError, ValueError) as error:
            invalid.append({"source_split": row["source_split"],
                            "index": row["index"], "reason": str(error)})
            continue
        durations.append(samples.size / 16000)
        peak = float(np.max(np.abs(samples)))
        silent += peak < 1e-5
        clipped += peak >= 0.999

    report = {
        "dataset_id": manifest["dataset_id"],
        "dataset_revision": manifest["dataset_revision"],
        "partition": args.split,
        "rows": sum(1 for row in manifest["rows"] if row["partition"] == args.split),
        "valid_audio": len(durations),
        "invalid": invalid,
        "channels": dict(channels),
        "sample_rates": dict(rates),
        "silent_peak_below_1e-5": silent,
        "peak_at_least_0_999": clipped,
        "duration_seconds": {
            "min": min(durations) if durations else None,
            "median": float(np.median(durations)) if durations else None,
            "max": max(durations) if durations else None,
            "total_hours": sum(durations) / 3600,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
