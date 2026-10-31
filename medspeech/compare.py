"""Paired ASR comparison with transcript-group bootstrap intervals."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from .metrics import score
from .protocol import normalize_text


def load_predictions(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError(f"No predictions in {path}")
    return rows


def paired_report(baseline: list[dict], candidate: list[dict],
                  terms: list[str], samples: int = 1000, seed: int = 42) -> dict:
    if samples < 1:
        raise ValueError("Bootstrap sample count must be positive")
    keys = lambda rows: [(r["source_split"], r["index"]) for r in rows]
    if keys(baseline) != keys(candidate):
        raise ValueError("Models must be scored on the same clips in the same order")
    references = [r["reference"] for r in baseline]
    if references != [r["reference"] for r in candidate]:
        raise ValueError("Reference transcripts differ between model runs")
    base_pred = [r["prediction"] for r in baseline]
    candidate_pred = [r["prediction"] for r in candidate]
    groups: dict[str, list[int]] = defaultdict(list)
    for index, reference in enumerate(references):
        groups[normalize_text(reference)].append(index)
    group_ids = sorted(groups)
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        selected = [i for _ in group_ids for i in groups[rng.choice(group_ids)]]
        refs = [references[i] for i in selected]
        base = [base_pred[i] for i in selected]
        other = [candidate_pred[i] for i in selected]
        deltas.append(score(refs, other)["wer"] - score(refs, base)["wer"])
    baseline_score = score(references, base_pred, terms)
    candidate_score = score(references, candidate_pred, terms)
    return {
        "clips": len(references),
        "unique_transcript_groups": len(group_ids),
        "baseline": baseline_score,
        "candidate": candidate_score,
        "candidate_minus_baseline_wer": candidate_score["wer"] - baseline_score["wer"],
        "paired_wer_delta_95pct_interval": [
            float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "bootstrap_samples": samples,
        "seed": seed,
        "interpretation": "Negative WER delta favors candidate; interval resamples normalized-transcript groups.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--terms", type=Path, default=Path("terms.txt"))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output file already exists; choose a new path")
    terms = [line.strip() for line in args.terms.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    result = paired_report(load_predictions(args.baseline),
                           load_predictions(args.candidate), terms,
                           samples=args.samples, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
