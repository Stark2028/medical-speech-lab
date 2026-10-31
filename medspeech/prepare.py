"""Make a deterministic index manifest without storing audio or transcripts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .protocol import normalize_text, partition, verify_no_text_overlap

DATASET_ID = "Hani89/medical_asr_recording_dataset"
DATASET_REVISION = "8b2b2bd4233140705f1f7bb48411b49cd188d89c"


def build_manifest(dataset_id: str = DATASET_ID,
                   revision: str = DATASET_REVISION) -> dict:
    from datasets import load_dataset

    if not revision:
        raise ValueError("A pinned dataset revision is required")
    dataset = load_dataset(dataset_id, revision=revision)
    rows = []
    for source_split, subset in dataset.items():
        for index, sentence in enumerate(subset["sentence"]):
            if not isinstance(sentence, str) or not sentence.strip():
                continue
            rows.append({"source_split": source_split, "index": index,
                         "sentence": sentence, "partition": partition(sentence)})
    verify_no_text_overlap(rows)
    counts = Counter(row["partition"] for row in rows)
    groups = {name: len({normalize_text(row["sentence"]) for row in rows
                         if row["partition"] == name})
              for name in ("train", "validation", "test")}
    if any(counts[name] == 0 for name in groups):
        raise ValueError("A partition is empty; inspect the dataset before continuing")
    # Manifest stores only row indices and source fingerprints, not transcript text.
    selected = [{"source_split": row["source_split"], "index": row["index"],
                 "partition": row["partition"]} for row in rows]
    return {"dataset_id": dataset_id,
            "dataset_revision": revision,
            "source_fingerprints": {name: subset._fingerprint for name, subset in dataset.items()},
            "partition_method": "sha256-normalized-transcript-80-10-10",
            "counts": dict(counts), "transcript_groups": groups, "rows": selected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--revision", default=DATASET_REVISION)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output file already exists; choose a new manifest path")
    manifest = build_manifest(args.dataset_id, args.revision)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(manifest['rows'])} rows: {manifest['counts']}")


if __name__ == "__main__":
    main()
