"""Deterministic transcript-disjoint partitioning and text normalization."""

from __future__ import annotations

import hashlib
import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normalize for ASR scoring; keep word boundaries and spoken content."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def partition(text: str) -> str:
    """Assign identical normalized transcripts to the same fixed partition."""
    key = normalize_text(text)
    if not key:
        raise ValueError("Empty transcript after normalization")
    bucket = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def verify_no_text_overlap(rows: list[dict]) -> None:
    """Fail if a normalized transcript occurs in multiple partitions."""
    seen: dict[str, str] = {}
    for row in rows:
        key = normalize_text(row["sentence"])
        current = row["partition"]
        previous = seen.setdefault(key, current)
        if previous != current:
            raise ValueError(f"Transcript crosses partitions: {key!r}")
