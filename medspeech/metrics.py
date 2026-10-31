"""Corpus-level ASR metrics and exact normalized term recall."""

from __future__ import annotations

import re

from .protocol import normalize_text


def score(references: list[str], predictions: list[str], terms: list[str] | None = None) -> dict:
    if len(references) != len(predictions) or not references:
        raise ValueError("References and predictions must have equal nonzero length")
    from jiwer import cer, wer

    refs = [normalize_text(text) for text in references]
    preds = [normalize_text(text) for text in predictions]
    result = {"n": len(refs), "wer": wer(refs, preds), "cer": cer(refs, preds)}
    if terms:
        normalized_terms = sorted({normalize_text(t) for t in terms if normalize_text(t)})
        present = 0
        retained = 0
        for ref, pred in zip(refs, preds):
            for term in normalized_terms:
                pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
                if re.search(pattern, ref):
                    present += 1
                    retained += bool(re.search(pattern, pred))
        result["term_occurrences"] = present
        result["term_recall"] = retained / present if present else None
    return result
