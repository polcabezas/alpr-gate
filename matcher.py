"""Fuzzy plate matching.

Real-world OCR regularly confuses O↔0, I↔1, B↔8, Z↔2, S↔5. A strict equality
check would reject valid reads over a single character; instead we normalize
then use a similarity ratio with a fairly tight threshold.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Iterable


_OCR_AMBIGUOUS = str.maketrans({
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
})


def normalize(plate: str) -> str:
    return "".join(c for c in plate.upper() if c.isalnum())


def _canonical(plate: str) -> str:
    """Collapse visually ambiguous characters to a single canonical form.
    Only used for matching, never shown to the user."""
    return normalize(plate).translate(_OCR_AMBIGUOUS)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def best_match(detected: str, allowed: Iterable[str], threshold: float = 0.85) -> str | None:
    """Return the allowed plate that best matches `detected`, else None.

    - Exact normalized match wins immediately.
    - Otherwise, canonical-form match (after ambiguous-char folding) wins immediately.
    - Otherwise, fall back to a similarity ratio above `threshold`.
    """
    det = normalize(detected)
    if not det:
        return None

    det_canon = _canonical(detected)

    best_plate: str | None = None
    best_score = 0.0
    for candidate in allowed:
        cand = normalize(candidate)
        if not cand:
            continue
        if det == cand:
            return candidate
        if det_canon == _canonical(candidate):
            return candidate
        score = similarity(det, cand)
        if score > best_score:
            best_score = score
            best_plate = candidate

    if best_score >= threshold:
        return best_plate
    return None
