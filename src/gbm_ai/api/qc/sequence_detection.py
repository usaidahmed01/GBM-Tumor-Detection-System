from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ALLOWED_SEQUENCE_LABELS = {
    "T1",
    "T1C",
    "T2",
    "FLAIR",
    "OTHER",
    "NOT_USABLE",
}


@dataclass(frozen=True)
class SequenceDetectionResult:
    state: str
    suggested_sequence: str | None
    confidence: float
    evidence: list[str]
    scores: dict[str, float]


def _as_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        return {str(v).strip().lower() for v in value if str(v).strip()}
    return {str(value).strip().lower()} if str(value).strip() else set()


def _numbers(value: Any) -> list[float]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    result = []
    for item in values:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            continue
    return result


def _first_number(value: Any) -> float | None:
    values = _numbers(value)
    return values[0] if values else None


def _add(scores: dict[str, float], label: str, amount: float) -> None:
    scores[label] = min(1.0, scores[label] + amount)


def detect_series_sequence(metadata: dict[str, Any]) -> SequenceDetectionResult:
    """
    Conservative engineering heuristic for MRI sequence suggestions.

    This is not a clinically validated sequence classifier. High-confidence
    mappings are exposed as automatic detections; ambiguous mappings remain
    NEEDS_CONFIRMATION so a clinician can review them.
    """
    scores = {"T1": 0.0, "T1C": 0.0, "T2": 0.0, "FLAIR": 0.0}
    evidence: list[str] = []

    desc_tokens = _as_set(metadata.get("series_description_tokens"))
    protocol_tokens = _as_set(metadata.get("protocol_name_tokens"))
    tokens = desc_tokens | protocol_tokens

    image_type = _as_set(metadata.get("image_type"))
    scan_options = _as_set(metadata.get("scan_options"))
    scanning_sequence = _as_set(metadata.get("scanning_sequence"))
    mr_acq = _as_set(metadata.get("mr_acquisition_type"))

    tr = _first_number(metadata.get("repetition_time_ms"))
    te = _first_number(metadata.get("echo_time_ms"))
    ti = _first_number(metadata.get("inversion_time_ms"))
    contrast = bool(metadata.get("contrast_metadata_present"))

    if "flair" in tokens:
        _add(scores, "FLAIR", 0.95)
        evidence.append("explicit_flair_token")

    if "t2" in tokens or "t2w" in tokens:
        _add(scores, "T2", 0.82)
        evidence.append("explicit_t2_token")

    if "t1" in tokens or "t1w" in tokens:
        _add(scores, "T1", 0.82)
        evidence.append("explicit_t1_token")

    if "mprage" in tokens or "spgr" in tokens:
        _add(scores, "T1", 0.68)
        evidence.append("t1_weighted_sequence_family_token")

    post_tokens = {
        "post",
        "contrast",
        "contrasted",
        "gad",
        "gadolinium",
        "ce",
        "enhanced",
    }
    explicit_post = bool(tokens & post_tokens)

    if explicit_post:
        evidence.append("post_contrast_token")

    if contrast:
        evidence.append("contrast_metadata_present")

    # Parameter-based supporting evidence. These ranges are broad engineering
    # heuristics only and intentionally do not override explicit ambiguity.
    if tr is not None and te is not None:
        if tr >= 4000 and te >= 60:
            _add(scores, "T2", 0.34)
            evidence.append("long_tr_te_supports_t2")
        if tr <= 1800 and te <= 40:
            _add(scores, "T1", 0.34)
            evidence.append("short_tr_te_supports_t1")

    if ti is not None and tr is not None and te is not None:
        if ti >= 1200 and tr >= 4000 and te >= 60:
            _add(scores, "FLAIR", 0.62)
            evidence.append("long_inversion_long_tr_te_supports_flair")

    # T1 post-contrast is a derived candidate from T1-like evidence plus
    # contrast evidence.
    t1_evidence = scores["T1"]
    if (explicit_post or contrast) and t1_evidence >= 0.30:
        scores["T1C"] = min(
            1.0,
            t1_evidence + (0.28 if explicit_post else 0.20),
        )
        evidence.append("t1_plus_contrast_supports_t1c")

    # If explicit post-contrast evidence exists, suppress plain T1 slightly.
    if scores["T1C"] > 0:
        scores["T1"] = max(0.0, scores["T1"] - 0.18)

    # FLAIR is technically T2-weighted, so explicit FLAIR should dominate a
    # generic T2 token.
    if scores["FLAIR"] >= 0.90:
        scores["T2"] = min(scores["T2"], 0.65)

    # Minor support from technical values.
    if "ir" in scanning_sequence or "ir" in scan_options:
        _add(scores, "FLAIR", 0.08)
    if "3d" in mr_acq:
        # Does not identify sequence, but commonly supports T1 structural data.
        _add(scores, "T1", 0.03)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_label, best_score = ordered[0]
    second_score = ordered[1][1]
    gap = best_score - second_score

    best_score = round(best_score, 4)
    rounded_scores = {
        label: round(value, 4)
        for label, value in scores.items()
    }

    if best_score < 0.45:
        return SequenceDetectionResult(
            state="UNKNOWN",
            suggested_sequence=None,
            confidence=best_score,
            evidence=evidence,
            scores=rounded_scores,
        )

    if best_score >= 0.80 and gap >= 0.15:
        return SequenceDetectionResult(
            state=best_label,
            suggested_sequence=best_label,
            confidence=best_score,
            evidence=evidence,
            scores=rounded_scores,
        )

    return SequenceDetectionResult(
        state="NEEDS_CONFIRMATION",
        suggested_sequence=best_label,
        confidence=best_score,
        evidence=evidence,
        scores=rounded_scores,
    )
