from __future__ import annotations

import pytest

from gbm_ai.safety.finalize_safety import (
    review_status,
    safety_metrics,
    validate_locked_test_not_used,
)


def test_safety_metrics_three_state_and_residual_errors():
    rows = [
        {
            "target": "1",
            "base_probability_state": "GBM_SUSPECTED",
            "final_safety_state": "GBM_SUSPECTED",
            "safety_override_applied": "False",
        },
        {
            "target": "1",
            "base_probability_state": "GBM_NOT_SUSPECTED",
            "final_safety_state": "INDETERMINATE",
            "safety_override_applied": "True",
        },
        {
            "target": "1",
            "base_probability_state": "GBM_NOT_SUSPECTED",
            "final_safety_state": "GBM_NOT_SUSPECTED",
            "safety_override_applied": "False",
        },
        {
            "target": "0",
            "base_probability_state": "GBM_NOT_SUSPECTED",
            "final_safety_state": "GBM_NOT_SUSPECTED",
            "safety_override_applied": "False",
        },
        {
            "target": "0",
            "base_probability_state": "GBM_SUSPECTED",
            "final_safety_state": "GBM_SUSPECTED",
            "safety_override_applied": "False",
        },
    ]

    result = safety_metrics(rows)
    assert result["total_oof_samples"] == 5
    assert result["safety_override_count"] == 1
    assert result["residual_safety_critical_false_negatives"] == 1
    assert result["residual_false_positive_suspected_cases"] == 1
    assert result["final_state_counts"]["INDETERMINATE"] == 1


def test_review_status_detects_pending_medical_review():
    rows = [
        {"medical_review_status": "PENDING"},
        {"medical_review_status": "REVIEWED"},
        {"medical_review_status": ""},
    ]
    result = review_status(rows)
    assert result["case_count"] == 3
    assert result["pending_count"] == 2
    assert result["complete"] is False


def test_review_status_complete_when_no_pending_cases():
    rows = [
        {"medical_review_status": "REVIEWED"},
        {"medical_review_status": "REVIEWED"},
    ]
    result = review_status(rows)
    assert result["pending_count"] == 0
    assert result["complete"] is True


def test_locked_test_gate_rejects_missing_or_true_flags():
    with pytest.raises(RuntimeError):
        validate_locked_test_not_used(
            {
                "a": {"locked_test_used": False},
                "b": {"locked_test_used": True},
            }
        )

    with pytest.raises(RuntimeError):
        validate_locked_test_not_used(
            {
                "a": {"locked_test_used": False},
                "b": {},
            }
        )


def test_locked_test_gate_accepts_all_false():
    validate_locked_test_not_used(
        {
            "a": {"locked_test_used": False},
            "b": {"locked_test_used": False},
        }
    )
