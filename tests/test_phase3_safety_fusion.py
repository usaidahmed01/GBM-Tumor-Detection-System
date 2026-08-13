from gbm_ai.safety.safety_fusion import fuse_case, probability_state


def test_probability_three_band_logic():
    assert probability_state(0.10, 0.13, 0.57) == "GBM_NOT_SUSPECTED"
    assert probability_state(0.30, 0.13, 0.57) == "INDETERMINATE"
    assert probability_state(0.70, 0.13, 0.57) == "GBM_SUSPECTED"


def test_ood_only_downgrades_to_indeterminate():
    result = fuse_case(0.75, 0.13, 0.57, "PASS", False, False, True)
    assert result["base_probability_state"] == "GBM_SUSPECTED"
    assert result["final_safety_state"] == "INDETERMINATE"
    assert result["safety_override_applied"] is True
    assert "OOD_LIKENESS" in result["safety_reason_codes"]


def test_clean_determinate_case_is_preserved():
    result = fuse_case(0.08, 0.13, 0.57, "PASS", False, False, False)
    assert result["final_safety_state"] == "GBM_NOT_SUSPECTED"
    assert result["safety_override_applied"] is False


def test_middle_band_stays_indeterminate():
    result = fuse_case(0.30, 0.13, 0.57, "PASS", False, False, False)
    assert result["final_safety_state"] == "INDETERMINATE"
    assert "PROBABILITY_MIDDLE_BAND" in result["safety_reason_codes"]


def test_qc_review_forces_indeterminate():
    result = fuse_case(0.05, 0.13, 0.57, "REVIEW", False, False, False)
    assert result["final_safety_state"] == "INDETERMINATE"
    assert "QC_REVIEW" in result["safety_reason_codes"]
