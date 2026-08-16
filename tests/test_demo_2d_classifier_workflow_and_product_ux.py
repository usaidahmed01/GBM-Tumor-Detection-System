from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from gbm_ai.api.models.analysis import BrainScopeStatus, QCState, StudyQCStatus
from gbm_ai.api.services.classifier_runtime import _effective_classifier_qc_state

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_confirmed_raster_scope_resolves_the_expected_partial_qc_reason():
    study = SimpleNamespace(
        qc_status=StudyQCStatus.PARTIAL,
        brain_scope_status=BrainScopeStatus.CLINICIAN_CONFIRMED,
        qc_summary={"partial_reasons": ["BRAIN_SCOPE_UNVERIFIED_FOR_RASTER"]},
    )
    assert _effective_classifier_qc_state(study) == QCState.PASS


def test_unresolved_raster_quality_reason_still_downgrades_classifier_qc():
    study = SimpleNamespace(
        qc_status=StudyQCStatus.PARTIAL,
        brain_scope_status=BrainScopeStatus.CLINICIAN_CONFIRMED,
        qc_summary={"partial_reasons": ["BRAIN_SCOPE_UNVERIFIED_FOR_RASTER", "RASTER_LOW_CONTRAST"]},
    )
    assert _effective_classifier_qc_state(study) == QCState.REVIEW


def test_classifier_release_manifest_pins_all_five_fold_checksums():
    payload = json.loads(_read("artifacts/deployment/efficientnetv2s_seed42/classifier_deployment_freeze.json"))
    assert len(payload["checkpoint_filenames"]) == 5
    assert set(payload["checkpoint_filenames"]) == set(payload["checkpoint_sha256"])
    assert all(len(value) == 64 for value in payload["checkpoint_sha256"].values())


def test_intake_ui_has_native_2d_classifier_workflow_and_no_3d_dead_end_for_raster():
    ui = _read("frontend/components/intake/AnalysisIntakeWorkspace.jsx")
    assert "Run 2D GBM analysis" in ui
    assert "2D classification is ready" in ui
    assert "Review required before classification" in ui
    assert "Save age & recheck" in ui
    assert "Estimated GBM likelihood" in ui
    assert "GBM not suspected" in ui
    assert "What this means" in ui
    assert "What to do next" in ui
    assert "Classifier running" in ui
    assert "classifierRuntime?.ready!==true" in ui
    assert "Restoring the latest workflow state" in ui
    assert "isTwoD ? <button" in ui


def test_intake_age_is_required_for_normal_adult_workflow():
    ui = _read("frontend/components/intake/AnalysisIntakeWorkspace.jsx")
    assert "required for adult scope" in ui
    assert 'min="18" max="100" required' in ui


def test_frontend_api_exposes_classifier_and_scope_resolution_calls():
    api = _read("frontend/lib/api.js")
    for symbol in (
        "updateStudyIntakeContext",
        "fetchClassifierRuntimeStatus",
        "runStudyClassifier",
        "fetchCurrentClassifierResult",
        "fetchCurrentStudyQc",
        "fetchCurrentCapabilities",
    ):
        assert f"function {symbol}" in api


def test_backend_has_current_classifier_result_and_context_update_endpoints():
    classifier_router = _read("src/gbm_ai/api/routers/classifier.py")
    intake_router = _read("src/gbm_ai/api/routers/intake.py")
    assert '"/studies/{study_uuid}/classifier/current"' in classifier_router
    assert '"/studies/{study_uuid}/intake-context"' in intake_router


def test_checkpoint_installer_uses_exact_hashes_not_filename_guessing():
    script = _read("scripts/install_classifier_runtime_assets.ps1")
    assert "Get-FileHash -Algorithm SHA256" in script
    assert "No arbitrary fold selection" in script
    assert "Classifier checkpoints ready" in script


def test_global_route_loading_and_interaction_feedback_are_present():
    loading = _read("frontend/app/loading.jsx")
    css = _read("frontend/app/globals.css")
    assert "Preparing the next workspace" in loading
    assert ".route-loading-shell" in css
    assert ".operation-progress" in css
    assert ".processing-confirm-strip" in css
    assert ".classification-summary-panels" in css
    assert ".classification-result__grid" in css
