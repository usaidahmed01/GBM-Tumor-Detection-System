from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from gbm_ai.api.models.analysis import (
    AnalysisRun,
    AnalysisStatus,
    CapabilityRoutingStatus,
    DecisionState,
    ModelVersion,
    QCState,
    SourceFormat,
    Study,
    StudyQCStatus,
)
from gbm_ai.api.models.audit import AuditAction, AuditActorType, AuditEntityType
from gbm_ai.api.models.localization import AnatomicalLocalization
from gbm_ai.api.models.quantification import TumorQuantification
from gbm_ai.api.models.segmentation import (
    Segmentation,
    SegmentationReviewStatus,
    SegmentationStatus,
)
from gbm_ai.api.services.audit import record_audit_event
from gbm_ai.api.services.classifier_runtime import deployment_strategy_frozen, load_deployment_manifest
from gbm_ai.api.services.current_segmentation import resolve_current_completed_segmentation


DECISION_FUSION_VERSION = "phase9_step1_guarded_decision_fusion_v1"
CLINICAL_NOTICE = (
    "AI-assisted decision support only. This result is not a definitive "
    "pathological diagnosis and does not replace specialist radiological, "
    "pathological, molecular or multidisciplinary review."
)


class DecisionFusionServiceError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _json_safe(value):
    if isinstance(value, uuid.UUID):
        return str(value)
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _current_segmentation(db: Session, study: Study) -> Segmentation | None:
    resolved = resolve_current_completed_segmentation(db, study, repair_summary=True)
    return resolved[1] if resolved is not None else None


def _current_quantification(
    db: Session,
    segmentation: Segmentation | None,
) -> TumorQuantification | None:
    if segmentation is None or segmentation.review_status == SegmentationReviewStatus.REJECTED:
        return None
    row = db.scalar(
        select(TumorQuantification)
        .where(TumorQuantification.segmentation_id == segmentation.id)
        .order_by(TumorQuantification.created_at.desc())
        .limit(1)
    )
    if row is None:
        return None
    current_checksums = {
        "WT": str(segmentation.wt_checksum_sha256).lower(),
        "TC": str(segmentation.tc_checksum_sha256).lower(),
        "ET": str(segmentation.et_checksum_sha256).lower(),
        "LABELMAP": str(segmentation.labelmap_checksum_sha256).lower(),
    }
    if (
        dict(row.source_mask_checksums or {}) != current_checksums
        or bool(row.source_clinician_modified) != bool(segmentation.clinician_modified)
        or not bool(row.physical_volume_generated)
    ):
        return None
    return row


def _current_localization(
    db: Session,
    segmentation: Segmentation | None,
    quantification: TumorQuantification | None,
) -> AnatomicalLocalization | None:
    if segmentation is None or quantification is None:
        return None
    row = db.scalar(
        select(AnatomicalLocalization)
        .where(AnatomicalLocalization.segmentation_id == segmentation.id)
        .order_by(AnatomicalLocalization.created_at.desc())
        .limit(1)
    )
    if (
        row is None
        or row.quantification_id != quantification.id
        or not bool(row.registration_qc_passed)
        or not bool(row.anatomical_localization_generated)
    ):
        return None
    return row


def _latest_classifier_run(db: Session, study: Study) -> AnalysisRun | None:
    return db.scalar(
        select(AnalysisRun)
        .where(
            AnalysisRun.study_id == study.id,
            AnalysisRun.classifier_model_version_id.is_not(None),
            AnalysisRun.status == AnalysisStatus.COMPLETE,
            AnalysisRun.calibrated_probability_gbm.is_not(None),
            AnalysisRun.decision_state.in_(
                [
                    DecisionState.GBM_SUSPECTED,
                    DecisionState.GBM_NOT_SUSPECTED,
                    DecisionState.INDETERMINATE,
                ]
            ),
            AnalysisRun.decision_fusion_version.is_(None),
        )
        .order_by(AnalysisRun.completed_at.desc(), AnalysisRun.created_at.desc())
        .limit(1)
    )


def _classifier_domain_allowed(study: Study) -> tuple[bool, str | None]:
    summary = dict(study.capability_summary or {})
    capabilities = dict(summary.get("capabilities") or {})
    if not deployment_strategy_frozen():
        return False, "CLASSIFIER_DEPLOYMENT_STRATEGY_NOT_FROZEN"

    if study.source_format == SourceFormat.IMAGE:
        state = str((capabilities.get("two_d_classification") or {}).get("state") or "")
        if state != "eligible":
            return False, "2D_CLASSIFIER_INPUT_NOT_ELIGIBLE"
        return True, None

    if study.source_format in {SourceFormat.DICOM, SourceFormat.NIFTI}:
        if not bool(summary.get("volumetric_to_2d_classifier_bridge_validated", False)):
            return False, "VOLUMETRIC_TO_2D_CLASSIFIER_BRIDGE_NOT_VALIDATED"
        return True, None

    return False, "CLASSIFIER_INPUT_DOMAIN_UNRESOLVED"


def _classifier_evidence(db: Session, study: Study) -> dict:
    domain_allowed, blocked_reason = _classifier_domain_allowed(study)
    if not domain_allowed:
        return {
            "available": False,
            "validated_for_current_input_domain": False,
            "analysis_run_uuid": None,
            "calibrated_probability_gbm": None,
            "classifier_state": None,
            "safety_reason_codes": [],
            "blocked_reason": blocked_reason,
            "model_version_uuid": None,
            "ood_likeness_candidate": None,
            "qc_state": None,
        }

    run = _latest_classifier_run(db, study)
    if run is None:
        return {
            "available": False,
            "validated_for_current_input_domain": True,
            "analysis_run_uuid": None,
            "calibrated_probability_gbm": None,
            "classifier_state": None,
            "safety_reason_codes": [],
            "blocked_reason": "CLASSIFIER_EVIDENCE_NOT_AVAILABLE",
            "model_version_uuid": None,
            "ood_likeness_candidate": None,
            "qc_state": None,
        }

    model = db.get(ModelVersion, run.classifier_model_version_id)
    if model is None or model.role.value != "classifier":
        return {
            "available": False,
            "validated_for_current_input_domain": True,
            "analysis_run_uuid": None,
            "calibrated_probability_gbm": None,
            "classifier_state": None,
            "safety_reason_codes": [],
            "blocked_reason": "CLASSIFIER_MODEL_PROVENANCE_INVALID",
            "model_version_uuid": None,
            "ood_likeness_candidate": None,
            "qc_state": None,
        }

    deployment = load_deployment_manifest()
    if model.version != deployment.deployment_version:
        return {
            "available": False,
            "validated_for_current_input_domain": True,
            "analysis_run_uuid": None,
            "calibrated_probability_gbm": None,
            "classifier_state": None,
            "safety_reason_codes": [],
            "blocked_reason": "CLASSIFIER_MODEL_NOT_FROZEN_DEPLOYMENT",
            "model_version_uuid": None,
            "ood_likeness_candidate": None,
            "qc_state": None,
        }

    return {
        "available": True,
        "validated_for_current_input_domain": True,
        "analysis_run_uuid": run.id,
        "calibrated_probability_gbm": float(run.calibrated_probability_gbm),
        "classifier_state": run.decision_state,
        "safety_reason_codes": list(run.safety_reason_codes or []),
        "blocked_reason": None,
        "model_version_uuid": run.classifier_model_version_id,
        "ood_likeness_candidate": run.ood_likeness_candidate,
        "qc_state": run.qc_state,
    }


def _segmentation_evidence(db: Session, study: Study) -> dict:
    segmentation = _current_segmentation(db, study)
    if segmentation is None:
        return {
            "available": False,
            "segmentation_uuid": None,
            "review_status": None,
            "clinician_modified": False,
            "lesion_evidence_present": None,
            "wt_voxel_count": None,
            "quantification_available": False,
            "localization_available": False,
            "model_version_uuid": None,
        }

    analysis = db.get(AnalysisRun, segmentation.analysis_run_id)
    quantification = _current_quantification(db, segmentation)
    localization = _current_localization(db, segmentation, quantification)
    review_status = segmentation.review_status.value
    reviewed_for_use = segmentation.review_status in {
        SegmentationReviewStatus.ACCEPTED,
        SegmentationReviewStatus.EDITED,
    }
    wt_voxels = int((segmentation.voxel_counts or {}).get("WT") or 0)

    return {
        "available": True,
        "segmentation_uuid": segmentation.id,
        "review_status": review_status,
        "clinician_modified": bool(segmentation.clinician_modified),
        "lesion_evidence_present": (wt_voxels > 0) if reviewed_for_use else None,
        "wt_voxel_count": wt_voxels,
        "quantification_available": quantification is not None,
        "localization_available": localization is not None,
        "model_version_uuid": analysis.segmentation_model_version_id if analysis else None,
    }


def _base_safety_reasons(study: Study, classifier: dict) -> list[str]:
    reasons: list[str] = []
    if study.qc_status == StudyQCStatus.FAIL:
        reasons.append("MRI_QC_FAILED")
    elif study.qc_status == StudyQCStatus.PARTIAL:
        reasons.append("MRI_QC_PARTIAL_REVIEW_REQUIRED")
    elif study.qc_status == StudyQCStatus.PENDING:
        reasons.append("MRI_QC_NOT_COMPLETE")

    if study.capability_routing_status != CapabilityRoutingStatus.READY:
        reasons.append("CAPABILITY_ROUTING_NOT_READY")

    for item in (study.capability_summary or {}).get("global_hard_block_reasons") or []:
        reasons.append(str(item))

    if classifier.get("available"):
        if classifier.get("qc_state") not in {QCState.PASS, "pass"}:
            reasons.append("CLASSIFIER_QC_NOT_PASS")
        if classifier.get("ood_likeness_candidate") is True:
            reasons.append("CLASSIFIER_OOD_LIKENESS")
        reasons.extend(str(item) for item in classifier.get("safety_reason_codes") or [])
    return sorted(set(reasons))


def _fuse(study: Study, classifier: dict, segmentation: dict) -> tuple[DecisionState, list[str]]:
    reasons = _base_safety_reasons(study, classifier)
    hard_safety = any(
        code in reasons
        for code in {
            "MRI_QC_FAILED",
            "MRI_QC_NOT_COMPLETE",
            "CAPABILITY_ROUTING_NOT_READY",
            "CLASSIFIER_QC_NOT_PASS",
            "CLASSIFIER_OOD_LIKENESS",
            "OUT_OF_SCOPE_PRIOR_TREATMENT",
            "OUT_OF_SCOPE_PEDIATRIC",
            "NON_BRAIN_OR_OUT_OF_SCOPE_STUDY",
        }
    )
    # A determinate classifier run carrying safety override reasons is internally
    # inconsistent. Fail safe to indeterminate rather than preserving a yes/no
    # state that its own safety layer says should be questioned.
    if classifier.get("available") and classifier.get("safety_reason_codes"):
        hard_safety = True
    if hard_safety:
        return DecisionState.INDETERMINATE, sorted(set(reasons))

    if not classifier.get("available"):
        blocked = str(classifier.get("blocked_reason") or "CLASSIFIER_EVIDENCE_NOT_AVAILABLE")
        reasons.append(blocked)
        if segmentation.get("available"):
            if segmentation.get("review_status") == SegmentationReviewStatus.REJECTED.value:
                reasons.append("SEGMENTATION_REJECTED_BY_CLINICIAN")
            elif segmentation.get("lesion_evidence_present") is True:
                reasons.append("SEGMENTATION_LESION_EVIDENCE_PRESENT")
                reasons.append("SEGMENTATION_IS_NOT_GBM_CLASSIFIER")
            elif segmentation.get("review_status") == SegmentationReviewStatus.UNREVIEWED.value:
                reasons.append("SEGMENTATION_REVIEW_REQUIRED")
        return DecisionState.INDETERMINATE, sorted(set(reasons))

    state: DecisionState = classifier["classifier_state"]
    lesion = segmentation.get("lesion_evidence_present")
    review_status = segmentation.get("review_status")

    # Conservative monotonic rule: segmentation/safety evidence can only
    # downgrade a determinate classifier state to indeterminate. It can never
    # flip suspected directly to not-suspected or vice versa.
    if state == DecisionState.GBM_SUSPECTED:
        if review_status == SegmentationReviewStatus.REJECTED.value:
            reasons.append("SEGMENTATION_REJECTED_BY_CLINICIAN")
            return DecisionState.INDETERMINATE, sorted(set(reasons))
        if lesion is False:
            reasons.append("CLASSIFIER_SEGMENTATION_DISCORDANCE_HIGH_PROBABILITY_NO_LESION")
            return DecisionState.INDETERMINATE, sorted(set(reasons))
        return state, sorted(set(reasons))

    if state == DecisionState.GBM_NOT_SUSPECTED:
        if lesion is True:
            reasons.append("OTHER_INTRACRANIAL_ABNORMALITY_CANNOT_BE_EXCLUDED")
        else:
            reasons.append("LOW_GBM_EVIDENCE_DOES_NOT_ESTABLISH_NORMAL_BRAIN")
        return state, sorted(set(reasons))

    reasons.append("CLASSIFIER_INDETERMINATE")
    if lesion is True:
        reasons.append("SEGMENTATION_LESION_EVIDENCE_PRESENT")
    return DecisionState.INDETERMINATE, sorted(set(reasons))


def _report_readiness(
    study: Study,
    classifier: dict,
    segmentation: dict,
    final_state: DecisionState,
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if final_state == DecisionState.PENDING:
        blockers.append("DECISION_FUSION_PENDING")
    if study.capability_routing_status == CapabilityRoutingStatus.PENDING:
        blockers.append("CAPABILITY_ROUTING_PENDING")

    # A final structured report is allowed to document a partial/indeterminate
    # analysis. Missing model branches are therefore report content, not an
    # automatic report blocker. The one hard human-review requirement here is
    # that any generated 3D segmentation must have an explicit review state.
    if (
        study.source_format in {SourceFormat.DICOM, SourceFormat.NIFTI}
        and segmentation.get("available")
        and segmentation.get("review_status") == SegmentationReviewStatus.UNREVIEWED.value
    ):
        blockers.append("SEGMENTATION_REVIEW_NOT_EXPLICIT")

    return not blockers, sorted(set(blockers))


def _wording(
    study: Study,
    final_state: DecisionState,
    classifier: dict,
    segmentation: dict,
) -> tuple[str, bool]:
    lesion = segmentation.get("lesion_evidence_present") is True
    if final_state == DecisionState.GBM_SUSPECTED:
        return (
            "GBM suspected by the current validated classifier evidence. "
            "Available segmentation and quantitative findings are supporting imaging outputs for clinician review, "
            "not definitive pathology.",
            False,
        )
    if final_state == DecisionState.GBM_NOT_SUSPECTED:
        return (
            "GBM not suspected by the current validated classifier. "
            "Another intracranial abnormality cannot be excluded and specialist radiological review remains required.",
            True,
        )

    if (
        study.source_format in {SourceFormat.DICOM, SourceFormat.NIFTI}
        and classifier.get("blocked_reason") == "VOLUMETRIC_TO_2D_CLASSIFIER_BRIDGE_NOT_VALIDATED"
    ):
        suffix = (
            " A reviewed glioma-like lesion segmentation is available, but segmentation is not a GBM-vs-other-tumor classifier."
            if lesion
            else ""
        )
        return (
            "GBM assessment is indeterminate for this volumetric study because the current GBM classifier was trained "
            "on standalone 2D images and the DICOM/NIfTI-to-classifier bridge has not been separately validated."
            + suffix,
            lesion,
        )

    return (
        "GBM assessment is indeterminate. Available evidence is incomplete, uncertain, out of scope, or discordant; "
        "manual clinician review is required.",
        lesion,
    )


def _public_payload(run: AnalysisRun) -> dict:
    summary = dict(run.decision_evidence_summary or {})
    classifier = dict(summary.get("classifier") or {})
    segmentation = dict(summary.get("segmentation") or {})
    # Internal model-version identifiers are retained in provenance JSON but
    # deliberately omitted from this concise public decision response.
    classifier.pop("model_version_uuid", None)
    classifier.pop("ood_likeness_candidate", None)
    classifier.pop("qc_state", None)
    classifier_state = classifier.get("classifier_state")
    if isinstance(classifier_state, DecisionState):
        classifier["classifier_state"] = classifier_state.value
    segmentation.pop("model_version_uuid", None)
    return {
        "version": run.decision_fusion_version,
        "analysis_run_uuid": run.id,
        "study_uuid": run.study_id,
        "source_format": summary.get("source_format"),
        "decision_state": run.decision_state,
        "calibrated_probability_gbm": run.calibrated_probability_gbm,
        "classifier": classifier,
        "segmentation": segmentation,
        "safety_reason_codes": list(run.safety_reason_codes or []),
        "other_intracranial_abnormality_not_excluded": bool(
            summary.get("other_intracranial_abnormality_not_excluded")
        ),
        "report_ready": bool(summary.get("report_ready")),
        "report_blockers": list(summary.get("report_blockers") or []),
        "user_facing_summary": str(summary.get("user_facing_summary") or ""),
        "clinical_notice": CLINICAL_NOTICE,
        "segmentation_is_gbm_diagnosis": False,
        "volumetric_classifier_bridge_validated": bool(
            summary.get("volumetric_classifier_bridge_validated")
        ),
        "clinical_validation_claimed": False,
        "fused_at": run.decision_fused_at,
    }


def fuse_study_decision(
    db: Session,
    study: Study,
    *,
    request_id: str | None = None,
) -> dict:
    classifier = _classifier_evidence(db, study)
    segmentation = _segmentation_evidence(db, study)
    final_state, reasons = _fuse(study, classifier, segmentation)
    report_ready, report_blockers = _report_readiness(
        study,
        classifier,
        segmentation,
        final_state,
    )
    wording, other_abnormality = _wording(
        study,
        final_state,
        classifier,
        segmentation,
    )
    now = datetime.now(timezone.utc)

    summary_classifier = dict(classifier)
    classifier_state = summary_classifier.get("classifier_state")
    if isinstance(classifier_state, DecisionState):
        summary_classifier["classifier_state"] = classifier_state.value
    for key in ("ood_likeness_candidate", "qc_state"):
        value = summary_classifier.get(key)
        if hasattr(value, "value"):
            summary_classifier[key] = value.value

    summary = {
        "version": DECISION_FUSION_VERSION,
        "source_format": study.source_format.value,
        "classifier": summary_classifier,
        "segmentation": dict(segmentation),
        "report_ready": report_ready,
        "report_blockers": report_blockers,
        "other_intracranial_abnormality_not_excluded": other_abnormality,
        "user_facing_summary": wording,
        "volumetric_classifier_bridge_validated": bool(
            (study.capability_summary or {}).get(
                "volumetric_to_2d_classifier_bridge_validated",
                False,
            )
        ),
        "segmentation_is_gbm_diagnosis": False,
        "clinical_validation_claimed": False,
    }

    classifier_model_version_id = classifier.get("model_version_uuid")
    segmentation_model_version_id = segmentation.get("model_version_uuid")
    run = AnalysisRun(
        study_id=study.id,
        classifier_model_version_id=classifier_model_version_id,
        segmentation_model_version_id=segmentation_model_version_id,
        status=AnalysisStatus.COMPLETE,
        qc_state=(
            QCState.FAIL
            if study.qc_status == StudyQCStatus.FAIL
            else QCState.REVIEW
            if study.qc_status in {StudyQCStatus.PARTIAL, StudyQCStatus.PENDING}
            else QCState.PASS
        ),
        ood_likeness_candidate=(
            classifier.get("ood_likeness_candidate")
            if classifier.get("available")
            else None
        ),
        calibrated_probability_gbm=(
            classifier.get("calibrated_probability_gbm")
            if classifier.get("available")
            else None
        ),
        decision_state=final_state,
        safety_reason_codes=reasons,
        decision_fusion_version=DECISION_FUSION_VERSION,
        decision_evidence_summary=_json_safe(summary),
        decision_fused_at=now,
        started_at=now,
        completed_at=now,
    )
    db.add(run)
    db.flush()
    record_audit_event(
        db,
        action=AuditAction.DECISION_FUSED,
        entity_type=AuditEntityType.ANALYSIS_RUN,
        entity_uuid=run.id,
        actor_type=AuditActorType.SYSTEM,
        request_id=request_id,
        technical_context={
            "decision_state": final_state.value,
            "decision_fusion_version": DECISION_FUSION_VERSION,
            "report_ready": report_ready,
            "classifier_evidence_available": bool(classifier.get("available")),
            "segmentation_evidence_available": bool(segmentation.get("available")),
        },
        commit=False,
    )
    db.commit()
    db.refresh(run)
    return _public_payload(run)


def get_current_fused_decision(db: Session, study: Study) -> dict:
    run = db.scalar(
        select(AnalysisRun)
        .where(
            AnalysisRun.study_id == study.id,
            AnalysisRun.decision_fusion_version == DECISION_FUSION_VERSION,
            AnalysisRun.status == AnalysisStatus.COMPLETE,
        )
        .order_by(AnalysisRun.decision_fused_at.desc(), AnalysisRun.created_at.desc())
        .limit(1)
    )
    if run is None:
        raise DecisionFusionServiceError(
            "DECISION_FUSION_NOT_AVAILABLE",
            "no fused decision has been generated for this study",
        )
    return _public_payload(run)
