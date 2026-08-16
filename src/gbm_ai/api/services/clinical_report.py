from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from gbm_ai.api.models.analysis import AnalysisRun, ModelVersion, Study
from gbm_ai.api.models.audit import AuditAction, AuditActorType, AuditEntityType
from gbm_ai.api.models.clinical import Assessment, Patient
from gbm_ai.api.models.localization import AnatomicalLocalization
from gbm_ai.api.models.quantification import TumorQuantification
from gbm_ai.api.models.report import ClinicalReport, ReportStatus
from gbm_ai.api.models.segmentation import Segmentation
from gbm_ai.api.services.audit import record_audit_event
from gbm_ai.api.services.decision_fusion import DECISION_FUSION_VERSION


CLINICAL_REPORT_VERSION = "phase9_step3_structured_clinical_report_v1"
CLINICAL_NOTICE = (
    "AI-assisted clinical decision support only. This report is not a definitive pathological diagnosis "
    "and does not replace specialist radiological, pathological, molecular, or multidisciplinary review."
)


class ClinicalReportError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _current_decision_run(db: Session, study: Study) -> AnalysisRun:
    run = db.scalar(
        select(AnalysisRun)
        .where(
            AnalysisRun.study_id == study.id,
            AnalysisRun.decision_fusion_version == DECISION_FUSION_VERSION,
        )
        .order_by(AnalysisRun.decision_fused_at.desc(), AnalysisRun.created_at.desc())
        .limit(1)
    )
    if run is None:
        raise ClinicalReportError(
            "DECISION_FUSION_REQUIRED",
            "Generate the current guarded decision before preparing a final report.",
        )
    return run


def _segmentation_from_decision(db: Session, run: AnalysisRun) -> Segmentation | None:
    summary = dict(run.decision_evidence_summary or {})
    raw = (summary.get("segmentation") or {}).get("segmentation_uuid")
    if not raw:
        return None
    try:
        return db.get(Segmentation, uuid.UUID(str(raw)))
    except (ValueError, TypeError):
        return None


def _latest_quantification(db: Session, segmentation: Segmentation | None) -> TumorQuantification | None:
    if segmentation is None:
        return None
    return db.scalar(
        select(TumorQuantification)
        .where(TumorQuantification.segmentation_id == segmentation.id)
        .order_by(TumorQuantification.created_at.desc())
        .limit(1)
    )


def _latest_localization(db: Session, segmentation: Segmentation | None) -> AnatomicalLocalization | None:
    if segmentation is None:
        return None
    return db.scalar(
        select(AnatomicalLocalization)
        .where(AnatomicalLocalization.segmentation_id == segmentation.id)
        .order_by(AnatomicalLocalization.created_at.desc())
        .limit(1)
    )


def _model_trace(db: Session, model_id) -> dict | None:
    if not model_id:
        return None
    model = db.get(ModelVersion, model_id)
    if model is None:
        return None
    return {
        "name": model.model_name,
        "version": model.version,
        "architecture": model.architecture,
        "preprocessing_version": model.preprocessing_version,
        "threshold_version": model.threshold_version,
        "calibration_version": model.calibration_version,
        "weights_checksum_sha256": model.weights_checksum_sha256,
    }


def _report_payload(db: Session, study: Study, decision: AnalysisRun) -> tuple[dict, list[str]]:
    assessment = db.get(Assessment, study.assessment_id)
    if assessment is None:
        raise ClinicalReportError("ASSESSMENT_NOT_FOUND", "Study assessment was not found.")
    patient = db.get(Patient, assessment.patient_id)
    if patient is None:
        raise ClinicalReportError("PATIENT_NOT_FOUND", "Assessment patient was not found.")

    summary = dict(decision.decision_evidence_summary or {})
    segmentation = _segmentation_from_decision(db, decision)
    quantification = _latest_quantification(db, segmentation)
    localization = _latest_localization(db, segmentation)

    report_blockers = list(summary.get("report_blockers") or [])
    report_ready = bool(summary.get("report_ready")) and not report_blockers

    tumor_analysis = None
    if segmentation is not None:
        tumor_analysis = {
            "segmentation_review_status": segmentation.review_status.value,
            "clinician_modified": bool(segmentation.clinician_modified),
            "wt_voxel_count": int((segmentation.voxel_counts or {}).get("WT") or 0),
            "tc_voxel_count": int((segmentation.voxel_counts or {}).get("TC") or 0),
            "et_voxel_count": int((segmentation.voxel_counts or {}).get("ET") or 0),
            "quantification": (
                {
                    "wt_volume_cm3": quantification.wt_volume_cm3,
                    "tc_volume_cm3": quantification.tc_volume_cm3,
                    "et_volume_cm3": quantification.et_volume_cm3,
                    "wt_max_axial_area_mm2": quantification.wt_max_axial_area_mm2,
                }
                if quantification is not None
                else None
            ),
            "localization": (
                {
                    "hemisphere": localization.hemisphere,
                    "primary_region": localization.primary_region,
                    "centroid_mni_mm": list(localization.centroid_mni_mm or []),
                    "registration_qc_passed": bool(localization.registration_qc_passed),
                }
                if localization is not None
                else None
            ),
        }

    classifier_summary = dict(summary.get("classifier") or {})
    segmentation_summary = dict(summary.get("segmentation") or {})

    payload = {
        "report_version": CLINICAL_REPORT_VERSION,
        "patient_study": {
            "case_reference": patient.patient_id,
            "patient_name": patient.patient_name,
            "age_years": patient.age_years,
            "sex": patient.sex.value,
            "mri_date": assessment.mri_date.isoformat(),
            "assessment_created_at": assessment.created_at.isoformat() if assessment.created_at else None,
        },
        "clinical_context": {
            "symptoms": list(assessment.symptoms or []),
            "symptom_duration": assessment.symptom_duration,
            "prior_treatment": bool(assessment.prior_treatment),
            "clinical_notes": assessment.clinical_notes,
        },
        "input_validation": {
            "source_format": study.source_format.value,
            "modality": study.modality,
            "qc_status": study.qc_status.value,
            "brain_scope_status": study.brain_scope_status.value,
            "capability_routing_status": study.capability_routing_status.value,
            "capabilities": dict((study.capability_summary or {}).get("capabilities") or {}),
        },
        "gbm_assessment": {
            "state": decision.decision_state.value,
            "calibrated_probability_gbm": decision.calibrated_probability_gbm,
            "safety_reason_codes": list(decision.safety_reason_codes or []),
            "summary": str(summary.get("user_facing_summary") or ""),
            "other_intracranial_abnormality_not_excluded": bool(
                summary.get("other_intracranial_abnormality_not_excluded")
            ),
        },
        "tumor_analysis": tumor_analysis,
        "human_review": {
            "segmentation_available": bool(segmentation_summary.get("available")),
            "segmentation_review_status": segmentation_summary.get("review_status"),
            "clinician_modified": bool(segmentation_summary.get("clinician_modified", False)),
        },
        "traceability": {
            "decision_fusion_version": decision.decision_fusion_version,
            "classifier_model": _model_trace(db, decision.classifier_model_version_id),
            "segmentation_model": _model_trace(db, decision.segmentation_model_version_id),
            "classifier_evidence_available": bool(classifier_summary.get("available")),
            "segmentation_evidence_available": bool(segmentation_summary.get("available")),
            "decision_fused_at": decision.decision_fused_at.isoformat() if decision.decision_fused_at else None,
        },
        "limitations": {
            "volumetric_classifier_bridge_validated": bool(
                summary.get("volumetric_classifier_bridge_validated", False)
            ),
            "segmentation_is_gbm_diagnosis": False,
            "clinical_validation_claimed": False,
        },
        "clinical_notice": CLINICAL_NOTICE,
        "report_ready": report_ready,
        "report_blockers": report_blockers,
    }
    return payload, report_blockers


def preview_report(db: Session, study: Study) -> dict:
    decision = _current_decision_run(db, study)
    report, blockers = _report_payload(db, study, decision)
    return {
        "report_version": CLINICAL_REPORT_VERSION,
        "study_uuid": study.id,
        "decision_analysis_run_uuid": decision.id,
        "finalization_ready": bool(report.get("report_ready")) and not blockers,
        "blockers": blockers,
        "report": report,
    }


def _checksum(report: dict, clinician_name: str, clinician_comment: str | None, signed_at: datetime) -> str:
    canonical = json.dumps(
        {
            "report": report,
            "clinician_name": clinician_name,
            "clinician_comment": clinician_comment,
            "signed_at": signed_at.isoformat(),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def finalize_report(
    db: Session,
    study: Study,
    *,
    clinician_name: str,
    clinician_comment: str | None,
    request_id: str | None = None,
) -> ClinicalReport:
    decision = _current_decision_run(db, study)
    report, blockers = _report_payload(db, study, decision)
    if not bool(report.get("report_ready")) or blockers:
        raise ClinicalReportError(
            "REPORT_NOT_READY",
            "Report cannot be finalized until required decision and human-review gates are complete.",
        )

    signed_at = datetime.now(timezone.utc)
    checksum = _checksum(report, clinician_name, clinician_comment, signed_at)
    row = ClinicalReport(
        study_id=study.id,
        decision_analysis_run_id=decision.id,
        report_version=CLINICAL_REPORT_VERSION,
        status=ReportStatus.FINALIZED,
        report_payload=report,
        report_checksum_sha256=checksum,
        clinician_name=clinician_name.strip(),
        clinician_comment=(clinician_comment.strip() if clinician_comment else None),
        signed_at=signed_at,
        # Authentication/RBAC remains intentionally outside current V1; do not
        # misrepresent typed attribution as cryptographically verified identity.
        signoff_identity_verified=False,
    )
    db.add(row)
    db.flush()
    record_audit_event(
        db,
        action=AuditAction.REPORT_FINALIZED,
        entity_type=AuditEntityType.REPORT,
        entity_uuid=row.id,
        actor_type=AuditActorType.DEMO_USER,
        actor_id=clinician_name.strip(),
        request_id=request_id,
        technical_context={
            "report_version": CLINICAL_REPORT_VERSION,
            "report_checksum_sha256": checksum,
            "decision_analysis_run_uuid": str(decision.id),
            "signoff_identity_verified": False,
        },
        commit=False,
    )
    db.commit()
    db.refresh(row)
    return row


def get_current_report(db: Session, study: Study) -> ClinicalReport:
    row = db.scalar(
        select(ClinicalReport)
        .where(ClinicalReport.study_id == study.id)
        .order_by(ClinicalReport.signed_at.desc(), ClinicalReport.created_at.desc())
        .limit(1)
    )
    if row is None:
        raise ClinicalReportError("REPORT_NOT_FOUND", "No finalized report exists for this study.")
    return row


def report_response(row: ClinicalReport) -> dict:
    return {
        "report_uuid": row.id,
        "report_version": row.report_version,
        "study_uuid": row.study_id,
        "decision_analysis_run_uuid": row.decision_analysis_run_id,
        "status": row.status.value,
        "report_checksum_sha256": row.report_checksum_sha256,
        "signed_at": row.signed_at,
        "clinician_name": row.clinician_name,
        "clinician_comment": row.clinician_comment,
        "signoff_identity_verified": bool(row.signoff_identity_verified),
        "report": dict(row.report_payload or {}),
    }
