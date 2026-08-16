from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from gbm_ai.api.models.analysis import AnalysisRun, AnalysisStatus, Study
from gbm_ai.api.models.segmentation import Segmentation, SegmentationStatus


CURRENT_SEGMENTATION_RESOLUTION_VERSION = "phase10_demo_current_segmentation_resolution_v1"


def _as_uuid(value: object) -> uuid.UUID | None:
    try:
        if value in {None, "", "None", "null"}:
            return None
        return uuid.UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        return None


def _validated_row(
    db: Session,
    study: Study,
    segmentation_uuid: uuid.UUID,
    *,
    model_input_checksum_sha256: str | None,
) -> tuple[AnalysisRun, Segmentation] | None:
    statement = (
        select(AnalysisRun, Segmentation)
        .join(Segmentation, Segmentation.analysis_run_id == AnalysisRun.id)
        .where(
            AnalysisRun.study_id == study.id,
            AnalysisRun.status == AnalysisStatus.COMPLETE,
            Segmentation.status == SegmentationStatus.GENERATED,
            Segmentation.id == segmentation_uuid,
        )
        .limit(1)
    )
    if model_input_checksum_sha256:
        statement = statement.where(
            Segmentation.model_input_checksum_sha256 == model_input_checksum_sha256
        )
    row = db.execute(statement).first()
    return (row[0], row[1]) if row is not None else None


def resolve_current_completed_segmentation(
    db: Session,
    study: Study,
    *,
    repair_summary: bool = True,
) -> tuple[AnalysisRun, Segmentation] | None:
    """Resolve the segmentation that belongs to the study's current model input.

    Older Phase 6 records could persist the literal string ``"None"`` in
    ``inference.segmentation_uuid`` because the ORM UUID default was assigned
    only when the segmentation row was flushed.  This resolver validates the
    current reference first, then safely recovers only from a completed
    segmentation whose model-input checksum matches the study's current model
    input.  It never selects a result from another study or another prepared
    input.
    """

    summary = dict(study.segmentation_preparation_summary or {})
    model_input = dict(summary.get("model_input") or {})
    inference = dict(summary.get("inference") or {})
    background_job = dict(summary.get("background_job") or {})

    current_checksum = str(model_input.get("checksum_sha256") or "").strip().lower()
    if len(current_checksum) != 64:
        current_checksum = ""

    candidate_ids: list[uuid.UUID] = []
    for raw in (
        inference.get("segmentation_uuid"),
        background_job.get("segmentation_uuid"),
    ):
        parsed = _as_uuid(raw)
        if parsed is not None and parsed not in candidate_ids:
            candidate_ids.append(parsed)

    resolved: tuple[AnalysisRun, Segmentation] | None = None
    for candidate in candidate_ids:
        resolved = _validated_row(
            db,
            study,
            candidate,
            model_input_checksum_sha256=current_checksum or None,
        )
        if resolved is not None:
            break

    if resolved is None and current_checksum:
        # Recovery is deliberately scoped to this study AND the immutable
        # current model-input checksum.  This repairs legacy metadata without
        # allowing an unrelated/stale segmentation to become current.
        row = db.execute(
            select(AnalysisRun, Segmentation)
            .join(Segmentation, Segmentation.analysis_run_id == AnalysisRun.id)
            .where(
                AnalysisRun.study_id == study.id,
                AnalysisRun.status == AnalysisStatus.COMPLETE,
                Segmentation.status == SegmentationStatus.GENERATED,
                Segmentation.model_input_checksum_sha256 == current_checksum,
            )
            .order_by(Segmentation.created_at.desc())
            .limit(1)
        ).first()
        if row is not None:
            resolved = (row[0], row[1])

    if resolved is None:
        return None

    analysis, segmentation = resolved
    if repair_summary:
        repaired = (
            inference.get("status") != "complete"
            or str(inference.get("segmentation_uuid") or "") != str(segmentation.id)
            or str(inference.get("analysis_run_uuid") or "") != str(analysis.id)
        )
        if repaired:
            inference.update(
                {
                    "status": "complete",
                    "analysis_run_uuid": str(analysis.id),
                    "segmentation_uuid": str(segmentation.id),
                    "segmentation_generated": True,
                    "model_input_checksum_sha256": segmentation.model_input_checksum_sha256,
                    "review_status": segmentation.review_status.value,
                    "reference_resolution_version": CURRENT_SEGMENTATION_RESOLUTION_VERSION,
                }
            )
            summary["inference"] = inference
            summary["segmentation_generated"] = True
            study.segmentation_preparation_summary = summary
            # Flush makes the repaired state available to the remainder of the
            # current request.  Mutating POST workflows will commit it normally;
            # read-only viewer requests do not need a side-effecting commit.
            db.flush()

    return analysis, segmentation
