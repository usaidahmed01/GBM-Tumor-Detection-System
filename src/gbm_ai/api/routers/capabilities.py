from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from gbm_ai.api.dependencies import get_db_session
from gbm_ai.api.models.analysis import CapabilityRoutingStatus
from gbm_ai.api.models.audit import AuditActorType
from gbm_ai.api.schemas.capabilities import (
    BrainScopeConfirmationRequest,
    BrainScopeConfirmationResponse,
    CapabilityRoutingResponse,
    NiftiSequenceMappingRequest,
    NiftiSequenceMappingResponse,
)
from gbm_ai.api.services.analysis_records import (
    StudyNotFoundError,
    get_study,
)
from gbm_ai.api.services.capability_routing import (
    CapabilityRoutingError,
    NiftiSequenceMappingError,
    StudyScopeConfirmationError,
    confirm_brain_scope,
    confirm_nifti_sequence_mapping,
    route_study_capabilities,
)

router = APIRouter(tags=["capabilities"])


def _next_step(summary: dict) -> str:
    status = summary["routing_status"]
    capabilities = summary["capabilities"]

    if status == "review_required":
        return "manual_review_or_confirmation"

    if status == "no_supported_analysis":
        return "unable_to_assess_current_pipeline"

    if (
        capabilities["three_d_segmentation"]["state"]
        == "eligible"
    ):
        return "phase6_3d_segmentation"

    if (
        capabilities["two_d_classification"]["state"]
        == "eligible"
    ):
        # The input is eligible, but the current project still has five
        # cross-validation fold checkpoints rather than a frozen deployment
        # inference strategy. Never silently choose one fold.
        return "freeze_classifier_deployment_strategy_before_inference"

    return "manual_review_or_confirmation"


def _routing_response(study, summary: dict) -> CapabilityRoutingResponse:
    return CapabilityRoutingResponse(
        study_uuid=study.id,
        study_status=study.status,
        routing_status=study.capability_routing_status,
        brain_scope_status=study.brain_scope_status,
        assessment_scope_status=summary["assessment_scope_status"],
        age_scope_status=summary["age_scope_status"],
        manual_review_required=summary["manual_review_required"],
        global_block_reasons=summary["global_block_reasons"],
        global_review_reasons=summary["global_review_reasons"],
        capabilities=summary["capabilities"],
        eligible_capability_count=summary["eligible_capability_count"],
        review_capability_count=summary["review_capability_count"],
        model_execution_started=False,
        classifier_deployment_strategy_frozen=False,
        volumetric_to_2d_classifier_bridge_validated=False,
        clinical_validation_claimed=False,
        next_step=_next_step(summary),
    )


@router.post(
    "/studies/{study_uuid}/capabilities/route",
    response_model=CapabilityRoutingResponse,
    summary="Route a QC-complete study to only supported analysis capabilities",
)
def route_capabilities(
    study_uuid: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db_session),
):
    try:
        study = get_study(db, study_uuid)
    except StudyNotFoundError:
        raise HTTPException(status_code=404, detail="study not found")

    try:
        summary = route_study_capabilities(
            db,
            study,
            request_id=getattr(request.state, "request_id", None),
            actor_type=AuditActorType.DEMO_USER,
        )
    except CapabilityRoutingError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return _routing_response(study, summary)


@router.get(
    "/studies/{study_uuid}/capabilities",
    response_model=CapabilityRoutingResponse,
    summary="Read the latest non-stale capability routing result",
)
def get_capabilities(
    study_uuid: uuid.UUID,
    db: Session = Depends(get_db_session),
):
    try:
        study = get_study(db, study_uuid)
    except StudyNotFoundError:
        raise HTTPException(status_code=404, detail="study not found")

    if (
        study.capability_routing_status == CapabilityRoutingStatus.PENDING
        or not study.capability_summary
        or study.capability_summary.get("stale")
    ):
        raise HTTPException(
            status_code=409,
            detail="capability routing has not been completed or is stale",
        )

    return _routing_response(
        study,
        dict(study.capability_summary),
    )


@router.put(
    "/studies/{study_uuid}/brain-scope-confirmation",
    response_model=BrainScopeConfirmationResponse,
    summary="Confirm whether an otherwise unverified upload is a brain MRI",
)
def brain_scope_confirmation(
    study_uuid: uuid.UUID,
    payload: BrainScopeConfirmationRequest,
    request: Request,
    db: Session = Depends(get_db_session),
):
    try:
        study = get_study(db, study_uuid)
    except StudyNotFoundError:
        raise HTTPException(status_code=404, detail="study not found")

    try:
        confirm_brain_scope(
            db,
            study,
            is_brain_mri=payload.is_brain_mri,
            request_id=getattr(request.state, "request_id", None),
            actor_type=AuditActorType.DEMO_USER,
        )
    except StudyScopeConfirmationError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return BrainScopeConfirmationResponse(
        study_uuid=study.id,
        brain_scope_status=study.brain_scope_status,
        capability_routing_status=study.capability_routing_status,
        study_status=study.status,
    )


@router.put(
    "/studies/{study_uuid}/nifti-sequence-mapping",
    response_model=NiftiSequenceMappingResponse,
    summary="Confirm T1/T1c/T2/FLAIR mapping by opaque NIfTI volume index",
)
def nifti_sequence_mapping(
    study_uuid: uuid.UUID,
    payload: NiftiSequenceMappingRequest,
    request: Request,
    db: Session = Depends(get_db_session),
):
    try:
        study = get_study(db, study_uuid)
    except StudyNotFoundError:
        raise HTTPException(status_code=404, detail="study not found")

    try:
        confirm_nifti_sequence_mapping(
            db,
            study,
            payload.normalized_mapping(),
            request_id=getattr(request.state, "request_id", None),
            actor_type=AuditActorType.DEMO_USER,
        )
    except NiftiSequenceMappingError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return NiftiSequenceMappingResponse(
        study_uuid=study.id,
        mapping=dict(study.nifti_sequence_mapping),
        capability_routing_status=study.capability_routing_status,
    )
