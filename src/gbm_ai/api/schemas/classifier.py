from __future__ import annotations

import uuid

from pydantic import BaseModel

from gbm_ai.api.models.analysis import DecisionState, QCState


class ClassifierCheckpointStatus(BaseModel):
    fold: int
    path: str
    exists: bool
    sha256: str | None = None
    expected_sha256: str | None = None
    checksum_ok: bool = False


class ClassifierRuntimeStatusResponse(BaseModel):
    runtime_version: str
    deployment_version: str
    deployment_strategy_frozen: bool
    selected_architecture: str
    ensemble_strategy: str
    checkpoint_root: str
    checkpoint_count_expected: int
    checkpoint_count_available: int
    checkpoints: list[ClassifierCheckpointStatus]
    threshold_low: float
    threshold_high: float
    safety_policy_name: str | None = None
    resolved_device: str
    ready: bool
    missing_assets: list[str]


class ClassifierRunResponse(BaseModel):
    analysis_run_uuid: uuid.UUID
    study_uuid: uuid.UUID
    model_version_uuid: uuid.UUID | None = None
    runtime_version: str
    raw_probability_gbm: float | None = None
    calibrated_probability_gbm: float | None = None
    decision_state: DecisionState
    qc_state: QCState
    safety_reason_codes: list[str]
