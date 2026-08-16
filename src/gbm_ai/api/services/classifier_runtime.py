from __future__ import annotations

import hashlib
import io
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session
from torchvision import transforms

from gbm_ai.api.config import Settings
from gbm_ai.api.models.analysis import (
    AnalysisRun,
    AnalysisStatus,
    DecisionState,
    ModelRole,
    ModelVersion,
    QCState,
    SourceFormat,
    Study,
    StudyQCStatus,
)
from gbm_ai.api.storage.local import LocalObjectStore
from gbm_ai.models.efficientnet_v2_gbm import GBMEfficientNetV2S


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEPLOYMENT_MANIFEST_RELATIVE = Path(
    "artifacts/deployment/efficientnetv2s_seed42/classifier_deployment_freeze.json"
)
SAFETY_POLICY_RELATIVE = Path(
    "artifacts/safety/fusion/classifier_safety_policy_v1.json"
)
CLASSIFIER_RUNTIME_VERSION = "phase9_step2_classifier_runtime_v1"


class ClassifierRuntimeError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class DeploymentManifest:
    deployment_version: str
    runtime_version: str
    deployment_strategy_frozen: bool
    selected_architecture: str
    ensemble_strategy: str
    folds: list[int]
    checkpoint_root_default: str
    checkpoint_filenames: list[str]
    checkpoint_sha256: dict[str, str]
    temperature: float
    threshold_low: float
    threshold_high: float
    input_height: int
    input_width: int


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_deployment_manifest(project_root: Path | None = None) -> DeploymentManifest:
    root = project_root or PROJECT_ROOT
    payload = _read_json(root / DEPLOYMENT_MANIFEST_RELATIVE)
    spec = payload["input_spec"]
    calibration = payload["calibration"]
    thresholds = payload["probability_thresholds"]
    return DeploymentManifest(
        deployment_version=str(payload["deployment_version"]),
        runtime_version=str(payload["runtime_version"]),
        deployment_strategy_frozen=bool(payload["deployment_strategy_frozen"]),
        selected_architecture=str(payload["selected_architecture"]),
        ensemble_strategy=str(payload["ensemble_strategy"]),
        folds=[int(x) for x in payload["folds"]],
        checkpoint_root_default=str(payload["checkpoint_root_default"]),
        checkpoint_filenames=[str(x) for x in payload["checkpoint_filenames"]],
        checkpoint_sha256={str(k): str(v).lower() for k, v in (payload.get("checkpoint_sha256") or {}).items()},
        temperature=float(calibration["temperature"]),
        threshold_low=float(thresholds["T_low"]),
        threshold_high=float(thresholds["T_high"]),
        input_height=int(spec["height"]),
        input_width=int(spec["width"]),
    )


def _safety_policy(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root or PROJECT_ROOT
    return _read_json(root / SAFETY_POLICY_RELATIVE)


def _resolve_device(preference: str) -> str:
    pref = (preference or "auto").lower()
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def checkpoint_paths(settings: Settings, manifest: DeploymentManifest | None = None) -> list[Path]:
    manifest = manifest or load_deployment_manifest()
    root = settings.classifier_checkpoint_root_resolved
    return [root / name for name in manifest.checkpoint_filenames]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classifier_runtime_status(settings: Settings) -> dict[str, Any]:
    manifest = load_deployment_manifest()
    policy = _safety_policy()
    paths = checkpoint_paths(settings, manifest)
    missing_assets: list[str] = []
    checkpoints = []
    for fold, path in zip(manifest.folds, paths):
        exists = path.exists()
        expected_sha = manifest.checkpoint_sha256.get(path.name)
        actual_sha = _sha256_file(path) if exists and expected_sha else None
        checksum_ok = bool(exists and (expected_sha is None or actual_sha == expected_sha))
        checkpoints.append({
            "fold": fold,
            "path": str(path),
            "exists": exists,
            "sha256": actual_sha,
            "expected_sha256": expected_sha,
            "checksum_ok": checksum_ok,
        })
        if not exists:
            missing_assets.append(str(path))
        elif not checksum_ok:
            missing_assets.append(f"checksum mismatch: {path}")
    return {
        "runtime_version": manifest.runtime_version,
        "deployment_version": manifest.deployment_version,
        "deployment_strategy_frozen": manifest.deployment_strategy_frozen,
        "selected_architecture": manifest.selected_architecture,
        "ensemble_strategy": manifest.ensemble_strategy,
        "checkpoint_root": str(settings.classifier_checkpoint_root_resolved),
        "checkpoint_count_expected": len(paths),
        "checkpoint_count_available": sum(1 for item in checkpoints if item["checksum_ok"]),
        "checkpoints": checkpoints,
        "threshold_low": manifest.threshold_low,
        "threshold_high": manifest.threshold_high,
        "safety_policy_name": policy.get("name"),
        "resolved_device": _resolve_device(settings.classifier_inference_device),
        "ready": not missing_assets,
        "missing_assets": missing_assets,
    }


def deployment_strategy_frozen() -> bool:
    manifest = load_deployment_manifest()
    return bool(manifest.deployment_strategy_frozen)


def preprocess_image_bytes(image_bytes: bytes, manifest: DeploymentManifest | None = None) -> torch.Tensor:
    manifest = manifest or load_deployment_manifest()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pipeline = transforms.Compose(
        [
            transforms.Resize((manifest.input_height, manifest.input_width)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return pipeline(image).unsqueeze(0)


def _load_checkpoint_state(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                return candidate
    if isinstance(payload, dict):
        return payload
    raise ClassifierRuntimeError("INVALID_CHECKPOINT_FORMAT", f"Unsupported checkpoint payload at {path}")


def _ensemble_probability(
    input_tensor: torch.Tensor,
    settings: Settings,
    manifest: DeploymentManifest,
) -> tuple[float, float]:
    device = _resolve_device(settings.classifier_inference_device)
    tensor = input_tensor.to(device)
    logits: list[float] = []
    for path in checkpoint_paths(settings, manifest):
        model = GBMEfficientNetV2S(pretrained=False, freeze_backbone=False)
        state = _load_checkpoint_state(path)
        model.load_state_dict(state, strict=False)
        model.eval()
        model.to(device)
        with torch.no_grad():
            value = model(tensor).detach().float().reshape(-1)[0].item()
        logits.append(float(value))
    if not logits:
        raise ClassifierRuntimeError("CLASSIFIER_RUNTIME_NOT_READY", "No classifier checkpoints are available")
    mean_logit = sum(logits) / len(logits)
    raw_probability = float(torch.sigmoid(torch.tensor(mean_logit)).item())
    calibrated_probability = float(torch.sigmoid(torch.tensor(mean_logit / manifest.temperature)).item())
    return raw_probability, calibrated_probability


def _decision_from_probability(probability: float, low: float, high: float) -> DecisionState:
    if probability <= low:
        return DecisionState.GBM_NOT_SUSPECTED
    if probability >= high:
        return DecisionState.GBM_SUSPECTED
    return DecisionState.INDETERMINATE


def _effective_classifier_qc_state(study: Study) -> QCState:
    """Return QC state after resolving manual brain-scope confirmation.

    Standalone raster QC is intentionally PARTIAL before a clinician confirms
    that the upload is a brain MRI because raster files do not carry reliable
    body-part metadata. Once that single partial reason is resolved, the 2D
    classifier should not be forced to indeterminate solely because the stored
    raw QC status remains PARTIAL. Other unresolved image-quality reasons still
    downgrade the classifier to REVIEW.
    """
    if study.qc_status == StudyQCStatus.FAIL:
        return QCState.FAIL
    if study.qc_status == StudyQCStatus.PASS:
        return QCState.PASS

    partial = set((study.qc_summary or {}).get("partial_reasons") or [])
    if getattr(study.brain_scope_status, "value", study.brain_scope_status) in {
        "clinician_confirmed",
        "supported_by_metadata",
    }:
        partial.discard("BRAIN_SCOPE_UNVERIFIED_FOR_RASTER")
    return QCState.PASS if not partial else QCState.REVIEW


def _qc_state_from_study(study: Study) -> QCState:
    return _effective_classifier_qc_state(study)


def _safety_reasons(study: Study, probability_state: DecisionState, qc_state: QCState | None = None) -> list[str]:
    reasons: list[str] = []
    effective_qc = qc_state or _effective_classifier_qc_state(study)
    if effective_qc != QCState.PASS:
        reasons.append("UPLOAD_QC_NOT_PASS")
    if probability_state == DecisionState.INDETERMINATE:
        reasons.append("PROBABILITY_BETWEEN_T_LOW_AND_T_HIGH")
    return reasons


def ensure_classifier_model_version(db: Session, settings: Settings) -> ModelVersion:
    manifest = load_deployment_manifest()
    row = db.scalar(
        select(ModelVersion).where(
            ModelVersion.model_name == manifest.selected_architecture,
            ModelVersion.version == manifest.deployment_version,
        )
    )
    if row is not None:
        return row

    row = ModelVersion(
        model_name=manifest.selected_architecture,
        version=manifest.deployment_version,
        role=ModelRole.CLASSIFIER,
        architecture=manifest.selected_architecture,
        weights_checksum_sha256=None,
        code_version=manifest.runtime_version,
        preprocessing_version="phase9_step2_rgb_resize384_imagenetnorm_v1",
        threshold_version=f"Tlow_{manifest.threshold_low:.2f}_Thigh_{manifest.threshold_high:.2f}",
        calibration_version=f"temperature_{manifest.temperature:.6f}",
        license_source_notes=(
            "Standalone 2D GBM classifier deployment manifest frozen in Phase 9 Step 2. "
            "Checkpoint files are mounted outside source control."
        ),
        is_active=True,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def run_classifier_for_study(
    db: Session,
    storage: LocalObjectStore,
    settings: Settings,
    study: Study,
) -> AnalysisRun:
    if study.source_format != SourceFormat.IMAGE:
        raise ClassifierRuntimeError(
            "STUDY_SOURCE_FORMAT_NOT_SUPPORTED",
            "Standalone 2D classifier runtime only supports source_format=image.",
        )
    runtime_status = classifier_runtime_status(settings)
    if not runtime_status["ready"]:
        raise ClassifierRuntimeError(
            "CLASSIFIER_RUNTIME_NOT_READY",
            "Classifier checkpoints are not fully installed for runtime inference.",
        )
    if not study.storage_key:
        raise ClassifierRuntimeError(
            "STUDY_SOURCE_OBJECT_MISSING",
            "Study has no stored image object.",
        )

    manifest = load_deployment_manifest()
    with storage.open_read(study.storage_key) as source:
        image_bytes = source.read()
    input_tensor = preprocess_image_bytes(image_bytes, manifest=manifest)
    raw_probability, calibrated_probability = _ensemble_probability(
        input_tensor,
        settings,
        manifest,
    )
    probability_state = _decision_from_probability(
        calibrated_probability,
        manifest.threshold_low,
        manifest.threshold_high,
    )
    qc_state = _qc_state_from_study(study)
    safety_reason_codes = _safety_reasons(study, probability_state, qc_state)
    final_state = probability_state if qc_state == QCState.PASS else DecisionState.INDETERMINATE

    model_version = ensure_classifier_model_version(db, settings)
    analysis = AnalysisRun(
        study_id=study.id,
        classifier_model_version_id=model_version.id,
        status=AnalysisStatus.COMPLETE,
        qc_state=qc_state,
        raw_probability_gbm=raw_probability,
        calibrated_probability_gbm=calibrated_probability,
        decision_state=final_state,
        safety_reason_codes=safety_reason_codes,
        decision_evidence_summary={
            "runtime_version": manifest.runtime_version,
            "deployment_version": manifest.deployment_version,
            "ensemble_strategy": manifest.ensemble_strategy,
            "threshold_low": manifest.threshold_low,
            "threshold_high": manifest.threshold_high,
            "source_format": study.source_format.value,
        },
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis
