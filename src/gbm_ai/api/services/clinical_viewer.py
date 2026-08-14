from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import BinaryIO

from sqlalchemy import select
from sqlalchemy.orm import Session

from gbm_ai.api.models.analysis import AnalysisRun, AnalysisStatus, SourceFormat, Study
from gbm_ai.api.models.localization import AnatomicalLocalization
from gbm_ai.api.models.quantification import TumorQuantification
from gbm_ai.api.models.segmentation import Segmentation, SegmentationStatus
from gbm_ai.api.storage.local import LocalObjectStore


CLINICAL_VIEWER_BACKEND_VERSION = "phase8_step1_clinical_viewer_backend_v1"
VIEWER_PRIMARY_REFERENCE_SEQUENCE = "T1C"
VIEWER_PLANES = ("axial", "coronal", "sagittal")
VIEWER_MODEL_SEQUENCES = ("T1C", "T1", "T2", "FLAIR")


class ClinicalViewerServiceError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ViewerAsset:
    alias: str
    storage_key: str
    checksum_sha256: str
    size_bytes: int
    kind: str
    format: str
    media_type: str
    coordinate_space: str
    filename: str
    sequence: str | None = None
    region: str | None = None

    def public_payload(self, *, study_uuid: uuid.UUID, api_prefix: str) -> dict:
        prefix = api_prefix.rstrip("/")
        return {
            "alias": self.alias,
            "kind": self.kind,
            "format": self.format,
            "media_type": self.media_type,
            "checksum_sha256": self.checksum_sha256,
            "size_bytes": int(self.size_bytes),
            "coordinate_space": self.coordinate_space,
            "sequence": self.sequence,
            "region": self.region,
            "download_url": f"{prefix}/studies/{study_uuid}/viewer/assets/{self.alias}",
        }


@dataclass(frozen=True)
class ViewerAssetStream:
    asset: ViewerAsset
    stream: BinaryIO


def _require_current_segmentation(db: Session, study: Study) -> tuple[AnalysisRun, Segmentation]:
    summary = dict(study.segmentation_preparation_summary or {})
    inference = dict(summary.get("inference") or {})
    if inference.get("status") != "complete" or not inference.get("segmentation_uuid"):
        raise ClinicalViewerServiceError(
            "VIEWER_SEGMENTATION_NOT_READY",
            "a current completed 3D segmentation is required before the clinical viewer can be opened",
        )

    try:
        segmentation_uuid = uuid.UUID(str(inference["segmentation_uuid"]))
    except (TypeError, ValueError) as exc:
        raise ClinicalViewerServiceError(
            "VIEWER_SEGMENTATION_REFERENCE_INVALID",
            "the current segmentation reference is invalid",
        ) from exc

    segmentation = db.get(Segmentation, segmentation_uuid)
    if segmentation is None or segmentation.status != SegmentationStatus.GENERATED:
        raise ClinicalViewerServiceError(
            "VIEWER_SEGMENTATION_NOT_FOUND",
            "the current persisted segmentation result is unavailable",
        )
    analysis = db.get(AnalysisRun, segmentation.analysis_run_id)
    if (
        analysis is None
        or analysis.study_id != study.id
        or analysis.status != AnalysisStatus.COMPLETE
    ):
        raise ClinicalViewerServiceError(
            "VIEWER_SEGMENTATION_PROVENANCE_INVALID",
            "the segmentation does not belong to a completed analysis for this study",
        )
    return analysis, segmentation


def _require_model_geometry(study: Study) -> dict:
    summary = dict(study.segmentation_preparation_summary or {})
    geometry = dict(summary.get("model_geometry") or {})
    if geometry.get("status") != "ready":
        raise ClinicalViewerServiceError(
            "VIEWER_MODEL_GEOMETRY_NOT_READY",
            "model-space MRI volumes are not ready for viewer overlay geometry",
        )
    channels = list(geometry.get("channels") or [])
    by_sequence = {str(item.get("sequence") or "").upper(): dict(item) for item in channels}
    if set(by_sequence) != set(VIEWER_MODEL_SEQUENCES):
        raise ClinicalViewerServiceError(
            "VIEWER_MODEL_CHANNELS_INVALID",
            "viewer requires exactly T1C, T1, T2 and FLAIR model-space volumes",
        )
    return by_sequence


def _asset(
    *,
    alias: str,
    storage_key: str | None,
    checksum: str | None,
    size_bytes: int | None,
    kind: str,
    coordinate_space: str,
    filename: str,
    sequence: str | None = None,
    region: str | None = None,
    format: str = "nifti_gzip",
    media_type: str = "application/octet-stream",
) -> ViewerAsset:
    key = str(storage_key or "").strip()
    digest = str(checksum or "").strip().lower()
    try:
        size = int(size_bytes or 0)
    except (TypeError, ValueError):
        size = -1
    if not key or len(digest) != 64 or size <= 0:
        raise ClinicalViewerServiceError(
            "VIEWER_ASSET_PROVENANCE_INCOMPLETE",
            f"viewer asset {alias!r} is missing protected-storage provenance",
        )
    return ViewerAsset(
        alias=alias,
        storage_key=key,
        checksum_sha256=digest,
        size_bytes=size,
        kind=kind,
        format=format,
        media_type=media_type,
        coordinate_space=coordinate_space,
        filename=filename,
        sequence=sequence,
        region=region,
    )


def _segmentation_checksums(segmentation: Segmentation) -> dict[str, str]:
    return {
        "WT": str(segmentation.wt_checksum_sha256).lower(),
        "TC": str(segmentation.tc_checksum_sha256).lower(),
        "ET": str(segmentation.et_checksum_sha256).lower(),
        "LABELMAP": str(segmentation.labelmap_checksum_sha256).lower(),
    }


def _latest_quantification_state(
    db: Session,
    segmentation: Segmentation,
) -> tuple[TumorQuantification | None, bool]:
    latest = db.scalar(
        select(TumorQuantification)
        .where(TumorQuantification.segmentation_id == segmentation.id)
        .order_by(TumorQuantification.created_at.desc())
        .limit(1)
    )
    if latest is None:
        return None, False

    expected_checksums = _segmentation_checksums(segmentation)
    current = (
        latest.source_review_status == segmentation.review_status.value
        and bool(latest.source_clinician_modified) == bool(segmentation.clinician_modified)
        and dict(latest.source_mask_checksums or {}) == expected_checksums
        and bool(latest.physical_volume_generated)
    )
    return (latest if current else None), (not current)


def _latest_localization_state(
    db: Session,
    segmentation: Segmentation,
    quantification: TumorQuantification | None,
) -> tuple[AnatomicalLocalization | None, bool]:
    latest = db.scalar(
        select(AnatomicalLocalization)
        .where(AnatomicalLocalization.segmentation_id == segmentation.id)
        .order_by(AnatomicalLocalization.created_at.desc())
        .limit(1)
    )
    if latest is None:
        return None, False
    current = (
        quantification is not None
        and latest.quantification_id == quantification.id
        and bool(latest.registration_qc_passed)
        and bool(latest.anatomical_localization_generated)
    )
    return (latest if current else None), (not current)


def build_viewer_assets(
    study: Study,
    segmentation: Segmentation,
    *,
    localization: AnatomicalLocalization | None,
    quantification: TumorQuantification | None,
) -> dict[str, ViewerAsset]:
    channels = _require_model_geometry(study)
    assets: dict[str, ViewerAsset] = {}
    for sequence in VIEWER_MODEL_SEQUENCES:
        item = channels[sequence]
        alias = f"mri_{sequence.lower()}"
        assets[alias] = _asset(
            alias=alias,
            storage_key=item.get("storage_key"),
            checksum=item.get("checksum_sha256"),
            size_bytes=item.get("size_bytes"),
            kind="mri_volume",
            coordinate_space="patient_model_space_ras",
            filename=f"{sequence.lower()}_model_space.nii.gz",
            sequence=sequence,
        )

    for region, key, checksum, size in (
        ("WT", segmentation.wt_storage_key, segmentation.wt_checksum_sha256, segmentation.wt_size_bytes),
        ("TC", segmentation.tc_storage_key, segmentation.tc_checksum_sha256, segmentation.tc_size_bytes),
        ("ET", segmentation.et_storage_key, segmentation.et_checksum_sha256, segmentation.et_size_bytes),
        (
            "BRATS_LABELMAP",
            segmentation.labelmap_storage_key,
            segmentation.labelmap_checksum_sha256,
            segmentation.labelmap_size_bytes,
        ),
    ):
        alias = "mask_labelmap" if region == "BRATS_LABELMAP" else f"mask_{region.lower()}"
        kind = "segmentation_labelmap" if region == "BRATS_LABELMAP" else "segmentation_mask"
        assets[alias] = _asset(
            alias=alias,
            storage_key=key,
            checksum=checksum,
            size_bytes=size,
            kind=kind,
            coordinate_space="patient_model_space_ras",
            filename=f"{region.lower()}_mask.nii.gz",
            region=region,
        )

    if quantification is not None:
        assets["quantification_areas"] = _asset(
            alias="quantification_areas",
            storage_key=quantification.per_slice_area_storage_key,
            checksum=quantification.per_slice_area_checksum_sha256,
            size_bytes=quantification.per_slice_area_size_bytes,
            kind="derived_metadata",
            coordinate_space="metadata",
            filename="quantification_areas.json",
            format="json",
            media_type="application/json",
        )

    if localization is not None:
        assets["mask_wt_mni"] = _asset(
            alias="mask_wt_mni",
            storage_key=localization.transformed_wt_storage_key,
            checksum=localization.transformed_wt_checksum_sha256,
            size_bytes=localization.transformed_wt_size_bytes,
            kind="standard_space_mask",
            coordinate_space="standard_space_mni152nlin6asym",
            filename="wt_standard_space_mni152nlin6asym.nii.gz",
            region="WT",
        )
    return assets


def _measurement(region: str, q: TumorQuantification) -> dict:
    prefix = region.lower()
    return {
        "region": region,
        "voxel_count": int(getattr(q, f"{prefix}_voxel_count")),
        "volume_mm3": float(getattr(q, f"{prefix}_volume_mm3")),
        "volume_cm3": float(getattr(q, f"{prefix}_volume_cm3")),
        "max_axial_area_mm2": float(getattr(q, f"{prefix}_max_axial_area_mm2")),
        "max_axial_slice_index": getattr(q, f"{prefix}_max_axial_slice_index"),
    }


def build_clinical_viewer_manifest(
    db: Session,
    study: Study,
    *,
    api_prefix: str,
) -> dict:
    if study.source_format not in {SourceFormat.DICOM, SourceFormat.NIFTI}:
        raise ClinicalViewerServiceError(
            "VIEWER_VOLUMETRIC_SOURCE_REQUIRED",
            "Phase 8 volumetric viewer requires a compatible DICOM or NIfTI study",
        )

    _, segmentation = _require_current_segmentation(db, study)
    quantification, quant_stale = _latest_quantification_state(db, segmentation)
    localization, localization_stale = _latest_localization_state(
        db,
        segmentation,
        quantification,
    )
    assets = build_viewer_assets(
        study,
        segmentation,
        localization=localization,
        quantification=quantification,
    )

    quant_payload = {
        "available": quantification is not None,
        "stale": bool(quant_stale),
        "primary_region": "WT",
        "voxel_spacing_mm": (
            list(quantification.voxel_spacing_mm) if quantification is not None else []
        ),
        "measurements": (
            [_measurement(region, quantification) for region in ("WT", "TC", "ET")]
            if quantification is not None
            else []
        ),
    }
    localization_payload = {
        "available": localization is not None,
        "stale": bool(localization_stale),
        "standard_space": localization.standard_space if localization is not None else None,
        "hemisphere": localization.hemisphere if localization is not None else None,
        "centroid_mni_mm": list(localization.centroid_mni_mm) if localization is not None else [],
        "primary_region": localization.primary_region if localization is not None else None,
        "secondary_regions": list(localization.secondary_regions) if localization is not None else [],
        "registration_qc_passed": (
            bool(localization.registration_qc_passed) if localization is not None else None
        ),
        "clinician_verification_required": True,
    }

    return {
        "version": CLINICAL_VIEWER_BACKEND_VERSION,
        "status": "ready",
        "study_uuid": study.id,
        "source_format": study.source_format.value,
        "segmentation_uuid": segmentation.id,
        "segmentation_review_status": segmentation.review_status.value,
        "clinician_modified": bool(segmentation.clinician_modified),
        "primary_reference_sequence": VIEWER_PRIMARY_REFERENCE_SEQUENCE,
        "canonical_orientation": "RAS",
        "viewer_planes": list(VIEWER_PLANES),
        "assets": [
            asset.public_payload(study_uuid=study.id, api_prefix=api_prefix)
            for asset in assets.values()
        ],
        "quantification": quant_payload,
        "localization": localization_payload,
        "raw_storage_keys_exposed": False,
        "patient_identifiers_in_manifest": False,
        "checksum_validation_required_before_streaming": True,
        "overlays_available": True,
        "three_dimensional_asset_basis_ready": True,
        "cornerstone_or_ohif_frontend_implemented": False,
        "manual_mask_editing_implemented": False,
        "clinician_verification_required": True,
        "segmentation_is_gbm_diagnosis": False,
        "clinical_validation_claimed": False,
        "next_step": "phase8_step2_clinical_viewer_ui",
    }


def resolve_viewer_asset(
    db: Session,
    study: Study,
    *,
    asset_alias: str,
) -> ViewerAsset:
    _, segmentation = _require_current_segmentation(db, study)
    quantification, _ = _latest_quantification_state(db, segmentation)
    localization, _ = _latest_localization_state(db, segmentation, quantification)
    assets = build_viewer_assets(
        study,
        segmentation,
        localization=localization,
        quantification=quantification,
    )
    asset = assets.get(str(asset_alias).strip().lower())
    if asset is None:
        raise ClinicalViewerServiceError(
            "VIEWER_ASSET_NOT_ALLOWED",
            "the requested viewer asset alias is not available for the current study state",
        )
    return asset


def open_verified_viewer_asset(
    storage: LocalObjectStore,
    asset: ViewerAsset,
) -> ViewerAssetStream:
    try:
        if not storage.verify_checksum(asset.storage_key, asset.checksum_sha256):
            raise ClinicalViewerServiceError(
                "VIEWER_ASSET_CHECKSUM_MISMATCH",
                "viewer asset checksum validation failed; the object will not be streamed",
            )
        stream = storage.open_read(asset.storage_key)
    except ClinicalViewerServiceError:
        raise
    except Exception as exc:
        raise ClinicalViewerServiceError(
            "VIEWER_ASSET_UNAVAILABLE",
            "viewer asset is unavailable in protected object storage",
        ) from exc
    return ViewerAssetStream(asset=asset, stream=stream)
