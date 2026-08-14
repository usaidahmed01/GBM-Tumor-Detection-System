from __future__ import annotations

import json
import math
import shutil
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from gbm_ai.api.localization import (
    ANATOMICAL_LOCALIZATION_VERSION,
    ATLAS_LICENSE,
    ATLAS_NAME,
    ATLAS_VERSION,
    REGISTRATION_METHOD,
    REGISTRATION_METRIC,
    REGISTRATION_RANDOM_SEED,
    REGISTRATION_SUPPORT_DICE_MIN,
    SECONDARY_REGION_MIN_FRACTION_OF_WT,
    STANDARD_SPACE,
    STANDARD_TEMPLATE_NAME,
    AnatomicalLocalizationError,
    centroid_world_mm,
    compute_region_overlaps,
    dice_coefficient,
    hemisphere_from_standard_mask,
    localization_source_fingerprint,
    merge_region_overlaps,
)
from gbm_ai.api.localization_assets import (
    LocalizationAssetError,
    load_and_verify_localization_assets,
)
from gbm_ai.api.models.analysis import Study
from gbm_ai.api.models.audit import AuditAction, AuditActorType, AuditEntityType
from gbm_ai.api.models.localization import AnatomicalLocalization
from gbm_ai.api.models.quantification import TumorQuantification
from gbm_ai.api.models.segmentation import Segmentation, SegmentationReviewStatus
from gbm_ai.api.services.audit import record_audit_event
from gbm_ai.api.services.tumor_quantification import (
    TumorQuantificationServiceError,
    get_latest_tumor_quantification,
)
from gbm_ai.api.storage.local import LocalObjectStore


class AnatomicalLocalizationServiceError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _find_t1_model_geometry(study: Study) -> dict:
    summary = dict(study.segmentation_preparation_summary or {})
    geometry = dict(summary.get("model_geometry") or {})
    if geometry.get("status") != "ready":
        raise AnatomicalLocalizationServiceError(
            "LOCALIZATION_MODEL_GEOMETRY_NOT_READY",
            "current model-geometry preprocessing is not ready for anatomical registration",
        )
    candidates = [
        dict(item)
        for item in list(geometry.get("channels") or [])
        if str(item.get("sequence") or "").upper() == "T1"
    ]
    if len(candidates) != 1:
        raise AnatomicalLocalizationServiceError(
            "LOCALIZATION_T1_REFERENCE_MISSING",
            "exactly one prepared T1 reference volume is required for standard-space registration",
        )
    item = candidates[0]
    if not item.get("storage_key") or len(str(item.get("checksum_sha256") or "")) != 64:
        raise AnatomicalLocalizationServiceError(
            "LOCALIZATION_T1_REFERENCE_INVALID",
            "prepared T1 reference metadata is incomplete",
        )
    return item


def _materialize_storage_object(storage: LocalObjectStore, key: str, checksum: str, suffix: str) -> Path:
    try:
        if not storage.verify_checksum(key, checksum):
            raise AnatomicalLocalizationServiceError(
                "LOCALIZATION_SOURCE_CHECKSUM_MISMATCH",
                "a localization source artifact failed protected-storage checksum validation",
            )
    except AnatomicalLocalizationServiceError:
        raise
    except Exception as exc:
        raise AnatomicalLocalizationServiceError(
            "LOCALIZATION_SOURCE_ARTIFACT_UNAVAILABLE",
            "a required localization source artifact is unavailable",
        ) from exc

    temp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_path = Path(temp.name)
    temp.close()
    try:
        with storage.open_read(key) as source, temp_path.open("wb") as target:
            shutil.copyfileobj(source, target)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def _register_t1_to_standard_space(moving_t1_path: Path, template_path: Path):
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise AnatomicalLocalizationServiceError(
            "SIMPLEITK_NOT_INSTALLED",
            "SimpleITK is required for standard-space anatomical registration",
        ) from exc

    fixed = sitk.Cast(sitk.ReadImage(str(template_path)), sitk.sitkFloat32)
    moving = sitk.Cast(sitk.ReadImage(str(moving_t1_path)), sitk.sitkFloat32)
    if fixed.GetDimension() != 3 or moving.GetDimension() != 3:
        raise AnatomicalLocalizationServiceError(
            "LOCALIZATION_REGISTRATION_VOLUME_INVALID",
            "standard-space registration requires two 3D MRI volumes",
        )

    fixed_norm = sitk.Normalize(fixed)
    moving_norm = sitk.Normalize(moving)

    rigid_initial = sitk.CenteredTransformInitializer(
        fixed_norm,
        moving_norm,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    rigid = sitk.ImageRegistrationMethod()
    rigid.SetMetricAsMattesMutualInformation(50)
    rigid.SetMetricSamplingStrategy(rigid.RANDOM)
    rigid.SetMetricSamplingPercentage(0.20, REGISTRATION_RANDOM_SEED)
    rigid.SetInterpolator(sitk.sitkLinear)
    rigid.SetOptimizerAsGradientDescent(1.0, 150, 1e-6, 10)
    rigid.SetOptimizerScalesFromPhysicalShift()
    rigid.SetShrinkFactorsPerLevel([4, 2, 1])
    rigid.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    rigid.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    rigid.SetInitialTransform(rigid_initial, inPlace=False)
    try:
        rigid_transform = rigid.Execute(fixed_norm, moving_norm)
    except RuntimeError as exc:
        raise AnatomicalLocalizationServiceError(
            "LOCALIZATION_RIGID_REGISTRATION_FAILED",
            "rigid initialization to standard space failed",
        ) from exc

    # SimpleITK can return a CompositeTransform when the initial transform was
    # supplied with inPlace=False. Extract the optimized rigid component before
    # converting it to an affine initialization; never assume the composite
    # itself exposes Euler3D matrix/center/translation accessors.
    rigid_component = rigid_transform
    if rigid_transform.GetName() == "CompositeTransform":
        try:
            if rigid_transform.GetNumberOfTransforms() < 1:
                raise RuntimeError("empty CompositeTransform")
            rigid_component = rigid_transform.GetBackTransform()
        except Exception as exc:
            raise AnatomicalLocalizationServiceError(
                "LOCALIZATION_RIGID_TRANSFORM_INVALID",
                "rigid registration returned an unusable composite transform",
            ) from exc

    if not all(
        hasattr(rigid_component, name)
        for name in ("GetCenter", "GetMatrix", "GetTranslation")
    ):
        raise AnatomicalLocalizationServiceError(
            "LOCALIZATION_RIGID_TRANSFORM_INVALID",
            "rigid registration did not return a transform that can initialize affine registration",
        )

    affine_initial = sitk.AffineTransform(3)
    affine_initial.SetCenter(rigid_component.GetCenter())
    affine_initial.SetMatrix(rigid_component.GetMatrix())
    affine_initial.SetTranslation(rigid_component.GetTranslation())

    affine = sitk.ImageRegistrationMethod()
    affine.SetMetricAsMattesMutualInformation(50)
    affine.SetMetricSamplingStrategy(affine.RANDOM)
    affine.SetMetricSamplingPercentage(0.20, REGISTRATION_RANDOM_SEED)
    affine.SetInterpolator(sitk.sitkLinear)
    affine.SetOptimizerAsGradientDescent(0.5, 200, 1e-6, 10)
    affine.SetOptimizerScalesFromPhysicalShift()
    affine.SetShrinkFactorsPerLevel([4, 2, 1])
    affine.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    affine.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    affine.SetInitialTransform(affine_initial, inPlace=False)
    try:
        transform = affine.Execute(fixed_norm, moving_norm)
    except RuntimeError as exc:
        raise AnatomicalLocalizationServiceError(
            "LOCALIZATION_AFFINE_REGISTRATION_FAILED",
            "affine registration to standard space failed",
        ) from exc

    metric = float(affine.GetMetricValue())
    if not math.isfinite(metric):
        raise AnatomicalLocalizationServiceError(
            "LOCALIZATION_REGISTRATION_METRIC_INVALID",
            "standard-space registration produced a non-finite similarity metric",
        )
    return fixed, moving, transform, metric


def _resample_mask_to_standard(mask_path: Path, fixed, transform):
    import SimpleITK as sitk

    mask = sitk.ReadImage(str(mask_path))
    return sitk.Resample(
        mask,
        fixed,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def _registration_support_dice(moving, fixed_brain_mask_path: Path, fixed, transform) -> float:
    import SimpleITK as sitk

    # Model-geometry MRI uses zero padding. Treat non-zero finite intensity as
    # a conservative patient support mask; this is an engineering registration
    # QC signal, not a tissue segmentation.
    moving_support = sitk.NotEqual(moving, 0.0)
    warped_support = sitk.Resample(
        moving_support,
        fixed,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    fixed_mask = sitk.Cast(sitk.ReadImage(str(fixed_brain_mask_path)), sitk.sitkUInt8)
    a = sitk.GetArrayFromImage(warped_support) > 0
    b = sitk.GetArrayFromImage(fixed_mask) > 0
    return dice_coefficient(a, b)


def _nib_mask_from_sitk(image):
    import nibabel as nib

    from gbm_ai.api.segmentation.model_geometry import sitk_affine_ras, sitk_to_numpy_xyz

    data = (sitk_to_numpy_xyz(image) > 0.5).astype(np.uint8)
    affine = sitk_affine_ras(image)
    return nib.Nifti1Image(data, affine), data, affine


def _localization_to_response(localization: AnatomicalLocalization, study: Study, segmentation: Segmentation, quantification: TumorQuantification) -> dict:
    return {
        "version": ANATOMICAL_LOCALIZATION_VERSION,
        "status": "complete",
        "study_uuid": study.id,
        "segmentation_uuid": segmentation.id,
        "quantification_uuid": quantification.id,
        "localization_uuid": localization.id,
        "standard_space": localization.standard_space,
        "template_name": localization.template_name,
        "atlas_name": localization.atlas_name,
        "atlas_version": localization.atlas_version,
        "atlas_license": localization.atlas_license,
        "registration": {
            "method": localization.registration_method,
            "metric": localization.registration_metric,
            "metric_value": localization.registration_metric_value,
            "support_dice": localization.registration_support_dice,
            "qc_passed": localization.registration_qc_passed,
            "engineering_qc_threshold": REGISTRATION_SUPPORT_DICE_MIN,
        },
        "hemisphere": localization.hemisphere,
        "centroid_mni_mm": list(localization.centroid_mni_mm),
        "primary_region": localization.primary_region,
        "primary_region_overlap_voxels": localization.primary_region_overlap_voxels,
        "primary_region_overlap_fraction_of_wt": localization.primary_region_overlap_fraction_of_wt,
        "secondary_regions": list(localization.secondary_regions),
        "clinician_verification_required": True,
        "anatomical_localization_generated": True,
        "localization_is_functional_deficit_prediction": False,
        "segmentation_is_gbm_diagnosis": False,
        "clinical_validation_claimed": False,
        "next_step": "phase8_clinical_viewer",
    }


def run_anatomical_localization(
    db: Session,
    storage: LocalObjectStore,
    study: Study,
    *,
    atlas_root: Path,
    request_id: str | None = None,
    actor_type: AuditActorType = AuditActorType.DEMO_USER,
    actor_id: str | None = None,
) -> dict:
    try:
        quant_response = get_latest_tumor_quantification(db, storage, study)
    except TumorQuantificationServiceError as exc:
        raise AnatomicalLocalizationServiceError(
            "LOCALIZATION_QUANTIFICATION_NOT_READY",
            str(exc),
        ) from exc

    segmentation = db.get(Segmentation, quant_response["segmentation_uuid"])
    quantification = db.get(TumorQuantification, quant_response["quantification_uuid"])
    if segmentation is None or quantification is None:
        raise AnatomicalLocalizationServiceError(
            "LOCALIZATION_SOURCE_RECORD_MISSING",
            "current segmentation/quantification records could not be resolved",
        )
    if segmentation.review_status == SegmentationReviewStatus.REJECTED:
        raise AnatomicalLocalizationServiceError(
            "LOCALIZATION_SEGMENTATION_REJECTED",
            "anatomical localization must not be generated from a rejected segmentation",
        )

    t1 = _find_t1_model_geometry(study)
    try:
        manifest, asset_paths = load_and_verify_localization_assets(atlas_root)
    except LocalizationAssetError as exc:
        raise AnatomicalLocalizationServiceError(exc.code, str(exc)) from exc

    fingerprint = localization_source_fingerprint(
        segmentation_uuid=str(segmentation.id),
        quantification_uuid=str(quantification.id),
        wt_checksum_sha256=segmentation.wt_checksum_sha256,
        t1_checksum_sha256=str(t1["checksum_sha256"]),
        atlas_manifest_checksum_sha256=str(manifest["manifest_checksum_sha256"]),
    )
    existing = db.scalar(
        select(AnatomicalLocalization).where(
            AnatomicalLocalization.source_fingerprint_sha256 == fingerprint
        )
    )
    if existing is not None:
        for key, checksum in (
            (existing.transformed_wt_storage_key, existing.transformed_wt_checksum_sha256),
            (existing.transform_storage_key, existing.transform_checksum_sha256),
            (existing.overlap_details_storage_key, existing.overlap_details_checksum_sha256),
        ):
            try:
                if not storage.verify_checksum(key, checksum):
                    raise AnatomicalLocalizationServiceError(
                        "LOCALIZATION_EXISTING_ARTIFACT_INVALID",
                        "existing localization artifact failed checksum validation",
                    )
            except AnatomicalLocalizationServiceError:
                raise
            except Exception as exc:
                raise AnatomicalLocalizationServiceError(
                    "LOCALIZATION_EXISTING_ARTIFACT_INVALID",
                    "existing localization artifact is unavailable",
                ) from exc
        return _localization_to_response(existing, study, segmentation, quantification)

    t1_path = _materialize_storage_object(
        storage,
        str(t1["storage_key"]),
        str(t1["checksum_sha256"]),
        ".nii.gz",
    )
    wt_path = _materialize_storage_object(
        storage,
        segmentation.wt_storage_key,
        segmentation.wt_checksum_sha256,
        ".nii.gz",
    )
    stored_keys: list[str] = []
    transform_temp: Path | None = None
    warped_temp: Path | None = None
    try:
        import nibabel as nib
        import SimpleITK as sitk

        fixed, moving, transform, metric = _register_t1_to_standard_space(
            t1_path,
            asset_paths["template"],
        )
        support_dice = _registration_support_dice(
            moving,
            asset_paths["brain_mask"],
            fixed,
            transform,
        )
        if support_dice < REGISTRATION_SUPPORT_DICE_MIN:
            raise AnatomicalLocalizationServiceError(
                "LOCALIZATION_REGISTRATION_QC_FAILED",
                "standard-space registration did not meet the frozen engineering support-overlap threshold; manual review is required",
            )

        warped_wt_sitk = _resample_mask_to_standard(wt_path, fixed, transform)
        warped_nib, warped_data, warped_affine = _nib_mask_from_sitk(warped_wt_sitk)
        if int(np.count_nonzero(warped_data)) == 0:
            raise AnatomicalLocalizationServiceError(
                "LOCALIZATION_TRANSFORMED_MASK_EMPTY",
                "WT mask became empty after standard-space transformation",
            )

        cortical_img = nib.load(str(asset_paths["cortical"]))
        subcortical_img = nib.load(str(asset_paths["subcortical"]))
        if cortical_img.shape != warped_data.shape or subcortical_img.shape != warped_data.shape:
            raise AnatomicalLocalizationServiceError(
                "LOCALIZATION_ATLAS_GEOMETRY_MISMATCH",
                "frozen atlas geometry no longer matches the standard-space WT mask",
            )
        if not np.allclose(cortical_img.affine, warped_affine, atol=1e-3, rtol=0.0):
            raise AnatomicalLocalizationServiceError(
                "LOCALIZATION_ATLAS_AFFINE_MISMATCH",
                "frozen cortical atlas affine does not match the standard-space WT geometry",
            )
        if not np.allclose(subcortical_img.affine, warped_affine, atol=1e-3, rtol=0.0):
            raise AnatomicalLocalizationServiceError(
                "LOCALIZATION_ATLAS_AFFINE_MISMATCH",
                "frozen subcortical atlas affine does not match the standard-space WT geometry",
            )

        labels = json.loads(asset_paths["labels"].read_text(encoding="utf-8"))
        cortical_overlaps = compute_region_overlaps(
            warped_data,
            np.asanyarray(cortical_img.dataobj),
            [str(v) for v in labels["cortical"]],
            atlas_name="Harvard-Oxford cortical",
        )
        subcortical_overlaps = compute_region_overlaps(
            warped_data,
            np.asanyarray(subcortical_img.dataobj),
            [str(v) for v in labels["subcortical"]],
            atlas_name="Harvard-Oxford subcortical",
        )
        overlaps = merge_region_overlaps(cortical_overlaps, subcortical_overlaps)
        if not overlaps:
            raise AnatomicalLocalizationServiceError(
                "LOCALIZATION_ATLAS_OVERLAP_EMPTY",
                "transformed WT mask did not overlap a labeled Harvard-Oxford region",
            )

        primary = overlaps[0]
        secondary = [
            item.as_dict()
            for item in overlaps[1:]
            if item.fraction_of_wt >= SECONDARY_REGION_MIN_FRACTION_OF_WT
        ][:10]
        hemisphere = hemisphere_from_standard_mask(warped_data, warped_affine)
        centroid = centroid_world_mm(warped_data, warped_affine)

        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temp:
            warped_temp = Path(temp.name)
        nib.save(warped_nib, str(warped_temp))
        warped_key = storage.generate_study_derived_key(
            study.id,
            "localization_wt_mni",
            suffix=".nii.gz",
        )
        with warped_temp.open("rb") as source:
            stored_warp = storage.put_stream(warped_key, source)
        stored_keys.append(stored_warp.storage_key)

        with tempfile.NamedTemporaryFile(suffix=".tfm", delete=False) as temp:
            transform_temp = Path(temp.name)
        sitk.WriteTransform(transform, str(transform_temp))
        transform_key = storage.generate_study_derived_key(
            study.id,
            "localization_transform",
            suffix=".tfm",
        )
        with transform_temp.open("rb") as source:
            stored_transform = storage.put_stream(transform_key, source)
        stored_keys.append(stored_transform.storage_key)

        overlap_payload = {
            "version": ANATOMICAL_LOCALIZATION_VERSION,
            "standard_space": STANDARD_SPACE,
            "atlas_name": ATLAS_NAME,
            "atlas_version": ATLAS_VERSION,
            "atlas_license": ATLAS_LICENSE,
            "hemisphere": hemisphere,
            "centroid_mni_mm": list(centroid),
            "primary_region": primary.as_dict(),
            "secondary_regions": secondary,
            "all_region_overlaps": [item.as_dict() for item in overlaps],
            "registration_support_dice": round(float(support_dice), 6),
            "clinician_verification_required": True,
            "clinical_validation_claimed": False,
        }
        overlap_bytes = json.dumps(
            overlap_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        overlap_key = storage.generate_study_derived_key(
            study.id,
            "localization_overlap",
            suffix=".json",
        )
        stored_overlap = storage.put_stream(overlap_key, BytesIO(overlap_bytes))
        stored_keys.append(stored_overlap.storage_key)

        localization = AnatomicalLocalization(
            segmentation_id=segmentation.id,
            quantification_id=quantification.id,
            localization_version=ANATOMICAL_LOCALIZATION_VERSION,
            source_fingerprint_sha256=fingerprint,
            standard_space=STANDARD_SPACE,
            template_name=STANDARD_TEMPLATE_NAME,
            template_checksum_sha256=str(manifest["files"]["template"]["sha256"]),
            atlas_name=ATLAS_NAME,
            atlas_version=ATLAS_VERSION,
            atlas_license=ATLAS_LICENSE,
            atlas_manifest_checksum_sha256=str(manifest["manifest_checksum_sha256"]),
            registration_method=REGISTRATION_METHOD,
            registration_metric=REGISTRATION_METRIC,
            registration_metric_value=round(float(metric), 8),
            registration_support_dice=round(float(support_dice), 6),
            registration_qc_passed=True,
            hemisphere=hemisphere,
            centroid_mni_mm=list(centroid),
            primary_region=primary.label,
            primary_region_overlap_voxels=primary.overlap_voxels,
            primary_region_overlap_fraction_of_wt=round(primary.fraction_of_wt, 6),
            secondary_regions=secondary,
            transformed_wt_storage_key=stored_warp.storage_key,
            transformed_wt_checksum_sha256=stored_warp.sha256,
            transformed_wt_size_bytes=stored_warp.size_bytes,
            transform_storage_key=stored_transform.storage_key,
            transform_checksum_sha256=stored_transform.sha256,
            transform_size_bytes=stored_transform.size_bytes,
            overlap_details_storage_key=stored_overlap.storage_key,
            overlap_details_checksum_sha256=stored_overlap.sha256,
            overlap_details_size_bytes=stored_overlap.size_bytes,
            clinician_verification_required=True,
            anatomical_localization_generated=True,
            clinical_validation_claimed=False,
        )
        db.add(localization)
        segmentation.anatomical_localization_generated = True
        quantification.anatomical_localization_generated = True

        current = dict(study.segmentation_preparation_summary or {})
        current["anatomical_localization_generated"] = True
        inference = dict(current.get("inference") or {})
        if str(inference.get("segmentation_uuid") or "") == str(segmentation.id):
            inference["anatomical_localization_generated"] = True
            current["inference"] = inference
        current["localization"] = {
            "version": ANATOMICAL_LOCALIZATION_VERSION,
            "status": "complete",
            "standard_space": STANDARD_SPACE,
            "atlas_version": ATLAS_VERSION,
            "source_fingerprint_sha256": fingerprint,
            "clinician_verification_required": True,
            "clinical_validation_claimed": False,
        }
        current["next_step"] = "phase8_clinical_viewer"
        study.segmentation_preparation_summary = current

        db.flush()
        record_audit_event(
            db,
            action=AuditAction.LOCALIZATION_COMPLETED,
            entity_type=AuditEntityType.LOCALIZATION,
            entity_uuid=localization.id,
            actor_type=actor_type,
            actor_id=actor_id,
            request_id=request_id,
            technical_context={
                "status": "complete",
                "operation": "atlas_based_anatomical_localization",
                "result": "hemisphere_primary_region_and_standard_space_centroid_generated",
                "manual_review_required": True,
            },
            commit=False,
        )
        db.commit()
        db.refresh(localization)
        return _localization_to_response(localization, study, segmentation, quantification)

    except AnatomicalLocalizationServiceError:
        db.rollback()
        for key in stored_keys:
            try:
                if storage.exists(key):
                    storage.delete(key)
            except Exception:
                pass
        raise
    except (AnatomicalLocalizationError, LocalizationAssetError) as exc:
        db.rollback()
        for key in stored_keys:
            try:
                if storage.exists(key):
                    storage.delete(key)
            except Exception:
                pass
        raise AnatomicalLocalizationServiceError(getattr(exc, "code", "LOCALIZATION_FAILED"), str(exc)) from exc
    except Exception as exc:
        db.rollback()
        for key in stored_keys:
            try:
                if storage.exists(key):
                    storage.delete(key)
            except Exception:
                pass
        raise AnatomicalLocalizationServiceError(
            "LOCALIZATION_UNEXPECTED_FAILURE",
            "unexpected anatomical localization failure",
        ) from exc
    finally:
        t1_path.unlink(missing_ok=True)
        wt_path.unlink(missing_ok=True)
        if transform_temp is not None:
            transform_temp.unlink(missing_ok=True)
        if warped_temp is not None:
            warped_temp.unlink(missing_ok=True)


def get_latest_anatomical_localization(
    db: Session,
    storage: LocalObjectStore,
    study: Study,
    *,
    atlas_root: Path,
) -> dict:
    try:
        quant_response = get_latest_tumor_quantification(db, storage, study)
    except TumorQuantificationServiceError as exc:
        raise AnatomicalLocalizationServiceError(
            "LOCALIZATION_QUANTIFICATION_NOT_READY",
            str(exc),
        ) from exc
    segmentation = db.get(Segmentation, quant_response["segmentation_uuid"])
    quantification = db.get(TumorQuantification, quant_response["quantification_uuid"])
    if segmentation is None or quantification is None:
        raise AnatomicalLocalizationServiceError(
            "LOCALIZATION_SOURCE_RECORD_MISSING",
            "current segmentation/quantification records could not be resolved",
        )
    t1 = _find_t1_model_geometry(study)
    try:
        manifest, _ = load_and_verify_localization_assets(atlas_root)
    except LocalizationAssetError as exc:
        raise AnatomicalLocalizationServiceError(exc.code, str(exc)) from exc
    fingerprint = localization_source_fingerprint(
        segmentation_uuid=str(segmentation.id),
        quantification_uuid=str(quantification.id),
        wt_checksum_sha256=segmentation.wt_checksum_sha256,
        t1_checksum_sha256=str(t1["checksum_sha256"]),
        atlas_manifest_checksum_sha256=str(manifest["manifest_checksum_sha256"]),
    )
    localization = db.scalar(
        select(AnatomicalLocalization).where(
            AnatomicalLocalization.source_fingerprint_sha256 == fingerprint
        )
    )
    if localization is None:
        raise AnatomicalLocalizationServiceError(
            "LOCALIZATION_RESULT_NOT_AVAILABLE",
            "no current anatomical localization exists for this segmentation",
        )
    for key, checksum in (
        (localization.transformed_wt_storage_key, localization.transformed_wt_checksum_sha256),
        (localization.transform_storage_key, localization.transform_checksum_sha256),
        (localization.overlap_details_storage_key, localization.overlap_details_checksum_sha256),
    ):
        try:
            if not storage.verify_checksum(key, checksum):
                raise AnatomicalLocalizationServiceError(
                    "LOCALIZATION_RESULT_ARTIFACT_INVALID",
                    "current localization artifact failed checksum validation",
                )
        except AnatomicalLocalizationServiceError:
            raise
        except Exception as exc:
            raise AnatomicalLocalizationServiceError(
                "LOCALIZATION_RESULT_ARTIFACT_INVALID",
                "current localization artifact is unavailable",
            ) from exc
    return _localization_to_response(localization, study, segmentation, quantification)
