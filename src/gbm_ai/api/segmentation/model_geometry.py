from __future__ import annotations

import io
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gbm_ai.api.segmentation.contract import SEGMENTATION_REFERENCE_SPACING_MM
from gbm_ai.api.segmentation.volume_loading import LoadedSegmentationVolume
from gbm_ai.api.storage.local import LocalObjectStore, StoredObject


MODEL_GEOMETRY_VERSION = "phase6_step3_model_geometry_v1"
MODEL_REFERENCE_SEQUENCE = "T1C"
CANONICAL_ORIENTATION = ("R", "A", "S")
LINEAR_INTERPOLATION = "linear"
REGISTRATION_METRIC = "mattes_mutual_information"
REGISTRATION_TRANSFORM = "rigid_euler3d"
REGISTRATION_RANDOM_SEED = 42
REGISTRATION_SAMPLING_PERCENTAGE = 0.10
REGISTRATION_MAX_ITERATIONS = 100
REGISTRATION_MAX_DISPLACEMENT_MM = 75.0
GEOMETRY_AFFINE_ATOL_MM = 1e-3


class SegmentationModelGeometryError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RegistrationSummary:
    performed: bool
    metric: str | None
    final_metric_value: float | None
    optimizer_stop_condition: str | None
    transform: str | None
    max_sampled_displacement_mm: float | None
    deterministic_seed: int | None

    def as_dict(self) -> dict:
        return {
            "performed": self.performed,
            "metric": self.metric,
            "final_metric_value": self.final_metric_value,
            "optimizer_stop_condition": self.optimizer_stop_condition,
            "transform": self.transform,
            "max_sampled_displacement_mm": self.max_sampled_displacement_mm,
            "deterministic_seed": self.deterministic_seed,
        }


def _require_simpleitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise SegmentationModelGeometryError(
            "SIMPLEITK_NOT_INSTALLED",
            "SimpleITK is required for Phase 6 registration/resampling; install the cumulative requirements.txt",
        ) from exc
    return sitk


def _require_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise SegmentationModelGeometryError(
            "NIBABEL_NOT_INSTALLED",
            "NiBabel is required to persist model-geometry NIfTI artifacts",
        ) from exc
    return nib


def _affine_spacing(affine: np.ndarray) -> tuple[float, float, float]:
    matrix = np.asarray(affine[:3, :3], dtype=np.float64)
    spacing = np.linalg.norm(matrix, axis=0)
    if not bool(np.isfinite(spacing).all()) or bool(np.any(spacing <= 0)):
        raise SegmentationModelGeometryError(
            "MODEL_GEOMETRY_SPACING_INVALID",
            "volume affine contains invalid voxel spacing",
        )
    return tuple(float(v) for v in spacing)


def loaded_volume_to_sitk(volume: LoadedSegmentationVolume):
    """Convert canonical-RAS NumPy/NiBabel geometry to a SimpleITK LPS image."""
    sitk = _require_simpleitk()

    data = np.asarray(volume.data, dtype=np.float32)
    if data.ndim != 3:
        raise SegmentationModelGeometryError(
            "MODEL_GEOMETRY_CHANNEL_NOT_3D",
            f"{volume.sequence} is not a 3D volume",
        )

    affine_ras = np.asarray(volume.affine_ras, dtype=np.float64)
    if affine_ras.shape != (4, 4) or not bool(np.isfinite(affine_ras).all()):
        raise SegmentationModelGeometryError(
            "MODEL_GEOMETRY_AFFINE_INVALID",
            f"{volume.sequence} has an invalid RAS affine",
        )

    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine_lps = ras_to_lps @ affine_ras

    spacing = np.asarray(_affine_spacing(affine_lps), dtype=np.float64)
    direction = affine_lps[:3, :3] / spacing[np.newaxis, :]

    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-3, rtol=0.0):
        raise SegmentationModelGeometryError(
            "MODEL_GEOMETRY_DIRECTION_INVALID",
            f"{volume.sequence} direction matrix is not orthonormal enough for safe SimpleITK conversion",
        )

    image = sitk.GetImageFromArray(np.transpose(data, (2, 1, 0)), isVector=False)
    image = sitk.Cast(image, sitk.sitkFloat32)
    image.SetSpacing(tuple(float(v) for v in spacing))
    image.SetOrigin(tuple(float(v) for v in affine_lps[:3, 3]))
    image.SetDirection(tuple(float(v) for v in direction.reshape(-1)))
    return image


def sitk_affine_ras(image) -> np.ndarray:
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)

    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, :3] = direction * spacing[np.newaxis, :]
    affine_lps[:3, 3] = origin

    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    return lps_to_ras @ affine_lps


def sitk_to_numpy_xyz(image) -> np.ndarray:
    sitk = _require_simpleitk()
    array_zyx = sitk.GetArrayFromImage(image)
    data = np.transpose(np.asarray(array_zyx, dtype=np.float32), (2, 1, 0))
    if data.ndim != 3 or not bool(np.isfinite(data).all()):
        raise SegmentationModelGeometryError(
            "MODEL_GEOMETRY_RESAMPLED_ARRAY_INVALID",
            "resampled volume is not a finite 3D float array",
        )
    return data


def geometries_match(
    reference: LoadedSegmentationVolume,
    moving: LoadedSegmentationVolume,
) -> bool:
    return (
        reference.shape == moving.shape
        and np.allclose(
            reference.affine_ras,
            moving.affine_ras,
            atol=GEOMETRY_AFFINE_ATOL_MM,
            rtol=0.0,
        )
        and reference.orientation_codes == CANONICAL_ORIENTATION
        and moving.orientation_codes == CANONICAL_ORIENTATION
    )


def create_isotropic_reference(
    reference_image,
    target_spacing_mm: tuple[float, float, float] = SEGMENTATION_REFERENCE_SPACING_MM,
):
    sitk = _require_simpleitk()

    old_size = np.asarray(reference_image.GetSize(), dtype=np.int64)
    old_spacing = np.asarray(reference_image.GetSpacing(), dtype=np.float64)
    target_spacing = np.asarray(target_spacing_mm, dtype=np.float64)

    if old_size.shape != (3,) or bool(np.any(old_size < 2)):
        raise SegmentationModelGeometryError(
            "MODEL_GEOMETRY_REFERENCE_SIZE_INVALID",
            "T1C reference volume has invalid 3D dimensions",
        )
    if not bool(np.isfinite(target_spacing).all()) or bool(np.any(target_spacing <= 0)):
        raise SegmentationModelGeometryError(
            "MODEL_GEOMETRY_TARGET_SPACING_INVALID",
            "target model spacing must contain three positive finite values",
        )

    physical_extent = (old_size - 1) * old_spacing
    new_size = np.ceil(physical_extent / target_spacing).astype(np.int64) + 1

    if bool(np.any(new_size <= 1)) or int(np.prod(new_size, dtype=np.int64)) > 100_000_000:
        raise SegmentationModelGeometryError(
            "MODEL_GEOMETRY_REFERENCE_VOXEL_LIMIT",
            "1 mm reference geometry would create an unsafe volume size",
        )

    reference = sitk.Image([int(v) for v in new_size], sitk.sitkFloat32)
    reference.SetSpacing(tuple(float(v) for v in target_spacing))
    reference.SetOrigin(reference_image.GetOrigin())
    reference.SetDirection(reference_image.GetDirection())
    return reference


def _sample_reference_points(image) -> list[tuple[float, float, float]]:
    size = image.GetSize()
    indices = [
        (0, 0, 0),
        (size[0] - 1, 0, 0),
        (0, size[1] - 1, 0),
        (0, 0, size[2] - 1),
        (size[0] - 1, size[1] - 1, 0),
        (size[0] - 1, 0, size[2] - 1),
        (0, size[1] - 1, size[2] - 1),
        (size[0] - 1, size[1] - 1, size[2] - 1),
        tuple(int((value - 1) / 2) for value in size),
    ]
    return [image.TransformIndexToPhysicalPoint(index) for index in indices]


def _max_transform_displacement_mm(transform, fixed_image) -> float:
    distances: list[float] = []
    for point in _sample_reference_points(fixed_image):
        mapped = np.asarray(transform.TransformPoint(point), dtype=np.float64)
        original = np.asarray(point, dtype=np.float64)
        distances.append(float(np.linalg.norm(mapped - original)))
    return max(distances, default=0.0)


def register_rigid_to_reference(fixed_image, moving_image):
    """Rigid multimodal registration with deterministic Mattes mutual information."""
    sitk = _require_simpleitk()

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(32)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(
        REGISTRATION_SAMPLING_PERCENTAGE,
        REGISTRATION_RANDOM_SEED,
    )
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        1.0,
        REGISTRATION_MAX_ITERATIONS,
        1e-6,
        10,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    initial = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration.SetInitialTransform(initial, inPlace=False)

    try:
        transform = registration.Execute(fixed_image, moving_image)
    except RuntimeError as exc:
        raise SegmentationModelGeometryError(
            "SEGMENTATION_REGISTRATION_FAILED",
            "rigid multimodal registration failed",
        ) from exc

    metric_value = float(registration.GetMetricValue())
    if not math.isfinite(metric_value):
        raise SegmentationModelGeometryError(
            "SEGMENTATION_REGISTRATION_METRIC_INVALID",
            "registration produced a non-finite similarity metric",
        )

    displacement = _max_transform_displacement_mm(transform, fixed_image)
    if not math.isfinite(displacement) or displacement > REGISTRATION_MAX_DISPLACEMENT_MM:
        raise SegmentationModelGeometryError(
            "SEGMENTATION_REGISTRATION_DISPLACEMENT_UNSAFE",
            "automatic rigid registration exceeded the configured geometric safety bound",
        )

    summary = RegistrationSummary(
        performed=True,
        metric=REGISTRATION_METRIC,
        final_metric_value=round(metric_value, 8),
        optimizer_stop_condition=str(registration.GetOptimizerStopConditionDescription()),
        transform=REGISTRATION_TRANSFORM,
        max_sampled_displacement_mm=round(displacement, 6),
        deterministic_seed=REGISTRATION_RANDOM_SEED,
    )
    return transform, summary


def identity_registration_summary() -> RegistrationSummary:
    return RegistrationSummary(
        performed=False,
        metric=None,
        final_metric_value=None,
        optimizer_stop_condition=None,
        transform=None,
        max_sampled_displacement_mm=0.0,
        deterministic_seed=None,
    )


def resample_to_reference(moving_image, reference_image, transform=None):
    sitk = _require_simpleitk()
    if transform is None:
        transform = sitk.Transform(3, sitk.sitkIdentity)

    try:
        output = sitk.Resample(
            moving_image,
            reference_image,
            transform,
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32,
        )
    except RuntimeError as exc:
        raise SegmentationModelGeometryError(
            "MODEL_GEOMETRY_RESAMPLING_FAILED",
            "MRI channel could not be resampled into the frozen model reference geometry",
        ) from exc

    if tuple(float(v) for v in output.GetSpacing()) != tuple(
        float(v) for v in reference_image.GetSpacing()
    ):
        raise SegmentationModelGeometryError(
            "MODEL_GEOMETRY_OUTPUT_SPACING_MISMATCH",
            "resampled output spacing does not match the model reference geometry",
        )
    if output.GetSize() != reference_image.GetSize():
        raise SegmentationModelGeometryError(
            "MODEL_GEOMETRY_OUTPUT_SIZE_MISMATCH",
            "resampled output size does not match the model reference geometry",
        )
    return output


def persist_resampled_nifti(
    storage: LocalObjectStore,
    study_uuid,
    *,
    sequence: str,
    image,
) -> tuple[StoredObject, dict]:
    nib = _require_nibabel()

    data = sitk_to_numpy_xyz(image)
    affine_ras = sitk_affine_ras(image)

    if not np.allclose(
        np.asarray(_affine_spacing(affine_ras)),
        np.asarray(SEGMENTATION_REFERENCE_SPACING_MM, dtype=np.float64),
        atol=1e-6,
        rtol=0.0,
    ):
        raise SegmentationModelGeometryError(
            "MODEL_GEOMETRY_PERSIST_SPACING_MISMATCH",
            f"{sequence} model-geometry artifact is not at the frozen 1 mm spacing",
        )

    image_nifti = nib.Nifti1Image(data, affine_ras)
    image_nifti.set_data_dtype(np.float32)

    key = storage.generate_study_derived_key(
        study_uuid,
        "segmentation_model_geometry",
        suffix=".nii.gz",
    )

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temp:
        temp_path = Path(temp.name)

    try:
        nib.save(image_nifti, str(temp_path))
        with temp_path.open("rb") as source:
            stored = storage.put_stream(key, source)
    except Exception as exc:
        raise SegmentationModelGeometryError(
            "MODEL_GEOMETRY_ARTIFACT_STORE_FAILED",
            f"{sequence} model-geometry artifact could not be stored safely",
        ) from exc
    finally:
        temp_path.unlink(missing_ok=True)

    summary = {
        "sequence": sequence,
        "storage_key": stored.storage_key,
        "checksum_sha256": stored.sha256,
        "size_bytes": stored.size_bytes,
        "shape": [int(v) for v in data.shape],
        "spacing_mm": [
            round(float(v), 6)
            for v in _affine_spacing(affine_ras)
        ],
        "affine_ras": [
            [round(float(value), 6) for value in row]
            for row in affine_ras
        ],
        "dtype": "float32",
        "interpolation": LINEAR_INTERPOLATION,
    }
    return stored, summary
