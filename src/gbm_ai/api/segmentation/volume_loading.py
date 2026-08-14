from __future__ import annotations

import gzip
import io
import math
import zipfile
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np
from sqlalchemy.orm import Session

from gbm_ai.api.models.analysis import Series, SourceFormat, Study
from gbm_ai.api.storage.local import LocalObjectStore


MAX_VOXELS_PER_CHANNEL = 50_000_000
AFFINE_ALIGNMENT_ATOL_MM = 1e-3
SPACING_ALIGNMENT_ATOL_MM = 1e-4
DICOM_DIRECTION_ATOL = 1e-3
DICOM_SLICE_SPACING_REL_TOL = 0.01
DICOM_SLICE_SPACING_ABS_TOL_MM = 0.05


class SegmentationVolumeLoadError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class LoadedSegmentationVolume:
    sequence: str
    source_kind: str
    source_reference: str
    data: np.ndarray
    affine_ras: np.ndarray
    orientation_codes: tuple[str, str, str]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.data.shape)

    @property
    def spacing_mm(self) -> tuple[float, float, float]:
        matrix = np.asarray(self.affine_ras[:3, :3], dtype=np.float64)
        return tuple(float(v) for v in np.linalg.norm(matrix, axis=0))


def _require_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise SegmentationVolumeLoadError(
            "NIBABEL_NOT_INSTALLED",
            "NiBabel is required for Phase 6 3D volume loading; install the cumulative requirements.txt",
        ) from exc
    return nib


def _require_pydicom():
    try:
        import pydicom
    except ImportError as exc:
        raise SegmentationVolumeLoadError(
            "PYDICOM_NOT_INSTALLED",
            "pydicom is required for Phase 6 DICOM volume reconstruction; install the cumulative requirements.txt",
        ) from exc
    return pydicom


def _decompress_if_gzip(raw: bytes) -> bytes:
    if raw.startswith(b"\x1f\x8b"):
        try:
            return gzip.decompress(raw)
        except OSError as exc:
            raise SegmentationVolumeLoadError(
                "NIFTI_GZIP_DECOMPRESSION_FAILED",
                "gzip-wrapped NIfTI payload could not be decompressed",
            ) from exc
    return raw


def _safe_nifti_payload_kind(raw: bytes) -> str | None:
    try:
        payload = _decompress_if_gzip(raw)
    except SegmentationVolumeLoadError:
        return None
    return _nifti_payload_kind(payload)


def _nifti_payload_kind(payload: bytes) -> str | None:
    if len(payload) >= 348 and payload[344:348] == b"n+1\x00":
        return "nifti1_single"
    if len(payload) >= 348 and payload[344:348] == b"ni1\x00":
        return "nifti1_pair"
    if len(payload) >= 540 and payload[4:12].startswith(b"n+2"):
        return "nifti2_single"
    if len(payload) >= 540 and payload[4:12].startswith(b"ni2"):
        return "nifti2_pair"
    return None


def _load_nifti_image_from_raw(raw: bytes):
    nib = _require_nibabel()
    payload = _decompress_if_gzip(raw)
    kind = _nifti_payload_kind(payload)

    if kind is None:
        raise SegmentationVolumeLoadError(
            "NIFTI_PAYLOAD_UNRECOGNIZED",
            "selected NIfTI volume is not a recognized NIfTI-1/2 payload",
        )
    if kind.endswith("_pair"):
        raise SegmentationVolumeLoadError(
            "NIFTI_DETACHED_PAIR_UNSUPPORTED",
            "detached NIfTI hdr/img pairs are not supported by the current Phase 6 loader",
        )

    image_class = nib.Nifti1Image if kind == "nifti1_single" else nib.Nifti2Image
    file_map = image_class.make_file_map()
    file_map["image"].fileobj = io.BytesIO(payload)

    try:
        return image_class.from_file_map(file_map)
    except Exception as exc:
        raise SegmentationVolumeLoadError(
            "NIFTI_FULL_VOLUME_LOAD_FAILED",
            "selected NIfTI payload passed header QC but its full 3D data could not be loaded",
        ) from exc


def _read_selected_nifti_raw(source: BinaryIO, target_volume_index: int) -> bytes:
    source.seek(0)
    prefix = source.read(4)
    source.seek(0)

    counted_index = 0

    if prefix.startswith(b"PK"):
        try:
            with zipfile.ZipFile(source, "r") as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    raw = archive.read(info)
                    if _safe_nifti_payload_kind(raw) is None:
                        continue
                    if counted_index == target_volume_index:
                        return raw
                    counted_index += 1
        except zipfile.BadZipFile as exc:
            raise SegmentationVolumeLoadError(
                "NIFTI_ARCHIVE_INVALID",
                "stored NIfTI archive is invalid",
            ) from exc
    else:
        raw = source.read()
        if _safe_nifti_payload_kind(raw) is not None and target_volume_index == 0:
            return raw

    raise SegmentationVolumeLoadError(
        "NIFTI_MAPPED_VOLUME_NOT_FOUND",
        f"confirmed NIfTI volume index {target_volume_index} could not be located in the protected source object",
    )


def _validate_loaded_array(sequence: str, data: np.ndarray) -> None:
    if data.ndim != 3:
        raise SegmentationVolumeLoadError(
            "SEGMENTATION_CHANNEL_NOT_3D",
            f"{sequence} did not load as one 3D volume",
        )

    voxel_count = int(np.prod(data.shape, dtype=np.int64))
    if voxel_count <= 0 or voxel_count > MAX_VOXELS_PER_CHANNEL:
        raise SegmentationVolumeLoadError(
            "SEGMENTATION_CHANNEL_VOXEL_LIMIT",
            f"{sequence} contains {voxel_count} voxels, outside the current safe Phase 6 loading limit",
        )

    if not bool(np.isfinite(data).all()):
        raise SegmentationVolumeLoadError(
            "SEGMENTATION_CHANNEL_NONFINITE",
            f"{sequence} contains non-finite voxel values",
        )

    if float(np.max(data) - np.min(data)) <= 1e-6:
        raise SegmentationVolumeLoadError(
            "SEGMENTATION_CHANNEL_BLANK",
            f"{sequence} is blank or effectively constant",
        )


def _canonicalize_image(sequence: str, image, *, source_kind: str, source_reference: str) -> LoadedSegmentationVolume:
    nib = _require_nibabel()

    try:
        canonical = nib.as_closest_canonical(image, enforce_diag=False)
        data = np.asarray(canonical.get_fdata(dtype=np.float32), dtype=np.float32)
        affine = np.asarray(canonical.affine, dtype=np.float64)
    except Exception as exc:
        raise SegmentationVolumeLoadError(
            "ORIENTATION_NORMALIZATION_FAILED",
            f"{sequence} could not be normalized to a canonical orientation",
        ) from exc

    _validate_loaded_array(sequence, data)

    if affine.shape != (4, 4) or not bool(np.isfinite(affine).all()):
        raise SegmentationVolumeLoadError(
            "SEGMENTATION_CHANNEL_INVALID_AFFINE",
            f"{sequence} has an invalid affine after orientation normalization",
        )

    determinant = float(np.linalg.det(affine[:3, :3]))
    if not math.isfinite(determinant) or abs(determinant) <= 1e-8:
        raise SegmentationVolumeLoadError(
            "SEGMENTATION_CHANNEL_SINGULAR_AFFINE",
            f"{sequence} has a singular affine after orientation normalization",
        )

    orientation = tuple(str(code) for code in nib.aff2axcodes(affine))
    if orientation != ("R", "A", "S"):
        raise SegmentationVolumeLoadError(
            "CANONICAL_ORIENTATION_NOT_RAS",
            f"{sequence} did not normalize to RAS orientation",
        )

    return LoadedSegmentationVolume(
        sequence=sequence,
        source_kind=source_kind,
        source_reference=source_reference,
        data=data,
        affine_ras=affine,
        orientation_codes=orientation,
    )


def load_nifti_channel_volume(
    storage: LocalObjectStore,
    study: Study,
    *,
    sequence: str,
    volume_index: int,
) -> LoadedSegmentationVolume:
    if not study.storage_key:
        raise SegmentationVolumeLoadError(
            "NIFTI_SOURCE_OBJECT_MISSING",
            "NIfTI study has no protected source object",
        )

    with storage.open_read(study.storage_key) as source:
        raw = _read_selected_nifti_raw(source, volume_index)

    image = _load_nifti_image_from_raw(raw)
    return _canonicalize_image(
        sequence,
        image,
        source_kind="nifti_volume",
        source_reference=f"volume_index:{volume_index}",
    )


def _float_tuple(value, *, length: int, field_name: str) -> tuple[float, ...]:
    if value is None:
        raise SegmentationVolumeLoadError(
            "DICOM_GEOMETRY_MISSING",
            f"DICOM {field_name} is required for safe 3D reconstruction",
        )
    try:
        parsed = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise SegmentationVolumeLoadError(
            "DICOM_GEOMETRY_INVALID",
            f"DICOM {field_name} contains invalid numeric values",
        ) from exc
    if len(parsed) != length or not all(math.isfinite(v) for v in parsed):
        raise SegmentationVolumeLoadError(
            "DICOM_GEOMETRY_INVALID",
            f"DICOM {field_name} must contain {length} finite numeric values",
        )
    return parsed


def _load_dicom_series_image(source: BinaryIO, series: Series, sequence: str):
    pydicom = _require_pydicom()
    nib = _require_nibabel()

    source.seek(0)
    try:
        archive = zipfile.ZipFile(source, "r")
    except zipfile.BadZipFile as exc:
        raise SegmentationVolumeLoadError(
            "DICOM_WORKING_ARCHIVE_INVALID",
            "de-identified DICOM working object is not a valid ZIP archive",
        ) from exc

    with archive:
        names = sorted(
            info.filename
            for info in archive.infolist()
            if not info.is_dir()
            and info.filename.startswith(series.working_member_prefix)
            and info.filename.lower().endswith(".dcm")
        )

        if not names:
            raise SegmentationVolumeLoadError(
                "DICOM_SERIES_MEMBERS_MISSING",
                f"no de-identified DICOM instances were found for {sequence}",
            )

        datasets = []
        for name in names:
            with archive.open(name, "r") as member:
                try:
                    ds = pydicom.dcmread(member, force=False)
                except Exception as exc:
                    raise SegmentationVolumeLoadError(
                        "DICOM_INSTANCE_READ_FAILED",
                        f"a de-identified {sequence} DICOM instance could not be read",
                    ) from exc

            frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
            if frames != 1:
                raise SegmentationVolumeLoadError(
                    "DICOM_MULTIFRAME_NOT_IMPLEMENTED",
                    f"{sequence} uses multi-frame DICOM, which requires a dedicated validated reconstruction path",
                )
            datasets.append(ds)

    first = datasets[0]
    rows = int(getattr(first, "Rows", 0) or 0)
    columns = int(getattr(first, "Columns", 0) or 0)
    if rows <= 0 or columns <= 0:
        raise SegmentationVolumeLoadError(
            "DICOM_MATRIX_INVALID",
            f"{sequence} has invalid Rows/Columns metadata",
        )

    pixel_spacing = _float_tuple(
        getattr(first, "PixelSpacing", None),
        length=2,
        field_name="PixelSpacing",
    )
    if min(pixel_spacing) <= 0:
        raise SegmentationVolumeLoadError(
            "DICOM_PIXEL_SPACING_INVALID",
            f"{sequence} has non-positive PixelSpacing",
        )

    orientation = _float_tuple(
        getattr(first, "ImageOrientationPatient", None),
        length=6,
        field_name="ImageOrientationPatient",
    )
    row_cos = np.asarray(orientation[:3], dtype=np.float64)
    col_cos = np.asarray(orientation[3:], dtype=np.float64)

    if (
        not np.isclose(np.linalg.norm(row_cos), 1.0, atol=DICOM_DIRECTION_ATOL)
        or not np.isclose(np.linalg.norm(col_cos), 1.0, atol=DICOM_DIRECTION_ATOL)
        or not np.isclose(float(np.dot(row_cos, col_cos)), 0.0, atol=DICOM_DIRECTION_ATOL)
    ):
        raise SegmentationVolumeLoadError(
            "DICOM_ORIENTATION_NOT_ORTHONORMAL",
            f"{sequence} ImageOrientationPatient is not a valid orthonormal orientation",
        )

    normal = np.cross(row_cos, col_cos)
    normal_norm = float(np.linalg.norm(normal))
    if not math.isfinite(normal_norm) or normal_norm <= 1e-8:
        raise SegmentationVolumeLoadError(
            "DICOM_SLICE_NORMAL_INVALID",
            f"{sequence} slice normal could not be derived",
        )
    normal = normal / normal_norm

    positioned: list[tuple[float, object, tuple[float, float, float]]] = []

    for ds in datasets:
        if int(getattr(ds, "Rows", 0) or 0) != rows or int(getattr(ds, "Columns", 0) or 0) != columns:
            raise SegmentationVolumeLoadError(
                "DICOM_MATRIX_INCONSISTENT",
                f"{sequence} contains inconsistent slice matrix sizes",
            )

        spacing = _float_tuple(
            getattr(ds, "PixelSpacing", None),
            length=2,
            field_name="PixelSpacing",
        )
        if not np.allclose(spacing, pixel_spacing, atol=1e-6, rtol=0.0):
            raise SegmentationVolumeLoadError(
                "DICOM_PIXEL_SPACING_INCONSISTENT",
                f"{sequence} contains inconsistent PixelSpacing values",
            )

        current_orientation = _float_tuple(
            getattr(ds, "ImageOrientationPatient", None),
            length=6,
            field_name="ImageOrientationPatient",
        )
        if not np.allclose(current_orientation, orientation, atol=1e-6, rtol=0.0):
            raise SegmentationVolumeLoadError(
                "DICOM_ORIENTATION_INCONSISTENT",
                f"{sequence} contains inconsistent ImageOrientationPatient values",
            )

        ipp = _float_tuple(
            getattr(ds, "ImagePositionPatient", None),
            length=3,
            field_name="ImagePositionPatient",
        )
        projection = float(np.dot(np.asarray(ipp, dtype=np.float64), normal))
        positioned.append((projection, ds, ipp))

    positioned.sort(key=lambda item: item[0])
    projections = np.asarray([item[0] for item in positioned], dtype=np.float64)

    if len(projections) < 2:
        raise SegmentationVolumeLoadError(
            "DICOM_TOO_FEW_SLICES_FOR_3D",
            f"{sequence} requires at least two spatially positioned slices",
        )

    diffs = np.diff(projections)
    if bool(np.any(diffs <= 1e-6)):
        raise SegmentationVolumeLoadError(
            "DICOM_DUPLICATE_OR_UNORDERED_POSITIONS",
            f"{sequence} contains duplicate or invalid slice positions",
        )

    slice_spacing = float(np.median(diffs))
    allowed_deviation = max(
        DICOM_SLICE_SPACING_ABS_TOL_MM,
        abs(slice_spacing) * DICOM_SLICE_SPACING_REL_TOL,
    )
    if float(np.max(np.abs(diffs - slice_spacing))) > allowed_deviation:
        raise SegmentationVolumeLoadError(
            "DICOM_IRREGULAR_SLICE_SPACING",
            f"{sequence} contains irregular slice spacing that is unsafe for direct volume reconstruction",
        )

    slices: list[np.ndarray] = []
    photometric = str(getattr(first, "PhotometricInterpretation", "") or "").upper()

    for _, ds, _ in positioned:
        try:
            array = np.asarray(ds.pixel_array, dtype=np.float32)
        except Exception as exc:
            raise SegmentationVolumeLoadError(
                "DICOM_PIXEL_DECODE_FAILED",
                f"{sequence} pixel data could not be decoded; an optional DICOM codec may be required",
            ) from exc

        if array.ndim != 2 or array.shape != (rows, columns):
            raise SegmentationVolumeLoadError(
                "DICOM_PIXEL_ARRAY_SHAPE_INVALID",
                f"{sequence} contains a non-2D or mismatched single-frame pixel array",
            )

        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        array = array * slope + intercept
        slices.append(array)

    volume = np.stack(slices, axis=2).astype(np.float32, copy=False)

    if photometric == "MONOCHROME1":
        low = float(np.min(volume))
        high = float(np.max(volume))
        volume = (low + high - volume).astype(np.float32, copy=False)
    elif photometric not in {"", "MONOCHROME2"}:
        raise SegmentationVolumeLoadError(
            "DICOM_PHOTOMETRIC_UNSUPPORTED",
            f"{sequence} uses unsupported PhotometricInterpretation {photometric!r}",
        )

    first_position = np.asarray(positioned[0][2], dtype=np.float64)

    affine_lps = np.eye(4, dtype=np.float64)
    # NumPy axis 0 is DICOM row index and therefore follows the second IOP
    # direction cosine. NumPy axis 1 follows the first IOP direction cosine.
    affine_lps[:3, 0] = col_cos * pixel_spacing[0]
    affine_lps[:3, 1] = row_cos * pixel_spacing[1]
    affine_lps[:3, 2] = normal * slice_spacing
    affine_lps[:3, 3] = first_position

    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine_ras = lps_to_ras @ affine_lps

    image = nib.Nifti1Image(volume, affine_ras)
    return image


def load_dicom_channel_volume(
    db: Session,
    storage: LocalObjectStore,
    study: Study,
    *,
    sequence: str,
    series_uuid,
) -> LoadedSegmentationVolume:
    if not study.deidentified_storage_key:
        raise SegmentationVolumeLoadError(
            "DICOM_DEIDENTIFIED_OBJECT_MISSING",
            "DICOM study has no de-identified AI working object",
        )

    series = db.get(Series, series_uuid)
    if series is None or series.study_id != study.id:
        raise SegmentationVolumeLoadError(
            "DICOM_SERIES_REFERENCE_INVALID",
            f"{sequence} references a DICOM series that does not belong to the study",
        )

    if bool((series.sequence_metadata or {}).get("multiframe_present")):
        raise SegmentationVolumeLoadError(
            "DICOM_MULTIFRAME_NOT_IMPLEMENTED",
            f"{sequence} is multi-frame DICOM and is not supported by the current Phase 6 reconstruction path",
        )

    with storage.open_read(study.deidentified_storage_key) as source:
        image = _load_dicom_series_image(source, series, sequence)

    return _canonicalize_image(
        sequence,
        image,
        source_kind="dicom_series",
        source_reference=f"series_uuid:{series.id}",
    )


def load_channel_volume(
    db: Session,
    storage: LocalObjectStore,
    study: Study,
    channel_plan: dict,
) -> LoadedSegmentationVolume:
    sequence = str(channel_plan["sequence"])

    if study.source_format == SourceFormat.NIFTI:
        volume_index = channel_plan.get("volume_index")
        if not isinstance(volume_index, int):
            raise SegmentationVolumeLoadError(
                "NIFTI_VOLUME_INDEX_INVALID",
                f"{sequence} does not have a valid NIfTI volume index",
            )
        return load_nifti_channel_volume(
            storage,
            study,
            sequence=sequence,
            volume_index=volume_index,
        )

    if study.source_format == SourceFormat.DICOM:
        series_uuid = channel_plan.get("series_uuid")
        if series_uuid is None:
            raise SegmentationVolumeLoadError(
                "DICOM_SERIES_REFERENCE_INVALID",
                f"{sequence} does not have a valid DICOM series reference",
            )
        return load_dicom_channel_volume(
            db,
            storage,
            study,
            sequence=sequence,
            series_uuid=series_uuid,
        )

    raise SegmentationVolumeLoadError(
        "SEGMENTATION_SOURCE_FORMAT_UNSUPPORTED",
        "3D volume loading accepts only DICOM or NIfTI studies",
    )


def summarize_loaded_volume(volume: LoadedSegmentationVolume) -> dict:
    spacing = volume.spacing_mm
    return {
        "sequence": volume.sequence,
        "source_kind": volume.source_kind,
        "source_reference": volume.source_reference,
        "shape": list(volume.shape),
        "spacing_mm": [round(float(v), 6) for v in spacing],
        "orientation_codes": list(volume.orientation_codes),
        "affine_ras": [
            [round(float(value), 6) for value in row]
            for row in np.asarray(volume.affine_ras, dtype=np.float64)
        ],
        "voxel_count": int(np.prod(volume.shape, dtype=np.int64)),
        "dtype": "float32",
        "orientation_normalized": True,
    }


def validate_channel_alignment(channel_summaries: list[dict]) -> dict:
    if len(channel_summaries) != 4:
        return {
            "aligned": False,
            "reference_sequence": "T1C",
            "reasons": ["FOUR_CHANNEL_GEOMETRY_REQUIRED"],
        }

    by_sequence = {
        str(item["sequence"]): item
        for item in channel_summaries
    }
    reference = by_sequence.get("T1C")
    if reference is None:
        return {
            "aligned": False,
            "reference_sequence": "T1C",
            "reasons": ["T1C_REFERENCE_GEOMETRY_MISSING"],
        }

    ref_shape = tuple(int(v) for v in reference["shape"])
    ref_spacing = np.asarray(reference["spacing_mm"], dtype=np.float64)
    ref_affine = np.asarray(reference["affine_ras"], dtype=np.float64)

    reasons: list[str] = []

    for sequence in ("T1C", "T1", "T2", "FLAIR"):
        item = by_sequence.get(sequence)
        if item is None:
            reasons.append(f"{sequence}_GEOMETRY_MISSING")
            continue

        if tuple(int(v) for v in item["shape"]) != ref_shape:
            reasons.append(f"{sequence}_SHAPE_DIFFERS_FROM_T1C")

        spacing = np.asarray(item["spacing_mm"], dtype=np.float64)
        if not np.allclose(
            spacing,
            ref_spacing,
            atol=SPACING_ALIGNMENT_ATOL_MM,
            rtol=0.0,
        ):
            reasons.append(f"{sequence}_SPACING_DIFFERS_FROM_T1C")

        affine = np.asarray(item["affine_ras"], dtype=np.float64)
        if not np.allclose(
            affine,
            ref_affine,
            atol=AFFINE_ALIGNMENT_ATOL_MM,
            rtol=0.0,
        ):
            reasons.append(f"{sequence}_AFFINE_DIFFERS_FROM_T1C")

        if tuple(item.get("orientation_codes") or ()) != ("R", "A", "S"):
            reasons.append(f"{sequence}_NOT_CANONICAL_RAS")

    reasons = sorted(set(reasons))
    return {
        "aligned": not reasons,
        "reference_sequence": "T1C",
        "reasons": reasons,
        "affine_tolerance_mm": AFFINE_ALIGNMENT_ATOL_MM,
        "spacing_tolerance_mm": SPACING_ALIGNMENT_ATOL_MM,
    }
