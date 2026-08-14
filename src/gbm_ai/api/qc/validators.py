from __future__ import annotations

import gzip
import io
import math
import zipfile
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np
from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class BasicQC:
    fail_reasons: list[str]
    partial_reasons: list[str]
    warnings: list[str]
    checks: dict


def qc_raster_image(source: BinaryIO) -> BasicQC:
    fail: list[str] = []
    partial: list[str] = []
    warnings: list[str] = []

    current = source.tell()
    try:
        source.seek(0)
        try:
            image = Image.open(source)
            image.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            return BasicQC(
                fail_reasons=["RASTER_UNREADABLE"],
                partial_reasons=[],
                warnings=[],
                checks={"readable": False, "error_type": exc.__class__.__name__},
            )

        width, height = image.size
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        finite = np.isfinite(gray)
        finite_fraction = float(finite.mean()) if gray.size else 0.0

        if gray.size == 0 or finite_fraction < 1.0:
            fail.append("RASTER_INVALID_PIXEL_ARRAY")
            std = 0.0
            p01 = 0.0
            p99 = 0.0
        else:
            std = float(np.std(gray))
            p01, p99 = [float(v) for v in np.percentile(gray, [1, 99])]

        dynamic_range = p99 - p01
        min_dimension = min(width, height)
        aspect_ratio = max(width, height) / max(min_dimension, 1)

        if min_dimension < 32:
            fail.append("RASTER_EXTREMELY_LOW_RESOLUTION")
        elif min_dimension < 96:
            partial.append("RASTER_LOW_RESOLUTION")

        if std < 1.0 or dynamic_range < 2.0:
            fail.append("RASTER_BLANK_OR_NEAR_BLANK")
        elif dynamic_range < 12.0:
            partial.append("RASTER_LOW_CONTRAST")

        if aspect_ratio > 4.0:
            partial.append("RASTER_EXTREME_ASPECT_RATIO")

        # A standalone raster has no reliable volumetric geometry or body-part
        # metadata. This requires manual scope confirmation rather than an
        # unsupported automatic brain claim.
        partial.append("BRAIN_SCOPE_UNVERIFIED_FOR_RASTER")
        warnings.append("NO_PHYSICAL_SPATIAL_METADATA_FOR_RASTER")

        return BasicQC(
            fail_reasons=sorted(set(fail)),
            partial_reasons=sorted(set(partial)),
            warnings=sorted(set(warnings)),
            checks={
                "readable": True,
                "width": int(width),
                "height": int(height),
                "mode": image.mode,
                "std_intensity": round(std, 4),
                "p01": round(p01, 4),
                "p99": round(p99, 4),
                "dynamic_range": round(dynamic_range, 4),
                "aspect_ratio": round(aspect_ratio, 4),
                "brain_scope_status": "UNVERIFIED",
                "physical_spatial_metadata": False,
            },
        )
    finally:
        source.seek(current)


def _read_nifti_header_bytes(source: BinaryIO, limit: int = 560) -> tuple[bytes, bool]:
    source.seek(0)
    prefix = source.read(2)
    source.seek(0)
    gzip_wrapped = prefix == b"\x1f\x8b"

    if gzip_wrapped:
        with gzip.GzipFile(fileobj=source, mode="rb") as gz:
            return gz.read(limit), True

    return source.read(limit), False


def _parse_nifti_header(source: BinaryIO):
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError(
            "NiBabel is required for NIfTI QC. Install cumulative requirements.txt."
        ) from exc

    header_bytes, gzip_wrapped = _read_nifti_header_bytes(source)

    if len(header_bytes) >= 348 and header_bytes[344:348] in {
        b"n+1\x00",
        b"ni1\x00",
    }:
        header = nib.Nifti1Header.from_fileobj(io.BytesIO(header_bytes), check=True)
        version = 1
    elif (
        len(header_bytes) >= 540
        and (
            header_bytes[4:12].startswith(b"n+2")
            or header_bytes[4:12].startswith(b"ni2")
        )
    ):
        header = nib.Nifti2Header.from_fileobj(io.BytesIO(header_bytes), check=True)
        version = 2
    else:
        raise ValueError("not a recognized NIfTI header")

    return header, version, gzip_wrapped


def _qc_one_nifti(source: BinaryIO) -> dict:
    current = source.tell()
    try:
        source.seek(0)
        header, version, gzip_wrapped = _parse_nifti_header(source)

        shape = tuple(int(v) for v in header.get_data_shape())
        zooms = tuple(float(v) for v in header.get_zooms())
        affine = np.asarray(header.get_best_affine(), dtype=np.float64)

        spatial_zooms = zooms[: min(3, len(zooms))]
        affine_finite = bool(np.isfinite(affine).all())
        determinant = (
            float(np.linalg.det(affine[:3, :3]))
            if affine.shape == (4, 4) and affine_finite
            else float("nan")
        )

        shape_valid = len(shape) >= 3 and all(v > 0 for v in shape[:3])
        spatial_size_sufficient = shape_valid and min(shape[:3]) >= 8
        spacing_valid = (
            len(spatial_zooms) == 3
            and all(math.isfinite(v) and v > 0 for v in spatial_zooms)
        )
        affine_valid = (
            affine.shape == (4, 4)
            and affine_finite
            and math.isfinite(determinant)
            and abs(determinant) > 1e-8
        )

        return {
            "nifti_version": version,
            "gzip_wrapped": gzip_wrapped,
            "shape": list(shape),
            "zooms": [round(v, 8) for v in zooms],
            "shape_valid": shape_valid,
            "spatial_size_sufficient": spatial_size_sufficient,
            "spacing_valid": spacing_valid,
            "affine_valid": affine_valid,
            "affine_determinant": (
                round(determinant, 8) if math.isfinite(determinant) else None
            ),
            "is_3d": len(shape) == 3,
            "is_4d": len(shape) == 4,
        }
    finally:
        source.seek(current)


def qc_nifti_object(source: BinaryIO) -> BasicQC:
    fail: list[str] = []
    partial: list[str] = []
    warnings: list[str] = []
    volume_checks: list[dict] = []

    current = source.tell()
    try:
        source.seek(0)
        prefix = source.read(4)
        source.seek(0)

        try:
            if prefix.startswith(b"PK"):
                with zipfile.ZipFile(source, "r") as archive:
                    for info in archive.infolist():
                        if info.is_dir():
                            continue
                        with archive.open(info, "r") as member:
                            try:
                                volume_checks.append(_qc_one_nifti(member))
                            except Exception:
                                continue
            else:
                volume_checks.append(_qc_one_nifti(source))
        except (OSError, ValueError, zipfile.BadZipFile):
            fail.append("NIFTI_HEADER_UNREADABLE")

        if not volume_checks:
            fail.append("NIFTI_NO_VALID_VOLUME")

        for item in volume_checks:
            if not item["shape_valid"] or not item["spatial_size_sufficient"]:
                fail.append("NIFTI_INVALID_OR_TINY_SPATIAL_SHAPE")
            if not item["spacing_valid"]:
                fail.append("NIFTI_INVALID_VOXEL_SPACING")
            if not item["affine_valid"]:
                fail.append("NIFTI_INVALID_AFFINE")
            if item["is_4d"]:
                partial.append("NIFTI_4D_REQUIRES_VOLUME_SELECTION")
            elif not item["is_3d"]:
                partial.append("NIFTI_DIMENSIONALITY_REQUIRES_REVIEW")

        # Container headers do not reliably identify T1/T1c/T2/FLAIR or prove
        # that the volume depicts the brain.
        partial.append("NIFTI_SEQUENCE_MAPPING_REQUIRES_CONFIRMATION")
        partial.append("BRAIN_SCOPE_UNVERIFIED_FOR_NIFTI")
        warnings.append("NIFTI_VOXEL_QUALITY_NOT_FULLY_SAMPLED")

        return BasicQC(
            fail_reasons=sorted(set(fail)),
            partial_reasons=sorted(set(partial)),
            warnings=sorted(set(warnings)),
            checks={
                "volume_count": len(volume_checks),
                "volumes": volume_checks,
                "brain_scope_status": "UNVERIFIED",
                "sequence_mapping_status": "REQUIRES_CONFIRMATION",
            },
        )
    finally:
        source.seek(current)


def sample_dicom_pixel_quality(
    source: BinaryIO,
    *,
    max_samples_per_series: int = 3,
) -> dict:
    """
    Decode a small sample from the de-identified DICOM ZIP.

    Failure to decode compressed transfer syntaxes is treated as unverified,
    not automatically as corrupt, because optional codec plugins may be needed.
    """
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError(
            "pydicom is required for DICOM pixel QC. Install cumulative requirements.txt."
        ) from exc

    current = source.tell()
    decoded = 0
    decode_errors = 0
    blank_samples = 0
    low_resolution_samples = 0
    sample_metrics: list[dict] = []
    per_prefix_count: dict[str, int] = {}

    try:
        source.seek(0)
        with zipfile.ZipFile(source, "r") as archive:
            for info in archive.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".dcm"):
                    continue

                prefix = info.filename.split("/", 1)[0]
                count = per_prefix_count.get(prefix, 0)
                if count >= max_samples_per_series:
                    continue
                per_prefix_count[prefix] = count + 1

                with archive.open(info, "r") as member:
                    try:
                        ds = pydicom.dcmread(member)
                        array = np.asarray(ds.pixel_array, dtype=np.float32)
                    except Exception:
                        decode_errors += 1
                        continue

                decoded += 1
                if array.size == 0 or not np.isfinite(array).all():
                    blank_samples += 1
                    continue

                std = float(np.std(array))
                p01, p99 = [float(v) for v in np.percentile(array, [1, 99])]
                dynamic_range = p99 - p01
                spatial_shape = array.shape[-2:] if array.ndim >= 2 else array.shape
                min_dimension = min(spatial_shape) if spatial_shape else 0

                is_blank = std < 1e-6 or dynamic_range < 1e-6
                if is_blank:
                    blank_samples += 1
                if min_dimension < 64:
                    low_resolution_samples += 1

                sample_metrics.append(
                    {
                        "std_intensity": round(std, 6),
                        "dynamic_range": round(dynamic_range, 6),
                        "spatial_shape": [int(v) for v in spatial_shape],
                        "blank_or_constant": is_blank,
                    }
                )

        return {
            "decoded_sample_count": decoded,
            "decode_error_count": decode_errors,
            "blank_sample_count": blank_samples,
            "low_resolution_sample_count": low_resolution_samples,
            "samples": sample_metrics,
        }
    finally:
        source.seek(current)
