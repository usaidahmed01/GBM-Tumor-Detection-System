from __future__ import annotations

import gzip
import io
import zipfile
from collections import Counter
from dataclasses import dataclass
from typing import BinaryIO, Callable

from PIL import Image, UnidentifiedImageError


ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
GZIP_MAGIC = b"\x1f\x8b"
ALLOWED_RASTER_FORMATS = {"JPEG", "PNG"}


class FormatDetectionError(ValueError):
    pass


class UnsupportedInputError(FormatDetectionError):
    pass


class AmbiguousArchiveError(FormatDetectionError):
    pass


class NonMRIInputError(FormatDetectionError):
    def __init__(self, modality: str) -> None:
        super().__init__(
            f"DICOM modality {modality!r} is outside the V1 MRI-only scope"
        )
        self.modality = modality


@dataclass(frozen=True)
class DetectedFormat:
    source_format: str
    parser: str
    modality: str
    technical_metadata: dict[str, object]


def _preserve_position(fn: Callable):
    def wrapper(source: BinaryIO, *args, **kwargs):
        current = source.tell()
        try:
            source.seek(0)
            return fn(source, *args, **kwargs)
        finally:
            source.seek(current)
    return wrapper


@_preserve_position
def has_zip_magic(source: BinaryIO) -> bool:
    return any(source.read(4).startswith(magic) for magic in ZIP_MAGIC)




def _has_nifti_signature(header_bytes: bytes) -> bool:
    if len(header_bytes) >= 348:
        nifti1_magic = header_bytes[344:348]
        if nifti1_magic in {b"n+1\x00", b"ni1\x00"}:
            return True

    if len(header_bytes) >= 12:
        nifti2_magic = header_bytes[4:12]
        if nifti2_magic.startswith(b"n+2") or nifti2_magic.startswith(b"ni2"):
            return True

    return False


def _read_nifti_header_bytes(source: BinaryIO, limit: int = 560) -> bytes:
    source.seek(0)
    prefix = source.read(2)
    source.seek(0)

    if prefix == GZIP_MAGIC:
        with gzip.GzipFile(fileobj=source, mode="rb") as gz:
            return gz.read(limit)

    return source.read(limit)


@_preserve_position
def detect_nifti(source: BinaryIO) -> DetectedFormat | None:
    """
    Parse the NIfTI header from content, including gzip-compressed NIfTI,
    without relying on a filename extension and without loading voxel data.
    """
    source.seek(0)
    gzip_wrapped = source.read(2) == GZIP_MAGIC
    source.seek(0)

    try:
        header_bytes = _read_nifti_header_bytes(source)
    except (OSError, EOFError):
        return None

    if not _has_nifti_signature(header_bytes):
        return None

    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError(
            "NiBabel is required for NIfTI format detection. "
            "Install cumulative requirements.txt."
        ) from exc

    # Select the NIfTI parser from the header signature before asking NiBabel
    # to parse it. Trying NIfTI-2 against a valid NIfTI-1 payload causes
    # noisy repair warnings (for example, "sizeof_hdr should be 540") even
    # though the NIfTI-1 file is valid.
    version: int
    header: object
    if len(header_bytes) >= 348 and header_bytes[344:348] in {
        b"n+1\x00",
        b"ni1\x00",
    }:
        try:
            header = nib.Nifti1Header.from_fileobj(
                io.BytesIO(header_bytes),
                check=True,
            )
        except Exception:
            return None
        version = 1
    elif (
        len(header_bytes) >= 540
        and (
            header_bytes[4:12].startswith(b"n+2")
            or header_bytes[4:12].startswith(b"ni2")
        )
    ):
        try:
            header = nib.Nifti2Header.from_fileobj(
                io.BytesIO(header_bytes),
                check=True,
            )
        except Exception:
            return None
        version = 2
    else:
        return None
    shape = tuple(int(v) for v in header.get_data_shape())
    dtype = str(header.get_data_dtype())
    magic = bytes(header["magic"])
    storage_form = "single_file" if magic.startswith(b"n+") else "pair_header"

    return DetectedFormat(
        source_format="nifti",
        parser=f"nibabel_nifti{version}",
        modality="UNKNOWN",
        technical_metadata={
            "nifti_version": version,
            "shape": list(shape),
            "dtype": dtype,
            "storage_form": storage_form,
            "gzip_wrapped": gzip_wrapped,
        },
    )


def _dataset_has_dicom_identity(ds) -> bool:
    identity_fields = (
        "SOPClassUID",
        "SOPInstanceUID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "Modality",
    )
    populated = sum(bool(getattr(ds, field, None)) for field in identity_fields)
    return populated >= 3 and bool(getattr(ds, "SOPClassUID", None))


@_preserve_position
def detect_dicom(source: BinaryIO) -> DetectedFormat | None:
    """
    Parse only selected technical DICOM tags and stop before pixel data.

    We first use strict File Format parsing. A conservative force=True fallback
    is allowed only when multiple DICOM identity fields are actually present;
    force=True alone is not accepted as evidence because arbitrary bytes can
    otherwise be interpreted as a deferred pydicom dataset.
    """
    source.seek(128)
    has_conformant_prefix = source.read(4) == b"DICM"
    source.seek(0)

    try:
        import pydicom
        from pydicom.errors import InvalidDicomError
    except ImportError as exc:
        # If the bytes explicitly carry the DICOM File Format prefix, fail
        # clearly because the required parser is missing. Otherwise this may
        # simply be an unrelated unsupported file, so allow later detection
        # logic to classify it as unknown.
        if has_conformant_prefix:
            raise RuntimeError(
                "pydicom is required for DICOM format detection. "
                "Install cumulative requirements.txt."
            ) from exc
        return None

    tags = [
        "SOPClassUID",
        "SOPInstanceUID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "Modality",
        "Rows",
        "Columns",
        "NumberOfFrames",
    ]

    ds = None
    try:
        ds = pydicom.dcmread(
            source,
            stop_before_pixels=True,
            defer_size="1 MB",
            force=False,
            specific_tags=tags,
        )
    except InvalidDicomError:
        source.seek(0)
        try:
            candidate = pydicom.dcmread(
                source,
                stop_before_pixels=True,
                defer_size="1 MB",
                force=True,
                specific_tags=tags,
            )
        except Exception:
            return None
        if _dataset_has_dicom_identity(candidate):
            ds = candidate
    except Exception:
        return None

    if ds is None or not _dataset_has_dicom_identity(ds):
        return None

    modality = str(getattr(ds, "Modality", "") or "").upper() or "UNKNOWN"

    metadata: dict[str, object] = {
        "modality": modality,
        "rows": int(ds.Rows) if getattr(ds, "Rows", None) is not None else None,
        "columns": (
            int(ds.Columns) if getattr(ds, "Columns", None) is not None else None
        ),
        "number_of_frames": (
            int(ds.NumberOfFrames)
            if getattr(ds, "NumberOfFrames", None) is not None
            else None
        ),
        # Deliberately do not persist PatientName, PatientID, DOB, institution,
        # accession number, original UIDs, or arbitrary raw metadata here.
        "dicom_identity_confirmed": True,
    }

    return DetectedFormat(
        source_format="dicom",
        parser="pydicom",
        modality=modality,
        technical_metadata=metadata,
    )


@_preserve_position
def detect_raster_image(source: BinaryIO) -> DetectedFormat | None:
    try:
        image = Image.open(source)
        image_format = str(image.format or "").upper()

        if image_format not in ALLOWED_RASTER_FORMATS:
            return None

        width, height = image.size
        mode = image.mode
        image.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        return None

    return DetectedFormat(
        source_format="image",
        parser="pillow",
        modality="UNKNOWN",
        technical_metadata={
            "raster_format": image_format,
            "width": int(width),
            "height": int(height),
            "mode": str(mode),
        },
    )


@_preserve_position
def detect_single_object(source: BinaryIO) -> DetectedFormat:
    if has_zip_magic(source):
        raise UnsupportedInputError(
            "ZIP object must be inspected as an archive, not as a single file"
        )

    # NIfTI first because compressed NIfTI has generic gzip framing.
    detected = detect_nifti(source)
    if detected is not None:
        return detected

    detected = detect_raster_image(source)
    if detected is not None:
        return detected

    detected = detect_dicom(source)
    if detected is not None:
        if detected.modality not in {"MR", "UNKNOWN"}:
            raise NonMRIInputError(detected.modality)
        return detected

    raise UnsupportedInputError(
        "stored object is not a supported JPG/PNG, DICOM, or NIfTI input"
    )


@_preserve_position
def detect_zip_contents(source: BinaryIO) -> DetectedFormat:
    if not has_zip_magic(source):
        raise UnsupportedInputError("stored object is not a ZIP archive")

    counters = Counter()
    modalities = Counter()
    parser_counts = Counter()
    example_metadata: dict[str, object] = {}
    unknown_entries = 0
    nifti_pair_headers = 0

    with zipfile.ZipFile(source, "r") as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if not infos:
            raise UnsupportedInputError("ZIP archive contains no files")

        for info in infos:
            with archive.open(info, "r") as member:
                # A content-level nested ZIP is rejected even if its filename
                # was disguised to bypass suffix checks in Step 1.
                if has_zip_magic(member):
                    raise UnsupportedInputError(
                        "nested ZIP content is not accepted"
                    )

                try:
                    detected = detect_single_object(member)
                except NonMRIInputError:
                    raise
                except UnsupportedInputError:
                    unknown_entries += 1
                    continue

                counters[detected.source_format] += 1
                parser_counts[detected.parser] += 1

                if detected.source_format == "dicom":
                    modalities[detected.modality] += 1

                if (
                    detected.source_format == "nifti"
                    and detected.technical_metadata.get("storage_form")
                    == "pair_header"
                ):
                    nifti_pair_headers += 1

                if not example_metadata:
                    example_metadata = dict(detected.technical_metadata)

    recognized_formats = [name for name, count in counters.items() if count > 0]

    if not recognized_formats:
        raise UnsupportedInputError(
            "ZIP archive contains no supported DICOM, NIfTI, JPG, or PNG input"
        )

    if len(recognized_formats) != 1:
        raise AmbiguousArchiveError(
            "ZIP archive mixes multiple recognized input formats: "
            + ", ".join(sorted(recognized_formats))
        )

    source_format = recognized_formats[0]

    if source_format == "dicom":
        explicit_non_mr = {
            modality
            for modality in modalities
            if modality not in {"MR", "UNKNOWN"}
        }
        if explicit_non_mr:
            raise NonMRIInputError(",".join(sorted(explicit_non_mr)))
        modality = "MR" if modalities.get("MR", 0) else "UNKNOWN"
    else:
        modality = "UNKNOWN"

    return DetectedFormat(
        source_format=source_format,
        parser="zip_content_scan",
        modality=modality,
        technical_metadata={
            "archive_recognized_format": source_format,
            "recognized_entry_count": int(counters[source_format]),
            "unknown_entry_count": int(unknown_entries),
            "parser_counts": dict(parser_counts),
            "dicom_modality_counts": dict(modalities),
            "nifti_pair_header_count": int(nifti_pair_headers),
            "example_format_metadata": example_metadata,
        },
    )


@_preserve_position
def detect_stored_object(source: BinaryIO) -> DetectedFormat:
    if has_zip_magic(source):
        return detect_zip_contents(source)
    return detect_single_object(source)
