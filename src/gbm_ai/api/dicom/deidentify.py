from __future__ import annotations

import io
import re
import shutil
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

try:
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.errors import InvalidDicomError
    from pydicom.multival import MultiValue
    from pydicom.tag import Tag
    from pydicom.uid import (
        ExplicitVRLittleEndian,
        PYDICOM_IMPLEMENTATION_UID,
        generate_uid,
    )
except ImportError:  # pragma: no cover - surfaced clearly at runtime
    pydicom = None
    Dataset = object
    FileMetaDataset = object
    InvalidDicomError = Exception
    MultiValue = tuple
    Tag = None
    ExplicitVRLittleEndian = None
    PYDICOM_IMPLEMENTATION_UID = None
    generate_uid = None


class DicomProcessingError(ValueError):
    pass


class DicomGroupingError(DicomProcessingError):
    pass


class DicomPixelPrivacyRiskError(DicomProcessingError):
    pass


class DicomModalityError(DicomProcessingError):
    pass


class DicomDeidentificationError(DicomProcessingError):
    pass


SAFE_SEQUENCE_TOKENS = {
    "t1",
    "t1w",
    "t2",
    "t2w",
    "flair",
    "fluid",
    "attenuated",
    "inversion",
    "recovery",
    "post",
    "pre",
    "contrast",
    "contrasted",
    "gad",
    "gadolinium",
    "ce",
    "enhanced",
    "mprage",
    "spgr",
    "tse",
    "fse",
    "se",
    "ir",
    "turbo",
    "spin",
    "echo",
    "axial",
    "sagittal",
    "coronal",
    "3d",
    "2d",
    "fs",
    "fat",
    "sat",
}

# Free-text VRs are removed from the AI working copy. This is intentionally
# conservative for a research prototype and avoids retaining scanner/site/user
# text strings that may contain identifiers.
FREE_TEXT_VRS = {
    "PN",
    "LO",
    "LT",
    "SH",
    "ST",
    "UT",
    "UC",
    "UR",
    "AE",
}

DATE_TIME_AGE_VRS = {
    "DA",
    "DT",
    "TM",
    "AS",
}

UID_CLASS_KEYWORDS = {
    "TransferSyntaxUID",
    "ImplementationClassUID",
}


@dataclass
class SeriesAccumulator:
    source_series_uid: str
    deidentified_series_uid: str
    series_number: int | None
    member_prefix: str
    instance_count: int = 0
    slice_count: int = 0
    series_description_tokens: set[str] = field(default_factory=set)
    protocol_name_tokens: set[str] = field(default_factory=set)
    scanning_sequence: set[str] = field(default_factory=set)
    sequence_variant: set[str] = field(default_factory=set)
    scan_options: set[str] = field(default_factory=set)
    image_type: set[str] = field(default_factory=set)
    mr_acquisition_type: set[str] = field(default_factory=set)
    repetition_times_ms: set[float] = field(default_factory=set)
    echo_times_ms: set[float] = field(default_factory=set)
    inversion_times_ms: set[float] = field(default_factory=set)
    flip_angles_deg: set[float] = field(default_factory=set)
    magnetic_field_strength_t: set[float] = field(default_factory=set)
    contrast_present: bool = False
    pixel_spacings: set[tuple[float, ...]] = field(default_factory=set)
    image_orientations: set[tuple[float, ...]] = field(default_factory=set)
    slice_thicknesses: set[float] = field(default_factory=set)
    spacing_between_slices: set[float] = field(default_factory=set)
    position_available_count: int = 0
    multiframe_present: bool = False
    body_part_hints: set[str] = field(default_factory=set)
    matrix_sizes: set[tuple[int, int]] = field(default_factory=set)

    def to_record(self) -> dict:
        return {
            "series_uid": self.deidentified_series_uid,
            "series_number": self.series_number,
            "detected_sequence": None,
            "confirmed_sequence": None,
            "sequence_confidence": None,
            "sequence_metadata": {
                "series_description_tokens": sorted(self.series_description_tokens),
                "protocol_name_tokens": sorted(self.protocol_name_tokens),
                "scanning_sequence": sorted(self.scanning_sequence),
                "sequence_variant": sorted(self.sequence_variant),
                "scan_options": sorted(self.scan_options),
                "image_type": sorted(self.image_type),
                "mr_acquisition_type": sorted(self.mr_acquisition_type),
                "repetition_time_ms": _single_or_list(self.repetition_times_ms),
                "echo_time_ms": _single_or_list(self.echo_times_ms),
                "inversion_time_ms": _single_or_list(self.inversion_times_ms),
                "flip_angle_deg": _single_or_list(self.flip_angles_deg),
                "magnetic_field_strength_t": _single_or_list(
                    self.magnetic_field_strength_t
                ),
                "contrast_metadata_present": self.contrast_present,
                "instance_count": self.instance_count,
                "multiframe_present": self.multiframe_present,
                "body_part_hints": sorted(self.body_part_hints),
                "matrix_sizes": [
                    list(item) for item in sorted(self.matrix_sizes)
                ],
            },
            "slice_count": self.slice_count,
            "spacing_orientation_metadata": {
                "pixel_spacing": _single_or_list(self.pixel_spacings),
                "pixel_spacing_consistent": len(self.pixel_spacings) <= 1,
                "image_orientation_patient": _single_or_list(
                    self.image_orientations
                ),
                "orientation_consistent": len(self.image_orientations) <= 1,
                "slice_thickness": _single_or_list(self.slice_thicknesses),
                "spacing_between_slices": _single_or_list(
                    self.spacing_between_slices
                ),
                "image_position_available_count": self.position_available_count,
            },
            "working_member_prefix": self.member_prefix,
        }


@dataclass(frozen=True)
class DeidentifiedDicomPackage:
    study_uid: str
    series_records: list[dict]
    output_stream: BinaryIO
    input_instance_count: int
    output_instance_count: int
    ignored_non_dicom_entries: int
    private_tags_removed: bool
    free_text_removed: bool
    uid_remapping_applied: bool
    pixel_data_modified: bool
    pixel_privacy_status: str


class UIDMapper:
    def __init__(self) -> None:
        self._mapping: dict[str, str] = {}

    def map(self, source_uid: str) -> str:
        source_uid = str(source_uid or "").strip()
        if not source_uid:
            raise DicomGroupingError("required DICOM UID is missing")
        if source_uid not in self._mapping:
            self._mapping[source_uid] = generate_uid()
        return self._mapping[source_uid]


def _single_or_list(values):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        value = ordered[0]
        if isinstance(value, tuple):
            return list(value)
        return value
    return [list(v) if isinstance(v, tuple) else v for v in ordered]


def _safe_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return round(float(value), 8)
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_tuple(value) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        return tuple(round(float(v), 8) for v in value)
    except (TypeError, ValueError):
        return None


def _string_values(value) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, MultiValue)):
        values = value
    else:
        values = [value]
    result = set()
    for item in values:
        text = str(item).strip()
        if text:
            result.add(text.upper())
    return result


def safe_sequence_tokens(value) -> set[str]:
    if value is None:
        return set()

    normalized = str(value).lower()
    raw_tokens = re.findall(r"[a-z]+\d+[a-z]*|\d+d|[a-z]+|\d+", normalized)

    tokens = {
        token
        for token in raw_tokens
        if token in SAFE_SEQUENCE_TOKENS
    }

    # Normalize common compact forms.
    if re.search(r"\bt1\s*(c|ce|post|gd|gad)\b", normalized):
        tokens.update({"t1", "post", "contrast"})
    if "t1c" in normalized:
        tokens.update({"t1", "post", "contrast"})
    if "t1ce" in normalized:
        tokens.update({"t1", "post", "contrast"})
    if "postcontrast" in normalized or "post-contrast" in normalized:
        tokens.update({"post", "contrast"})

    return tokens



def safe_body_part_hint(value) -> str:
    """
    Convert uncontrolled DICOM body-part text to a tiny privacy-safe scope hint.
    Raw source text is never persisted.
    """
    text = str(value or "").strip().lower()
    if not text:
        return "UNKNOWN"

    if any(token in text for token in ("brain", "head", "cranium", "skull")):
        return "BRAIN_OR_HEAD"

    explicit_non_brain = (
        "chest",
        "thorax",
        "abdomen",
        "pelvis",
        "knee",
        "shoulder",
        "spine",
        "lumbar",
        "cervical",
        "ankle",
        "wrist",
        "elbow",
        "hip",
        "foot",
        "hand",
        "breast",
        "cardiac",
        "heart",
    )
    if any(token in text for token in explicit_non_brain):
        return "NON_BRAIN"

    return "UNKNOWN"


def _require_pydicom() -> None:
    if pydicom is None:
        raise RuntimeError(
            "pydicom is required for DICOM grouping/de-identification. "
            "Install cumulative requirements.txt."
        )


def read_dicom_dataset(source: BinaryIO):
    _require_pydicom()

    source.seek(0)
    try:
        return pydicom.dcmread(
            source,
            force=False,
            defer_size="8 MB",
        )
    except InvalidDicomError:
        source.seek(0)
        candidate = pydicom.dcmread(
            source,
            force=True,
            defer_size="8 MB",
        )

        identity = (
            bool(getattr(candidate, "SOPClassUID", None))
            and bool(getattr(candidate, "SOPInstanceUID", None))
            and bool(getattr(candidate, "StudyInstanceUID", None))
            and bool(getattr(candidate, "SeriesInstanceUID", None))
            and bool(getattr(candidate, "Modality", None))
        )
        if not identity:
            raise DicomProcessingError(
                "forced DICOM parse did not contain required identity fields"
            )
        return candidate


def _remove_overlay_curve_groups(ds) -> None:
    for tag in list(ds.keys()):
        element = ds[tag]

        if 0x5000 <= tag.group <= 0x50FF or 0x6000 <= tag.group <= 0x60FF:
            del ds[tag]
            continue

        if element.VR == "SQ":
            for item in element.value:
                _remove_overlay_curve_groups(item)


def _scrub_dataset_recursive(ds, uid_mapper: UIDMapper) -> None:
    for tag in list(ds.keys()):
        element = ds[tag]

        # Entire Patient group is excluded from the AI working copy.
        if tag.group == 0x0010:
            del ds[tag]
            continue

        if element.VR == "SQ":
            for item in element.value:
                _scrub_dataset_recursive(item, uid_mapper)
            continue

        if element.VR in FREE_TEXT_VRS or element.VR in DATE_TIME_AGE_VRS:
            del ds[tag]
            continue

        if element.VR == "UI":
            keyword = element.keyword or ""

            # SOP Class UIDs identify object classes, not instances.
            if keyword.endswith("SOPClassUID") or keyword in UID_CLASS_KEYWORDS:
                continue

            value = element.value
            if isinstance(value, (list, tuple, MultiValue)):
                element.value = [
                    uid_mapper.map(str(item))
                    for item in value
                    if str(item).strip()
                ]
            elif value:
                element.value = uid_mapper.map(str(value))


def _rebuild_file_meta(ds, original_transfer_syntax) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    file_meta.TransferSyntaxUID = (
        original_transfer_syntax or ExplicitVRLittleEndian
    )
    file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

    ds.file_meta = file_meta
    ds.preamble = b"\0" * 128


def extract_safe_series_metadata(ds, accumulator: SeriesAccumulator) -> None:
    accumulator.instance_count += 1

    frames = _safe_int(getattr(ds, "NumberOfFrames", None)) or 1
    accumulator.slice_count += max(frames, 1)
    accumulator.multiframe_present = (
        accumulator.multiframe_present or frames > 1
    )

    accumulator.series_description_tokens.update(
        safe_sequence_tokens(getattr(ds, "SeriesDescription", None))
    )
    accumulator.protocol_name_tokens.update(
        safe_sequence_tokens(getattr(ds, "ProtocolName", None))
    )
    accumulator.scanning_sequence.update(
        _string_values(getattr(ds, "ScanningSequence", None))
    )
    accumulator.sequence_variant.update(
        _string_values(getattr(ds, "SequenceVariant", None))
    )
    accumulator.scan_options.update(
        _string_values(getattr(ds, "ScanOptions", None))
    )
    accumulator.image_type.update(
        _string_values(getattr(ds, "ImageType", None))
    )
    accumulator.mr_acquisition_type.update(
        _string_values(getattr(ds, "MRAcquisitionType", None))
    )

    numeric_targets = (
        ("RepetitionTime", accumulator.repetition_times_ms),
        ("EchoTime", accumulator.echo_times_ms),
        ("InversionTime", accumulator.inversion_times_ms),
        ("FlipAngle", accumulator.flip_angles_deg),
        ("MagneticFieldStrength", accumulator.magnetic_field_strength_t),
    )
    for keyword, target in numeric_targets:
        value = _safe_float(getattr(ds, keyword, None))
        if value is not None:
            target.add(value)

    accumulator.contrast_present = (
        accumulator.contrast_present
        or bool(getattr(ds, "ContrastBolusAgent", None))
        or bool(getattr(ds, "ContrastBolusVolume", None))
    )

    accumulator.body_part_hints.add(
        safe_body_part_hint(getattr(ds, "BodyPartExamined", None))
    )

    rows = _safe_int(getattr(ds, "Rows", None))
    columns = _safe_int(getattr(ds, "Columns", None))
    if rows is not None and columns is not None:
        accumulator.matrix_sizes.add((rows, columns))

    spacing = _float_tuple(getattr(ds, "PixelSpacing", None))
    if spacing:
        accumulator.pixel_spacings.add(spacing)

    orientation = _float_tuple(
        getattr(ds, "ImageOrientationPatient", None)
    )
    if orientation:
        accumulator.image_orientations.add(orientation)

    thickness = _safe_float(getattr(ds, "SliceThickness", None))
    if thickness is not None:
        accumulator.slice_thicknesses.add(thickness)

    spacing_between = _safe_float(
        getattr(ds, "SpacingBetweenSlices", None)
    )
    if spacing_between is not None:
        accumulator.spacing_between_slices.add(spacing_between)

    if getattr(ds, "ImagePositionPatient", None) is not None:
        accumulator.position_available_count += 1


def _pixel_privacy_flag(ds) -> str | None:
    burned = str(
        getattr(ds, "BurnedInAnnotation", "") or ""
    ).strip().upper()
    recognizable = str(
        getattr(ds, "RecognizableVisualFeatures", "") or ""
    ).strip().upper()

    if burned == "YES":
        return "burned_in_annotation_yes"
    if recognizable == "YES":
        return "recognizable_visual_features_yes"
    return None


def deidentify_dataset(
    ds,
    uid_mapper: UIDMapper,
):
    """
    Create a metadata-reduced AI working copy.

    This is a prototype research de-identification layer and deliberately does
    not claim full DICOM PS3.15 Basic Application Level Confidentiality Profile
    conformance.
    """
    _require_pydicom()

    pixel_risk = _pixel_privacy_flag(ds)
    if pixel_risk:
        raise DicomPixelPrivacyRiskError(
            f"DICOM pixel privacy flag blocks AI working copy: {pixel_risk}"
        )

    original_transfer_syntax = getattr(
        getattr(ds, "file_meta", None),
        "TransferSyntaxUID",
        None,
    )

    # Copy so the protected original object is never mutated.
    working = ds.copy()

    # pydicom's documented helper recursively removes private data elements.
    working.remove_private_tags()
    _remove_overlay_curve_groups(working)

    # Apply UID remapping and conservative text/date/person removal recursively.
    _scrub_dataset_recursive(working, uid_mapper)

    # Original free-text sequence/protocol fields have already been converted
    # to safe tokens in PostgreSQL before scrubbing.
    _rebuild_file_meta(working, original_transfer_syntax)

    return working


def _iter_source_members(source: BinaryIO):
    source.seek(0)
    prefix = source.read(4)
    source.seek(0)

    if prefix.startswith(b"PK"):
        with zipfile.ZipFile(source, "r") as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                with archive.open(info, "r") as member:
                    yield member
    else:
        yield source


def build_deidentified_dicom_package(
    source: BinaryIO,
) -> DeidentifiedDicomPackage:
    _require_pydicom()

    uid_mapper = UIDMapper()
    series_accumulators: dict[str, SeriesAccumulator] = {}
    source_study_uids: set[str] = set()
    seen_sop_uids: set[str] = set()
    ignored_non_dicom = 0
    input_instances = 0
    output_instances = 0

    output = tempfile.SpooledTemporaryFile(
        max_size=64 * 1024 * 1024,
        mode="w+b",
    )

    try:
        with zipfile.ZipFile(
            output,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            allowZip64=True,
        ) as out_zip:
            for member in _iter_source_members(source):
                try:
                    ds = read_dicom_dataset(member)
                except Exception:
                    ignored_non_dicom += 1
                    continue

                input_instances += 1

                modality = str(
                    getattr(ds, "Modality", "") or ""
                ).strip().upper()
                if modality != "MR":
                    raise DicomModalityError(
                        f"DICOM package contains non-MR modality {modality!r}"
                    )

                source_study_uid = str(
                    getattr(ds, "StudyInstanceUID", "") or ""
                ).strip()
                source_series_uid = str(
                    getattr(ds, "SeriesInstanceUID", "") or ""
                ).strip()
                source_sop_uid = str(
                    getattr(ds, "SOPInstanceUID", "") or ""
                ).strip()

                if not source_study_uid or not source_series_uid or not source_sop_uid:
                    raise DicomGroupingError(
                        "DICOM instance missing Study/Series/SOP Instance UID"
                    )

                source_study_uids.add(source_study_uid)
                if len(source_study_uids) > 1:
                    raise DicomGroupingError(
                        "one upload contains multiple DICOM StudyInstanceUID values"
                    )

                if source_sop_uid in seen_sop_uids:
                    raise DicomGroupingError(
                        "duplicate SOPInstanceUID found in DICOM upload"
                    )
                seen_sop_uids.add(source_sop_uid)

                if source_series_uid not in series_accumulators:
                    index = len(series_accumulators) + 1
                    series_accumulators[source_series_uid] = SeriesAccumulator(
                        source_series_uid=source_series_uid,
                        deidentified_series_uid=uid_mapper.map(
                            source_series_uid
                        ),
                        series_number=_safe_int(
                            getattr(ds, "SeriesNumber", None)
                        ),
                        member_prefix=f"series_{index:03d}/",
                    )

                accumulator = series_accumulators[source_series_uid]
                extract_safe_series_metadata(ds, accumulator)

                working = deidentify_dataset(ds, uid_mapper)

                instance_index = accumulator.instance_count
                member_name = (
                    f"{accumulator.member_prefix}"
                    f"instance_{instance_index:06d}.dcm"
                )

                with tempfile.SpooledTemporaryFile(
                    max_size=16 * 1024 * 1024,
                    mode="w+b",
                ) as temp_dicom:
                    pydicom.dcmwrite(
                        temp_dicom,
                        working,
                        enforce_file_format=True,
                    )
                    temp_dicom.seek(0)
                    with out_zip.open(member_name, "w") as target:
                        shutil.copyfileobj(
                            temp_dicom,
                            target,
                            length=1024 * 1024,
                        )

                output_instances += 1

        if input_instances == 0:
            raise DicomGroupingError(
                "stored object contains no readable DICOM instances"
            )

        source_study_uid = next(iter(source_study_uids))
        deidentified_study_uid = uid_mapper.map(source_study_uid)

        output.seek(0)

        return DeidentifiedDicomPackage(
            study_uid=deidentified_study_uid,
            series_records=[
                accumulator.to_record()
                for accumulator in series_accumulators.values()
            ],
            output_stream=output,
            input_instance_count=input_instances,
            output_instance_count=output_instances,
            ignored_non_dicom_entries=ignored_non_dicom,
            private_tags_removed=True,
            free_text_removed=True,
            uid_remapping_applied=True,
            pixel_data_modified=False,
            pixel_privacy_status=(
                "metadata_only_deidentification_pixel_content_not_cleaned"
            ),
        )
    except Exception:
        output.close()
        raise
