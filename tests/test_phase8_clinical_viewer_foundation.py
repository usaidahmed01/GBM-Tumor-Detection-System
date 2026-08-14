from __future__ import annotations

import uuid
from io import BytesIO
from types import SimpleNamespace

import pytest

from gbm_ai.api.main import create_app
from gbm_ai.api.models.analysis import SourceFormat
from gbm_ai.api.services.clinical_viewer import (
    CLINICAL_VIEWER_BACKEND_VERSION,
    ClinicalViewerServiceError,
    ViewerAsset,
    build_viewer_assets,
    open_verified_viewer_asset,
)
from gbm_ai.api.storage.local import LocalObjectStore


def _study():
    channels = []
    for sequence in ("T1C", "T1", "T2", "FLAIR"):
        channels.append(
            {
                "sequence": sequence,
                "storage_key": f"studies/s/derived/model_geometry/{sequence.lower()}.nii.gz",
                "checksum_sha256": (sequence[0].lower() if sequence != "FLAIR" else "f") * 64,
                "size_bytes": 100 + len(sequence),
            }
        )
    return SimpleNamespace(
        id=uuid.uuid4(),
        source_format=SourceFormat.NIFTI,
        segmentation_preparation_summary={
            "model_geometry": {"status": "ready", "channels": channels}
        },
    )


def _segmentation(review_status="unreviewed"):
    return SimpleNamespace(
        id=uuid.uuid4(),
        wt_storage_key="studies/s/derived/wt/wt.nii.gz",
        wt_checksum_sha256="a" * 64,
        wt_size_bytes=101,
        tc_storage_key="studies/s/derived/tc/tc.nii.gz",
        tc_checksum_sha256="b" * 64,
        tc_size_bytes=102,
        et_storage_key="studies/s/derived/et/et.nii.gz",
        et_checksum_sha256="c" * 64,
        et_size_bytes=103,
        labelmap_storage_key="studies/s/derived/label/label.nii.gz",
        labelmap_checksum_sha256="d" * 64,
        labelmap_size_bytes=104,
        review_status=SimpleNamespace(value=review_status),
        clinician_modified=False,
    )


def test_phase8_viewer_contract_routes_are_registered_without_frontend_claim():
    assert CLINICAL_VIEWER_BACKEND_VERSION == "phase8_step1_clinical_viewer_backend_v1"
    app = create_app()
    paths = {route.path for route in app.routes}
    assert "/api/v1/studies/{study_uuid}/viewer/manifest" in paths
    assert "/api/v1/studies/{study_uuid}/viewer/assets/{asset_alias}" in paths


def test_viewer_asset_catalog_contains_four_mri_volumes_and_four_overlays_without_raw_keys_in_public_payload():
    study = _study()
    segmentation = _segmentation()
    assets = build_viewer_assets(
        study,
        segmentation,
        localization=None,
        quantification=None,
    )
    assert set(assets) == {
        "mri_t1c",
        "mri_t1",
        "mri_t2",
        "mri_flair",
        "mask_wt",
        "mask_tc",
        "mask_et",
        "mask_labelmap",
    }
    public = assets["mask_wt"].public_payload(
        study_uuid=study.id,
        api_prefix="/api/v1",
    )
    assert "storage_key" not in public
    assert public["download_url"].endswith("/viewer/assets/mask_wt")
    assert public["coordinate_space"] == "patient_model_space_ras"


def test_rejected_segmentation_is_still_viewable_for_human_review():
    assets = build_viewer_assets(
        _study(),
        _segmentation("rejected"),
        localization=None,
        quantification=None,
    )
    assert "mask_wt" in assets
    assert "mri_t1c" in assets


def test_viewer_requires_exact_four_model_space_channels():
    study = _study()
    study.segmentation_preparation_summary["model_geometry"]["channels"] = study.segmentation_preparation_summary["model_geometry"]["channels"][:-1]
    with pytest.raises(ClinicalViewerServiceError, match="exactly T1C"):
        build_viewer_assets(
            study,
            _segmentation(),
            localization=None,
            quantification=None,
        )


def test_verified_asset_stream_blocks_checksum_mismatch(tmp_path):
    storage = LocalObjectStore(tmp_path / "storage", max_object_bytes=1024 * 1024)
    key = "studies/00000000-0000-0000-0000-000000000001/derived/viewer/test.bin"
    stored = storage.put_stream(key, BytesIO(b"viewer-fixture"))
    asset = ViewerAsset(
        alias="mri_t1c",
        storage_key=stored.storage_key,
        checksum_sha256="0" * 64,
        size_bytes=stored.size_bytes,
        kind="mri_volume",
        format="nifti_gzip",
        media_type="application/octet-stream",
        coordinate_space="patient_model_space_ras",
        filename="t1c_model_space.nii.gz",
        sequence="T1C",
    )
    with pytest.raises(ClinicalViewerServiceError, match="checksum"):
        open_verified_viewer_asset(storage, asset)


def test_verified_asset_stream_opens_only_after_checksum_pass(tmp_path):
    storage = LocalObjectStore(tmp_path / "storage", max_object_bytes=1024 * 1024)
    key = "studies/00000000-0000-0000-0000-000000000001/derived/viewer/test.bin"
    stored = storage.put_stream(key, BytesIO(b"viewer-fixture"))
    asset = ViewerAsset(
        alias="mri_t1c",
        storage_key=stored.storage_key,
        checksum_sha256=stored.sha256,
        size_bytes=stored.size_bytes,
        kind="mri_volume",
        format="nifti_gzip",
        media_type="application/octet-stream",
        coordinate_space="patient_model_space_ras",
        filename="t1c_model_space.nii.gz",
        sequence="T1C",
    )
    opened = open_verified_viewer_asset(storage, asset)
    try:
        assert opened.stream.read() == b"viewer-fixture"
    finally:
        opened.stream.close()
