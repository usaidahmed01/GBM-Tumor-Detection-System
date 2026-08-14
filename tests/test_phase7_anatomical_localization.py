from __future__ import annotations

import json

import numpy as np
import pytest

from gbm_ai.api.localization import (
    ANATOMICAL_LOCALIZATION_VERSION,
    ATLAS_LICENSE,
    ATLAS_VERSION,
    REGISTRATION_SUPPORT_DICE_MIN,
    AnatomicalLocalizationError,
    centroid_world_mm,
    compute_region_overlaps,
    dice_coefficient,
    hemisphere_from_standard_mask,
    localization_source_fingerprint,
    merge_region_overlaps,
)
from gbm_ai.api.localization_assets import (
    LOCALIZATION_ASSET_MANIFEST_VERSION,
    LocalizationAssetError,
    load_and_verify_localization_assets,
)


def test_phase7_localization_contract_is_frozen_and_nonclinical():
    assert ANATOMICAL_LOCALIZATION_VERSION == "phase7_step2_anatomical_localization_v1"
    assert "Harvard" in ATLAS_VERSION or ATLAS_VERSION.startswith("FSL5")
    assert ATLAS_LICENSE == "CC BY-SA 4.0"
    assert REGISTRATION_SUPPORT_DICE_MIN == pytest.approx(0.40)


def test_atlas_overlap_primary_secondary_and_fraction_are_deterministic():
    wt = np.zeros((4, 4, 4), dtype=np.uint8)
    wt[0:2, 0:2, 0:2] = 1  # 8 voxels
    cortical = np.zeros_like(wt, dtype=np.int16)
    sub = np.zeros_like(wt, dtype=np.int16)
    cortical[0:2, 0:2, 0] = 1  # 4/8
    cortical[0:2, 0:2, 1] = 2  # 4/8
    sub[0, 0, 0] = 1           # 1/8

    cort = compute_region_overlaps(
        wt,
        cortical,
        ["Background", "Frontal Pole", "Parietal Region"],
        atlas_name="cortical",
    )
    subc = compute_region_overlaps(
        wt,
        sub,
        ["Background", "Left Thalamus"],
        atlas_name="subcortical",
    )
    merged = merge_region_overlaps(cort, subc)
    assert merged[0].overlap_voxels == 4
    assert merged[0].fraction_of_wt == pytest.approx(0.5)
    assert {item.label for item in merged[:2]} == {"Frontal Pole", "Parietal Region"}
    assert any(item.label == "Left Thalamus" and item.overlap_voxels == 1 for item in merged)


def test_hemisphere_and_centroid_use_standard_space_world_coordinates():
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[0:2, 2:4, 2:4] = 1
    affine = np.eye(4)
    affine[0, 3] = -10.0
    assert hemisphere_from_standard_mask(mask, affine) == "left"
    centroid = centroid_world_mm(mask, affine)
    assert centroid[0] < 0

    right_affine = np.eye(4)
    right_affine[0, 3] = 10.0
    assert hemisphere_from_standard_mask(mask, right_affine) == "right"


def test_bilateral_mask_is_not_forced_into_one_hemisphere():
    mask = np.zeros((5, 3, 3), dtype=np.uint8)
    mask[0, 1, 1] = 1
    mask[4, 1, 1] = 1
    affine = np.eye(4)
    affine[0, 3] = -2.0
    assert hemisphere_from_standard_mask(mask, affine) in {"bilateral", "midline"}


def test_registration_support_dice_is_bounded_and_shape_checked():
    a = np.zeros((3, 3, 3), dtype=np.uint8)
    b = np.zeros_like(a)
    a[0:2, 0:2, 0:2] = 1
    b[0:2, 0:2, 0:2] = 1
    assert dice_coefficient(a, b) == pytest.approx(1.0)
    b[1:3, 1:3, 1:3] = 1
    score = dice_coefficient(a, b)
    assert 0.0 <= score <= 1.0
    with pytest.raises(AnatomicalLocalizationError, match="shape"):
        dice_coefficient(a, np.zeros((2, 2, 2), dtype=np.uint8))


def test_localization_fingerprint_changes_when_atlas_or_source_changes():
    base = dict(
        segmentation_uuid="00000000-0000-0000-0000-000000000001",
        quantification_uuid="00000000-0000-0000-0000-000000000002",
        wt_checksum_sha256="a" * 64,
        t1_checksum_sha256="b" * 64,
        atlas_manifest_checksum_sha256="c" * 64,
    )
    first = localization_source_fingerprint(**base)
    second = localization_source_fingerprint(**{**base, "atlas_manifest_checksum_sha256": "d" * 64})
    assert len(first) == 64
    assert first != second


def test_asset_manifest_checksum_and_file_hashes_are_enforced(tmp_path):
    import hashlib

    root = tmp_path / "atlas"
    root.mkdir()
    files = {}
    for key, name in {
        "template": "MNI152NLin6Asym_res-02_desc-brain_T1w.nii.gz",
        "brain_mask": "MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz",
        "cortical": "harvard_oxford_cortical_lateralized_2mm.nii.gz",
        "subcortical": "harvard_oxford_subcortical_lateralized_2mm.nii.gz",
        "labels": "harvard_oxford_labels.json",
    }.items():
        path = root / name
        path.write_bytes(f"fixture-{key}".encode())
        files[key] = {"name": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    core = {
        "manifest_version": LOCALIZATION_ASSET_MANIFEST_VERSION,
        "standard_space": "MNI152NLin6Asym",
        "template_name": "TemplateFlow_MNI152NLin6Asym_res-02_desc-brain_T1w",
        "template_source": "TemplateFlow",
        "templateflow_identifier": "MNI152NLin6Asym",
        "templateflow_metadata_license": "See LICENSE file",
        "atlas_name": "Harvard-Oxford cortical + subcortical structural atlases",
        "atlas_version": ATLAS_VERSION,
        "atlas_license": ATLAS_LICENSE,
        "cortical_atlas_id": "cort-maxprob-thr25-2mm",
        "subcortical_atlas_id": "sub-maxprob-thr25-2mm",
        "nilearn_atlas_template": "MNI152NLin6Asym",
        "symmetric_split": True,
        "files": files,
        "clinical_validation_claimed": False,
    }
    checksum = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {**core, "manifest_checksum_sha256": checksum}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    loaded, _ = load_and_verify_localization_assets(root)
    assert loaded["manifest_checksum_sha256"] == checksum

    (root / "harvard_oxford_labels.json").write_bytes(b"tampered")
    with pytest.raises(LocalizationAssetError, match="checksum"):
        load_and_verify_localization_assets(root)
