from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from gbm_ai.api.localization import (
    ATLAS_CORTICAL_ID,
    ATLAS_LICENSE,
    ATLAS_NAME,
    ATLAS_SUBCORTICAL_ID,
    ATLAS_SYMMETRIC_SPLIT,
    ATLAS_VERSION,
    STANDARD_SPACE,
    STANDARD_TEMPLATE_NAME,
)


LOCALIZATION_ASSET_MANIFEST_VERSION = "phase7_step2_atlas_assets_v1"


class LocalizationAssetError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def localization_asset_paths(root: Path) -> dict[str, Path]:
    root = Path(root).expanduser().resolve()
    return {
        "root": root,
        "template": root / "MNI152NLin6Asym_res-02_desc-brain_T1w.nii.gz",
        "brain_mask": root / "MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz",
        "cortical": root / "harvard_oxford_cortical_lateralized_2mm.nii.gz",
        "subcortical": root / "harvard_oxford_subcortical_lateralized_2mm.nii.gz",
        "labels": root / "harvard_oxford_labels.json",
        "manifest": root / "manifest.json",
    }


def _save_img(img, path: Path) -> None:
    import nibabel as nib

    nib.save(img, str(path))


def _one_path(value, *, label: str) -> Path:
    if value is None:
        raise LocalizationAssetError(
            "LOCALIZATION_TEMPLATEFLOW_ASSET_MISSING",
            f"TemplateFlow did not provide the required {label} asset",
        )
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise LocalizationAssetError(
                "LOCALIZATION_TEMPLATEFLOW_ASSET_AMBIGUOUS",
                f"TemplateFlow returned more than one candidate for {label}",
            )
        value = value[0]
    path = Path(value)
    if not path.is_file():
        raise LocalizationAssetError(
            "LOCALIZATION_TEMPLATEFLOW_ASSET_MISSING",
            f"TemplateFlow {label} asset is unavailable after retrieval",
        )
    return path


def prepare_localization_assets(root: Path) -> dict:
    """Fetch and freeze one template/atlas combination for Phase 7.

    The Harvard-Oxford atlas returned by current Nilearn identifies its
    standardized space as MNI152NLin6Asym. The fixed T1 template and brain mask
    are therefore retrieved from the matching TemplateFlow space rather than
    mixing the atlas with Nilearn's different ICBM152-2009a built-in template.

    Downloaded assets are not committed to Git. This function writes canonical
    local copies plus SHA-256 checksums that are revalidated before every
    localization run.
    """
    try:
        import nibabel as nib
        from nibabel.processing import resample_from_to
        from nilearn.datasets import fetch_atlas_harvard_oxford
        import templateflow.api as tflow
    except ImportError as exc:
        raise LocalizationAssetError(
            "LOCALIZATION_DEPENDENCY_MISSING",
            "NiBabel, Nilearn and TemplateFlow are required to prepare localization assets",
        ) from exc

    paths = localization_asset_paths(root)
    paths["root"].mkdir(parents=True, exist_ok=True)
    nilearn_cache = paths["root"] / "nilearn_cache"

    cortical = fetch_atlas_harvard_oxford(
        ATLAS_CORTICAL_ID,
        data_dir=str(nilearn_cache),
        symmetric_split=ATLAS_SYMMETRIC_SPLIT,
        verbose=0,
    )
    subcortical = fetch_atlas_harvard_oxford(
        ATLAS_SUBCORTICAL_ID,
        data_dir=str(nilearn_cache),
        symmetric_split=ATLAS_SYMMETRIC_SPLIT,
        verbose=0,
    )

    cortical_space = str(getattr(cortical, "template", "") or "")
    subcortical_space = str(getattr(subcortical, "template", "") or "")
    if cortical_space != STANDARD_SPACE or subcortical_space != STANDARD_SPACE:
        raise LocalizationAssetError(
            "LOCALIZATION_ATLAS_STANDARD_SPACE_MISMATCH",
            "Harvard-Oxford atlas metadata does not match the frozen standard-space contract",
        )

    try:
        template_source = _one_path(
            tflow.get(
                STANDARD_SPACE,
                resolution=2,
                desc="brain",
                suffix="T1w",
            ),
            label="MNI152NLin6Asym brain T1w template",
        )
        mask_source = _one_path(
            tflow.get(
                STANDARD_SPACE,
                resolution=2,
                desc="brain",
                suffix="mask",
            ),
            label="MNI152NLin6Asym brain mask",
        )
        template_metadata = dict(tflow.get_metadata(STANDARD_SPACE) or {})
    except LocalizationAssetError:
        raise
    except Exception as exc:
        raise LocalizationAssetError(
            "LOCALIZATION_TEMPLATEFLOW_DOWNLOAD_FAILED",
            "matching MNI152NLin6Asym template assets could not be retrieved from TemplateFlow",
        ) from exc

    cortical_img = (
        nib.load(str(cortical.maps))
        if isinstance(cortical.maps, (str, Path))
        else cortical.maps
    )
    subcortical_img = (
        nib.load(str(subcortical.maps))
        if isinstance(subcortical.maps, (str, Path))
        else subcortical.maps
    )
    template_img = nib.load(str(template_source))
    brain_mask_img = nib.load(str(mask_source))

    # Use the cortical atlas geometry as the exact discrete-label grid. The
    # fixed template, brain mask and subcortical atlas are all resources in the
    # same declared MNI152NLin6Asym space and are resampled only to that grid.
    target = (cortical_img.shape, cortical_img.affine)
    template_on_atlas = resample_from_to(template_img, target, order=1)
    mask_on_atlas = resample_from_to(brain_mask_img, target, order=0)
    subcortical_on_atlas = resample_from_to(subcortical_img, target, order=0)

    cortical_data = np.rint(np.asanyarray(cortical_img.dataobj)).astype(np.int16)
    subcortical_data = np.rint(np.asanyarray(subcortical_on_atlas.dataobj)).astype(np.int16)
    mask_data = (np.asanyarray(mask_on_atlas.dataobj) > 0).astype(np.uint8)

    if cortical_data.ndim != 3 or subcortical_data.ndim != 3 or mask_data.ndim != 3:
        raise LocalizationAssetError(
            "LOCALIZATION_ASSET_GEOMETRY_INVALID",
            "frozen localization template/atlas assets must be 3D",
        )

    _save_img(template_on_atlas, paths["template"])
    _save_img(nib.Nifti1Image(mask_data, cortical_img.affine), paths["brain_mask"])
    _save_img(nib.Nifti1Image(cortical_data, cortical_img.affine), paths["cortical"])
    _save_img(nib.Nifti1Image(subcortical_data, cortical_img.affine), paths["subcortical"])

    labels_payload = {
        "cortical": [str(value) for value in cortical.labels],
        "subcortical": [str(value) for value in subcortical.labels],
    }
    paths["labels"].write_text(
        json.dumps(labels_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    files = {
        key: {
            "name": path.name,
            "sha256": _sha256(path),
        }
        for key, path in paths.items()
        if key in {"template", "brain_mask", "cortical", "subcortical", "labels"}
    }
    manifest_core = {
        "manifest_version": LOCALIZATION_ASSET_MANIFEST_VERSION,
        "standard_space": STANDARD_SPACE,
        "template_name": STANDARD_TEMPLATE_NAME,
        "template_source": "TemplateFlow",
        "templateflow_identifier": STANDARD_SPACE,
        "templateflow_metadata_license": str(template_metadata.get("License") or "unspecified"),
        "atlas_name": ATLAS_NAME,
        "atlas_version": ATLAS_VERSION,
        "atlas_license": ATLAS_LICENSE,
        "cortical_atlas_id": ATLAS_CORTICAL_ID,
        "subcortical_atlas_id": ATLAS_SUBCORTICAL_ID,
        "nilearn_atlas_template": cortical_space,
        "symmetric_split": ATLAS_SYMMETRIC_SPLIT,
        "files": files,
        "clinician_verification_required": True,
        "clinical_validation_claimed": False,
    }
    manifest_core["manifest_checksum_sha256"] = _canonical_json_sha256(manifest_core)
    paths["manifest"].write_text(
        json.dumps(manifest_core, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Nilearn's downloader cache is only an implementation detail after the
    # canonical atlas assets have been frozen into this ignored local folder.
    shutil.rmtree(nilearn_cache, ignore_errors=True)
    return manifest_core


def load_and_verify_localization_assets(root: Path) -> tuple[dict, dict[str, Path]]:
    paths = localization_asset_paths(root)
    if not paths["manifest"].is_file():
        raise LocalizationAssetError(
            "LOCALIZATION_ASSETS_NOT_PREPARED",
            "localization assets are missing; run the verification command with --download-assets first",
        )
    try:
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    except Exception as exc:
        raise LocalizationAssetError(
            "LOCALIZATION_ASSET_MANIFEST_INVALID",
            "localization asset manifest could not be read",
        ) from exc

    manifest_checksum = str(manifest.get("manifest_checksum_sha256") or "")
    core = dict(manifest)
    core.pop("manifest_checksum_sha256", None)
    if len(manifest_checksum) != 64 or _canonical_json_sha256(core) != manifest_checksum:
        raise LocalizationAssetError(
            "LOCALIZATION_ASSET_MANIFEST_CHECKSUM_MISMATCH",
            "localization asset manifest failed checksum validation",
        )
    if (
        manifest.get("manifest_version") != LOCALIZATION_ASSET_MANIFEST_VERSION
        or manifest.get("standard_space") != STANDARD_SPACE
        or manifest.get("template_name") != STANDARD_TEMPLATE_NAME
        or manifest.get("templateflow_identifier") != STANDARD_SPACE
        or manifest.get("nilearn_atlas_template") != STANDARD_SPACE
        or manifest.get("atlas_name") != ATLAS_NAME
        or manifest.get("atlas_version") != ATLAS_VERSION
        or manifest.get("atlas_license") != ATLAS_LICENSE
    ):
        raise LocalizationAssetError(
            "LOCALIZATION_ASSET_CONTRACT_MISMATCH",
            "installed localization assets do not match the frozen Phase 7 contract",
        )

    for key in ("template", "brain_mask", "cortical", "subcortical", "labels"):
        path = paths[key]
        expected = str((manifest.get("files") or {}).get(key, {}).get("sha256") or "")
        if not path.is_file() or len(expected) != 64 or _sha256(path) != expected:
            raise LocalizationAssetError(
                "LOCALIZATION_ASSET_FILE_CHECKSUM_MISMATCH",
                f"localization asset {key} is missing or failed checksum validation",
            )
    return manifest, paths
