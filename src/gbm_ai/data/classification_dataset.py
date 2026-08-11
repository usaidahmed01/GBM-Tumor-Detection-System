from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Callable, Literal

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from gbm_ai.training.reproducibility import make_generator, seed_worker

DatasetSplit = Literal["train", "validation", "test"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen_release(
    project_root: Path,
    release_name: str = "classification_v1.0",
) -> tuple[Path, dict]:
    release_dir = project_root / "data" / "releases" / release_name
    manifest = release_dir / "classification_split_manifest.csv"
    metadata_path = release_dir / "dataset_release.json"

    if not manifest.exists() or not metadata_path.exists():
        raise RuntimeError(
            f"Frozen release '{release_name}' is incomplete. "
            "Run Phase 1 final dataset freeze first."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_hash = metadata.get("frozen_manifest_sha256")
    actual_hash = sha256_file(manifest)

    if expected_hash and actual_hash != expected_hash:
        raise RuntimeError(
            "Frozen classification manifest hash mismatch. "
            "Do not train on a modified Phase 1 release."
        )

    return manifest, metadata


def build_transform(training: bool) -> Callable:
    # Images were already standardized to 384x384 in Phase 1.
    # ImageNet normalization matches the pretrained EfficientNetV2-S weights.
    if training:
        return transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=5,
                    translate=(0.02, 0.02),
                    scale=(0.98, 1.02),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                ),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class GBMClassificationDataset(Dataset):
    """Dataset backed only by the frozen Phase 1 manifest."""

    def __init__(
        self,
        project_root: str | Path,
        split: DatasetSplit,
        fold: int = 0,
        release_name: str = "classification_v1.0",
        transform: Callable | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.split = split
        self.fold = int(fold)

        manifest_path, self.release_metadata = verify_frozen_release(
            self.project_root, release_name
        )

        with manifest_path.open("r", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

        selected: list[dict[str, str]] = []
        for row in rows:
            holdout = row["holdout_split"].strip().lower()
            cv_fold = row.get("cv_fold", "").strip()

            if split == "test":
                if holdout == "test":
                    selected.append(row)
                continue

            if holdout != "development":
                continue

            if split == "validation" and cv_fold == str(self.fold):
                selected.append(row)
            elif split == "train" and cv_fold != str(self.fold):
                selected.append(row)

        if not selected:
            raise RuntimeError(
                f"No samples found for split={split!r}, fold={fold}. "
                "Check the frozen manifest."
            )

        self.rows = selected
        self.transform = transform or build_transform(training=split == "train")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        image_path = self.project_root / row["standardized_relative_path"]

        if not image_path.exists():
            raise FileNotFoundError(f"Model input image missing: {image_path}")

        with Image.open(image_path) as im:
            image = im.convert("RGB")
            image_tensor = self.transform(image)

        return {
            "image": image_tensor,
            "target": torch.tensor(float(row["label"]), dtype=torch.float32),
            "sample_id": row["sample_id"],
            "class_name": row["class_name"],
        }


def create_dataloader(
    project_root: str | Path,
    split: DatasetSplit,
    fold: int = 0,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 42,
    release_name: str = "classification_v1.0",
) -> DataLoader:
    dataset = GBMClassificationDataset(
        project_root=project_root,
        split=split,
        fold=fold,
        release_name=release_name,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=make_generator(seed),
        drop_last=False,
    )
