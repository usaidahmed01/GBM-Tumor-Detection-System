from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from scipy.fft import dctn
import numpy as np
from PIL import Image, ImageOps, ImageDraw

LOGGER = logging.getLogger("gbm.dataset.near_duplicates")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    class_name: str
    label: int
    original_name: str
    clean_relative_path: str


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    left_sample_id: str
    right_sample_id: str
    left_class: str
    right_class: str
    phash_distance: int
    dhash_distance: int
    correlation: float
    rmse: float
    same_class: bool
    review_image: str


def read_manifest(path: Path) -> list[Sample]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    rows: list[Sample] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                Sample(
                    sample_id=row["sample_id"],
                    class_name=row["class_name"],
                    label=int(row["label"]),
                    original_name=row["original_name"],
                    clean_relative_path=row["clean_relative_path"],
                )
            )
    if not rows:
        raise RuntimeError("Classification manifest is empty")
    return rows


def letterbox_grayscale(path: Path, size: int) -> Image.Image:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("L")
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (size, size), 0)
        x = (size - img.width) // 2
        y = (size - img.height) // 2
        canvas.paste(img, (x, y))
        return canvas


def perceptual_hash(img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    size = hash_size * highfreq_factor
    normalized = ImageOps.fit(img.convert("L"), (size, size), method=Image.Resampling.LANCZOS)
    pixels = np.asarray(normalized, dtype=np.float32)
    spectrum = dctn(pixels, type=2, norm="ortho")
    low = spectrum[:hash_size, :hash_size]
    median = float(np.median(low[1:, :]))
    return (low > median).reshape(-1)


def difference_hash(img: Image.Image, hash_size: int = 8) -> np.ndarray:
    normalized = ImageOps.fit(
        img.convert("L"), (hash_size + 1, hash_size), method=Image.Resampling.LANCZOS
    )
    pixels = np.asarray(normalized, dtype=np.int16)
    return (pixels[:, 1:] > pixels[:, :-1]).reshape(-1)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def normalized_array(path: Path, size: int = 128) -> np.ndarray:
    img = letterbox_grayscale(path, size)
    return np.asarray(img, dtype=np.float32) / 255.0


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    av = a.reshape(-1).astype(np.float64)
    bv = b.reshape(-1).astype(np.float64)
    av -= av.mean()
    bv -= bv.mean()
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    if denom == 0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(av, bv) / denom)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a - b))))


def build_review_image(
    left_path: Path,
    right_path: Path,
    left_title: str,
    right_title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel_w, panel_h = 420, 460
    image_box = 390

    def panel(path: Path, title: str) -> Image.Image:
        with Image.open(path) as img:
            rgb = ImageOps.exif_transpose(img).convert("RGB")
            rgb.thumbnail((image_box, image_box), Image.Resampling.LANCZOS)
            p = Image.new("RGB", (panel_w, panel_h), "white")
            p.paste(rgb, ((panel_w - rgb.width) // 2, 10))
            ImageDraw.Draw(p).text((12, 420), title[:64], fill="black")
            return p

    out = Image.new("RGB", (panel_w * 2, panel_h), "white")
    out.paste(panel(left_path, left_title), (0, 0))
    out.paste(panel(right_path, right_title), (panel_w, 0))
    out.save(output_path, quality=92)


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            nxt = self.parent[item]
            self.parent[item] = root
            item = nxt
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit_near_duplicates(
    project_root: Path,
    *,
    phash_max: int = 4,
    dhash_max: int = 4,
    correlation_min: float = 0.98,
    rmse_max: float = 0.08,
    fail_on_cross_class: bool = True,
) -> dict:
    project_root = project_root.resolve()
    manifest_path = project_root / "data" / "manifests" / "classification_manifest.csv"
    reports_root = project_root / "data" / "reports"
    manifests_root = project_root / "data" / "manifests"
    review_root = reports_root / "near_duplicate_review"
    review_root.mkdir(parents=True, exist_ok=True)

    # Clear old review images to make repeated runs deterministic.
    for old in review_root.glob("candidate_*.jpg"):
        old.unlink()

    samples = read_manifest(manifest_path)
    paths: dict[str, Path] = {}
    phashes: dict[str, np.ndarray] = {}
    dhashes: dict[str, np.ndarray] = {}
    arrays: dict[str, np.ndarray] = {}

    for sample in samples:
        path = project_root / sample.clean_relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Clean image missing for {sample.sample_id}: {path}")
        paths[sample.sample_id] = path
        with Image.open(path) as img:
            gray = ImageOps.exif_transpose(img).convert("L")
            phashes[sample.sample_id] = perceptual_hash(gray)
            dhashes[sample.sample_id] = difference_hash(gray)
        arrays[sample.sample_id] = normalized_array(path)

    candidates: list[Candidate] = []
    uf = UnionFind([s.sample_id for s in samples])

    candidate_number = 0
    for i, left in enumerate(samples):
        for right in samples[i + 1 :]:
            pd = hamming_distance(phashes[left.sample_id], phashes[right.sample_id])
            dd = hamming_distance(dhashes[left.sample_id], dhashes[right.sample_id])
            if pd > phash_max or dd > dhash_max:
                continue

            corr = correlation(arrays[left.sample_id], arrays[right.sample_id])
            error = rmse(arrays[left.sample_id], arrays[right.sample_id])
            if corr < correlation_min or error > rmse_max:
                continue

            candidate_number += 1
            candidate_id = f"nd_{candidate_number:04d}"
            review_rel = Path("data") / "reports" / "near_duplicate_review" / f"candidate_{candidate_number:04d}.jpg"
            build_review_image(
                paths[left.sample_id],
                paths[right.sample_id],
                f"{left.class_name} | {left.original_name} | {left.sample_id}",
                f"{right.class_name} | {right.original_name} | {right.sample_id}",
                project_root / review_rel,
            )
            candidates.append(
                Candidate(
                    candidate_id=candidate_id,
                    left_sample_id=left.sample_id,
                    right_sample_id=right.sample_id,
                    left_class=left.class_name,
                    right_class=right.class_name,
                    phash_distance=int(pd),
                    dhash_distance=int(dd),
                    correlation=round(corr, 6),
                    rmse=round(error, 6),
                    same_class=left.class_name == right.class_name,
                    review_image=review_rel.as_posix(),
                )
            )
            uf.union(left.sample_id, right.sample_id)

    candidate_rows = [asdict(c) for c in candidates]
    write_csv(
        manifests_root / "near_duplicate_candidates.csv",
        candidate_rows,
        list(Candidate.__dataclass_fields__.keys()),
    )

    grouped: dict[str, list[Sample]] = {}
    for sample in samples:
        root = uf.find(sample.sample_id)
        grouped.setdefault(root, []).append(sample)

    duplicate_groups = [group for group in grouped.values() if len(group) > 1]
    duplicate_groups.sort(key=lambda group: min(s.sample_id for s in group))

    group_id_by_sample: dict[str, str] = {}
    group_rows: list[dict] = []
    for index, group in enumerate(duplicate_groups, start=1):
        group_id = f"ndg_{index:03d}"
        for sample in sorted(group, key=lambda s: s.sample_id):
            group_id_by_sample[sample.sample_id] = group_id
            group_rows.append(
                {
                    "near_duplicate_group_id": group_id,
                    "sample_id": sample.sample_id,
                    "class_name": sample.class_name,
                    "label": sample.label,
                    "original_name": sample.original_name,
                    "clean_relative_path": sample.clean_relative_path,
                }
            )

    write_csv(
        manifests_root / "near_duplicate_groups.csv",
        group_rows,
        [
            "near_duplicate_group_id",
            "sample_id",
            "class_name",
            "label",
            "original_name",
            "clean_relative_path",
        ],
    )

    # Create an enriched manifest for the later leakage-safe split step.
    enriched_rows: list[dict] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["near_duplicate_group_id"] = group_id_by_sample.get(row["sample_id"], "")
            enriched_rows.append(row)
    fieldnames = list(enriched_rows[0].keys()) if enriched_rows else ["near_duplicate_group_id"]
    write_csv(
        manifests_root / "classification_manifest_grouped.csv",
        enriched_rows,
        fieldnames,
    )

    cross_class_candidates = [c for c in candidates if not c.same_class]
    report = {
        "input_manifest": manifest_path.relative_to(project_root).as_posix(),
        "sample_count": len(samples),
        "method": {
            "phash_max_hamming": phash_max,
            "dhash_max_hamming": dhash_max,
            "correlation_min": correlation_min,
            "rmse_max": rmse_max,
            "comparison_image_size": "128x128 grayscale letterbox",
            "policy": "candidate grouping only; no automatic deletion",
        },
        "near_duplicate_candidate_pairs": len(candidates),
        "near_duplicate_groups": len(duplicate_groups),
        "samples_in_near_duplicate_groups": sum(len(g) for g in duplicate_groups),
        "cross_class_candidate_pairs": len(cross_class_candidates),
        "outputs": {
            "candidates": "data/manifests/near_duplicate_candidates.csv",
            "groups": "data/manifests/near_duplicate_groups.csv",
            "grouped_manifest": "data/manifests/classification_manifest_grouped.csv",
            "review_images": "data/reports/near_duplicate_review/",
        },
    }
    report_path = reports_root / "phase1_step2_near_duplicate_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if cross_class_candidates and fail_on_cross_class:
        raise RuntimeError(
            "Cross-class near-duplicate candidates detected. Manual label review is required before splitting. "
            f"See {manifests_root / 'near_duplicate_candidates.csv'}"
        )

    LOGGER.info("Near-duplicate audit complete: %s", json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect and group perceptual near-duplicate MRI images before dataset splitting."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--phash-max", type=int, default=4)
    parser.add_argument("--dhash-max", type=int, default=4)
    parser.add_argument("--correlation-min", type=float, default=0.98)
    parser.add_argument("--rmse-max", type=float, default=0.08)
    parser.add_argument(
        "--allow-cross-class-candidates",
        action="store_true",
        help="Do not fail on cross-class perceptual matches (not recommended).",
    )
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    try:
        audit_near_duplicates(
            args.project_root,
            phash_max=args.phash_max,
            dhash_max=args.dhash_max,
            correlation_min=args.correlation_min,
            rmse_max=args.rmse_max,
            fail_on_cross_class=not args.allow_cross_class_candidates,
        )
    except Exception as exc:
        LOGGER.error("Near-duplicate audit failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
