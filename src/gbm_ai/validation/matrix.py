from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VALIDATION_MATRIX_VERSION = "phase10_step2_validation_matrix_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MATRIX_PATH = PROJECT_ROOT / "artifacts" / "validation" / "phase10" / "validation_matrix_v1.json"


def load_validation_matrix() -> dict[str, Any]:
    payload = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    if payload.get("version") != VALIDATION_MATRIX_VERSION:
        raise RuntimeError("Phase 10 validation matrix version mismatch")
    return payload


def automated_test_paths() -> list[str]:
    matrix = load_validation_matrix()
    paths: list[str] = []
    for group in matrix.get("automated_groups", []):
        for value in group.get("tests", []):
            item = str(value)
            if item not in paths:
                paths.append(item)
    return paths


def validate_matrix_files_exist() -> list[str]:
    missing: list[str] = []
    for relative in automated_test_paths():
        if not (PROJECT_ROOT / relative).is_file():
            missing.append(relative)
    return missing
