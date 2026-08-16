from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gbm_ai.api.config import Settings
from gbm_ai.api.localization_assets import localization_asset_paths
from gbm_ai.api.segmentation.bundle_runtime import frozen_bundle_dir
from gbm_ai.api.services.classifier_runtime import classifier_runtime_status
from gbm_ai.validation.matrix import (
    PROJECT_ROOT,
    VALIDATION_MATRIX_VERSION,
    load_validation_matrix,
    validate_matrix_files_exist,
)


REPORT_VERSION = "phase10_step2_automated_validation_report_v1"
DEFAULT_REPORT = PROJECT_ROOT / "var" / "validation" / "phase10_step2_validation_report.json"


def _run(command: list[str], *, cwd: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        errors="replace",
    )
    ended = datetime.now(timezone.utc)
    return {
        "command": command,
        "returncode": result.returncode,
        "status": "pass" if result.returncode == 0 else "fail",
        "started_at_utc": started.isoformat(),
        "ended_at_utc": ended.isoformat(),
        "stdout": result.stdout[-12000:],
        "stderr": result.stderr[-12000:],
    }


def _runtime_prerequisites() -> dict[str, Any]:
    settings = Settings()
    classifier = classifier_runtime_status(settings)

    bundle_dir = frozen_bundle_dir(settings.segmentation_bundle_root_resolved)
    bundle_files = [
        bundle_dir / "models" / "model.pt",
        bundle_dir / "configs" / "metadata.json",
        bundle_dir / "configs" / "inference.json",
    ]
    segmentation_bundle_ready = all(path.is_file() for path in bundle_files)

    atlas = localization_asset_paths(settings.localization_atlas_root_resolved)
    localization_assets_ready = all(
        atlas[name].is_file()
        for name in ("template", "brain_mask", "cortical", "subcortical", "labels", "manifest")
    )

    return {
        "classifier_runtime_ready": bool(classifier.get("ready")),
        "classifier_checkpoint_count_available": classifier.get("checkpoint_count_available"),
        "classifier_checkpoint_count_expected": classifier.get("checkpoint_count_expected"),
        "segmentation_bundle_ready": segmentation_bundle_ready,
        "localization_assets_ready": localization_assets_ready,
        "database_url_configured": bool(settings.database_url_value),
        "frontend_node_modules_present": (PROJECT_ROOT / "frontend" / "node_modules").is_dir(),
        "npm_available": shutil.which("npm.cmd" if os.name == "nt" else "npm") is not None,
    }


def _alembic_heads() -> dict[str, Any]:
    command = [sys.executable, "-m", "alembic", "heads"]
    result = _run(command, cwd=PROJECT_ROOT)
    heads = [line.strip() for line in result["stdout"].splitlines() if "(head)" in line]
    result["heads"] = heads
    result["single_head"] = result["status"] == "pass" and len(heads) == 1
    return result


def _group_tests(group: dict[str, Any]) -> dict[str, Any]:
    tests = [str(value) for value in group.get("tests", [])]
    command = [sys.executable, "-m", "pytest", "-q", *tests]
    result = _run(command, cwd=PROJECT_ROOT)
    result["id"] = group.get("id")
    result["title"] = group.get("title")
    result["tests"] = tests
    return result


def _frontend_build() -> dict[str, Any]:
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
    if not npm:
        return {
            "status": "blocked",
            "reason": "npm_not_available",
            "returncode": None,
        }
    if not (PROJECT_ROOT / "frontend" / "node_modules").is_dir():
        return {
            "status": "blocked",
            "reason": "frontend_node_modules_missing",
            "returncode": None,
        }
    return _run([npm, "run", "build"], cwd=PROJECT_ROOT / "frontend")


def run_validation(*, full: bool, frontend_build: bool, report_path: Path) -> dict[str, Any]:
    matrix = load_validation_matrix()
    missing_tests = validate_matrix_files_exist()
    if missing_tests:
        raise RuntimeError("Validation matrix references missing tests: " + ", ".join(missing_tests))

    groups = list(matrix["automated_groups"])
    if not full:
        selected_ids = {"upload_qc_routing", "classifier_safety", "segmentation_pipeline", "viewer_review_report"}
        groups = [group for group in groups if group.get("id") in selected_ids]

    group_results = [_group_tests(group) for group in groups]
    alembic = _alembic_heads()
    frontend = _frontend_build() if frontend_build else {"status": "not_requested"}
    prerequisites = _runtime_prerequisites()

    automated_pass = all(item["status"] == "pass" for item in group_results)
    schema_pass = bool(alembic.get("single_head"))
    frontend_pass = frontend.get("status") in {"pass", "not_requested"}

    if automated_pass and schema_pass and frontend_pass:
        overall = "AUTOMATED_PASS_EXTERNAL_VALIDATION_PENDING"
    else:
        overall = "FAIL"

    report = {
        "report_version": REPORT_VERSION,
        "matrix_version": VALIDATION_MATRIX_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "full" if full else "quick",
        "overall_status": overall,
        "automated_groups": group_results,
        "alembic_heads": alembic,
        "frontend_build": frontend,
        "runtime_prerequisites": prerequisites,
        "critical_failure_modes": matrix.get("critical_failure_modes", []),
        "external_or_manual_cases": matrix.get("external_or_manual_cases", []),
        "clinical_validation_claimed": False,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 10 automated validation matrix.")
    parser.add_argument("--full", action="store_true", help="Run every automated validation group.")
    parser.add_argument("--frontend-build", action="store_true", help="Also run npm run build when frontend dependencies are installed.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Output JSON report path.")
    args = parser.parse_args()

    report = run_validation(full=args.full, frontend_build=args.frontend_build, report_path=args.report)
    print("PHASE 10 STEP 2 — END-TO-END VALIDATION MATRIX")
    print("=" * 78)
    print(f"Mode:                         {report['mode'].upper()}")
    print(f"Automated groups executed:    {len(report['automated_groups'])}")
    print(f"Alembic single head:          {'YES' if report['alembic_heads'].get('single_head') else 'NO'}")
    print(f"Frontend build:               {report['frontend_build'].get('status', 'unknown').upper()}")
    runtime = report["runtime_prerequisites"]
    print(f"2D classifier runtime assets: {'READY' if runtime['classifier_runtime_ready'] else 'PENDING LOCAL CHECKPOINTS'}")
    print(f"MONAI bundle assets:          {'READY' if runtime['segmentation_bundle_ready'] else 'PENDING LOCAL BUNDLE'}")
    print(f"Localization atlas assets:    {'READY' if runtime['localization_assets_ready'] else 'PENDING LOCAL ASSETS'}")
    print(f"Overall automated status:     {report['overall_status']}")
    print(f"Report written:               {args.report}")
    print("Clinical validation claimed:  NO")

    if report["overall_status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
