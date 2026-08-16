from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gbm_ai.api.storage.local import LocalObjectStore
from gbm_ai.validation.matrix import PROJECT_ROOT


PERFORMANCE_PROFILE_VERSION = "phase10_step3_engineering_performance_profile_v1"
BUDGET_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "validation"
    / "phase10"
    / "performance_budget_v1.json"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "var"
    / "validation"
    / "phase10_step3_performance_report.json"
)


def _load_budget() -> dict[str, Any]:
    return json.loads(BUDGET_PATH.read_text(encoding="utf-8"))


def _mib_per_second(size_bytes: int, seconds: float) -> float:
    if seconds <= 0:
        return float("inf")
    return (size_bytes / (1024 * 1024)) / seconds


def benchmark_protected_storage(payload_mib: int) -> dict[str, Any]:
    size_bytes = int(payload_mib) * 1024 * 1024
    block = bytes(range(256)) * 4096
    repeats, remainder = divmod(size_bytes, len(block))
    payload = block * repeats + block[:remainder]
    expected = hashlib.sha256(payload).hexdigest()

    with tempfile.TemporaryDirectory(prefix="ngai-perf-storage-") as tmp:
        store = LocalObjectStore(
            Path(tmp),
            max_object_bytes=max(size_bytes * 2, 1024 * 1024),
            chunk_bytes=1024 * 1024,
        )
        key = store.generate_study_derived_key(
            uuid.uuid4(), "performance", suffix=".bin"
        )

        start = time.perf_counter()
        stored = store.put_stream(key, io.BytesIO(payload))
        write_seconds = time.perf_counter() - start

        start = time.perf_counter()
        checksum_ok = store.verify_checksum(key, expected)
        checksum_seconds = time.perf_counter() - start

        start = time.perf_counter()
        with store.open_read(key) as source:
            loaded = source.read()
        read_seconds = time.perf_counter() - start

    return {
        "payload_bytes": size_bytes,
        "sha256_match": stored.sha256 == expected and checksum_ok,
        "roundtrip_bytes_match": loaded == payload,
        "write_seconds": write_seconds,
        "write_mib_per_second": _mib_per_second(size_bytes, write_seconds),
        "checksum_read_seconds": checksum_seconds,
        "checksum_read_mib_per_second": _mib_per_second(
            size_bytes, checksum_seconds
        ),
        "plain_read_seconds": read_seconds,
        "plain_read_mib_per_second": _mib_per_second(size_bytes, read_seconds),
    }


def benchmark_json_serialization(iterations: int = 3000) -> dict[str, Any]:
    payload = {
        "case_reference": "NGAI-PERF-SYNTHETIC",
        "decision_state": "indeterminate",
        "safety_reason_codes": [
            "SYNTHETIC_ENGINEERING_SMOKE",
            "NO_CLINICAL_CLAIM",
        ],
        "tumor_analysis": {
            "wt_volume_cm3": 12.5,
            "tc_volume_cm3": 7.2,
            "et_volume_cm3": 2.1,
            "hemisphere": "left",
            "primary_region": "synthetic-region",
        },
        "traceability": {
            "classifier": "EfficientNetV2-S",
            "segmentation": "SegResNet",
            "review_status": "accepted",
        },
    }
    start = time.perf_counter()
    output = ""
    for _ in range(iterations):
        output = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    elapsed = time.perf_counter() - start
    return {
        "iterations": iterations,
        "seconds": elapsed,
        "last_payload_bytes": len(output.encode("utf-8")),
    }


def _run_timed(command: list[str], cwd: Path) -> dict[str, Any]:
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        errors="replace",
    )
    seconds = time.perf_counter() - start
    return {
        "command": command,
        "returncode": completed.returncode,
        "status": "pass" if completed.returncode == 0 else "fail",
        "seconds": seconds,
        "stdout": completed.stdout[-8000:],
        "stderr": completed.stderr[-8000:],
    }


def benchmark_queue_regression() -> dict[str, Any]:
    return _run_timed(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_phase6_segmentation_background_jobs.py",
        ],
        PROJECT_ROOT,
    )


def benchmark_frontend_build() -> dict[str, Any]:
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
    if npm is None:
        return {"status": "blocked", "reason": "npm_not_available"}
    if not (PROJECT_ROOT / "frontend" / "node_modules").is_dir():
        return {
            "status": "blocked",
            "reason": "frontend_node_modules_missing",
        }
    return _run_timed([npm, "run", "build"], PROJECT_ROOT / "frontend")


def run_performance_profile(
    *, include_frontend_build: bool, report_path: Path
) -> dict[str, Any]:
    budget = _load_budget()
    gates = budget["hard_gates"]
    storage = benchmark_protected_storage(
        int(budget["synthetic_storage_payload_mib"])
    )
    serialization = benchmark_json_serialization()
    queue = benchmark_queue_regression()
    frontend = (
        benchmark_frontend_build()
        if include_frontend_build
        else {"status": "not_requested"}
    )

    checks = {
        "storage_integrity": bool(
            storage["sha256_match"] and storage["roundtrip_bytes_match"]
        ),
        "storage_write_budget": storage["write_mib_per_second"]
        >= float(gates["storage_write_min_mib_per_second"]),
        "storage_checksum_budget": storage["checksum_read_mib_per_second"]
        >= float(gates["storage_checksum_read_min_mib_per_second"]),
        "json_serialization_budget": serialization["seconds"]
        <= float(gates["json_serialization_max_seconds"]),
        "queue_regression_passed": queue["status"] == "pass",
        "queue_regression_budget": queue["seconds"]
        <= float(gates["queue_regression_max_seconds"]),
        "frontend_build_passed": frontend.get("status")
        in {"pass", "not_requested"},
        "frontend_build_budget": (
            frontend.get("status") == "not_requested"
            or (
                frontend.get("status") == "pass"
                and float(frontend.get("seconds", 0.0))
                <= float(gates["frontend_build_max_seconds"])
            )
        ),
    }
    passed = all(checks.values())
    report = {
        "profile_version": PERFORMANCE_PROFILE_VERSION,
        "budget_version": budget["version"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "engineering_smoke_status": "PASS" if passed else "FAIL",
        "checks": checks,
        "storage": storage,
        "json_serialization": serialization,
        "queue_regression": queue,
        "frontend_build": frontend,
        "full_segmentation_latency_measured": False,
        "reason_full_segmentation_latency_not_measured": (
            "Requires a real compatible T1C/T1/T2/FLAIR validation case and "
            "is intentionally kept out of the synthetic smoke profile."
        ),
        "clinical_performance_claimed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NeuroGlioma AI Phase 10 engineering performance smoke checks."
    )
    parser.add_argument(
        "--frontend-build",
        action="store_true",
        help="Also time npm run build when frontend dependencies are present.",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run_performance_profile(
        include_frontend_build=args.frontend_build,
        report_path=args.report,
    )

    print("PHASE 10 STEP 3 — ENGINEERING PERFORMANCE SMOKE")
    print("=" * 78)
    print(f"Protected storage integrity:   {'PASS' if report['checks']['storage_integrity'] else 'FAIL'}")
    print(f"Storage write throughput:      {report['storage']['write_mib_per_second']:.2f} MiB/s")
    print(f"Checksum read throughput:      {report['storage']['checksum_read_mib_per_second']:.2f} MiB/s")
    print(f"Queue regression:              {report['queue_regression']['status'].upper()} ({report['queue_regression']['seconds']:.2f}s)")
    print(f"Frontend build:                {report['frontend_build'].get('status', 'unknown').upper()}")
    print("Full-study SegResNet latency:  NOT MEASURED IN SYNTHETIC SMOKE")
    print(f"Engineering smoke status:      {report['engineering_smoke_status']}")
    print(f"Report written:                {args.report}")
    print("Clinical performance claimed:  NO")
    if report["engineering_smoke_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
