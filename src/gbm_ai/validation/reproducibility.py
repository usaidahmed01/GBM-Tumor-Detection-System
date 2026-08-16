from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gbm_ai.validation.matrix import PROJECT_ROOT


REPRODUCIBILITY_VERSION = "phase10_step3_reproducibility_check_v1"
CONTRACT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "validation"
    / "phase10"
    / "reproducibility_contract_v1.json"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "var"
    / "validation"
    / "phase10_step3_reproducibility_report.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], cwd: Path = PROJECT_ROOT) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        errors="replace",
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "status": "pass" if result.returncode == 0 else "fail",
        "stdout": result.stdout[-8000:],
        "stderr": result.stderr[-8000:],
    }


def _python_version_ok(minimum: str) -> bool:
    major, minor = [int(x) for x in minimum.split(".")[:2]]
    return sys.version_info >= (major, minor)


def _node_version() -> dict[str, Any]:
    node = shutil.which("node.exe" if os.name == "nt" else "node")
    if node is None:
        return {"available": False, "version": None, "major": None}
    result = _run([node, "--version"])
    value = result["stdout"].strip().lstrip("v")
    try:
        major = int(value.split(".", 1)[0])
    except Exception:
        major = None
    return {"available": True, "version": value, "major": major}


def _requirements_policy() -> dict[str, Any]:
    candidates = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in PROJECT_ROOT.glob("requirements*.txt")
        if path.is_file()
    )
    return {
        "files": candidates,
        "exactly_one": candidates == ["requirements.txt"],
    }


def _alembic_heads() -> dict[str, Any]:
    result = _run([sys.executable, "-m", "alembic", "heads"])
    heads = [line.strip() for line in result["stdout"].splitlines() if "(head)" in line]
    return {
        **result,
        "heads": heads,
        "single_head": result["status"] == "pass" and len(heads) == 1,
    }


def _git_generated_tracking(contract: dict[str, Any]) -> dict[str, Any]:
    git = shutil.which("git.exe" if os.name == "nt" else "git")
    if git is None or not (PROJECT_ROOT / ".git").exists():
        return {
            "git_repository_available": False,
            "tracked_generated_paths": [],
            "clean": True,
            "note": "Git index check skipped outside a Git working tree.",
        }
    result = _run([git, "ls-files"])
    tracked = [line.strip().replace("\\", "/") for line in result["stdout"].splitlines()]
    bad: list[str] = []
    prefixes = [str(x).strip("/") for x in contract["generated_paths_that_must_not_be_tracked"]]
    for path in tracked:
        normalized = path.strip("/")
        if any(normalized == prefix or normalized.startswith(prefix + "/") for prefix in prefixes):
            bad.append(path)
    return {
        "git_repository_available": True,
        "tracked_generated_paths": sorted(set(bad)),
        "clean": not bad,
    }


def collect_reproducibility_report(report_path: Path) -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    required = {
        rel: (PROJECT_ROOT / rel).is_file()
        for rel in contract["required_repository_files"]
    }
    lockfile = PROJECT_ROOT / contract["frontend_lockfile"]
    requirements = _requirements_policy()
    pip_check = _run([sys.executable, "-m", "pip", "check"])
    alembic = _alembic_heads()
    node = _node_version()
    git = _git_generated_tracking(contract)

    fingerprints: dict[str, str] = {}
    for rel in ["requirements.txt", "frontend/package.json", ".env.example"]:
        path = PROJECT_ROOT / rel
        if path.is_file():
            fingerprints[rel] = _sha256(path)
    if lockfile.is_file():
        fingerprints[contract["frontend_lockfile"]] = _sha256(lockfile)

    python_ok = _python_version_ok(contract["python_minimum"])
    node_ok = bool(
        node["available"]
        and node["major"] is not None
        and node["major"] >= int(contract["node_minimum_major"])
    )
    checks = {
        "required_repository_files_present": all(required.values()),
        "single_python_requirements_file": requirements["exactly_one"],
        "python_version_supported": python_ok,
        "node_version_supported": node_ok,
        "frontend_lockfile_present": lockfile.is_file(),
        "pip_dependency_check": pip_check["status"] == "pass",
        "alembic_single_head": alembic["single_head"],
        "generated_artifacts_not_tracked": git["clean"],
    }
    ready = all(checks.values())
    report = {
        "reproducibility_version": REPRODUCIBILITY_VERSION,
        "contract_version": contract["version"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "READY" if ready else "ACTION_REQUIRED",
        "checks": checks,
        "required_repository_files": required,
        "requirements_policy": requirements,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "minimum": contract["python_minimum"],
        },
        "node": node,
        "frontend_lockfile": str(lockfile.relative_to(PROJECT_ROOT)),
        "pip_check": pip_check,
        "alembic": alembic,
        "git_generated_artifacts": git,
        "dependency_fingerprints_sha256": fingerprints,
        "external_runtime_assets_are_environment_managed": True,
        "clinical_validation_claimed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check NeuroGlioma AI clean-environment reproducibility prerequisites."
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = collect_reproducibility_report(args.report)
    checks = report["checks"]
    print("PHASE 10 STEP 3 — CLEAN-ENVIRONMENT REPRODUCIBILITY CHECK")
    print("=" * 78)
    print(f"Python supported:              {'YES' if checks['python_version_supported'] else 'NO'} ({report['python']['version']})")
    print(f"Node supported:                {'YES' if checks['node_version_supported'] else 'NO'} ({report['node']['version'] or 'missing'})")
    print(f"Single requirements.txt:       {'YES' if checks['single_python_requirements_file'] else 'NO'}")
    print(f"Frontend package-lock.json:    {'READY' if checks['frontend_lockfile_present'] else 'MISSING'}")
    print(f"pip check:                     {'PASS' if checks['pip_dependency_check'] else 'FAIL'}")
    print(f"Alembic single head:           {'YES' if checks['alembic_single_head'] else 'NO'}")
    print(f"Generated artifacts tracked:   {'NO' if checks['generated_artifacts_not_tracked'] else 'YES'}")
    print(f"Reproducibility status:        {report['status']}")
    print(f"Report written:                {args.report}")
    print("Clinical validation claimed:   NO")
    if report["status"] != "READY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
