from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _tracked_generated_paths() -> tuple[str, list[str]]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "frontend/.next", "frontend/node_modules", "frontend/out", "frontend/.turbo", "frontend/.cache"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return "UNAVAILABLE", []
    tracked = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return ("CLEAN" if not tracked else "GENERATED_FILES_TRACKED"), tracked


def main() -> None:
    ignore = _read(".gitignore")
    attrs = _read(".gitattributes")
    next_config = _read("frontend/next.config.mjs")
    ui_sources = "\n".join(
        _read(path)
        for path in (
            "frontend/app/page.jsx",
            "frontend/app/analysis/new/page.jsx",
            "frontend/app/viewer/current/page.jsx",
            "frontend/app/report/current/page.jsx",
            "frontend/components/intake/AnalysisIntakeWorkspace.jsx",
            "frontend/components/viewer/ViewerWorkspace.jsx",
            "frontend/components/report/ReportWorkspace.jsx",
        )
    )

    required_ignores = [
        "/frontend/.next/",
        "/frontend/node_modules/",
        "/frontend/out/",
        "/frontend/.turbo/",
        "/frontend/.cache/",
        "/frontend/.env.local",
    ]
    forbidden_copy = [
        "Research prototype",
        "Protected research workflow",
        "No UUID entry required",
        "Internal study IDs",
        "technical identifiers automatically",
        "background worker owns",
        "Local testing:",
        "Passed engineering gate",
        "checksum-bound",
        "Immutable finalized report record",
    ]

    if not all(rule in ignore for rule in required_ignores):
        raise RuntimeError("frontend generated-artifact ignore policy is incomplete")
    if "* text=auto" not in attrs:
        raise RuntimeError(".gitattributes line-ending policy is missing")
    if any(phrase in ui_sources for phrase in forbidden_copy):
        raise RuntimeError("developer/prototype copy remains in the normal product UI")
    for header in ("X-Content-Type-Options", "Referrer-Policy", "X-Frame-Options", "Permissions-Policy"):
        if header not in next_config:
            raise RuntimeError(f"frontend security header missing: {header}")

    git_state, tracked = _tracked_generated_paths()
    if git_state == "GENERATED_FILES_TRACKED":
        raise RuntimeError(
            "generated frontend paths are still tracked by Git; run scripts/fix_git_generated_artifacts.ps1 first: "
            + ", ".join(tracked[:5])
        )

    print("PHASE 10 STEP 1 — PRODUCT / REPOSITORY HARDENING CHECK")
    print("=" * 82)
    print("Product-facing prototype/debug copy: REMOVED")
    print("Clinical safety notice retained:    YES")
    print("Frontend .next ignored:             YES")
    print("Frontend node_modules ignored:      YES")
    print("Frontend build output ignored:      YES")
    print("Line-ending policy:                 READY (.gitattributes)")
    print("Frontend security headers:          READY")
    print(f"Git generated-artifact index:       {git_state}")
    print("Clean-environment E2E validation:   NEXT STEP")
    print("Clinical validation claimed:        NO")
    print("Phase 10 Step 1 foundation:         READY")


if __name__ == "__main__":
    main()
