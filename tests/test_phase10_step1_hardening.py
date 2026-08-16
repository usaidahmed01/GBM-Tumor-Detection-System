from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_generated_frontend_artifacts_are_explicitly_ignored():
    content = _read(".gitignore")
    for rule in (
        "/frontend/.next/",
        "/frontend/node_modules/",
        "/frontend/out/",
        "/frontend/.turbo/",
        "/frontend/.cache/",
        "/frontend/.env.local",
    ):
        assert rule in content


def test_repository_has_explicit_line_ending_policy():
    content = _read(".gitattributes")
    assert "* text=auto" in content
    assert "*.py text eol=lf" in content
    assert "*.jsx text eol=lf" in content
    assert "*.ps1 text eol=crlf" in content


def test_product_ui_does_not_show_developer_or_prototype_copy():
    files = [
        ROOT / "frontend/app/page.jsx",
        ROOT / "frontend/app/analysis/new/page.jsx",
        ROOT / "frontend/app/viewer/current/page.jsx",
        ROOT / "frontend/app/report/current/page.jsx",
        ROOT / "frontend/components/intake/AnalysisIntakeWorkspace.jsx",
        ROOT / "frontend/components/viewer/ViewerWorkspace.jsx",
        ROOT / "frontend/components/report/ReportWorkspace.jsx",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in files)
    forbidden = (
        "Research prototype",
        "Protected research workflow",
        "No UUID entry required",
        "Internal study IDs",
        "technical identifiers automatically",
        "documentation only — not classifier features",
        "background worker owns",
        "Local testing:",
        "Passed engineering gate",
        "checksum-bound",
        "Immutable finalized report record",
    )
    for phrase in forbidden:
        assert phrase not in source


def test_clinically_important_safety_boundary_remains_visible():
    viewer = _read("frontend/components/viewer/ViewerWorkspace.jsx")
    report = _read("frontend/components/report/ReportWorkspace.jsx")
    assert "not a definitive GBM diagnosis" in viewer
    assert "report.clinical_notice" in report


def test_next_frontend_has_baseline_security_headers():
    config = _read("frontend/next.config.mjs")
    for header in (
        "X-Content-Type-Options",
        "Referrer-Policy",
        "X-Frame-Options",
        "Permissions-Policy",
    ):
        assert header in config


def test_git_cleanup_helper_is_present():
    content = _read("scripts/fix_git_generated_artifacts.ps1")
    assert "git rm -r --cached --ignore-unmatch frontend/.next" in content
    assert "git check-ignore -v frontend/.next/dev/lock" in content
    assert "git ls-files frontend/.next" in content
