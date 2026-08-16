from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "frontend" / "app" / "analysis" / "new" / "page.jsx"
WORKSPACE = ROOT / "frontend" / "components" / "intake" / "AnalysisIntakeWorkspace.jsx"


def test_analysis_new_wraps_search_param_client_tree_in_suspense():
    page = PAGE.read_text(encoding="utf-8")
    workspace = WORKSPACE.read_text(encoding="utf-8")
    assert "useSearchParams" in workspace
    assert "import { Suspense } from 'react'" in page
    assert "<Suspense" in page
    assert "fallback=" in page
    assert "<AnalysisIntakeWorkspace" in page
