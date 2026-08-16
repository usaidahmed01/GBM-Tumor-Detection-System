from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_qc_confirmation_bar_is_prominent_and_actionable():
    ui = _read("frontend/components/intake/AnalysisIntakeWorkspace.jsx")
    css = _read("frontend/app/globals.css")
    assert "Required action" in ui
    assert "Click anywhere on this bar" in ui
    assert "brain-confirm--attention" in ui
    assert "brain-confirm__pulse-dot" in ui
    assert "@keyframes brain-confirm-attention" in css
    assert "@keyframes brain-confirm-sheen" in css
    assert "@keyframes brain-confirm-dot" in css


def test_qc_continue_button_is_locked_until_required_review_is_complete():
    ui = _read("frontend/components/intake/AnalysisIntakeWorkspace.jsx")
    assert "qcReadyForRouting" in ui
    assert "1 required confirmation remaining" in ui
    assert "Ready for AI eligibility" in ui
    assert "Confirm brain MRI above" in ui
    assert "disabled={busy || !qcReadyForRouting}" in ui


def test_confirmation_checkbox_remains_keyboard_focusable():
    css = _read("frontend/app/globals.css")
    assert ".brain-confirm input{position:absolute;opacity:0" in css
    assert ".brain-confirm:has(input:focus-visible)" in css
