from __future__ import annotations

from gbm_ai.validation import runner


def test_runner_writes_pass_report_without_upgrading_external_cases(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner,
        "_group_tests",
        lambda group: {
            "id": group["id"],
            "title": group["title"],
            "tests": group["tests"],
            "status": "pass",
            "returncode": 0,
            "stdout": "ok",
            "stderr": "",
        },
    )
    monkeypatch.setattr(
        runner,
        "_alembic_heads",
        lambda: {"status": "pass", "single_head": True, "heads": ["20260816_0014 (head)"]},
    )
    monkeypatch.setattr(
        runner,
        "_runtime_prerequisites",
        lambda: {
            "classifier_runtime_ready": False,
            "classifier_checkpoint_count_available": 0,
            "classifier_checkpoint_count_expected": 5,
            "segmentation_bundle_ready": False,
            "localization_assets_ready": False,
            "database_url_configured": True,
            "frontend_node_modules_present": False,
            "npm_available": False,
        },
    )

    path = tmp_path / "report.json"
    report = runner.run_validation(full=False, frontend_build=False, report_path=path)
    assert report["overall_status"] == "AUTOMATED_PASS_EXTERNAL_VALIDATION_PENDING"
    assert path.is_file()
    assert any(item["status"] == "requires_local_classifier_checkpoints" for item in report["external_or_manual_cases"])
    assert report["clinical_validation_claimed"] is False
