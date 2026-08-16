from __future__ import annotations

import json
from pathlib import Path

from gbm_ai.validation.matrix import PROJECT_ROOT
from gbm_ai.validation.performance import (
    BUDGET_PATH,
    benchmark_json_serialization,
    benchmark_protected_storage,
)
from gbm_ai.validation.reproducibility import (
    CONTRACT_PATH,
    _requirements_policy,
)


def test_phase10_step3_contract_files_exist_and_are_versioned():
    performance = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
    reproducibility = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert performance["version"] == "phase10_step3_engineering_performance_budget_v1"
    assert reproducibility["version"] == "phase10_step3_reproducibility_contract_v1"
    assert performance["scope"].startswith("engineering smoke budgets")
    assert reproducibility["clinical_validation_claimed"] is False


def test_protected_storage_smoke_preserves_bytes_and_checksum():
    result = benchmark_protected_storage(1)
    assert result["sha256_match"] is True
    assert result["roundtrip_bytes_match"] is True
    assert result["write_mib_per_second"] > 0
    assert result["checksum_read_mib_per_second"] > 0


def test_json_serialization_smoke_completes():
    result = benchmark_json_serialization(iterations=50)
    assert result["iterations"] == 50
    assert result["seconds"] >= 0
    assert result["last_payload_bytes"] > 0


def test_single_cumulative_requirements_policy_is_preserved():
    policy = _requirements_policy()
    assert policy["exactly_one"] is True
    assert policy["files"] == ["requirements.txt"]


def test_reproducibility_scripts_and_readme_are_present():
    expected = [
        "scripts/run_phase10_performance.ps1",
        "scripts/run_phase10_reproducibility.ps1",
        "README.md",
    ]
    for relative in expected:
        assert (PROJECT_ROOT / relative).is_file(), relative


def test_clean_reproducibility_venv_is_ignored():
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/.phase10-clean-venv/" in gitignore
