from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_release_manifest_free_tier_policy_is_explicit():
    payload = json.loads((ROOT / "artifacts/release/phase10_release_manifest_v1.json").read_text(encoding="utf-8"))
    assert payload["deployment_policy"]["paid_model_api_required"] is False
    assert payload["deployment_policy"]["free_tier_first"] is True
    assert payload["deployment_policy"]["clinical_deployment_claimed"] is False


def test_backend_dockerfile_uses_src_layout_and_does_not_copy_secrets():
    content = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "PYTHONPATH=/app/src" in content
    assert "gbm_ai.api.main:app" in content
    assert "COPY .env" not in content


def test_dockerignore_blocks_private_runtime_and_large_model_assets():
    content = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for token in (".env", "var/storage", "var/model_bundles", "var/localization_atlas", "*.pt", "frontend"):
        assert token in content


def test_free_deployment_document_does_not_overclaim_persistent_hosted_storage():
    content = (ROOT / "docs/FREE_DEPLOYMENT.md").read_text(encoding="utf-8")
    assert "free-tier-first" in content.lower()
    assert "paid model api" in content.lower()
    assert "demonstration-only" in content.lower()
    assert "do **not** claim durable cloud mri storage" in content.lower()


def test_git_lock_recovery_script_never_deletes_git_index():
    content = (ROOT / "scripts/fix_git_index_lock.ps1").read_text(encoding="utf-8")
    assert '"index.lock"' in content
    assert 'Remove-Item $lockPath -Force' in content
    assert 'Remove-Item $gitDir' not in content
    assert r'Remove-Item ".git\\index"' not in content


def test_frontend_vercel_descriptor_exists():
    payload = json.loads((ROOT / "frontend/vercel.json").read_text(encoding="utf-8"))
    assert payload["framework"] == "nextjs"
