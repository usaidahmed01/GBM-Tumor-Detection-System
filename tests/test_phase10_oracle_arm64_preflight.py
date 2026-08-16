from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_oracle_contract_is_hard_free_tier_first():
    payload = json.loads((ROOT / "artifacts/deployment/oracle_always_free_arm64_v1.json").read_text(encoding="utf-8"))
    assert payload["compute_target"] == "VM.Standard.A1.Flex"
    assert payload["architecture"] == "linux/arm64"
    assert payload["account_policy"]["always_free_only"] is True
    assert payload["account_policy"]["upgrade_to_paid_account"] is False
    assert payload["account_policy"]["paid_model_api_required"] is False
    assert payload["conservative_always_free_compute_budget"] == {"ocpus": 2, "memory_gb": 12, "notes": payload["conservative_always_free_compute_budget"]["notes"]}


def test_dockerfile_has_real_arm64_preflight_target_and_runtime_target():
    content = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "AS oracle-arm64-preflight" in content
    assert "python -m pip check" in content
    assert "gbm_ai.deployment.oracle_arm64_preflight" in content
    assert "AS api" in content
    assert "gbm_ai.api.main:app" in content


def test_windows_preflight_uses_buildx_linux_arm64_and_inspects_architecture():
    content = (ROOT / "scripts/run_oracle_arm64_preflight.ps1").read_text(encoding="utf-8")
    assert '"--platform", "linux/arm64"' in content
    assert '"--target", "oracle-arm64-preflight"' in content
    assert "docker image inspect" in content
    assert "arm64" in content
    assert "ORACLE ARM64 DOCKER PREFLIGHT: PASS" in content


def test_runtime_preflight_checks_native_imaging_and_both_model_architectures():
    content = (ROOT / "src/gbm_ai/deployment/oracle_arm64_preflight.py").read_text(encoding="utf-8")
    for token in ("SimpleITK", "Nifti1Image", "efficientnet_v2_s", "SegResNet", "gbm_ai.api.main"):
        assert token in content
    assert "weights=None" in content
    assert "full_forward_inference_executed" in content
    assert '"aarch64", "arm64"' in content


def test_oracle_deployment_document_keeps_private_assets_out_of_git_and_image():
    content = (ROOT / "docs/ORACLE_ALWAYS_FREE_DEPLOYMENT.md").read_text(encoding="utf-8").lower()
    assert "always free" in content
    assert "do not upgrade" in content
    assert "2 ocpus" in content
    assert "12 gb" in content
    assert "outside git" in content
    assert "outside the image" in content
    assert "full 3d mri inference fits" in content


def test_release_manifest_now_points_api_to_oracle_not_google_cloud():
    payload = json.loads((ROOT / "artifacts/release/phase10_release_manifest_v1.json").read_text(encoding="utf-8"))
    deployment = payload["deployment_policy"]
    assert "Oracle Cloud Always Free" in deployment["recommended_api"]
    assert deployment["oracle_account_upgrade_to_paid"] is False
    assert deployment["oracle_arm64_preflight_required"] is True
    assert "Google Cloud" not in deployment["recommended_api"]


def test_oracle_dockerfile_forces_official_cpu_only_pytorch_wheels():
    content = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "https://download.pytorch.org/whl/cpu" in content
    assert "2.13.0+cpu" in content
    assert "0.28.0+cpu" in content
    assert "requirements-without-torch.txt" in content
    assert "nvidia-" in content.lower()


def test_arm64_runtime_preflight_rejects_cuda_nvidia_packages():
    content = (ROOT / "src/gbm_ai/deployment/oracle_arm64_preflight.py").read_text(encoding="utf-8")
    assert "ORACLE_ARM64_GPU_PACKAGES_PRESENT" in content
    assert "ORACLE_ARM64_NON_CPU_TORCH" in content
    assert '"+cpu"' in content
    assert "torch.version" in content
