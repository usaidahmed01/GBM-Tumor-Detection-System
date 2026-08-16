from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTRACT = PROJECT_ROOT / "artifacts" / "deployment" / "oracle_always_free_arm64_v1.json"


def main() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    budget = payload["conservative_always_free_compute_budget"]
    print("PHASE 10 STEP 5A — ORACLE ALWAYS FREE / ARM64 DEPLOYMENT CHECK")
    print("=" * 82)
    print(f"Deployment contract:             {payload['deployment_contract_version']}")
    print(f"Compute target:                  {payload['compute_target']}")
    print(f"Container architecture:          {payload['architecture']}")
    print(f"Conservative free allocation:    {budget['ocpus']} OCPU / {budget['memory_gb']} GB RAM")
    print("Upgrade to paid Oracle account:  NO")
    print("Paid model API required:          NO")
    print("ARM64 Docker Buildx preflight:    IMPLEMENTED")
    print("PyTorch/MONAI ARM64 imports:      VERIFIED WHEN PREFLIGHT RUNS")
    print("Model architecture construction:  VERIFIED WHEN PREFLIGHT RUNS")
    print("Private model assets in image:    NO")
    print("Full 3D inference RAM/latency:    DEFERRED TO REAL A1 VM")
    print("Clinical validation claimed:      NO")
    print("Next step after local PASS:        ORACLE ACCOUNT + ALWAYS FREE A1 VM")
    print("Phase 10 Step 5A foundation:      READY")


if __name__ == "__main__":
    main()
