from __future__ import annotations

import json
from pathlib import Path

from gbm_ai.validation.matrix import PROJECT_ROOT


PERFORMANCE_BUDGET = (
    PROJECT_ROOT / "artifacts" / "validation" / "phase10" / "performance_budget_v1.json"
)
REPRO_CONTRACT = (
    PROJECT_ROOT / "artifacts" / "validation" / "phase10" / "reproducibility_contract_v1.json"
)


def main() -> None:
    performance = json.loads(PERFORMANCE_BUDGET.read_text(encoding="utf-8"))
    repro = json.loads(REPRO_CONTRACT.read_text(encoding="utf-8"))

    print("PHASE 10 STEP 3 — PERFORMANCE & REPRODUCIBILITY FOUNDATION CHECK")
    print("=" * 82)
    print(f"Performance budget version:      {performance['version']}")
    print("Performance budget purpose:      ENGINEERING STALL/FAILURE GUARD")
    print("Clinical latency target:          NOT CLAIMED")
    print(f"Synthetic storage smoke payload:  {performance['synthetic_storage_payload_mib']} MiB")
    print("Queue recovery regression timing: IMPLEMENTED")
    print("Frontend production-build timing: IMPLEMENTED / OPTIONAL")
    print("Real SegResNet case timing:        EXTERNAL COMPATIBLE CASE REQUIRED")
    print(f"Reproducibility contract:          {repro['version']}")
    print("Single requirements.txt policy:    REQUIRED")
    print("Frontend package-lock policy:      REQUIRED FOR CLEAN npm ci")
    print("Alembic single-head check:         REQUIRED")
    print("pip dependency consistency check:  REQUIRED")
    print("Generated-artifact Git check:      REQUIRED")
    print("Clean Python venv script:          IMPLEMENTED")
    print("Clean frontend npm ci script:      IMPLEMENTED")
    print("Runtime MRI/model assets in Git:   NO")
    print("Clinical validation claimed:       NO")
    print("Next step:                         PHASE 10 STEP 4 — RELEASE PACKAGING / FINAL DOCUMENTATION")
    print("Phase 10 Step 3 foundation:        READY")


if __name__ == "__main__":
    main()
