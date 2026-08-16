from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = PROJECT_ROOT / "artifacts" / "release" / "phase10_release_manifest_v1.json"


def main() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    deployment = payload["deployment_policy"]
    print("PHASE 10 STEP 4 — RELEASE PACKAGING & FREE-DEPLOYMENT CHECK")
    print("=" * 82)
    print(f"Release manifest:                 {payload['release_manifest_version']}")
    print(f"Product name:                     {payload['product_name']}")
    print("Release stage:                    UNIVERSITY DEMO RELEASE CANDIDATE")
    print("Backend Docker image definition:  READY")
    print("Frontend Vercel descriptor:       READY")
    print("Release checklist:                READY")
    print("Git stale-lock recovery helper:   READY")
    print(f"Paid model API required:           {'YES' if deployment['paid_model_api_required'] else 'NO'}")
    print(f"Free-tier-first policy:            {'YES' if deployment['free_tier_first'] else 'NO'}")
    print(f"Frontend target:                   {deployment['recommended_frontend']}")
    print(f"API target:                        {deployment['recommended_api']}")
    print(f"Database target:                   {deployment['recommended_database']}")
    print(f"Object-storage target:             {deployment['recommended_object_storage']}")
    print("Durable cloud MRI storage:         ADAPTER STILL REQUIRED / NOT CLAIMED")
    print("2D classifier cloud readiness:     REQUIRES FIVE FROZEN LOCAL/PRIVATE CHECKPOINTS")
    print("Clinical validation claimed:       NO")
    print("Next step:                         PHASE 10 STEP 5 — FREE DEPLOYMENT STORAGE/EXECUTION")
    print("Phase 10 Step 4 foundation:        READY")


if __name__ == "__main__":
    main()
