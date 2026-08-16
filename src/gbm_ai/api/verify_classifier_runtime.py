from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from gbm_ai.api.config import Settings
from gbm_ai.api.services.classifier_runtime import classifier_runtime_status, load_deployment_manifest, preprocess_image_bytes


def main() -> None:
    manifest = load_deployment_manifest()
    settings = Settings()
    status = classifier_runtime_status(settings)

    buffer = BytesIO()
    Image.new("RGB", (4, 4), color=(30, 180, 220)).save(buffer, format="PNG")
    tensor = preprocess_image_bytes(buffer.getvalue(), manifest=manifest)

    payload = {
        "deployment_strategy_frozen": manifest.deployment_strategy_frozen,
        "ensemble_folds": manifest.folds,
        "runtime_ready": status["ready"],
        "missing_asset_count": len(status["missing_assets"]),
        "preprocess_shape": list(tensor.shape),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
