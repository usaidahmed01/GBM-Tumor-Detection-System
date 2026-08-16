from __future__ import annotations

from io import BytesIO

from PIL import Image

from gbm_ai.api.config import Settings
from gbm_ai.api.services.classifier_runtime import (
    classifier_runtime_status,
    load_deployment_manifest,
    preprocess_image_bytes,
)


def test_classifier_deployment_manifest_is_frozen():
    manifest = load_deployment_manifest()
    assert manifest.deployment_strategy_frozen is True
    assert manifest.folds == [0, 1, 2, 3, 4]
    assert manifest.threshold_low < manifest.threshold_high


def test_classifier_runtime_status_exposes_readiness_and_missing_assets():
    settings = Settings()
    status = classifier_runtime_status(settings)
    assert status["deployment_strategy_frozen"] is True
    assert status["checkpoint_count_expected"] == 5
    assert isinstance(status["missing_assets"], list)


def test_classifier_preprocess_outputs_expected_tensor_shape():
    image = Image.new("RGB", (24, 24), color=(0, 128, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    tensor = preprocess_image_bytes(buffer.getvalue())
    assert tuple(tensor.shape) == (1, 3, 384, 384)
