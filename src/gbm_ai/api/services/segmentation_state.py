from __future__ import annotations

from gbm_ai.api.models.analysis import (
    SegmentationPreparationStatus,
    Study,
)


def invalidate_segmentation_preparation(study: Study) -> None:
    """Clear derived Phase 6 volume-preparation state after an upstream change."""
    study.segmentation_preparation_status = SegmentationPreparationStatus.PENDING
    study.segmentation_preparation_summary = {}
