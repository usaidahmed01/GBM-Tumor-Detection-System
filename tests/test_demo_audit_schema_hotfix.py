from pathlib import Path

from gbm_ai.api.models.audit import AuditAction, AuditEntityType, AuditLog


ROOT = Path(__file__).resolve().parents[1]


def test_audit_model_storage_can_hold_every_current_label():
    action_length = AuditLog.__table__.c.action.type.length
    entity_length = AuditLog.__table__.c.entity_type.type.length

    assert action_length == 64
    assert entity_length == 64
    assert max(len(item.name) for item in AuditAction) <= action_length
    assert max(len(item.name) for item in AuditEntityType) <= entity_length


def test_segmentation_preparation_action_no_longer_exceeds_storage():
    label = AuditAction.SEGMENTATION_PREPARATION_COMPLETED.name
    assert label == "SEGMENTATION_PREPARATION_COMPLETED"
    assert len(label) == 34
    assert len(label) <= AuditLog.__table__.c.action.type.length


def test_migration_expands_both_audit_enum_columns():
    migration = (ROOT / "migrations/versions/20260816_0015_expand_audit_enum_storage.py").read_text(
        encoding="utf-8"
    )
    assert '"20260816_0014"' in migration
    assert '"action"' in migration
    assert '"entity_type"' in migration
    assert "String(length=64)" in migration


def test_nifti_detection_does_not_probe_both_header_versions():
    source = (ROOT / "src/gbm_ai/api/upload/format_detection.py").read_text(encoding="utf-8")
    assert "Select the NIfTI parser from the header signature" in source
    assert "candidates: list[tuple[int, object]]" not in source
