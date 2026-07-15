"""Offline TaskRecord schema validation (VAL-SKEL-003)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from swe_factory.schema import (
    VALID_SOURCE_TRACKS,
    SourceTrack,
    TaskRecord,
)


def _full_record_kwargs(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "instance_id": "owner-repo__synthetic_grounded__deadbeefcafe",
        "source_track": "synthetic_grounded",
        "repo": "owner/repo",
        "base_commit": "0123456789abcdef0123456789abcdef01234567",
        "language": "python",
        "problem_statement": "Module foo.bar incorrectly handles multi-file cases.",
        "fail_to_pass": ["pytest tests/test_bug.py::test_case"],
        "pass_to_pass": ["pytest tests/test_ok.py"],
        "gold_patch": (
            "diff --git a/foo/bar.py b/foo/bar.py\n"
            "--- a/foo/bar.py\n"
            "+++ b/foo/bar.py\n"
            "@@ -1,2 +1,3 @@\n"
            " def f():\n"
            "+    return 1\n"
            "     pass\n"
        ),
        "environment": {"image_digest": "sha256:abc123def456"},
        "license": "MIT",
    }
    data.update(overrides)
    return data


def test_valid_full_record_serializes_round_trip() -> None:
    record = TaskRecord.model_validate(_full_record_kwargs())
    payload = record.model_dump(mode="json")
    restored = TaskRecord.model_validate(payload)
    assert restored.instance_id == record.instance_id
    assert restored.source_track == SourceTrack.SYNTHETIC_GROUNDED
    assert restored.gold_patch.startswith("diff --git")
    assert restored.environment.image_digest.startswith("sha256:")
    assert restored.fail_to_pass == ["pytest tests/test_bug.py::test_case"]
    assert restored.pass_to_pass == ["pytest tests/test_ok.py"]
    as_json = record.model_dump_json()
    assert "gold_patch" in as_json
    assert "image_digest" in as_json


def test_real_pr_source_track_accepted() -> None:
    record = TaskRecord.model_validate(_full_record_kwargs(source_track="real_pr", language="go"))
    assert record.source_track == SourceTrack.REAL_PR
    assert record.language == "go"


def test_optional_panel_hardness_fields() -> None:
    record = TaskRecord.model_validate(
        _full_record_kwargs(
            panel={
                "grok_4_5": 0.25,
                "opus_4_8": 0.125,
                "pass_at_k": 0.1875,
                "discrimination": 1.5,
            }
        )
    )
    assert record.panel is not None
    assert record.panel.pass_at_k == 0.1875
    assert record.panel.discrimination == 1.5
    assert record.panel.grok_4_5 == 0.25
    assert record.panel.opus_4_8 == 0.125


def test_missing_gold_patch_raises() -> None:
    data = _full_record_kwargs()
    del data["gold_patch"]
    with pytest.raises(ValidationError) as exc:
        TaskRecord.model_validate(data)
    assert "gold_patch" in str(exc.value)


def test_empty_gold_patch_raises() -> None:
    with pytest.raises(ValidationError) as exc:
        TaskRecord.model_validate(_full_record_kwargs(gold_patch=""))
    assert "gold_patch" in str(exc.value)


def test_missing_fail_to_pass_raises() -> None:
    data = _full_record_kwargs()
    del data["fail_to_pass"]
    with pytest.raises(ValidationError) as exc:
        TaskRecord.model_validate(data)
    assert "fail_to_pass" in str(exc.value)


def test_empty_fail_to_pass_raises() -> None:
    with pytest.raises(ValidationError) as exc:
        TaskRecord.model_validate(_full_record_kwargs(fail_to_pass=[]))
    assert "fail_to_pass" in str(exc.value)


def test_missing_base_commit_raises() -> None:
    data = _full_record_kwargs()
    del data["base_commit"]
    with pytest.raises(ValidationError) as exc:
        TaskRecord.model_validate(data)
    assert "base_commit" in str(exc.value)


def test_empty_base_commit_raises() -> None:
    with pytest.raises(ValidationError) as exc:
        TaskRecord.model_validate(_full_record_kwargs(base_commit=""))
    assert "base_commit" in str(exc.value)


def test_missing_source_track_raises() -> None:
    data = _full_record_kwargs()
    del data["source_track"]
    with pytest.raises(ValidationError) as exc:
        TaskRecord.model_validate(data)
    assert "source_track" in str(exc.value)


def test_invalid_source_track_raises() -> None:
    with pytest.raises(ValidationError) as exc:
        TaskRecord.model_validate(_full_record_kwargs(source_track="mixed_unknown"))
    assert "source_track" in str(exc.value)
    for track in VALID_SOURCE_TRACKS:
        TaskRecord.model_validate(_full_record_kwargs(source_track=track))


def test_missing_environment_image_digest_raises() -> None:
    data = _full_record_kwargs()
    del data["environment"]
    with pytest.raises(ValidationError) as exc:
        TaskRecord.model_validate(data)
    assert "environment" in str(exc.value)

    with pytest.raises(ValidationError) as exc2:
        TaskRecord.model_validate(_full_record_kwargs(environment={"image_digest": ""}))
    assert "image_digest" in str(exc2.value)


def test_missing_license_raises() -> None:
    data = _full_record_kwargs()
    del data["license"]
    with pytest.raises(ValidationError) as exc:
        TaskRecord.model_validate(data)
    assert "license" in str(exc.value)


def test_gold_patch_is_present_but_marked_hidden_in_schema() -> None:
    """Gold remains on the full record for oracle/export; agents never see it."""
    field = TaskRecord.model_fields["gold_patch"]
    assert field.is_required()
    # Annotated for harness/export hidden handling
    extras = field.json_schema_extra or {}
    if callable(extras):
        collected: dict[str, Any] = {}
        extras(collected, field)  # type: ignore[call-arg, arg-type]
        extras = collected
    assert extras.get("hidden") is True or extras.get("x_hidden_from_agent") is True


def test_jsonl_line_parse() -> None:
    record = TaskRecord.model_validate(_full_record_kwargs())
    line = record.model_dump_json()
    parsed = TaskRecord.model_validate_json(line)
    assert parsed.instance_id == record.instance_id
    assert parsed.source_track == record.source_track
