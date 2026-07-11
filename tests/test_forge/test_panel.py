"""Offline panel tests using child-authority test responses."""

from __future__ import annotations

import pytest

from swe_forge.forge.panel import (
    InvalidTierError,
    PanelError,
    PanelModel,
    build_panel,
    resolve_panel_endpoint,
    run_rollouts,
)
from swe_forge.forge.secrets import key_fingerprint
from swe_forge.forge.teacher import TeacherClient

SECRET = "sk-super-secret-do-not-print"


def _model(**overrides: object) -> PanelModel:
    values: dict[str, object] = {
        "id": "m",
        "model_string": "anthropic/claude-x",
        "tier": "frontier",
        "base_url": "https://host.example",
        "api_key": SECRET,
    }
    values.update(overrides)
    return PanelModel(**values)  # type: ignore[arg-type]


def _response(text: str) -> dict[str, object]:
    return {
        "text": text,
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        "cost": 0.00012,
        "request_id": f"panel-{text}",
    }


def test_panel_model_validates_tiers_and_never_serializes_raw_key() -> None:
    with pytest.raises(InvalidTierError):
        _model(tier="invalid")
    data = _model().to_dict()
    assert "api_key" not in data
    assert data["key_fingerprint"] == key_fingerprint(SECRET)
    assert SECRET not in repr(_model())


def test_panel_endpoint_inheritance_and_override() -> None:
    assert resolve_panel_endpoint(
        {"TEACHER_LLM_BASE_URL": "https://teacher", "TEACHER_LLM_API_KEY": "key"}
    ) == ("https://teacher", "key")
    assert resolve_panel_endpoint(
        {
            "TEACHER_LLM_BASE_URL": "https://teacher",
            "TEACHER_LLM_API_KEY": "teacher-key",
            "PANEL_LLM_BASE_URL": "https://panel",
            "PANEL_LLM_API_KEY": "panel-key",
        }
    ) == ("https://panel", "panel-key")
    assert {model.tier for model in build_panel("https://teacher", "key")} == {
        "weak",
        "mid",
        "frontier",
    }


async def test_rollouts_are_bounded_and_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    client = TeacherClient(
        base_url=model.base_url,
        api_key=model.api_key,
        model=model.model_string,
        authority_test_responses=[_response(str(i)) for i in range(4)],
    )
    monkeypatch.setattr(PanelModel, "client", lambda _self, **_kwargs: client)
    results = await run_rollouts("task", model, 4, concurrency=2)
    await client.aclose()

    assert [result.index for result in results] == [0, 1, 2, 3]
    assert [result.text for result in results] == ["0", "1", "2", "3"]
    assert all(result.usage.total_tokens == 15 for result in results)


async def test_rollout_failures_are_secret_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    client = TeacherClient(
        base_url=model.base_url,
        api_key=model.api_key,
        model=model.model_string,
        authority_test_responses=[{"crash": True}],
    )
    monkeypatch.setattr(PanelModel, "client", lambda _self, **_kwargs: client)
    result = (await run_rollouts("task", model, 1))[0]
    await client.aclose()

    assert result.error is not None
    assert SECRET not in result.error


async def test_zero_and_negative_rollout_counts_are_handled() -> None:
    assert await run_rollouts("task", _model(), 0) == []
    with pytest.raises(PanelError, match="non-negative"):
        await run_rollouts("task", _model(), -1)
