"""Unit tests for the calibration panel + `forge panel-*` CLI (offline, mocked).

All LLM access is mocked (``litellm.acompletion`` patched) so no network call is
made. Live-endpoint exercises are gated behind ``@pytest.mark.integration``.

Covered invariants (validation contract VAL-LLM-013..015, 019):
- a panel carries >=2 tiered, distinct model ids sharing the teacher endpoint by
  default and honoring ``PANEL_LLM_*`` overrides when set;
- model ids are validated with a single live probe before any bulk rollout;
- ``run_rollouts`` returns exactly k independent (uncached) results, each with its
  own usage/cost, under a respected concurrency cap.
"""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from swe_forge.forge import teacher as teacher_mod
from swe_forge.forge.cli import app as forge_app
from swe_forge.forge.panel import (
    DEFAULT_PANEL_SPECS,
    VALID_TIERS,
    InvalidTierError,
    PanelModel,
    build_panel,
    build_panel_from_env,
    resolve_panel_endpoint,
    run_rollouts,
    select_default_model,
    validate_model,
    validate_models,
)
from swe_forge.forge.secrets import key_fingerprint

runner = CliRunner()

SECRET = "sk-super-secret-do-not-print"


def _fake_response(
    *,
    content: str | None = "ok",
    cost: float | None = 0.00012,
    prompt_tokens: int = 12,
    completion_tokens: int = 3,
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    hidden = {} if cost is None else {"response_cost": cost}
    return SimpleNamespace(choices=[choice], usage=usage, _hidden_params=hidden)


def _route_good_or_raise(*good_substrings: str) -> Any:
    """A side_effect that returns a response for known ids and raises otherwise."""

    def _se(**kwargs: Any) -> SimpleNamespace:
        model = kwargs.get("model", "")
        if any(sub in model for sub in good_substrings):
            return _fake_response()
        raise RuntimeError("NotFoundError: unknown model id")

    return _se


def _model(**overrides: Any) -> PanelModel:
    params: dict[str, Any] = {
        "id": "m",
        "model_string": "anthropic/claude-x",
        "tier": "frontier",
        "base_url": "https://host.example",
        "api_key": SECRET,
    }
    params.update(overrides)
    return PanelModel(**params)


# --------------------------------------------------------------------------- #
# PanelModel
# --------------------------------------------------------------------------- #


class TestPanelModel:
    def test_invalid_tier_rejected(self) -> None:
        with pytest.raises(InvalidTierError):
            _model(tier="genius")

    def test_valid_tiers_accepted(self) -> None:
        for tier in VALID_TIERS:
            assert _model(tier=tier).tier == tier

    def test_routing_anthropic_is_host_only(self) -> None:
        m = _model(model_string="anthropic/x", base_url="https://host/v1/")
        assert m.routing.api_base == "https://host"

    def test_routing_openai_appends_v1(self) -> None:
        m = _model(model_string="openai/x", base_url="https://host")
        assert m.routing.api_base == "https://host/v1"

    def test_to_dict_omits_api_key_by_default(self) -> None:
        data = _model().to_dict()
        assert "api_key" not in data
        assert data["model_string"] == "anthropic/claude-x"
        # exposes a non-reversible fingerprint instead of the raw key
        assert data["key_fingerprint"] == key_fingerprint(SECRET)
        assert SECRET not in data["key_fingerprint"]

    def test_to_dict_includes_api_key_when_requested(self) -> None:
        data = _model().to_dict(include_api_key=True)
        assert data["api_key"] == SECRET
        # the fingerprint is still present alongside the in-process raw key
        assert data["key_fingerprint"] == key_fingerprint(SECRET)

    def test_key_fingerprint_property_is_non_reversible(self) -> None:
        m = _model()
        assert m.key_fingerprint == key_fingerprint(SECRET)
        assert SECRET not in m.key_fingerprint

    def test_repr_never_leaks_api_key(self) -> None:
        assert SECRET not in repr(_model())


# --------------------------------------------------------------------------- #
# Panel registry + endpoint resolution
# --------------------------------------------------------------------------- #


class TestPanelRegistry:
    def test_default_panel_has_distinct_tiered_models(self) -> None:
        panel = build_panel("https://teacher.example", "sk-teacher")
        assert len(panel) >= 2
        model_strings = [m.model_string for m in panel]
        assert len(set(model_strings)) == len(model_strings)  # all distinct
        assert all(m.tier in VALID_TIERS for m in panel)
        # tiers span weak/mid/frontier for discrimination downstream
        assert {m.tier for m in panel} == set(VALID_TIERS)

    def test_default_specs_are_provider_prefixed(self) -> None:
        for spec in DEFAULT_PANEL_SPECS:
            assert "/" in spec.model_string

    def test_resolve_endpoint_inherits_teacher_when_panel_unset(self) -> None:
        env = {
            "TEACHER_LLM_BASE_URL": "https://teacher.example",
            "TEACHER_LLM_API_KEY": "sk-teacher",
        }
        base_url, api_key = resolve_panel_endpoint(env)
        assert base_url == "https://teacher.example"
        assert api_key == "sk-teacher"

    def test_resolve_endpoint_honors_panel_override(self) -> None:
        env = {
            "TEACHER_LLM_BASE_URL": "https://teacher.example",
            "TEACHER_LLM_API_KEY": "sk-teacher",
            "PANEL_LLM_BASE_URL": "https://panel.example",
            "PANEL_LLM_API_KEY": "sk-panel",
        }
        base_url, api_key = resolve_panel_endpoint(env)
        assert base_url == "https://panel.example"
        assert api_key == "sk-panel"

    def test_build_from_env_inherits_teacher(self) -> None:
        env = {
            "TEACHER_LLM_BASE_URL": "https://teacher.example",
            "TEACHER_LLM_API_KEY": "sk-teacher",
        }
        panel = build_panel_from_env(env=env)
        assert all(m.base_url == "https://teacher.example" for m in panel)
        assert all(m.api_key == "sk-teacher" for m in panel)

    def test_build_from_env_honors_override(self) -> None:
        env = {
            "TEACHER_LLM_BASE_URL": "https://teacher.example",
            "TEACHER_LLM_API_KEY": "sk-teacher",
            "PANEL_LLM_BASE_URL": "https://panel.example",
            "PANEL_LLM_API_KEY": "sk-panel",
        }
        panel = build_panel_from_env(env=env)
        assert all(m.base_url == "https://panel.example" for m in panel)
        assert all(m.api_key == "sk-panel" for m in panel)

    def test_select_default_model_prefers_tier(self) -> None:
        panel = build_panel("https://h", "k")
        assert select_default_model(panel, tier="frontier").tier == "frontier"
        assert select_default_model(panel, tier="weak").tier == "weak"


# --------------------------------------------------------------------------- #
# Live model-id validation (mocked) - one probe, no bulk
# --------------------------------------------------------------------------- #


class TestValidation:
    async def test_good_id_is_valid(self) -> None:
        mock = AsyncMock(return_value=_fake_response())
        with patch.object(teacher_mod.litellm, "acompletion", mock):
            result = await validate_model(
                "anthropic/claude-opus-4-8",
                base_url="https://host",
                api_key=SECRET,
            )
        assert result.valid is True
        assert result.usage is not None and result.usage.total_tokens > 0
        assert mock.call_count == 1  # single probe

    async def test_bogus_id_is_invalid_without_crash(self) -> None:
        mock = AsyncMock(side_effect=_route_good_or_raise("claude-opus-4-8"))
        with patch.object(teacher_mod.litellm, "acompletion", mock):
            result = await validate_model(
                "anthropic/definitely-not-a-real-model-xyz",
                base_url="https://host",
                api_key=SECRET,
            )
        assert result.valid is False
        assert result.error and "NotFoundError" in result.error
        assert SECRET not in (result.error or "")

    async def test_unprefixed_model_is_invalid(self) -> None:
        mock = AsyncMock(return_value=_fake_response())
        with patch.object(teacher_mod.litellm, "acompletion", mock):
            result = await validate_model(
                "claude-opus-4-8", base_url="https://host", api_key=SECRET
            )
        assert result.valid is False
        assert "provider-prefixed" in (result.error or "")
        assert mock.call_count == 0  # rejected before any call

    async def test_validate_models_marks_good_and_bogus(self) -> None:
        mock = AsyncMock(side_effect=_route_good_or_raise("claude-opus-4-8"))
        with patch.object(teacher_mod.litellm, "acompletion", mock):
            results = await validate_models(
                [
                    "anthropic/claude-opus-4-8",
                    "anthropic/definitely-not-a-real-model-xyz",
                ],
                base_url="https://host",
                api_key=SECRET,
            )
        assert results[0].valid is True
        assert results[1].valid is False
        # one probe per id; never a k-burst on the bogus id
        assert mock.call_count == 2


# --------------------------------------------------------------------------- #
# run_rollouts - k independent, uncached, concurrency-bounded
# --------------------------------------------------------------------------- #


class TestRunRollouts:
    async def test_returns_exactly_k_results(self) -> None:
        mock = AsyncMock(return_value=_fake_response())
        with patch.object(teacher_mod.litellm, "acompletion", mock):
            results = await run_rollouts("task", _model(), 5)
        assert len(results) == 5
        assert mock.call_count == 5  # k separate calls (no cache dedup)

    async def test_each_result_has_own_usage_and_cost(self) -> None:
        mock = AsyncMock(
            side_effect=[
                _fake_response(content="111111", completion_tokens=6, cost=0.001),
                _fake_response(content="222222", completion_tokens=7, cost=0.002),
                _fake_response(content="333333", completion_tokens=8, cost=0.003),
            ]
        )
        with patch.object(teacher_mod.litellm, "acompletion", mock):
            results = await run_rollouts("task", _model(), 3)
        assert [r.text for r in results] == ["111111", "222222", "333333"]
        assert [r.cost for r in results] == [0.001, 0.002, 0.003]
        assert all(r.usage.total_tokens > 0 for r in results)
        # independent usage records (distinct completion token counts)
        assert len({r.usage.completion_tokens for r in results}) == 3

    async def test_results_are_index_ordered(self) -> None:
        mock = AsyncMock(return_value=_fake_response())
        with patch.object(teacher_mod.litellm, "acompletion", mock):
            results = await run_rollouts("task", _model(), 4)
        assert [r.index for r in results] == [0, 1, 2, 3]

    async def test_concurrency_cap_is_respected(self) -> None:
        state = {"current": 0, "max": 0}

        async def fake_acompletion(**kwargs: Any) -> SimpleNamespace:
            state["current"] += 1
            state["max"] = max(state["max"], state["current"])
            await asyncio.sleep(0.02)
            state["current"] -= 1
            return _fake_response()

        with patch.object(teacher_mod.litellm, "acompletion", new=fake_acompletion):
            results = await run_rollouts("task", _model(), 8, concurrency=3)
        assert len(results) == 8
        assert state["max"] <= 3  # cap never exceeded
        assert state["max"] >= 2  # genuinely concurrent

    async def test_k_zero_returns_empty(self) -> None:
        mock = AsyncMock(return_value=_fake_response())
        with patch.object(teacher_mod.litellm, "acompletion", mock):
            results = await run_rollouts("task", _model(), 0)
        assert results == []
        assert mock.call_count == 0

    async def test_failing_rollout_recorded_without_aborting_batch(self) -> None:
        mock = AsyncMock(
            side_effect=[
                _fake_response(content="a"),
                RuntimeError("APIError: transient"),
                _fake_response(content="c"),
            ]
        )
        with patch.object(teacher_mod.litellm, "acompletion", mock):
            results = await run_rollouts("task", _model(), 3, concurrency=1)
        assert len(results) == 3
        assert results[1].error and "APIError" in results[1].error
        assert results[0].text == "a" and results[2].text == "c"

    async def test_rollout_error_redacts_secret(self) -> None:
        mock = AsyncMock(side_effect=RuntimeError(f"boom key={SECRET}"))
        with patch.object(teacher_mod.litellm, "acompletion", mock):
            results = await run_rollouts("task", _model(), 1)
        assert results[0].error is not None
        assert SECRET not in results[0].error


# --------------------------------------------------------------------------- #
# CLI: panel-info / panel-validate / panel-rollouts
# --------------------------------------------------------------------------- #

_TEACHER_ENV = {
    "TEACHER_LLM_BASE_URL": "https://teacher.example",
    "TEACHER_LLM_API_KEY": "sk-teacher",
}
_OVERRIDE_ENV = {
    **_TEACHER_ENV,
    "PANEL_LLM_BASE_URL": "https://panel.example",
    "PANEL_LLM_API_KEY": "sk-panel",
}


class TestPanelInfoCli:
    def test_inherits_teacher_endpoint(self) -> None:
        with (
            runner.isolated_filesystem(),
            patch.dict(os.environ, _TEACHER_ENV, clear=True),
        ):
            result = runner.invoke(forge_app, ["panel-info", "--json"])
        assert result.exit_code == 0, result.output
        models = json.loads(result.output)
        assert len(models) >= 2
        assert len({m["model_string"] for m in models}) == len(models)
        assert all(m["tier"] in VALID_TIERS for m in models)
        assert all(m["base_url"] == "https://teacher.example" for m in models)
        # raw key is NEVER emitted; inheritance is proven via base_url + fingerprint
        assert all("api_key" not in m for m in models)
        teacher_fp = key_fingerprint("sk-teacher")
        assert all(m["key_fingerprint"] == teacher_fp for m in models)

    def test_panel_override_is_honored(self) -> None:
        with (
            runner.isolated_filesystem(),
            patch.dict(os.environ, _OVERRIDE_ENV, clear=True),
        ):
            result = runner.invoke(forge_app, ["panel-info", "--json"])
        assert result.exit_code == 0, result.output
        models = json.loads(result.output)
        assert all(m["base_url"] == "https://panel.example" for m in models)
        # raw key never emitted; override divergence proven via base_url + fingerprint
        assert all("api_key" not in m for m in models)
        panel_fp = key_fingerprint("sk-panel")
        teacher_fp = key_fingerprint("sk-teacher")
        assert all(m["key_fingerprint"] == panel_fp for m in models)
        # diverges from the teacher endpoint
        assert all(m["base_url"] != "https://teacher.example" for m in models)
        assert all(m["key_fingerprint"] != teacher_fp for m in models)

    def test_human_view_does_not_echo_secret(self) -> None:
        env = {
            "TEACHER_LLM_BASE_URL": "https://teacher.example",
            "TEACHER_LLM_API_KEY": SECRET,
        }
        with runner.isolated_filesystem(), patch.dict(os.environ, env, clear=True):
            result = runner.invoke(forge_app, ["panel-info"])
        assert result.exit_code == 0, result.output
        assert SECRET not in result.output


class TestPanelValidateCli:
    def test_good_and_bogus_ids(self) -> None:
        mock = AsyncMock(side_effect=_route_good_or_raise("claude-opus-4-8"))
        with (
            runner.isolated_filesystem(),
            patch.dict(
                os.environ, {**_TEACHER_ENV, "TEACHER_LLM_API_KEY": SECRET}, clear=True
            ),
            patch.object(teacher_mod.litellm, "acompletion", mock),
        ):
            result = runner.invoke(
                forge_app,
                [
                    "panel-validate",
                    "--models",
                    "anthropic/claude-opus-4-8,anthropic/definitely-not-a-real-model-xyz",
                    "--json",
                ],
            )
        # status reflects the invalid id
        assert result.exit_code == 1
        # JSON is still emitted (last stdout line)
        payload = json.loads(result.output.strip().splitlines()[-1])
        assert payload[0]["model"] == "anthropic/claude-opus-4-8"
        assert payload[0]["valid"] is True
        assert payload[1]["valid"] is False
        assert payload[1]["error"]
        # only one probe per id (no k-burst on the bogus id)
        assert mock.call_count == 2
        assert SECRET not in result.output

    def test_all_valid_exits_zero(self) -> None:
        mock = AsyncMock(return_value=_fake_response())
        with (
            runner.isolated_filesystem(),
            patch.dict(os.environ, _TEACHER_ENV, clear=True),
            patch.object(teacher_mod.litellm, "acompletion", mock),
        ):
            result = runner.invoke(
                forge_app,
                ["panel-validate", "--models", "anthropic/a,openai/b", "--json"],
            )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert all(r["valid"] for r in payload)

    def test_missing_creds_fails_fast_naming_var(self) -> None:
        with runner.isolated_filesystem(), patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(
                forge_app, ["panel-validate", "--models", "anthropic/a"]
            )
        assert result.exit_code != 0
        assert "TEACHER_LLM_BASE_URL" in result.output


class TestPanelRolloutsCli:
    def test_emits_k_independent_results(self) -> None:
        mock = AsyncMock(
            side_effect=[
                _fake_response(content="probe"),  # validation probe
                _fake_response(content="111111", cost=0.001),
                _fake_response(content="222222", cost=0.002),
                _fake_response(content="333333", cost=0.003),
            ]
        )
        with (
            runner.isolated_filesystem(),
            patch.dict(
                os.environ, {**_TEACHER_ENV, "TEACHER_LLM_API_KEY": SECRET}, clear=True
            ),
            patch.object(teacher_mod.litellm, "acompletion", mock),
        ):
            result = runner.invoke(forge_app, ["panel-rollouts", "--k", "3", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert isinstance(payload, list) and len(payload) == 3
        assert all(r["usage"]["total_tokens"] > 0 for r in payload)
        assert len({r["cost"] for r in payload}) == 3  # each its own cost
        assert SECRET not in result.output

    def test_no_validate_skips_probe(self) -> None:
        mock = AsyncMock(return_value=_fake_response())
        with (
            runner.isolated_filesystem(),
            patch.dict(os.environ, _TEACHER_ENV, clear=True),
            patch.object(teacher_mod.litellm, "acompletion", mock),
        ):
            result = runner.invoke(
                forge_app, ["panel-rollouts", "--k", "2", "--no-validate", "--json"]
            )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert len(payload) == 2
        assert mock.call_count == 2  # no extra validation probe

    def test_invalid_model_aborts_before_bulk(self) -> None:
        mock = AsyncMock(side_effect=_route_good_or_raise("claude-opus-4-8"))
        with (
            runner.isolated_filesystem(),
            patch.dict(os.environ, _TEACHER_ENV, clear=True),
            patch.object(teacher_mod.litellm, "acompletion", mock),
        ):
            result = runner.invoke(
                forge_app,
                ["panel-rollouts", "--k", "5", "--model", "anthropic/bogus-xyz"],
            )
        assert result.exit_code != 0
        # only the single validation probe ran, never the k-burst
        assert mock.call_count == 1

    def test_missing_creds_fails_fast(self) -> None:
        with runner.isolated_filesystem(), patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(forge_app, ["panel-rollouts", "--k", "2"])
        assert result.exit_code != 0
        assert "TEACHER_LLM_BASE_URL" in result.output


# --------------------------------------------------------------------------- #
# Secret hardening: no CLI path leaks the raw key; inheritance via fingerprint
# --------------------------------------------------------------------------- #

_LIVE_ENV = {
    "TEACHER_LLM_BASE_URL": "https://teacher.example",
    "TEACHER_LLM_API_KEY": SECRET,
    "TEACHER_LLM_MODEL": "anthropic/claude-x",
    "TEACHER_LLM_PROVIDER": "anthropic",
}


class TestSecretHardeningNoLeak:
    """No panel/teacher CLI path may emit the raw TEACHER_LLM_API_KEY value."""

    def test_panel_info_json_does_not_leak_key(self) -> None:
        with (
            runner.isolated_filesystem(),
            patch.dict(os.environ, _LIVE_ENV, clear=True),
        ):
            result = runner.invoke(forge_app, ["panel-info", "--json"])
        assert result.exit_code == 0, result.output
        assert SECRET not in result.output

    def test_panel_info_human_does_not_leak_key(self) -> None:
        with (
            runner.isolated_filesystem(),
            patch.dict(os.environ, _LIVE_ENV, clear=True),
        ):
            result = runner.invoke(forge_app, ["panel-info"])
        assert result.exit_code == 0, result.output
        assert SECRET not in result.output

    def test_panel_validate_json_does_not_leak_key(self) -> None:
        mock = AsyncMock(return_value=_fake_response())
        with (
            runner.isolated_filesystem(),
            patch.dict(os.environ, _LIVE_ENV, clear=True),
            patch.object(teacher_mod.litellm, "acompletion", mock),
        ):
            result = runner.invoke(
                forge_app, ["panel-validate", "--models", "anthropic/a", "--json"]
            )
        assert result.exit_code == 0, result.output
        assert SECRET not in result.output

    def test_panel_validate_error_path_does_not_leak_key(self) -> None:
        # The endpoint error message itself echoes the key; it must be redacted.
        mock = AsyncMock(side_effect=RuntimeError(f"auth failed for key={SECRET}"))
        with (
            runner.isolated_filesystem(),
            patch.dict(os.environ, _LIVE_ENV, clear=True),
            patch.object(teacher_mod.litellm, "acompletion", mock),
        ):
            result = runner.invoke(
                forge_app, ["panel-validate", "--models", "anthropic/a", "--json"]
            )
        # an invalid probe exits non-zero but must never print the raw key
        assert SECRET not in result.output

    def test_panel_rollouts_json_does_not_leak_key(self) -> None:
        mock = AsyncMock(return_value=_fake_response())
        with (
            runner.isolated_filesystem(),
            patch.dict(os.environ, _LIVE_ENV, clear=True),
            patch.object(teacher_mod.litellm, "acompletion", mock),
        ):
            result = runner.invoke(forge_app, ["panel-rollouts", "--k", "1", "--json"])
        assert result.exit_code == 0, result.output
        assert SECRET not in result.output

    def test_panel_rollouts_error_path_does_not_leak_key(self) -> None:
        mock = AsyncMock(side_effect=RuntimeError(f"boom key={SECRET}"))
        with (
            runner.isolated_filesystem(),
            patch.dict(os.environ, _LIVE_ENV, clear=True),
            patch.object(teacher_mod.litellm, "acompletion", mock),
        ):
            result = runner.invoke(
                forge_app, ["panel-rollouts", "--k", "1", "--no-validate", "--json"]
            )
        assert SECRET not in result.output

    def test_teacher_info_does_not_leak_key(self) -> None:
        with (
            runner.isolated_filesystem(),
            patch.dict(os.environ, _LIVE_ENV, clear=True),
        ):
            result = runner.invoke(forge_app, ["info"])
        assert result.exit_code == 0, result.output
        assert SECRET not in result.output

    def test_teacher_llm_check_dry_run_does_not_leak_key(self) -> None:
        with (
            runner.isolated_filesystem(),
            patch.dict(os.environ, _LIVE_ENV, clear=True),
        ):
            result = runner.invoke(forge_app, ["llm-check", "--dry-run", "--json"])
        assert result.exit_code == 0, result.output
        assert SECRET not in result.output

    def test_teacher_llm_check_error_path_does_not_leak_key(self) -> None:
        mock = AsyncMock(side_effect=RuntimeError(f"upstream rejected key={SECRET}"))
        with (
            runner.isolated_filesystem(),
            patch.dict(os.environ, _LIVE_ENV, clear=True),
            patch.object(teacher_mod.litellm, "acompletion", mock),
        ):
            result = runner.invoke(forge_app, ["llm-check"])
        assert result.exit_code != 0
        assert SECRET not in result.output


class TestSecretHardeningFingerprint:
    """Fingerprint proves inheritance/override without exposing the key."""

    def test_panel_fingerprint_equals_teacher_when_inherited(self) -> None:
        with (
            runner.isolated_filesystem(),
            patch.dict(os.environ, _LIVE_ENV, clear=True),
        ):
            result = runner.invoke(forge_app, ["panel-info", "--json"])
        assert result.exit_code == 0, result.output
        models = json.loads(result.output)
        teacher_fp = key_fingerprint(SECRET)
        assert teacher_fp != ""
        # every inherited panel model matches the teacher base_url + fingerprint
        assert all(m["base_url"] == "https://teacher.example" for m in models)
        assert all(m["key_fingerprint"] == teacher_fp for m in models)

    def test_panel_fingerprint_differs_under_override(self) -> None:
        override_env = {
            **_LIVE_ENV,
            "PANEL_LLM_BASE_URL": "https://panel.example",
            "PANEL_LLM_API_KEY": "sk-panel-override",
        }
        with (
            runner.isolated_filesystem(),
            patch.dict(os.environ, override_env, clear=True),
        ):
            result = runner.invoke(forge_app, ["panel-info", "--json"])
        assert result.exit_code == 0, result.output
        models = json.loads(result.output)
        teacher_fp = key_fingerprint(SECRET)
        override_fp = key_fingerprint("sk-panel-override")
        assert all(m["base_url"] == "https://panel.example" for m in models)
        assert all(m["key_fingerprint"] == override_fp for m in models)
        # the override diverges from the teacher endpoint + key
        assert override_fp != teacher_fp
        assert all(m["key_fingerprint"] != teacher_fp for m in models)
        # and the override secret value is never printed
        assert "sk-panel-override" not in result.output
