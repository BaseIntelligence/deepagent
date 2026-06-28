"""Unit tests for the LiteLLM teacher client + `forge llm-check` CLI.

All tests are offline: ``litellm.acompletion`` is mocked everywhere so no network
call is made. Live-endpoint exercises are gated behind ``@pytest.mark.integration``
and deselected from the milestone gate.

Covered invariants (validation contract VAL-LLM-005..018):
- single ``litellm.acompletion`` surface; ``drop_params`` on; ``max_tokens`` always
- ``api_base``/``api_key`` + per-call no-cache passed as args (no global mutable state)
- anthropic path host-only (never ``/v1``); openai path exactly one ``/v1``; unprefixed rejected
- text + usage + cost returned; structured json; normalized tool_calls; multi-turn agentic
- identical/repeat calls are independent (no cache dedup)
- fail-fast naming the missing var; the key is never echoed
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import litellm
import pytest
from typer.testing import CliRunner

from swe_forge.forge import teacher as teacher_mod
from swe_forge.forge.cli import app as forge_app
from swe_forge.forge.teacher import (
    MissingCredentialsError,
    ModelRoutingError,
    NormalizedToolCall,
    TeacherClient,
    Usage,
    normalize_base_url,
    resolve_routing,
    split_model,
)

runner = CliRunner()

SECRET = "sk-super-secret-do-not-print"


def _fake_response(
    *,
    content: str | None = "pong",
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
    cost: float | None = 0.00018,
    prompt_tokens: int = 16,
    completion_tokens: int = 4,
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    hidden = {} if cost is None else {"response_cost": cost}
    return SimpleNamespace(choices=[choice], usage=usage, _hidden_params=hidden)


def _tool_call(name: str, arguments: str, call_id: str = "call_1") -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _client(**overrides: Any) -> TeacherClient:
    params: dict[str, Any] = {
        "base_url": "https://host.example",
        "api_key": SECRET,
        "model": "anthropic/claude-x",
    }
    params.update(overrides)
    return TeacherClient(**params)


# --------------------------------------------------------------------------- #
# Routing + base-URL normalization (pure, no network)
# --------------------------------------------------------------------------- #


class TestRouting:
    def test_split_model_valid(self) -> None:
        assert split_model("anthropic/claude-opus-4-8") == (
            "anthropic",
            "claude-opus-4-8",
        )
        assert split_model("openai/gpt-4o-mini") == ("openai", "gpt-4o-mini")

    def test_unprefixed_model_rejected_with_clear_message(self) -> None:
        with pytest.raises(ModelRoutingError) as exc:
            split_model("gpt-4o-mini")
        assert "provider-prefixed" in str(exc.value)

    def test_empty_model_rejected(self) -> None:
        with pytest.raises(ModelRoutingError):
            split_model("")

    def test_anthropic_base_url_is_host_only(self) -> None:
        # trailing slash stripped, never /v1
        assert normalize_base_url("https://host/", "anthropic") == "https://host"
        assert normalize_base_url("https://host/v1", "anthropic") == "https://host"
        assert normalize_base_url("https://host/v1/", "anthropic") == "https://host"

    def test_openai_base_url_has_exactly_one_v1(self) -> None:
        assert normalize_base_url("https://host", "openai") == "https://host/v1"
        assert normalize_base_url("https://host/", "openai") == "https://host/v1"
        assert normalize_base_url("https://host/v1", "openai") == "https://host/v1"

    def test_resolve_routing_combines_model_and_base(self) -> None:
        r_anth = resolve_routing("anthropic/x", "https://host/")
        assert r_anth.provider == "anthropic"
        assert r_anth.api_base == "https://host"
        r_oai = resolve_routing("openai/x", "https://host")
        assert r_oai.provider == "openai"
        assert r_oai.api_base == "https://host/v1"


class TestUsage:
    def test_usage_addition(self) -> None:
        total = Usage(1, 2, 3) + Usage(10, 20, 30)
        assert total.to_dict() == {
            "prompt_tokens": 11,
            "completion_tokens": 22,
            "total_tokens": 33,
        }


# --------------------------------------------------------------------------- #
# Teacher client call contract (mocked litellm.acompletion)
# --------------------------------------------------------------------------- #


class TestTeacherCallContract:
    async def test_complete_text_returns_text_usage_cost(self) -> None:
        mock = AsyncMock(return_value=_fake_response())
        with patch.object(teacher_mod.litellm, "acompletion", mock):
            result = await _client().complete_text("hello")
        assert result.text == "pong"
        assert result.usage.total_tokens == 20
        assert result.cost == 0.00018

    async def test_call_kwargs_satisfy_contract(self) -> None:
        mock = AsyncMock(return_value=_fake_response())
        with patch.object(teacher_mod.litellm, "acompletion", mock):
            await _client(model="anthropic/claude-x", max_tokens=321).complete_text(
                "hi"
            )
        kwargs = mock.call_args.kwargs
        assert kwargs["model"] == "anthropic/claude-x"
        assert kwargs["api_base"] == "https://host.example"  # host-only, no /v1
        assert kwargs["api_key"] == SECRET
        assert kwargs["max_tokens"] == 321
        assert kwargs["cache"] == {"no-cache": True, "no-store": True}
        assert kwargs["num_retries"] >= 1
        assert "timeout" in kwargs

    async def test_anthropic_path_never_gets_v1(self) -> None:
        mock = AsyncMock(return_value=_fake_response())
        with patch.object(teacher_mod.litellm, "acompletion", mock):
            await _client(
                model="anthropic/x", base_url="https://host/v1/"
            ).complete_text("hi")
        assert mock.call_args.kwargs["api_base"] == "https://host"

    async def test_openai_path_gets_single_v1(self) -> None:
        mock = AsyncMock(return_value=_fake_response())
        with patch.object(teacher_mod.litellm, "acompletion", mock):
            await _client(model="openai/x", base_url="https://host").complete_text("hi")
        assert mock.call_args.kwargs["api_base"] == "https://host/v1"

    async def test_drop_params_enabled(self) -> None:
        assert litellm.drop_params is True

    async def test_no_global_litellm_credentials_or_cache_mutated(self) -> None:
        before = (
            getattr(litellm, "api_base", None),
            getattr(litellm, "api_key", None),
            getattr(litellm, "cache", None),
        )
        mock = AsyncMock(return_value=_fake_response())
        with patch.object(teacher_mod.litellm, "acompletion", mock):
            await _client().complete_text("hi")
        after = (
            getattr(litellm, "api_base", None),
            getattr(litellm, "api_key", None),
            getattr(litellm, "cache", None),
        )
        assert before == after

    async def test_repeat_calls_are_independent(self) -> None:
        mock = AsyncMock(
            side_effect=[
                _fake_response(content="123456", completion_tokens=6),
                _fake_response(content="987654", completion_tokens=6),
            ]
        )
        client = _client()
        with patch.object(teacher_mod.litellm, "acompletion", mock):
            first = await client.complete_text("Say a random 6-digit number")
            second = await client.complete_text("Say a random 6-digit number")
        assert mock.call_count == 2
        assert first.text != second.text
        assert first.usage.total_tokens > 0 and second.usage.total_tokens > 0

    async def test_cost_falls_back_to_zero_when_hidden_params_missing(self) -> None:
        mock = AsyncMock(return_value=_fake_response(cost=None))
        with (
            patch.object(teacher_mod.litellm, "acompletion", mock),
            patch.object(
                teacher_mod.litellm,
                "completion_cost",
                side_effect=Exception("no price"),
            ),
        ):
            result = await _client().complete_text("hi")
        assert result.cost == 0.0


class TestStructuredAndTools:
    async def test_complete_json_sets_response_format(self) -> None:
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        mock = AsyncMock(return_value=_fake_response(content='{"answer": "ok"}'))
        with patch.object(teacher_mod.litellm, "acompletion", mock):
            result = await _client().complete_json("q", schema)
        rf = mock.call_args.kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["schema"] == schema
        assert json.loads(result.text) == {"answer": "ok"}

    async def test_complete_with_tools_normalizes_tool_calls(self) -> None:
        tc = _tool_call("get_weather", '{"city": "Paris"}')
        mock = AsyncMock(
            return_value=_fake_response(
                content="", tool_calls=[tc], finish_reason="tool_calls"
            )
        )
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        with patch.object(teacher_mod.litellm, "acompletion", mock):
            result = await _client().complete_with_tools("weather?", tools)
        assert len(result.tool_calls) == 1
        call = result.tool_calls[0]
        assert call.name == "get_weather"
        assert call.arguments == {"city": "Paris"}

    async def test_tool_call_with_unparsable_arguments_is_graceful(self) -> None:
        tc = _tool_call("get_weather", "not-json")
        mock = AsyncMock(return_value=_fake_response(content="", tool_calls=[tc]))
        with patch.object(teacher_mod.litellm, "acompletion", mock):
            result = await _client().complete_with_tools("weather?", [])
        assert result.tool_calls[0].arguments == {"__unparsed__": "not-json"}


class TestAgenticTurn:
    async def test_two_step_tool_exchange(self) -> None:
        tc = _tool_call("get_weather", '{"city": "Paris"}')
        mock = AsyncMock(
            side_effect=[
                _fake_response(content="", tool_calls=[tc], finish_reason="tool_calls"),
                _fake_response(
                    content="It is 25C and sunny in Paris.", finish_reason="stop"
                ),
            ]
        )
        observed: list[str] = []

        def executor(call: NormalizedToolCall) -> str:
            observed.append(call.name)
            city = call.arguments["city"]
            return f"It is 25 degrees Celsius and sunny in {city}."

        with patch.object(teacher_mod.litellm, "acompletion", mock):
            result = await _client().agentic_turn(
                [{"role": "user", "content": "weather in Paris?"}],
                [{"type": "function", "function": {"name": "get_weather"}}],
                executor,
            )

        assert result.turns == 2
        assert mock.call_count == 2
        assert "Paris" in result.text
        assert observed == ["get_weather"]
        # The tool result was fed back on the second call.
        second_messages = mock.call_args_list[1].kwargs["messages"]
        assert any(m.get("role") == "tool" for m in second_messages)
        assert result.usage.total_tokens > 0

    async def test_async_tool_executor_supported(self) -> None:
        tc = _tool_call("get_weather", '{"city": "Paris"}')
        mock = AsyncMock(
            side_effect=[
                _fake_response(content="", tool_calls=[tc], finish_reason="tool_calls"),
                _fake_response(content="done in Paris", finish_reason="stop"),
            ]
        )

        async def executor(call: NormalizedToolCall) -> str:
            return "sunny"

        with patch.object(teacher_mod.litellm, "acompletion", mock):
            result = await _client().agentic_turn(
                [{"role": "user", "content": "weather?"}],
                [{"type": "function", "function": {"name": "get_weather"}}],
                executor,
            )
        assert result.turns == 2
        assert "Paris" in result.text


# --------------------------------------------------------------------------- #
# Fail-fast credential handling (no key echoed)
# --------------------------------------------------------------------------- #


class TestFailFast:
    async def test_missing_base_url_names_var_without_key(self) -> None:
        client = _client(base_url="")
        with pytest.raises(MissingCredentialsError) as exc:
            await client.complete_text("hi")
        message = str(exc.value)
        assert "TEACHER_LLM_BASE_URL" in message
        assert SECRET not in message

    async def test_missing_api_key_names_var_without_key(self) -> None:
        client = _client(api_key="")
        with pytest.raises(MissingCredentialsError) as exc:
            await client.complete_text("hi")
        message = str(exc.value)
        assert "TEACHER_LLM_API_KEY" in message
        assert SECRET not in message


# --------------------------------------------------------------------------- #
# CLI: `forge llm-check`
# --------------------------------------------------------------------------- #


class TestLlmCheckCli:
    def test_dry_run_anthropic_strips_v1_and_slash(self) -> None:
        with runner.isolated_filesystem(), patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(
                forge_app,
                [
                    "llm-check",
                    "--mode",
                    "anthropic",
                    "--dry-run",
                    "--base-url",
                    "https://host/",
                    "--json",
                ],
            )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["api_base"] == "https://host"

    def test_dry_run_openai_appends_v1(self) -> None:
        with runner.isolated_filesystem(), patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(
                forge_app,
                [
                    "llm-check",
                    "--mode",
                    "openai",
                    "--dry-run",
                    "--base-url",
                    "https://host",
                    "--json",
                ],
            )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["api_base"] == "https://host/v1"

    def test_unprefixed_model_rejected(self) -> None:
        with runner.isolated_filesystem(), patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(
                forge_app,
                [
                    "llm-check",
                    "--dry-run",
                    "--base-url",
                    "https://host",
                    "--model",
                    "x",
                ],
            )
        assert result.exit_code != 0
        assert "provider-prefixed" in result.output

    def test_missing_api_key_fails_fast_naming_var(self) -> None:
        env = {
            "TEACHER_LLM_BASE_URL": "https://host",
            "TEACHER_LLM_MODEL": "anthropic/x",
        }
        with runner.isolated_filesystem(), patch.dict(os.environ, env, clear=True):
            result = runner.invoke(forge_app, ["llm-check", "--json"])
        assert result.exit_code != 0
        assert "TEACHER_LLM_API_KEY" in result.output

    def test_missing_base_url_fails_fast_naming_var(self) -> None:
        env = {"TEACHER_LLM_API_KEY": SECRET, "TEACHER_LLM_MODEL": "anthropic/x"}
        with runner.isolated_filesystem(), patch.dict(os.environ, env, clear=True):
            result = runner.invoke(forge_app, ["llm-check", "--json"])
        assert result.exit_code != 0
        assert "TEACHER_LLM_BASE_URL" in result.output
        assert SECRET not in result.output

    def test_text_call_does_not_echo_secret(self) -> None:
        env = {
            "TEACHER_LLM_BASE_URL": "https://host",
            "TEACHER_LLM_API_KEY": SECRET,
            "TEACHER_LLM_MODEL": "anthropic/x",
        }
        mock = AsyncMock(return_value=_fake_response(content="pong"))
        with (
            runner.isolated_filesystem(),
            patch.dict(os.environ, env, clear=True),
            patch.object(teacher_mod.litellm, "acompletion", mock),
        ):
            result = runner.invoke(forge_app, ["llm-check", "--json"])
        assert result.exit_code == 0, result.output
        assert SECRET not in result.output
        payload = json.loads(result.output)
        assert payload["text"] == "pong"
        assert payload["usage"]["total_tokens"] > 0
        assert isinstance(payload["cost"], (int, float))

    def test_repeat_emits_two_independent_results(self) -> None:
        env = {
            "TEACHER_LLM_BASE_URL": "https://host",
            "TEACHER_LLM_API_KEY": SECRET,
            "TEACHER_LLM_MODEL": "anthropic/x",
        }
        mock = AsyncMock(
            side_effect=[
                _fake_response(content="111111"),
                _fake_response(content="222222"),
            ]
        )
        with (
            runner.isolated_filesystem(),
            patch.dict(os.environ, env, clear=True),
            patch.object(teacher_mod.litellm, "acompletion", mock),
        ):
            result = runner.invoke(forge_app, ["llm-check", "--repeat", "2", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert isinstance(payload, list) and len(payload) == 2
        assert mock.call_count == 2

    def test_tool_demo_emits_normalized_tool_calls(self) -> None:
        env = {
            "TEACHER_LLM_BASE_URL": "https://host",
            "TEACHER_LLM_API_KEY": SECRET,
            "TEACHER_LLM_MODEL": "anthropic/x",
        }
        tc = _tool_call("get_weather", '{"city": "Paris"}')
        mock = AsyncMock(
            return_value=_fake_response(
                content="", tool_calls=[tc], finish_reason="tool_calls"
            )
        )
        with (
            runner.isolated_filesystem(),
            patch.dict(os.environ, env, clear=True),
            patch.object(teacher_mod.litellm, "acompletion", mock),
        ):
            result = runner.invoke(
                forge_app,
                [
                    "llm-check",
                    "--tool-demo",
                    "weather",
                    "--prompt",
                    "weather in Paris?",
                ],
            )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["tool_calls"] == [
            {"name": "get_weather", "arguments": {"city": "Paris"}}
        ]

    def test_agentic_demo_completes_two_turns(self) -> None:
        env = {
            "TEACHER_LLM_BASE_URL": "https://host",
            "TEACHER_LLM_API_KEY": SECRET,
            "TEACHER_LLM_MODEL": "anthropic/x",
        }
        tc = _tool_call("get_weather", '{"city": "Paris"}')
        mock = AsyncMock(
            side_effect=[
                _fake_response(content="", tool_calls=[tc], finish_reason="tool_calls"),
                _fake_response(
                    content="It is 25 degrees and sunny in Paris.", finish_reason="stop"
                ),
            ]
        )
        with (
            runner.isolated_filesystem(),
            patch.dict(os.environ, env, clear=True),
            patch.object(teacher_mod.litellm, "acompletion", mock),
        ):
            result = runner.invoke(forge_app, ["llm-check", "--agentic-demo"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["turns"] >= 2
        assert "Paris" in payload["text"]
