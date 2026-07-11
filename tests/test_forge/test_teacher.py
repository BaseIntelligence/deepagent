"""Offline tests for the child-owned Forge teacher transport."""

from __future__ import annotations

import json
from typing import Any

import pytest

from swe_forge.forge import teacher as teacher_mod
from swe_forge.forge.teacher import (
    FORCED_TOOL_CHOICE,
    MissingCredentialsError,
    ModelRoutingError,
    NormalizedToolCall,
    TeacherClient,
    normalize_base_url,
    resolve_routing,
    split_model,
)

SECRET = "sk-super-secret-do-not-print"


def _response(
    text: str = "pong", *, tool_calls: list[dict[str, object]] | None = None
) -> dict[str, object]:
    return {
        "text": text,
        "usage": {"prompt_tokens": 16, "completion_tokens": 4, "total_tokens": 20},
        "cost": 0.00018,
        "request_id": "test-request",
        "tool_calls": tool_calls or [],
    }


def _client(*responses: dict[str, object], **overrides: Any) -> TeacherClient:
    values: dict[str, Any] = {
        "base_url": "https://host.example",
        "api_key": SECRET,
        "model": "anthropic/claude-x",
        "authority_test_responses": list(responses or (_response(),)),
    }
    values.update(overrides)
    return TeacherClient(**values)


def test_routing_stays_provider_prefixed_and_endpoint_normalized() -> None:
    assert split_model("anthropic/claude") == ("anthropic", "claude")
    assert normalize_base_url("https://host/v1/", "anthropic") == "https://host"
    assert normalize_base_url("https://host", "openai") == "https://host/v1"
    assert resolve_routing("openai/x", "https://host").api_base == "https://host/v1"
    with pytest.raises(ModelRoutingError, match="provider-prefixed"):
        split_model("unprefixed")


async def test_child_authority_returns_normalized_text_usage_cost_and_receipt() -> None:
    client = _client(_response("child response"))
    result = await client.complete_text("hello")
    await client.aclose()

    assert result.text == "child response"
    assert result.usage.total_tokens == 20
    assert result.cost == pytest.approx(0.00018)
    assert result.transport_receipt is None  # non-gate calls receive no evidence
    assert "litellm" not in teacher_mod.__dict__


async def test_json_and_tool_results_are_normalized_from_child_response() -> None:
    tool = {
        "id": "call_1",
        "name": "get_weather",
        "arguments": {"city": "Paris"},
        "raw_arguments": '{"city":"Paris"}',
    }
    client = _client(_response('{"answer":"ok"}'), _response("", tool_calls=[tool]))
    structured = await client.complete_json(
        "question", {"type": "object", "properties": {"answer": {"type": "string"}}}
    )
    tools = await client.complete_with_tools("weather", [{"type": "function"}])
    await client.aclose()

    assert json.loads(structured.text) == {"answer": "ok"}
    assert tools.tool_calls == [
        NormalizedToolCall(
            id="call_1",
            name="get_weather",
            arguments={"city": "Paris"},
            raw_arguments='{"city":"Paris"}',
        )
    ]
    assert FORCED_TOOL_CHOICE == "required"


async def test_agentic_turn_uses_independent_child_transports() -> None:
    tool = {
        "id": "call_1",
        "name": "get_weather",
        "arguments": {"city": "Paris"},
        "raw_arguments": '{"city":"Paris"}',
    }
    client = _client(_response("", tool_calls=[tool]), _response("Paris is sunny"))
    observed: list[str] = []

    def executor(call: NormalizedToolCall) -> str:
        observed.append(call.name)
        return "sunny"

    result = await client.agentic_turn(
        [{"role": "user", "content": "weather"}],
        [{"type": "function"}],
        executor,
    )
    await client.aclose()

    assert result.turns == 2
    assert result.text == "Paris is sunny"
    assert observed == ["get_weather"]
    assert result.usage.total_tokens == 40


async def test_missing_credentials_fail_before_authority_start() -> None:
    with pytest.raises(MissingCredentialsError, match="TEACHER_LLM_BASE_URL"):
        await _client(base_url="").complete_text("hello")
    with pytest.raises(MissingCredentialsError, match="TEACHER_LLM_API_KEY"):
        await _client(api_key="").complete_text("hello")
