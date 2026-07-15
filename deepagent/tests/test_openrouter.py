"""Offline unit tests for OpenRouter client (httpx mock, no live network)."""

from __future__ import annotations

import json
from decimal import Decimal

import httpx
import pytest

from swe_factory.openrouter import (
    ChatResult,
    OpenRouterAuthError,
    OpenRouterBillingError,
    OpenRouterClient,
    OpenRouterError,
    ScriptedChatClient,
    TokenUsage,
)


def test_scripted_client_records_calls() -> None:
    client = ScriptedChatClient(
        responses=[
            ChatResult(
                model="x-ai/grok-4.5",
                text="diff",
                usage=TokenUsage(1, 2, 3),
                request_id="id-1",
                cost_usd=Decimal("0.01"),
                finish_reason="stop",
                raw_usage={},
            )
        ]
    )
    result = client.complete(
        model="x-ai/grok-4.5",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert result.text == "diff"
    assert client.calls is not None
    assert client.calls[0]["model"] == "x-ai/grok-4.5"


def test_openrouter_client_parses_usage_and_cost() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "Authorization" in request.headers
        assert request.headers["Authorization"].startswith("Bearer ")
        body = json.loads(request.content.decode())
        assert body["model"] == "anthropic/claude-opus-4.8"
        return httpx.Response(
            200,
            json={
                "id": "gen-abc",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "hello"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                    "cost": 0.0025,
                },
            },
        )

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    client = OpenRouterClient(api_key="sk-or-test", http_client=http)
    result = client.complete(
        model="anthropic/claude-opus-4.8",
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=16,
    )
    assert result.text == "hello"
    assert result.request_id == "gen-abc"
    assert result.usage.total_tokens == 18
    assert result.cost_usd == Decimal("0.0025")
    client.close()


def test_openrouter_auth_error_on_401() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad"})

    transport = httpx.MockTransport(handler)
    client = OpenRouterClient(api_key="sk-or-bad", http_client=httpx.Client(transport=transport))
    with pytest.raises(OpenRouterAuthError):
        client.complete(
            model="x-ai/grok-4.5",
            messages=[{"role": "user", "content": "x"}],
        )
    client.close()


def test_openrouter_missing_key_raises() -> None:
    with pytest.raises(OpenRouterAuthError):
        OpenRouterClient(api_key="   ")


def test_fetch_generation_cost() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "generation" in str(request.url)
        return httpx.Response(200, json={"data": {"total_cost": 0.0042}})

    client = OpenRouterClient(
        api_key="sk-or-test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    cost = client.fetch_generation_cost("gen-xyz")
    assert cost == Decimal("0.0042")
    client.close()


def test_fetch_generation_cost_missing_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {}})

    client = OpenRouterClient(
        api_key="sk-or-test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(OpenRouterBillingError):
        client.fetch_generation_cost("gen-xyz")
    client.close()


def test_http_error_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = OpenRouterClient(
        api_key="sk-or-test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(OpenRouterError, match="HTTP 500"):
        client.complete(
            model="x-ai/grok-4.5",
            messages=[{"role": "user", "content": "x"}],
        )
    client.close()
