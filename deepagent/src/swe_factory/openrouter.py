"""OpenRouter chat client (OpenAI-compatible HTTP) for factory teacher/panel.

Uses httpx only (already a package dependency). Secrets are never logged.
Physical spend is expected to go through :class:`BudgetLedger` reserve/settle
outside this module — the client returns usage + optional exact cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

import httpx

from swe_factory.config import (
    DEFAULT_OPENROUTER_BASE_URL,
    FactorySettings,
    load_settings,
)


class OpenRouterError(RuntimeError):
    """Base error for OpenRouter HTTP transport."""


class OpenRouterAuthError(OpenRouterError):
    """Missing or rejected API credentials."""


class OpenRouterBillingError(OpenRouterError):
    """Provider metering missing or unusable for exact settlement."""


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class ChatResult:
    """Normalized chat completion (no secrets, no full key material)."""

    model: str
    text: str
    usage: TokenUsage
    request_id: str
    cost_usd: Decimal | None
    finish_reason: str | None
    raw_usage: dict[str, Any]


class ChatClient(Protocol):
    """Injectable completion surface for offline panel tests."""

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ChatResult: ...


def _parse_usage(payload: dict[str, Any]) -> tuple[TokenUsage, dict[str, Any], Decimal | None]:
    raw_usage = payload.get("usage")
    usage_map: dict[str, Any] = dict(raw_usage) if isinstance(raw_usage, dict) else {}
    prompt = int(usage_map.get("prompt_tokens") or 0)
    completion = int(usage_map.get("completion_tokens") or 0)
    total = int(usage_map.get("total_tokens") or (prompt + completion))
    usage = TokenUsage(
        prompt_tokens=max(0, prompt),
        completion_tokens=max(0, completion),
        total_tokens=max(0, total),
    )
    cost: Decimal | None = None
    for key in ("cost", "total_cost"):
        if key in usage_map and usage_map[key] is not None:
            try:
                cost = Decimal(str(usage_map[key]))
                if cost.is_finite() and cost >= 0:
                    return usage, usage_map, cost
            except (InvalidOperation, ValueError):
                pass
    # Some OpenRouter responses nest cost under usage.cost
    return usage, usage_map, cost


def _extract_text(payload: dict[str, Any]) -> tuple[str, str | None]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", None
    first_raw = choices[0]
    first: dict[str, Any] = first_raw if isinstance(first_raw, dict) else {}
    finish = first.get("finish_reason")
    raw_message = first.get("message")
    message: dict[str, Any] = raw_message if isinstance(raw_message, dict) else {}
    content = message.get("content")
    if isinstance(content, str):
        return content, finish if isinstance(finish, str) else None
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts), finish if isinstance(finish, str) else None
    # tool/legacy form
    if isinstance(first.get("text"), str):
        return first["text"], finish if isinstance(finish, str) else None
    return "", finish if isinstance(finish, str) else None


class OpenRouterClient:
    """Synchronous OpenRouter chat completions client."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        timeout: float = 120.0,
        http_client: httpx.Client | None = None,
        app_title: str = "deepagent",
    ) -> None:
        key = api_key.strip()
        if not key:
            raise OpenRouterAuthError("OPENROUTER_API_KEY is missing or empty")
        self._api_key = key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=timeout)
        self.app_title = app_title

    @classmethod
    def from_settings(cls, settings: FactorySettings | None = None) -> OpenRouterClient:
        cfg = settings or load_settings()
        if not cfg.has_api_key():
            raise OpenRouterAuthError("OPENROUTER_API_KEY is missing or empty")
        assert cfg.openrouter_api_key is not None
        return cls(
            api_key=cfg.openrouter_api_key.get_secret_value(),
            base_url=cfg.openrouter_base_url,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OpenRouterClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/local/deepagent",
            "X-Title": self.app_title,
        }

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ChatResult:
        """Issue one chat completion. Never logs the API key."""
        model_s = model.strip()
        if not model_s:
            raise OpenRouterError("model must be non-empty")
        if not messages:
            raise OpenRouterError("messages must be non-empty")
        body = {
            "model": model_s,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        url = f"{self.base_url}/chat/completions"
        try:
            response = self._client.post(url, headers=self._headers(), json=body)
        except httpx.HTTPError as exc:
            # Do not embed headers/body that could include secrets.
            raise OpenRouterError(f"OpenRouter transport error: {type(exc).__name__}") from exc
        if response.status_code in (401, 403):
            raise OpenRouterAuthError(f"OpenRouter auth rejected (status={response.status_code})")
        if response.status_code >= 400:
            raise OpenRouterError(f"OpenRouter HTTP {response.status_code} for model {model_s}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise OpenRouterError("OpenRouter returned non-JSON body") from exc
        if not isinstance(payload, dict):
            raise OpenRouterError("OpenRouter JSON payload must be an object")
        text, finish = _extract_text(payload)
        usage, raw_usage, cost = _parse_usage(payload)
        request_id = ""
        if isinstance(payload.get("id"), str):
            request_id = payload["id"]
        elif isinstance(response.headers.get("x-request-id"), str):
            request_id = response.headers["x-request-id"]
        return ChatResult(
            model=model_s,
            text=text,
            usage=usage,
            request_id=request_id,
            cost_usd=cost,
            finish_reason=finish,
            raw_usage=raw_usage,
        )

    def fetch_generation_cost(self, generation_id: str) -> Decimal:
        """Look up exact generation cost by OpenRouter generation id."""
        gid = generation_id.strip()
        if not gid:
            raise OpenRouterBillingError("generation_id is empty")
        url = f"{self.base_url}/generation"
        try:
            response = self._client.get(url, headers=self._headers(), params={"id": gid})
        except httpx.HTTPError as exc:
            raise OpenRouterBillingError(
                f"generation cost lookup failed: {type(exc).__name__}"
            ) from exc
        if response.status_code >= 400:
            raise OpenRouterBillingError(f"generation cost lookup HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise OpenRouterBillingError("generation cost non-JSON") from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            data = payload if isinstance(payload, dict) else {}
        for key in ("total_cost", "cost", "native_tokens_cost"):
            if key in data and data[key] is not None:
                try:
                    cost = Decimal(str(data[key]))
                except (InvalidOperation, ValueError) as exc:
                    raise OpenRouterBillingError(f"unparseable cost field {key!r}") from exc
                if cost.is_finite() and cost >= 0:
                    return cost
        raise OpenRouterBillingError("generation response missing total_cost")


@dataclass
class ScriptedChatClient:
    """Offline/fake chat client for unit tests (no network)."""

    responses: list[ChatResult | Exception]
    calls: list[dict[str, Any]] | None = None
    _index: int = 0

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ChatResult:
        assert self.calls is not None
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if self._index >= len(self.responses):
            raise OpenRouterError("ScriptedChatClient exhausted responses")
        item = self.responses[self._index]
        self._index += 1
        if isinstance(item, Exception):
            raise item
        return item


__all__ = [
    "ChatClient",
    "ChatResult",
    "OpenRouterAuthError",
    "OpenRouterBillingError",
    "OpenRouterClient",
    "OpenRouterError",
    "ScriptedChatClient",
    "TokenUsage",
]
