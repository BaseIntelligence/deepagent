"""LiteLLM-based async teacher client for the forge pipeline.

The teacher is the single LLM surface used to author bugs, specs, and tests. It
talks to an OpenAI- or Anthropic-compatible endpoint exclusively through
``litellm.acompletion``; it never imports the repository's bespoke LLM clients
or response cache, and no provider hostname/brand string is hardcoded here.

Routing is driven by the provider-prefixed model id:

* ``anthropic/<id>`` -> ``api_base`` is host-only (trailing slash stripped, never
  ``/v1``); LiteLLM appends ``/v1/messages`` itself.
* ``openai/<id>`` -> ``api_base`` is ``<base>/v1`` (exactly one ``/v1``).

Endpoint credentials and a per-call no-cache directive are passed as arguments to
every call; no mutable ``litellm.*`` global is ever written (apart from the
required ``litellm.drop_params`` flag). Every call passes ``max_tokens`` and is
bounded by retries + a timeout. Missing credentials fail fast with a message that
names the absent environment variable and never echoes the key.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Union

# LiteLLM auto-loads a local .env on import while in its default "DEV" mode. That
# would repopulate credentials from .env even after an explicit `env -u`, which
# defeats fail-fast on missing credentials. Credentials must come from the
# process environment only, so disable the implicit .env load before importing.
os.environ.setdefault("LITELLM_MODE", "PRODUCTION")

import litellm  # noqa: E402  (must follow the LITELLM_MODE guard above)

from swe_forge.forge.config import ForgeSettings  # noqa: E402

# Drop provider-incompatible params instead of erroring (e.g. an unsupported
# sampling field on one of the two protocols). Required by the LLM contract.
litellm.drop_params = True

# Environment variable names surfaced in fail-fast messages (no values logged).
TEACHER_BASE_URL_VAR = "TEACHER_LLM_BASE_URL"
TEACHER_API_KEY_VAR = "TEACHER_LLM_API_KEY"

DEFAULT_MAX_TOKENS = 1024
DEFAULT_NUM_RETRIES = 3
DEFAULT_TIMEOUT = 120.0
DEFAULT_AGENTIC_MAX_TURNS = 8

Message = dict[str, Any]


class TeacherError(RuntimeError):
    """Base error for the teacher client."""


class MissingCredentialsError(TeacherError):
    """Raised when the endpoint base URL or API key is absent/empty."""


class ModelRoutingError(TeacherError):
    """Raised when a model id is not provider-prefixed (``provider/<id>``)."""


@dataclass
class Usage:
    """Token usage for a single completion (or aggregated across turns)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class NormalizedToolCall:
    """A provider-agnostic tool call: a name plus parsed JSON arguments."""

    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class Routing:
    """Resolved request routing for a single call (no secrets)."""

    model: str
    provider: str
    api_base: str

    def to_dict(self) -> dict[str, str]:
        return {
            "model": self.model,
            "provider": self.provider,
            "api_base": self.api_base,
        }


@dataclass
class LLMResult:
    """The text + usage + cost of a single completion."""

    text: str
    usage: Usage
    cost: float
    finish_reason: str | None = None
    tool_calls: list[NormalizedToolCall] = field(default_factory=list)
    raw: Any = field(default=None, repr=False)

    def to_dict(self, *, include_tools: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "text": self.text,
            "usage": self.usage.to_dict(),
            "cost": self.cost,
            "finish_reason": self.finish_reason,
        }
        if include_tools:
            data["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return data


@dataclass
class AgenticResult:
    """The outcome of a multi-turn agentic exchange."""

    text: str
    turns: int
    usage: Usage
    cost: float
    tool_calls: list[NormalizedToolCall] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "turns": self.turns,
            "usage": self.usage.to_dict(),
            "cost": self.cost,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
        }


ToolExecutor = Callable[[NormalizedToolCall], Union[str, Awaitable[str]]]


def split_model(model: str) -> tuple[str, str]:
    """Split a provider-prefixed model id into ``(provider, model_id)``.

    Raises :class:`ModelRoutingError` for an unprefixed/empty model id.
    """
    cleaned = (model or "").strip()
    if not cleaned or "/" not in cleaned:
        raise ModelRoutingError(
            f"model {cleaned or '(empty)'!r} is not provider-prefixed; "
            "expected 'anthropic/<id>' or 'openai/<id>'"
        )
    provider, _, model_id = cleaned.partition("/")
    provider = provider.strip().lower()
    if not provider or not model_id.strip():
        raise ModelRoutingError(
            f"model {cleaned!r} is not provider-prefixed; "
            "expected 'anthropic/<id>' or 'openai/<id>'"
        )
    return provider, model_id.strip()


def normalize_base_url(base_url: str, provider: str) -> str:
    """Normalize ``base_url`` for the routing protocol selected by ``provider``.

    * ``openai`` -> ensure the base ends with exactly one ``/v1``.
    * anything else (``anthropic`` and other host-only protocols) -> host-only:
      strip trailing slashes and a trailing ``/v1`` so ``/v1`` never leaks.
    """
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return base
    if provider == "openai":
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return base
    if base.endswith("/v1"):
        base = base[: -len("/v1")].rstrip("/")
    return base


def resolve_routing(model: str, base_url: str) -> Routing:
    """Resolve the model + normalized ``api_base`` for a call (no creds checked)."""
    provider, _ = split_model(model)
    return Routing(
        model=model.strip(),
        provider=provider,
        api_base=normalize_base_url(base_url, provider),
    )


def _usage_from_response(resp: Any) -> Usage:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return Usage()
    return Usage(
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
    )


def _cost_from_response(resp: Any) -> float:
    hidden = getattr(resp, "_hidden_params", None) or {}
    cost = hidden.get("response_cost") if isinstance(hidden, dict) else None
    if cost is None:
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost = 0.0
    try:
        return float(cost or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _normalize_tool_calls(message: Any) -> list[NormalizedToolCall]:
    raw_calls = getattr(message, "tool_calls", None) or []
    normalized: list[NormalizedToolCall] = []
    for call in raw_calls:
        function = getattr(call, "function", None)
        name = getattr(function, "name", "") or ""
        raw_args = getattr(function, "arguments", "") or ""
        try:
            parsed = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            parsed = {"__unparsed__": raw_args}
        if not isinstance(parsed, dict):
            parsed = {"value": parsed}
        normalized.append(
            NormalizedToolCall(
                id=getattr(call, "id", "") or "",
                name=name,
                arguments=parsed,
                raw_arguments=raw_args,
            )
        )
    return normalized


class TeacherClient:
    """Async LiteLLM teacher client (env-driven endpoint, no caching)."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        num_retries: int = DEFAULT_NUM_RETRIES,
        timeout: float = DEFAULT_TIMEOUT,
        base_url_var: str = TEACHER_BASE_URL_VAR,
        api_key_var: str = TEACHER_API_KEY_VAR,
    ) -> None:
        self.base_url = (base_url or "").strip()
        self.api_key = api_key or ""
        self.model = (model or "").strip()
        self.max_tokens = max_tokens
        self.num_retries = num_retries
        self.timeout = timeout
        self._base_url_var = base_url_var
        self._api_key_var = api_key_var

    @classmethod
    def from_settings(
        cls,
        settings: ForgeSettings | None = None,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        num_retries: int = DEFAULT_NUM_RETRIES,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> "TeacherClient":
        settings = settings or ForgeSettings()
        return cls(
            base_url=settings.teacher_llm_base_url if base_url is None else base_url,
            api_key=settings.teacher_llm_api_key if api_key is None else api_key,
            model=settings.teacher_llm_model if model is None else model,
            max_tokens=max_tokens,
            num_retries=num_retries,
            timeout=timeout,
        )

    @property
    def routing(self) -> Routing:
        """Resolve routing for the configured model/base URL (no creds needed)."""
        return resolve_routing(self.model, self.base_url)

    def _require_credentials(self) -> None:
        if not self.base_url:
            raise MissingCredentialsError(
                f"{self._base_url_var} is not set; export it (e.g. in .env) "
                "before calling the LLM endpoint"
            )
        if not self.api_key:
            raise MissingCredentialsError(
                f"{self._api_key_var} is not set; export it (e.g. in .env) "
                "before calling the LLM endpoint"
            )

    @staticmethod
    def _as_messages(
        prompt: str | Sequence[Message], system: str | None = None
    ) -> list[Message]:
        if isinstance(prompt, str):
            messages: list[Message] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            return messages
        return [dict(message) for message in prompt]

    async def _acompletion(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        """The single LiteLLM call surface; credentials/no-cache passed per call."""
        self._require_credentials()
        routing = self.routing
        kwargs: dict[str, Any] = {
            "model": routing.model,
            "api_base": routing.api_base,
            "api_key": self.api_key,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "cache": {"no-cache": True, "no-store": True},
            "num_retries": self.num_retries,
            "timeout": self.timeout,
        }
        if tools is not None:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        if response_format is not None:
            kwargs["response_format"] = response_format
        return await litellm.acompletion(**kwargs)

    @staticmethod
    def _result(resp: Any) -> LLMResult:
        choice = resp.choices[0]
        message = choice.message
        return LLMResult(
            text=getattr(message, "content", None) or "",
            usage=_usage_from_response(resp),
            cost=_cost_from_response(resp),
            finish_reason=getattr(choice, "finish_reason", None),
            tool_calls=_normalize_tool_calls(message),
            raw=resp,
        )

    async def complete_text(
        self,
        prompt: str | Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """Return assistant text + usage + cost for a plain completion."""
        resp = await self._acompletion(
            self._as_messages(prompt, system), max_tokens=max_tokens
        )
        return self._result(resp)

    async def complete_json(
        self,
        prompt: str | Sequence[Message],
        schema: dict[str, Any],
        *,
        system: str | None = None,
        schema_name: str = "response",
        max_tokens: int | None = None,
    ) -> LLMResult:
        """Request structured output via ``response_format`` json_schema."""
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema},
        }
        resp = await self._acompletion(
            self._as_messages(prompt, system),
            response_format=response_format,
            max_tokens=max_tokens,
        )
        return self._result(resp)

    async def complete_with_tools(
        self,
        prompt: str | Sequence[Message],
        tools: list[dict[str, Any]],
        *,
        system: str | None = None,
        tool_choice: str = "auto",
        max_tokens: int | None = None,
    ) -> LLMResult:
        """Complete with function tools; ``tool_calls`` are normalized."""
        resp = await self._acompletion(
            self._as_messages(prompt, system),
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
        )
        return self._result(resp)

    async def agentic_turn(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]],
        tool_executor: ToolExecutor,
        *,
        max_turns: int = DEFAULT_AGENTIC_MAX_TURNS,
        tool_choice: str = "auto",
        max_tokens: int | None = None,
    ) -> AgenticResult:
        """Run a multi-turn tool exchange until the model returns a final answer.

        On each turn the model may request tools; ``tool_executor`` produces a
        result string that is fed back, and the loop continues until the model
        responds without tool calls (or ``max_turns`` is reached). Usage and cost
        are accumulated across every turn (no caching/dedup).
        """
        conversation: list[Message] = [dict(message) for message in messages]
        total_usage = Usage()
        total_cost = 0.0
        turns = 0
        final_text = ""
        last_tool_calls: list[NormalizedToolCall] = []

        for _ in range(max_turns):
            resp = await self._acompletion(
                conversation,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
            )
            turns += 1
            total_usage = total_usage + _usage_from_response(resp)
            total_cost += _cost_from_response(resp)
            message = resp.choices[0].message
            tool_calls = _normalize_tool_calls(message)
            final_text = getattr(message, "content", None) or ""

            if not tool_calls:
                break

            last_tool_calls = tool_calls
            conversation.append(
                {
                    "role": "assistant",
                    "content": getattr(message, "content", None) or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.raw_arguments
                                or json.dumps(call.arguments),
                            },
                        }
                        for call in tool_calls
                    ],
                }
            )
            for call in tool_calls:
                outcome = tool_executor(call)
                if isinstance(outcome, Awaitable):
                    outcome = await outcome
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": str(outcome),
                    }
                )

        return AgenticResult(
            text=final_text,
            turns=turns,
            usage=total_usage,
            cost=total_cost,
            tool_calls=last_tool_calls,
            messages=conversation,
        )
