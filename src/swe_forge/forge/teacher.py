"""LiteLLM-based async teacher client for the forge pipeline.

The teacher is the single LLM surface used to author bugs, specs, and tests. It
proxies every provider request through an isolated receipt-authority child,
which alone invokes ``litellm.acompletion``. It never imports the repository's
bespoke LLM clients or response cache, and no provider hostname/brand string is
hardcoded here.

Routing is driven by the provider-prefixed model id:

* ``anthropic/<id>`` -> ``api_base`` is host-only (trailing slash stripped, never
  ``/v1``); LiteLLM appends ``/v1/messages`` itself.
* ``openai/<id>`` -> ``api_base`` is ``<base>/v1`` (exactly one ``/v1``).

Endpoint credentials and a per-call no-cache directive traverse only bounded
local IPC to the child. Every call passes ``max_tokens`` and is bounded by
retries + a timeout. Missing credentials fail fast with a message that names the
absent environment variable and never echoes the key.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import math
from decimal import Decimal, InvalidOperation
import uuid
from types import SimpleNamespace
from collections.abc import Awaitable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator, Union

from swe_forge.forge.config import ForgeSettings  # noqa: E402
from swe_forge.forge import receipt_authority

if TYPE_CHECKING:
    from swe_forge.forge.recovery_accounting import RecoveryBudgetLedger

# Environment variable names surfaced in fail-fast messages (no values logged).
TEACHER_BASE_URL_VAR = "TEACHER_LLM_BASE_URL"
TEACHER_API_KEY_VAR = "TEACHER_LLM_API_KEY"

DEFAULT_MAX_TOKENS = 1024
DEFAULT_NUM_RETRIES = 3
DEFAULT_TIMEOUT = 120.0
DEFAULT_AGENTIC_MAX_TURNS = 8

# Provider-agnostic forcing choice: when tools are supplied to a one-shot
# completion, require the model to emit a tool call so the normalized result is
# deterministic. LiteLLM translates this per protocol (OpenAI "required" /
# Anthropic ``{"type": "any"}``).
FORCED_TOOL_CHOICE = "required"

Message = dict[str, Any]


class TeacherError(RuntimeError):
    """Base error for the teacher client."""


class MissingCredentialsError(TeacherError):
    """Raised when the endpoint base URL or API key is absent/empty."""


class ModelRoutingError(TeacherError):
    """Raised when a model id is not provider-prefixed (``provider/<id>``)."""


class UnknownBillingError(TeacherError):
    """Raised when a possibly-sent request has no exact provider metering."""


class _AuthorityProviderError(TeacherError):
    """Child-reported provider error retaining only an exact safe response."""

    def __init__(self, error_type: str, response: object) -> None:
        self.response = response
        self.error_type = error_type
        super().__init__(error_type)


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
    recovery_accounting: dict[str, object] | None = None
    transport_receipt: "TransportReceipt | None" = field(default=None, repr=False)
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
        if self.recovery_accounting is not None:
            data["recovery_accounting"] = dict(self.recovery_accounting)
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
    recovery_accounting: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "turns": self.turns,
            "usage": self.usage.to_dict(),
            "cost": self.cost,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "recovery_accounting": [
                dict(record) for record in self.recovery_accounting
            ],
        }


ToolExecutor = Callable[[NormalizedToolCall], Union[str, Awaitable[str]]]


@dataclass(frozen=True)
class TransportReceipt:
    """Source-free proof returned only by the isolated transport authority."""

    call_id: str
    candidate_fingerprint: str
    gate: str
    call_kind: str
    model: str
    usage: Usage
    cost: float
    provider_request_id: str
    response_commitment: str
    ledger_linkage: str
    issuer_key_id: str
    signature: str
    authority_domain: str = "test"
    authority_root_id: str = "0" * 64
    version: int = 4

    def __post_init__(self) -> None:
        if self.version != 4:
            raise TeacherError("transport receipt version is unsupported")
        for field_name, length in (
            ("call_id", 32),
            ("candidate_fingerprint", 64),
            ("response_commitment", 64),
            ("issuer_key_id", 64),
            ("authority_root_id", 64),
        ):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or len(value) != length
                or any(ch not in "0123456789abcdef" for ch in value)
            ):
                raise TeacherError(f"transport receipt {field_name} is malformed")
        if self.authority_domain not in {"production", "test"}:
            raise TeacherError("transport receipt trust domain is malformed")
        if not all(
            isinstance(value, str) and value
            for value in (
                self.gate,
                self.call_kind,
                self.model,
                self.provider_request_id,
                self.ledger_linkage,
                self.signature,
            )
        ):
            raise TeacherError("transport receipt binding is incomplete")
        if (
            not isinstance(self.cost, (int, float))
            or isinstance(self.cost, bool)
            or not math.isfinite(float(self.cost))
            or float(self.cost) < 0.0
        ):
            raise TeacherError("transport receipt cost is malformed")

    def _signed_claims_payload(self) -> dict[str, object]:
        return {
            "version": self.version,
            "call_id": self.call_id,
            "candidate_fingerprint": self.candidate_fingerprint,
            "gate": self.gate,
            "call_kind": self.call_kind,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "cost": float(self.cost),
            "provider_request_id": self.provider_request_id,
            "response_commitment": self.response_commitment,
            "ledger_linkage": self.ledger_linkage,
            "issuer_key_id": self.issuer_key_id,
            "authority_domain": self.authority_domain,
            "authority_root_id": self.authority_root_id,
        }

    def _canonical_claims(self) -> bytes:
        return json.dumps(
            self._signed_claims_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    @property
    def commitment(self) -> str:
        encoded = json.dumps(
            self.to_private_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_private_dict(self) -> dict[str, object]:
        return {**self._signed_claims_payload(), "signature": self.signature}

    @classmethod
    def from_private_dict(cls, data: object) -> "TransportReceipt":
        if not isinstance(data, dict):
            raise TeacherError("transport receipt is not an object")
        expected = {
            "version",
            "call_id",
            "candidate_fingerprint",
            "gate",
            "call_kind",
            "model",
            "usage",
            "cost",
            "provider_request_id",
            "response_commitment",
            "ledger_linkage",
            "issuer_key_id",
            "authority_domain",
            "authority_root_id",
            "signature",
        }
        if set(data) != expected:
            raise TeacherError("transport receipt has an unsafe schema")
        usage = data["usage"]
        if not isinstance(usage, dict) or set(usage) != {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        }:
            raise TeacherError("transport receipt usage is malformed")
        values: dict[str, int] = {}
        for field_name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage[field_name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise TeacherError("transport receipt usage is malformed")
            values[field_name] = value
        if not isinstance(data["version"], int) or isinstance(data["version"], bool):
            raise TeacherError("transport receipt version is malformed")
        for field_name in expected - {"version", "usage", "cost"}:
            if not isinstance(data[field_name], str):
                raise TeacherError("transport receipt binding is malformed")
        cost = data["cost"]
        if not isinstance(cost, (int, float)) or isinstance(cost, bool):
            raise TeacherError("transport receipt cost is malformed")
        return cls(
            version=data["version"],
            call_id=data["call_id"],
            candidate_fingerprint=data["candidate_fingerprint"],
            gate=data["gate"],
            call_kind=data["call_kind"],
            model=data["model"],
            usage=Usage(**values),
            cost=float(cost),
            provider_request_id=data["provider_request_id"],
            response_commitment=data["response_commitment"],
            ledger_linkage=data["ledger_linkage"],
            issuer_key_id=data["issuer_key_id"],
            authority_domain=data["authority_domain"],
            authority_root_id=data["authority_root_id"],
            signature=data["signature"],
        )


def verify_transport_receipt(receipt: TransportReceipt | None) -> bool:
    """Verify a receipt against the durable host-held issuer authority."""
    return receipt is not None and receipt_authority.verify_signature(
        key_id=receipt.issuer_key_id,
        claims=receipt._canonical_claims(),
        signature=receipt.signature,
    )


@dataclass(frozen=True)
class _TransportReceiptContext:
    """Request binding supplied to the child after a gate selects a candidate."""

    candidate_fingerprint: str
    gate: str
    call_kind: str


_TRANSPORT_RECEIPT_CONTEXT: contextvars.ContextVar[_TransportReceiptContext | None] = (
    contextvars.ContextVar("forge_teacher_transport_receipt_context", default=None)
)


def is_authoritative_transport_receipt(receipt: TransportReceipt | None) -> bool:
    """A signature rooted in the durable public key is the sole authority test."""
    return receipt is not None and verify_transport_receipt(receipt)


def candidate_transport_fingerprint(candidate: object) -> str:
    """Hash a Candidate's canonical public data without retaining source bytes."""
    to_dict = getattr(candidate, "to_dict", None)
    if not callable(to_dict):
        raise TeacherError(
            "teacher transport receipt requires a serializable candidate"
        )
    try:
        encoded = json.dumps(
            to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TeacherError(
            "teacher transport receipt cannot canonicalize candidate"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


@contextmanager
def transport_receipt_context(
    candidate: object, *, gate: str, call_kind: str
) -> Iterator[None]:
    """Bind a request to a candidate, gate, and call kind for child attestation."""
    if not isinstance(gate, str) or not gate.strip():
        raise TeacherError("teacher transport receipt requires a non-empty gate")
    if not isinstance(call_kind, str) or not call_kind.strip():
        raise TeacherError("teacher transport receipt requires a non-empty call kind")
    token = _TRANSPORT_RECEIPT_CONTEXT.set(
        _TransportReceiptContext(
            candidate_fingerprint=candidate_transport_fingerprint(candidate),
            gate=gate.strip(),
            call_kind=call_kind.strip(),
        )
    )
    try:
        yield
    finally:
        _TRANSPORT_RECEIPT_CONTEXT.reset(token)


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
        cost = 0.0
    try:
        return float(cost or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _response_field(value: object, field: str) -> object:
    """Read structured provider metadata without parsing response body content."""
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def _exact_response_metering(
    resp: object,
) -> tuple[Usage, float | str | Decimal] | None:
    """Return only provider-supplied usage and cost, never inferred defaults."""
    raw_usage = _response_field(resp, "usage")
    values: dict[str, int] = {}
    for usage_field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        raw_value = _response_field(raw_usage, usage_field)
        if (
            not isinstance(raw_value, int)
            or isinstance(raw_value, bool)
            or raw_value < 0
        ):
            return None
        values[usage_field] = raw_value
    hidden = _response_field(resp, "_hidden_params")
    raw_cost = _response_field(hidden, "response_cost")
    if isinstance(raw_cost, bool) or raw_cost is None:
        return None
    try:
        parsed_cost = Decimal(str(raw_cost))
    except (InvalidOperation, ValueError):
        return None
    if not parsed_cost.is_finite() or parsed_cost < 0:
        return None
    if isinstance(raw_cost, (float, str, Decimal)):
        exact_cost: float | str | Decimal = raw_cost
    elif isinstance(raw_cost, int):
        exact_cost = Decimal(raw_cost)
    else:
        return None
    return Usage(**values), exact_cost


def _response_from_exception(exc: BaseException) -> object | None:
    """Find a structured provider response without reading exception text/body."""
    for response_field in ("llm_response", "completion_response", "response"):
        response = getattr(exc, response_field, None)
        if response is not None:
            return response
    return None


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
        recovery_ledger: RecoveryBudgetLedger | None = None,
        recovery_stage: str = "",
        recovery_logical_call_id: str = "",
        authority_test_responses: Sequence[dict[str, object]] | None = None,
    ) -> None:
        self.base_url = (base_url or "").strip()
        self.api_key = api_key or ""
        self.model = (model or "").strip()
        self.max_tokens = max_tokens
        self.num_retries = num_retries
        self.timeout = timeout
        self._base_url_var = base_url_var
        self._api_key_var = api_key_var
        self._recovery_ledger = recovery_ledger
        self._recovery_stage = recovery_stage.strip()
        self._recovery_logical_call_id = recovery_logical_call_id.strip()
        self.last_recovery_accounting: dict[str, object] | None = None
        self.last_agentic_recovery_accounting: list[dict[str, object]] = []
        self._recovery_history: list[dict[str, object]] = []
        self._recovery_invocations = 0
        self._authority = receipt_authority.acquire_shared_authority(
            request_timeout=timeout
        )
        self._authority_test_responses = list(authority_test_responses or ())
        if self._recovery_ledger is not None and not self._recovery_stage:
            raise TeacherError(
                "recovery_stage is required when a recovery budget ledger is active"
            )

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
        recovery_ledger: RecoveryBudgetLedger | None = None,
        recovery_stage: str = "",
        recovery_logical_call_id: str = "",
    ) -> "TeacherClient":
        settings = settings or ForgeSettings()
        return cls(
            base_url=settings.teacher_llm_base_url if base_url is None else base_url,
            api_key=settings.teacher_llm_api_key if api_key is None else api_key,
            model=settings.teacher_llm_model if model is None else model,
            max_tokens=max_tokens,
            num_retries=num_retries,
            timeout=timeout,
            recovery_ledger=recovery_ledger,
            recovery_stage=recovery_stage,
            recovery_logical_call_id=recovery_logical_call_id,
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
        """Proxy one completion to the isolated authority over bounded IPC."""
        self._require_credentials()
        routing = self.routing
        context = _TRANSPORT_RECEIPT_CONTEXT.get()
        request_context = (
            {
                "candidate_fingerprint": context.candidate_fingerprint,
                "gate": context.gate,
                "call_kind": context.call_kind,
            }
            if context is not None
            else None
        )
        if self._recovery_ledger is None:
            payload = await self._authority_complete(
                self._authority_request(
                    routing=routing,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    max_tokens=max_tokens or self.max_tokens,
                    context=request_context,
                    recovery=None,
                ),
                timeout=self.timeout,
            )
            if payload.get("type") == "provider_error":
                raise _AuthorityProviderError(
                    str(payload.get("error_type", "ProviderError")),
                    _authority_response(payload),
                )
            return _authority_response(payload)

        if self._recovery_logical_call_id:
            suffix = self._recovery_invocations
            logical_call_id = (
                self._recovery_logical_call_id
                if suffix == 0
                else f"{self._recovery_logical_call_id}:{suffix}"
            )
        else:
            logical_call_id = uuid.uuid4().hex
        self._recovery_invocations += 1
        self.last_recovery_accounting = None
        for retry in range(self.num_retries + 1):
            physical_call_id = self._recovery_ledger.reserve(
                logical_call_id=logical_call_id,
                stage=self._recovery_stage,
                model=routing.model,
                retry=retry,
            )
            recovery = {
                "logical_call_id": logical_call_id,
                "physical_call_id": physical_call_id,
                "stage": self._recovery_stage,
                "model": routing.model,
                "retry": retry,
            }
            try:
                payload = await self._authority_complete(
                    self._authority_request(
                        routing=routing,
                        messages=messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        response_format=response_format,
                        max_tokens=max_tokens or self.max_tokens,
                        context=request_context,
                        recovery=recovery,
                    ),
                    timeout=self.timeout,
                )
                if (
                    payload.get("type") == "provider_error"
                    and payload.get("error_type") != "UnsafeRequestId"
                ):
                    raise _AuthorityProviderError(
                        str(payload.get("error_type", "ProviderError")),
                        _authority_response(payload),
                    )
                response: Any = _authority_response(payload)
                raw_request_id = _response_request_id(response)
                from swe_forge.forge.recovery_accounting import sanitize_request_id

                request_id = sanitize_request_id(raw_request_id)
                metering = _exact_response_metering(response)
                if metering is None:
                    self._recovery_ledger.mark_unknown_billing(
                        physical_call_id, error_type="MissingExactProviderMetering"
                    )
                    raise UnknownBillingError(
                        "recovery request has unknown provider billing"
                    )
                usage, cost = metering
                if not request_id:
                    self._recovery_ledger.settle(
                        physical_call_id,
                        usage=usage,
                        cost=cost,
                        status="error",
                        error_type=(
                            "UnsafeRequestId" if raw_request_id else "MissingRequestId"
                        ),
                    )
                    raise TeacherError(
                        "recovery LLM response omitted a safe provider request id"
                    )
                self._recovery_ledger.settle(
                    physical_call_id,
                    request_id=request_id,
                    usage=usage,
                    cost=cost,
                    status="success",
                    finish_reason=getattr(response.choices[0], "finish_reason", None),
                )
                accounting = _recovery_call_record(
                    self._recovery_ledger, physical_call_id, logical_call_id
                )
                self.last_recovery_accounting = accounting
                self._recovery_history.append(accounting)
                _set_recovery_accounting(response, accounting)
                return response
            except asyncio.CancelledError:
                self._recovery_ledger.mark_unknown_billing(
                    physical_call_id, error_type="CancelledError"
                )
                raise
            except Exception as exc:
                from swe_forge.forge.recovery_accounting import RecoveryAccountingError

                settled = {
                    record["physical_call_id"]
                    for record in self._recovery_ledger.settled_calls()
                }
                if isinstance(exc, UnknownBillingError):
                    raise
                if physical_call_id not in settled:
                    response = _response_from_exception(exc)
                    metering = (
                        _exact_response_metering(response)
                        if response is not None
                        else None
                    )
                    if metering is None:
                        self._recovery_ledger.mark_unknown_billing(
                            physical_call_id, error_type=type(exc).__name__
                        )
                        raise UnknownBillingError(
                            "recovery request has unknown provider billing"
                        ) from None
                    usage, cost = metering
                    from swe_forge.forge.recovery_accounting import sanitize_request_id

                    self._recovery_ledger.settle(
                        physical_call_id,
                        request_id=sanitize_request_id(_response_request_id(response)),
                        usage=usage,
                        cost=cost,
                        status="error",
                        error_type=(
                            exc.error_type
                            if isinstance(exc, _AuthorityProviderError)
                            else type(exc).__name__
                        ),
                    )
                if isinstance(exc, RecoveryAccountingError):
                    self.last_recovery_accounting = _recovery_call_record(
                        self._recovery_ledger, physical_call_id, logical_call_id
                    )
                    self._recovery_history.append(self.last_recovery_accounting)
                    raise
                if retry >= self.num_retries:
                    self.last_recovery_accounting = _recovery_call_record(
                        self._recovery_ledger, physical_call_id, logical_call_id
                    )
                    self._recovery_history.append(self.last_recovery_accounting)
                    raise
        raise AssertionError("unreachable recovery retry loop")

    async def _authority_complete(
        self, request: dict[str, object], *, timeout: float
    ) -> dict[str, object]:
        """Use the child-only transport without blocking unrelated coroutines."""
        try:
            return await asyncio.to_thread(
                self._authority.complete, request, timeout=timeout
            )
        except receipt_authority.ReceiptAuthorityError as exc:
            raise TeacherError(str(exc)) from exc

    def _authority_request(
        self,
        *,
        routing: Routing,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | None,
        response_format: dict[str, Any] | None,
        max_tokens: int,
        context: dict[str, str] | None,
        recovery: dict[str, object] | None,
    ) -> dict[str, object]:
        test_response: dict[str, object] | None = None
        if self._authority_test_responses:
            test_response = self._authority_test_responses.pop(0)
        return {
            "type": "complete",
            "routing": {
                "model": routing.model,
                "api_base": routing.api_base,
                "api_key": self.api_key,
                "num_retries": self.num_retries,
                "timeout": self.timeout,
            },
            "messages": messages,
            "max_tokens": max_tokens,
            "tools": tools,
            "tool_choice": tool_choice,
            "response_format": response_format,
            "context": context,
            "recovery": recovery,
            "test_provider_response": test_response,
        }

    async def aclose(self) -> None:
        """Reap the authority child owned by this client."""
        receipt_authority.release_shared_authority(self._authority)

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
            recovery_accounting=_get_recovery_accounting(resp),
            transport_receipt=_get_transport_receipt(resp),
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
        tool_choice: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """Complete with function tools; ``tool_calls`` are normalized.

        When one or more ``tools`` are supplied and no explicit ``tool_choice``
        is given, the call forces tool use (:data:`FORCED_TOOL_CHOICE`) so the
        normalized ``tool_calls`` are deterministic. With no tools supplied the
        choice stays ``"auto"`` (behavior unchanged); an explicit ``tool_choice``
        always wins.
        """
        if tool_choice is None:
            tool_choice = FORCED_TOOL_CHOICE if tools else "auto"
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
        recovery_accounting: list[dict[str, object]] = []
        history_start = len(self._recovery_history)

        try:
            for _ in range(max_turns):
                resp = await self._acompletion(
                    conversation,
                    tools=tools,
                    tool_choice=tool_choice,
                    max_tokens=max_tokens,
                )
                turns += 1
                call_accounting = getattr(resp, "_forge_recovery_accounting", None)
                if isinstance(call_accounting, dict):
                    recovery_accounting.append(dict(call_accounting))
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
        except Exception:
            self.last_agentic_recovery_accounting = [
                dict(record) for record in self._recovery_history[history_start:]
            ]
            raise

        self.last_agentic_recovery_accounting = list(recovery_accounting)

        return AgenticResult(
            text=final_text,
            turns=turns,
            usage=total_usage,
            cost=total_cost,
            tool_calls=last_tool_calls,
            messages=conversation,
            recovery_accounting=recovery_accounting,
        )


def is_concrete_teacher_client(client: object) -> bool:
    """Return whether an invocation uses the unmodified Forge transport type.

    Oracle shipping evidence is authoritative only for a call made through the
    concrete transport. Injected parsers and test doubles, including subclasses
    that can replace the transport method, remain supported but are deliberately
    non-authoritative.
    """
    return type(client) is TeacherClient


def _authority_response(payload: dict[str, object]) -> SimpleNamespace:
    """Convert a child-normalized result without retaining provider response data."""
    normalized = payload.get("normalized")
    if not isinstance(normalized, dict):
        raise TeacherError("teacher receipt authority response is malformed")
    usage = normalized.get("usage")
    tool_calls = normalized.get("tool_calls")
    if (
        not isinstance(usage, dict)
        or not isinstance(tool_calls, list)
        or not isinstance(normalized.get("text"), str)
        or not isinstance(normalized.get("cost"), (int, float))
        or isinstance(normalized.get("cost"), bool)
        or not isinstance(normalized.get("provider_request_id"), str)
    ):
        raise TeacherError("teacher receipt authority response is malformed")
    raw_calls = []
    for call in tool_calls:
        if not isinstance(call, dict):
            raise TeacherError("teacher receipt authority response is malformed")
        raw_calls.append(
            SimpleNamespace(
                id=call.get("id", ""),
                function=SimpleNamespace(
                    name=call.get("name", ""),
                    arguments=call.get("raw_arguments", ""),
                ),
            )
        )
    response = SimpleNamespace(
        id=normalized["provider_request_id"],
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=normalized["text"], tool_calls=raw_calls
                ),
                finish_reason=normalized.get("finish_reason"),
            )
        ],
        usage=SimpleNamespace(**usage),
        _hidden_params={"response_cost": normalized["cost"]},
    )
    receipt_payload = payload.get("receipt")
    if receipt_payload is not None:
        receipt = TransportReceipt.from_private_dict(receipt_payload)
        signed_commitment = normalized.get("response_commitment")
        normalized_without_commitment = {
            key: value
            for key, value in normalized.items()
            if key != "response_commitment"
        }
        computed_commitment = hashlib.sha256(
            json.dumps(
                normalized_without_commitment,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if (
            not isinstance(signed_commitment, str)
            or signed_commitment != computed_commitment
            or receipt.response_commitment != computed_commitment
        ):
            raise TeacherError("teacher receipt response commitment mismatches")
        setattr(response, "_forge_transport_receipt", receipt)
    return response


def _response_request_id(resp: Any) -> str:
    """Extract the provider's request id without retaining any response body."""
    for value in (
        getattr(resp, "id", None),
        (getattr(resp, "_hidden_params", None) or {}).get("request_id"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _recovery_call_record(
    ledger: RecoveryBudgetLedger, physical_call_id: str, logical_call_id: str
) -> dict[str, object]:
    """Return one terminal ledger call in the compact evidence shape."""
    calls = [
        call
        for call in ledger.settled_calls()
        if call["logical_call_id"] == logical_call_id
    ]
    if any(call["physical_call_id"] == physical_call_id for call in calls):
        return {
            "logical_call_id": logical_call_id,
            "physical_calls": [
                {
                    "physical_call_id": call["physical_call_id"],
                    "run_id": call["run_id"],
                    "stage": call["stage"],
                    "model": call["model"],
                    "request_id": call["request_id"],
                    "retry": call["retry"],
                    "usage": call["usage"],
                    "cost": call["cost"],
                    "status": call["status"],
                    "finish_reason": call["finish_reason"],
                    "error_type": call["error_type"],
                }
                for call in calls
            ],
        }
    raise TeacherError(
        f"recovery ledger lost settled physical request {physical_call_id!r}"
    )


def _set_recovery_accounting(resp: Any, accounting: dict[str, object]) -> None:
    """Attach safe accounting to a provider response without exposing contents."""
    try:
        setattr(resp, "_forge_recovery_accounting", accounting)
        return
    except (AttributeError, TypeError):
        pass
    hidden = getattr(resp, "_hidden_params", None)
    if isinstance(hidden, dict):
        hidden["_forge_recovery_accounting"] = accounting
        return
    raise TeacherError("cannot retain required recovery accounting on LLM response")


def _get_recovery_accounting(resp: Any) -> dict[str, object] | None:
    """Read the safe accounting attachment from either supported response shape."""
    direct = getattr(resp, "_forge_recovery_accounting", None)
    if isinstance(direct, dict):
        return direct
    hidden = getattr(resp, "_hidden_params", None)
    if isinstance(hidden, dict):
        value = hidden.get("_forge_recovery_accounting")
        if isinstance(value, dict):
            return value
    return None


def _get_transport_receipt(resp: Any) -> TransportReceipt | None:
    """Read the source-free receipt returned by the isolated authority."""
    receipt = getattr(resp, "_forge_transport_receipt", None)
    return receipt if isinstance(receipt, TransportReceipt) else None
