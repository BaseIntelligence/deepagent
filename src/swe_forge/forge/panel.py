"""Calibration panel: a tiered model registry + k-rollout runner over LiteLLM.

A *panel* is a list of solver models spanning difficulty tiers (``weak`` /
``mid`` / ``frontier``). Every model is reached through the same env-driven
LiteLLM surface as the teacher (:mod:`swe_forge.forge.teacher`); no provider
hostname/brand string is hardcoded here and no response caching is used.

Endpoint resolution mirrors the documented contract:

* with ``PANEL_LLM_BASE_URL`` / ``PANEL_LLM_API_KEY`` unset, every panel model
  inherits the teacher endpoint (``TEACHER_LLM_BASE_URL`` / ``TEACHER_LLM_API_KEY``);
* when those overrides are exported, the panel uses them instead, so it can be
  pointed at a different endpoint/key than the teacher without code changes.

Credentials are read from the process environment (the entrypoint scrubs any
implicit ``.env`` injection) so that an explicit ``env -u`` stays authoritative.

``run_rollouts(task, model, k)`` issues exactly ``k`` independent, uncached
rollouts concurrently under an :class:`asyncio.Semaphore`; each rollout is a
separate completion with its own usage/cost. Model ids are validated with a
single live probe before any bulk rollouts (cost discipline).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from dataclasses import dataclass, field

from swe_forge.forge.secrets import key_fingerprint
from swe_forge.forge.teacher import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_NUM_RETRIES,
    DEFAULT_TIMEOUT,
    LLMResult,
    MissingCredentialsError,
    ModelRoutingError,
    Routing,
    TeacherClient,
    Usage,
    resolve_routing,
)
from swe_forge.forge.teacher import (
    Message as Message,
)

TEACHER_BASE_URL_VAR = "TEACHER_LLM_BASE_URL"
TEACHER_API_KEY_VAR = "TEACHER_LLM_API_KEY"
PANEL_BASE_URL_VAR = "PANEL_LLM_BASE_URL"
PANEL_API_KEY_VAR = "PANEL_LLM_API_KEY"

VALID_TIERS: tuple[str, ...] = ("weak", "mid", "frontier")

DEFAULT_ROLLOUT_CONCURRENCY = 4
DEFAULT_VALIDATE_MAX_TOKENS = 8
DEFAULT_VALIDATE_NUM_RETRIES = 1
DEFAULT_VALIDATE_TIMEOUT = 60.0

TaskInput = str | Sequence[Message]


class PanelError(RuntimeError):
    """Base error for the panel layer."""


class InvalidTierError(PanelError):
    """Raised when a panel model declares a tier outside the allowed set."""


def _redact(text: str, secret: str) -> str:
    """Strip a secret from a string before it is surfaced anywhere."""
    if secret and secret in text:
        return text.replace(secret, "***redacted***")
    return text


@dataclass(frozen=True)
class PanelSpec:
    """A static panel entry: a friendly id, a routing model string, and a tier."""

    id: str
    model_string: str
    tier: str


# Default solver panel: provider-prefixed model ids spanning the three tiers.
# Model ids only (no provider host/brand). Ids are validated live before bulk use.
DEFAULT_PANEL_SPECS = (
    PanelSpec(id="gpt-4o-mini", model_string="openai/gpt-4o-mini", tier="weak"),
    PanelSpec(
        id="claude-sonnet-4-6",
        model_string="anthropic/claude-sonnet-4-6",
        tier="mid",
    ),
    PanelSpec(
        id="claude-opus-4-8",
        model_string="anthropic/claude-opus-4-8",
        tier="frontier",
    ),
    PanelSpec(id="gpt-5.5", model_string="openai/gpt-5.5", tier="frontier"),
)


@dataclass(frozen=True)
class PanelModel:
    """A solver model bound to an endpoint.

    ``base_url``/``api_key`` are the *configured* endpoint values (panel override
    when set, otherwise inherited from the teacher). Per-call routing normalizes
    ``base_url`` for the provider protocol at request time via :attr:`routing`.
    The API key is excluded from ``repr`` so it never leaks into tracebacks/logs.
    """

    id: str
    model_string: str
    tier: str
    base_url: str
    api_key: str = field(repr=False, default="")

    def __post_init__(self) -> None:
        if self.tier not in VALID_TIERS:
            raise InvalidTierError(
                f"tier {self.tier!r} for model {self.model_string!r} is not one of "
                f"{VALID_TIERS}"
            )

    @property
    def routing(self) -> Routing:
        """Resolve model + normalized ``api_base`` for a call (no creds needed)."""
        return resolve_routing(self.model_string, self.base_url)

    def client(
        self,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        num_retries: int = DEFAULT_NUM_RETRIES,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> TeacherClient:
        """Build a LiteLLM client bound to this model's endpoint."""
        return TeacherClient(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model_string,
            max_tokens=max_tokens,
            num_retries=num_retries,
            timeout=timeout,
            base_url_var=TEACHER_BASE_URL_VAR,
            api_key_var=TEACHER_API_KEY_VAR,
        )

    @property
    def key_fingerprint(self) -> str:
        """Non-reversible, stable fingerprint of the configured key (never the key)."""
        return key_fingerprint(self.api_key)

    def to_dict(self, *, include_api_key: bool = False) -> dict[str, str]:
        """Serialize the model.

        The default serialization exposes ``base_url`` plus a non-reversible
        ``key_fingerprint`` so endpoint inheritance/override can be verified
        without leaking the secret; this is what the CLI emits. ``include_api_key``
        additionally exposes the raw key value and is for in-process tests ONLY -
        it must never reach any CLI output path.
        """
        data: dict[str, str] = {
            "id": self.id,
            "model_string": self.model_string,
            "tier": self.tier,
            "base_url": self.base_url,
            "key_fingerprint": self.key_fingerprint,
        }
        if include_api_key:
            data["api_key"] = self.api_key
        return data


@dataclass
class ModelValidation:
    """The outcome of a single live model-id probe."""

    model: str
    valid: bool
    error: str | None = None
    usage: Usage | None = None
    cost: float = 0.0

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "model": self.model,
            "valid": self.valid,
            "error": self.error,
        }
        if self.usage is not None:
            data["usage"] = self.usage.to_dict()
            data["cost"] = self.cost
        return data


@dataclass
class RolloutResult:
    """The text + usage + cost of one independent rollout."""

    index: int
    model: str
    text: str
    usage: Usage
    cost: float
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "model": self.model,
            "text": self.text,
            "usage": self.usage.to_dict(),
            "cost": self.cost,
            "error": self.error,
        }


def resolve_panel_endpoint(env: dict[str, str] | None = None) -> tuple[str, str]:
    """Resolve ``(base_url, api_key)`` for the panel from the environment.

    Honors ``PANEL_LLM_*`` overrides when set; otherwise inherits the teacher
    endpoint. Reads the process environment (not a re-read of ``.env``) so an
    explicit ``env -u`` stays authoritative.
    """
    source = os.environ if env is None else env
    teacher_base = (source.get(TEACHER_BASE_URL_VAR) or "").strip()
    teacher_key = source.get(TEACHER_API_KEY_VAR) or ""
    panel_base = (source.get(PANEL_BASE_URL_VAR) or "").strip() or teacher_base
    panel_key = (source.get(PANEL_API_KEY_VAR) or "") or teacher_key
    return panel_base, panel_key


def build_panel(
    base_url: str,
    api_key: str,
    specs: Sequence[PanelSpec] = DEFAULT_PANEL_SPECS,
) -> list[PanelModel]:
    """Bind a set of :class:`PanelSpec` to an endpoint, yielding panel models."""
    return [
        PanelModel(
            id=spec.id,
            model_string=spec.model_string,
            tier=spec.tier,
            base_url=base_url,
            api_key=api_key,
        )
        for spec in specs
    ]


def build_panel_from_env(
    specs: Sequence[PanelSpec] = DEFAULT_PANEL_SPECS,
    env: dict[str, str] | None = None,
) -> list[PanelModel]:
    """Build the panel using the env-resolved endpoint (override or teacher)."""
    base_url, api_key = resolve_panel_endpoint(env)
    return build_panel(base_url, api_key, specs)


def select_default_model(
    panel: Sequence[PanelModel], tier: str = "frontier"
) -> PanelModel:
    """Pick a representative model for ``tier`` (falls back to the first model)."""
    if not panel:
        raise PanelError("panel is empty; cannot select a default model")
    for model in panel:
        if model.tier == tier:
            return model
    return panel[0]


async def validate_model(
    model_string: str,
    *,
    base_url: str,
    api_key: str,
    prompt: str = "ping",
    max_tokens: int = DEFAULT_VALIDATE_MAX_TOKENS,
    num_retries: int = DEFAULT_VALIDATE_NUM_RETRIES,
    timeout: float = DEFAULT_VALIDATE_TIMEOUT,
) -> ModelValidation:
    """Probe a model id with a single live call; never runs bulk rollouts.

    Returns ``valid=True`` with usage/cost on success, or ``valid=False`` with a
    secret-free error on any failure (bad id, routing, or transient error).
    """
    secret = api_key or ""
    try:
        resolve_routing(model_string, base_url)
    except ModelRoutingError as exc:
        return ModelValidation(
            model=model_string, valid=False, error=_redact(str(exc), secret)
        )
    client = TeacherClient(
        base_url=base_url,
        api_key=api_key,
        model=model_string,
        max_tokens=max_tokens,
        num_retries=num_retries,
        timeout=timeout,
    )
    try:
        result = await client.complete_text(prompt)
    except MissingCredentialsError as exc:
        return ModelValidation(
            model=model_string, valid=False, error=_redact(str(exc), secret)
        )
    except Exception as exc:  # bad model id / transient: flag invalid, never crash
        return ModelValidation(
            model=model_string,
            valid=False,
            error=_redact(f"{type(exc).__name__}: {exc}", secret),
        )
    return ModelValidation(
        model=model_string, valid=True, usage=result.usage, cost=result.cost
    )


async def validate_models(
    model_strings: Sequence[str],
    *,
    base_url: str,
    api_key: str,
    prompt: str = "ping",
    concurrency: int = DEFAULT_ROLLOUT_CONCURRENCY,
    max_tokens: int = DEFAULT_VALIDATE_MAX_TOKENS,
    num_retries: int = DEFAULT_VALIDATE_NUM_RETRIES,
    timeout: float = DEFAULT_VALIDATE_TIMEOUT,
) -> list[ModelValidation]:
    """Validate several model ids concurrently with one probe each (no bulk)."""
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(model_string: str) -> ModelValidation:
        async with semaphore:
            return await validate_model(
                model_string,
                base_url=base_url,
                api_key=api_key,
                prompt=prompt,
                max_tokens=max_tokens,
                num_retries=num_retries,
                timeout=timeout,
            )

    return list(await asyncio.gather(*(one(m) for m in model_strings)))


async def run_rollouts(
    task: TaskInput,
    model: PanelModel,
    k: int,
    *,
    concurrency: int = DEFAULT_ROLLOUT_CONCURRENCY,
    system: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    num_retries: int = DEFAULT_NUM_RETRIES,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[RolloutResult]:
    """Issue exactly ``k`` independent, uncached rollouts of ``task`` on ``model``.

    Rollouts run concurrently bounded by an :class:`asyncio.Semaphore` of size
    ``concurrency`` (never exceeded). Each rollout is a separate completion with
    its own usage/cost (no caching/dedup). A failing rollout is recorded as a
    result with ``error`` set rather than aborting the batch.
    """
    if k < 0:
        raise PanelError(f"k must be non-negative, got {k}")
    if k == 0:
        return []

    client = model.client(
        max_tokens=max_tokens, num_retries=num_retries, timeout=timeout
    )
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(index: int) -> RolloutResult:
        async with semaphore:
            try:
                result: LLMResult = await client.complete_text(task, system=system)
            except Exception as exc:
                return RolloutResult(
                    index=index,
                    model=model.model_string,
                    text="",
                    usage=Usage(),
                    cost=0.0,
                    error=_redact(f"{type(exc).__name__}: {exc}", model.api_key or ""),
                )
            return RolloutResult(
                index=index,
                model=model.model_string,
                text=result.text,
                usage=result.usage,
                cost=result.cost,
            )

    return list(await asyncio.gather(*(one(i) for i in range(k))))
