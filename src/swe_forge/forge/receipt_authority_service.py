"""Child-process implementation for isolated teacher transport receipts.

Only this module invokes LiteLLM and holds an Ed25519 private key.  It receives
bounded JSON messages over an inherited pipe, normalizes a provider response,
then signs claims derived from that response.  The parent never receives the
private key or a claims-signing operation.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import stat
import sys
import uuid
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from multiprocessing.connection import Connection
from pathlib import Path

os.environ.setdefault("LITELLM_MODE", "PRODUCTION")

import litellm  # noqa: E402
from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: E402
    Ed25519PrivateKey,
)

litellm.drop_params = True


MAX_IPC_BYTES = 1_000_000
ROOT_NAME = "authority-v1.json"
TEST_MARKER_NAME = "test-authority-v1.json"
PRODUCTION_MARKER_NAME = "production-authority-v1.json"
TRUST_DOMAINS = frozenset(("production", "test"))
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SAFE_FINISH_REASONS = frozenset(
    ("stop", "length", "tool_calls", "function_call", "content_filter")
)


class AuthorityServiceError(RuntimeError):
    """Raised inside the authority child for an unsafe request or root."""


class ProviderTransportError(RuntimeError):
    """A failed provider request with an exact normalized response."""

    def __init__(self, error_type: str, normalized: dict[str, object]) -> None:
        self.error_type = error_type
        self.normalized = normalized
        super().__init__(error_type)


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def response_commitment(value: dict[str, object]) -> str:
    """Hash the complete normalized provider response without persisting it."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _open_root(root: Path, *, create: bool) -> int:
    """Pin every root ancestor and return an fd for its private directory."""
    if not root.is_absolute():
        raise AuthorityServiceError("authority root must be absolute")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in root.parts[1:]:
            try:
                child = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create:
                    raise AuthorityServiceError(
                        "authority root is unavailable"
                    ) from None
                os.mkdir(component, mode=0o700, dir_fd=descriptor)
                child = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise AuthorityServiceError("authority root is not a private directory")
        if metadata.st_mode & 0o077:
            raise AuthorityServiceError("authority root has unsafe permissions")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _safe_file(root_fd: int, name: str, *, required: bool = False) -> bool:
    try:
        metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        if required:
            raise AuthorityServiceError("authority root is incomplete")
        return False
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o077
    ):
        raise AuthorityServiceError("authority metadata is unsafe")
    return True


def _read_json(root_fd: int, name: str) -> dict[str, object]:
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
    except OSError as exc:
        raise AuthorityServiceError("authority metadata is malformed") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            raise AuthorityServiceError("authority metadata is unsafe")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            if sum(len(item) for item in chunks) > MAX_IPC_BYTES:
                raise AuthorityServiceError("authority metadata is malformed")
        data = json.loads(b"".join(chunks).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorityServiceError("authority metadata is malformed") from exc
    finally:
        os.close(descriptor)
    if not isinstance(data, dict):
        raise AuthorityServiceError("authority metadata is malformed")
    return data


def _write_json_exclusive(root_fd: int, name: str, payload: dict[str, object]) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=root_fd,
    )
    try:
        os.write(descriptor, canonical_json(payload) + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(root_fd)


def initialize_test_root(root: Path) -> None:
    """Create a non-production root marker used only by hermetic tests."""
    root_fd = _open_root(root, create=True)
    try:
        if _safe_file(root_fd, ROOT_NAME):
            raise AuthorityServiceError("test root already has authority metadata")
        if _safe_file(root_fd, PRODUCTION_MARKER_NAME):
            raise AuthorityServiceError("root is already a production authority")
        marker = _safe_file(root_fd, TEST_MARKER_NAME)
        if not marker:
            _write_json_exclusive(root_fd, TEST_MARKER_NAME, {"environment": "test"})
        elif _read_json(root_fd, TEST_MARKER_NAME) != {"environment": "test"}:
            raise AuthorityServiceError("test authority marker is malformed")
    finally:
        os.close(root_fd)


def _root_environment(root_fd: int, *, create_production: bool = False) -> str:
    """Read the immutable domain marker, creating production state only in-child."""
    has_test = _safe_file(root_fd, TEST_MARKER_NAME)
    has_production = _safe_file(root_fd, PRODUCTION_MARKER_NAME)
    if has_test and has_production:
        raise AuthorityServiceError("authority root has conflicting trust domains")
    if has_test:
        if _read_json(root_fd, TEST_MARKER_NAME) != {"environment": "test"}:
            raise AuthorityServiceError("test authority marker is malformed")
        return "test"
    if has_production:
        if _read_json(root_fd, PRODUCTION_MARKER_NAME) != {"environment": "production"}:
            raise AuthorityServiceError("production authority marker is malformed")
        return "production"
    if not create_production:
        raise AuthorityServiceError("authority root has no trust-domain marker")
    if os.listdir(root_fd):
        raise AuthorityServiceError("authority root has no trust-domain marker")
    _write_json_exclusive(
        root_fd, PRODUCTION_MARKER_NAME, {"environment": "production"}
    )
    return "production"


def _require_public_root_contents(root_fd: int, environment: str) -> None:
    """Reject any durable material other than public authority metadata."""
    allowed = {ROOT_NAME}
    if environment == "test":
        allowed.add(TEST_MARKER_NAME)
    elif environment == "production":
        allowed.add(PRODUCTION_MARKER_NAME)
    else:
        raise AuthorityServiceError("authority root trust domain is malformed")
    try:
        entries = os.listdir(root_fd)
    except OSError as exc:
        raise AuthorityServiceError("authority root is unavailable") from exc
    if any(entry not in allowed for entry in entries):
        raise AuthorityServiceError(
            "authority root contains private or unknown material"
        )


def _public_key_bytes(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )


def _root_identity(root: Path, environment: str) -> str:
    """Derive a stable identity for this exact root and trust domain."""
    canonical_root = os.path.abspath(os.fspath(root))
    return hashlib.sha256(
        canonical_json(
            {
                "version": 1,
                "environment": environment,
                "root": canonical_root,
            }
        )
    ).hexdigest()


def _key_id(public_key: bytes, *, environment: str, root_id: str) -> str:
    """Bind key identity to both the trust domain and the root instance."""
    return hashlib.sha256(
        canonical_json(
            {
                "algorithm": "Ed25519",
                "environment": environment,
                "root_id": root_id,
                "public_key": base64.b64encode(public_key).decode("ascii"),
            }
        )
    ).hexdigest()


def _public_metadata(
    *, public_key: bytes, environment: str, root_id: str
) -> dict[str, object]:
    return {
        "version": 1,
        "algorithm": "Ed25519",
        "environment": environment,
        "root_id": root_id,
        "key_id": _key_id(public_key, environment=environment, root_id=root_id),
        "public_key": base64.b64encode(public_key).decode("ascii"),
    }


def _require_pinned_key(
    root: Path, private_key: Ed25519PrivateKey
) -> tuple[str, str, str]:
    """Pin this child key once, then reject any replacement or restart."""
    root_fd = _open_root(root, create=True)
    try:
        environment = _root_environment(root_fd, create_production=True)
        _require_public_root_contents(root_fd, environment)
        root_id = _root_identity(root, environment)
        public_key = _public_key_bytes(private_key)
        expected = _public_metadata(
            public_key=public_key,
            environment=environment,
            root_id=root_id,
        )
        metadata_exists = _safe_file(root_fd, ROOT_NAME)
        if not metadata_exists:
            try:
                _write_json_exclusive(root_fd, ROOT_NAME, expected)
            except FileExistsError:
                _safe_file(root_fd, ROOT_NAME, required=True)
            else:
                return (
                    expected["key_id"],
                    environment,
                    root_id,
                )  # type: ignore[return-value]
        if _read_json(root_fd, ROOT_NAME) != expected:
            raise AuthorityServiceError(
                "authority root key does not match this authority"
            )
        _require_public_root_contents(root_fd, environment)
        return (
            expected["key_id"],
            environment,
            root_id,
        )  # type: ignore[return-value]
    finally:
        os.close(root_fd)


def _get_field(value: object, field: str) -> object:
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def _usage(response: object) -> dict[str, int]:
    raw = _get_field(response, "usage")
    values: dict[str, int] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = _get_field(raw, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise AuthorityServiceError("provider response has incomplete metering")
        values[name] = value
    return values


def _cost(response: object) -> float:
    hidden = _get_field(response, "_hidden_params")
    value = _get_field(hidden, "response_cost")
    if isinstance(value, bool) or value is None:
        raise AuthorityServiceError("provider response has incomplete metering")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise AuthorityServiceError("provider response cost is malformed") from exc
    if not parsed.is_finite() or parsed < 0:
        raise AuthorityServiceError("provider response cost is malformed")
    return float(parsed)


def _request_id(response: object) -> str:
    hidden = _get_field(response, "_hidden_params")
    for candidate in (
        _get_field(response, "id"),
        _get_field(hidden, "request_id"),
    ):
        if isinstance(candidate, str) and _SAFE_REQUEST_ID.fullmatch(candidate):
            return candidate
    raise AuthorityServiceError("provider response omitted a safe request identity")


def _tool_calls(response: object) -> list[dict[str, object]]:
    choices = _get_field(response, "choices")
    if not isinstance(choices, (list, tuple)) or not choices:
        raise AuthorityServiceError("provider response has no completion choice")
    message = _get_field(choices[0], "message")
    raw_calls = _get_field(message, "tool_calls") or []
    if not isinstance(raw_calls, (list, tuple)):
        raise AuthorityServiceError("provider response tool calls are malformed")
    normalized: list[dict[str, object]] = []
    for raw_call in raw_calls:
        function = _get_field(raw_call, "function")
        name = _get_field(function, "name")
        arguments = _get_field(function, "arguments")
        call_id = _get_field(raw_call, "id")
        if not all(isinstance(value, str) for value in (name, arguments, call_id)):
            raise AuthorityServiceError("provider response tool call is malformed")
        assert isinstance(name, str)
        assert isinstance(arguments, str)
        assert isinstance(call_id, str)
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            parsed = {"__unparsed__": arguments}
        if not isinstance(parsed, dict):
            parsed = {"value": parsed}
        normalized.append(
            {
                "id": call_id,
                "name": name,
                "arguments": parsed,
                "raw_arguments": arguments,
            }
        )
    return normalized


def _normalize_response(response: object) -> dict[str, object]:
    choices = _get_field(response, "choices")
    if not isinstance(choices, (list, tuple)) or not choices:
        raise AuthorityServiceError("provider response has no completion choice")
    choice = choices[0]
    message = _get_field(choice, "message")
    text = _get_field(message, "content")
    if text is not None and not isinstance(text, str):
        raise AuthorityServiceError("provider response content is malformed")
    finish_reason = _get_field(choice, "finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise AuthorityServiceError("provider response finish reason is malformed")
    return {
        "text": text or "",
        "usage": _usage(response),
        "cost": _cost(response),
        "finish_reason": finish_reason,
        "tool_calls": _tool_calls(response),
        "provider_request_id": _request_id(response),
    }


def _response_from_exception(exc: BaseException) -> object | None:
    for field in ("llm_response", "completion_response", "response"):
        response = getattr(exc, field, None)
        if response is not None:
            return response
    return None


def _normalize_provider_error(response: object) -> dict[str, object]:
    """Retain only exact error metering, never provider content."""
    return {
        "text": "",
        "usage": _usage(response),
        "cost": _cost(response),
        "finish_reason": None,
        "tool_calls": [],
        "provider_request_id": _request_id(response),
    }


class _TestProviderError(RuntimeError):
    """Hermetic provider error that remains wholly inside the child."""

    def __init__(self, response: object, error_type: str) -> None:
        self.response = response
        self.error_type = error_type
        super().__init__(error_type)


def _test_provider_response(value: object) -> object:
    """Return a response-shaped object inside the child for hermetic tests."""
    if not isinstance(value, dict):
        raise AuthorityServiceError("test provider response is malformed")
    if value.get("crash") is True:
        os._exit(70)
    error = value.get("error")
    if error is not None:
        if not isinstance(error, dict) or set(error) != {"response", "error_type"}:
            raise AuthorityServiceError("test provider response is malformed")
        if not isinstance(error["error_type"], str):
            raise AuthorityServiceError("test provider response is malformed")
        raise _TestProviderError(
            _test_provider_response(error["response"]), error["error_type"]
        )
    required = {
        "text",
        "usage",
        "cost",
        "request_id",
    }
    if not required <= set(value):
        raise AuthorityServiceError("test provider response is incomplete")
    usage = value["usage"]
    if not isinstance(usage, dict):
        raise AuthorityServiceError("test provider response is malformed")
    tool_calls = value.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        raise AuthorityServiceError("test provider response is malformed")

    class _Value:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    calls = [
        _Value(
            id=item.get("id", ""),
            function=_Value(
                name=item.get("name", ""),
                arguments=item.get(
                    "raw_arguments", json.dumps(item.get("arguments", {}))
                ),
            ),
        )
        for item in tool_calls
        if isinstance(item, dict)
    ]
    return _Value(
        id=value["request_id"],
        choices=[
            _Value(
                message=_Value(content=value["text"], tool_calls=calls),
                finish_reason=value.get("finish_reason", "stop"),
            )
        ],
        usage=_Value(**usage),
        _hidden_params={"response_cost": value["cost"]},
    )


def _validate_request(request: object) -> dict[str, object]:
    if not isinstance(request, dict) or set(request) != {
        "type",
        "routing",
        "messages",
        "max_tokens",
        "tools",
        "tool_choice",
        "response_format",
        "context",
        "recovery",
        "test_provider_response",
    }:
        raise AuthorityServiceError("authority IPC request is malformed")
    if request["type"] != "complete":
        raise AuthorityServiceError("authority IPC request type is invalid")
    if not isinstance(request["routing"], dict):
        raise AuthorityServiceError("authority IPC routing is malformed")
    if not isinstance(request["messages"], list):
        raise AuthorityServiceError("authority IPC messages are malformed")
    if (
        not isinstance(request["max_tokens"], int)
        or isinstance(request["max_tokens"], bool)
        or request["max_tokens"] <= 0
    ):
        raise AuthorityServiceError("authority IPC max_tokens is malformed")
    context = request["context"]
    if context is not None:
        if not isinstance(context, dict) or set(context) != {
            "candidate_fingerprint",
            "gate",
            "call_kind",
        }:
            raise AuthorityServiceError("authority receipt context is malformed")
        fingerprint = context["candidate_fingerprint"]
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in fingerprint)
            or not isinstance(context["gate"], str)
            or not context["gate"].strip()
            or not isinstance(context["call_kind"], str)
            or not context["call_kind"].strip()
        ):
            raise AuthorityServiceError("authority receipt context is malformed")
    return request


async def _provider_response(request: dict[str, object], *, environment: str) -> object:
    test_response = request["test_provider_response"]
    if test_response is not None:
        if environment != "test":
            raise AuthorityServiceError("test provider is unavailable in production")
        return _test_provider_response(test_response)

    routing = request["routing"]
    assert isinstance(routing, dict)
    kwargs: dict[str, object] = {
        "model": routing["model"],
        "api_base": routing["api_base"],
        "api_key": routing["api_key"],
        "messages": request["messages"],
        "max_tokens": request["max_tokens"],
        "cache": {"no-cache": True, "no-store": True},
        "num_retries": routing.get("num_retries", 0),
        "timeout": routing.get("timeout"),
    }
    if request["tools"] is not None:
        kwargs["tools"] = request["tools"]
        if request["tool_choice"] is not None:
            kwargs["tool_choice"] = request["tool_choice"]
    if request["response_format"] is not None:
        kwargs["response_format"] = request["response_format"]
    try:
        return await litellm.acompletion(**kwargs)
    except Exception as exc:
        response = _response_from_exception(exc)
        if response is None:
            raise
        raise ProviderTransportError(
            type(exc).__name__, _normalize_provider_error(response)
        ) from None


def _ledger_linkage(request: dict[str, object]) -> str:
    """Bind claims to a supplied child-created recovery accounting identity."""
    recovery = request["recovery"]
    if recovery is None:
        return "not_applicable"
    if not isinstance(recovery, dict):
        raise AuthorityServiceError("authority recovery linkage is malformed")
    fields = ("logical_call_id", "physical_call_id", "stage", "model", "retry")
    linkage = {field: recovery.get(field) for field in fields}
    if (
        not all(
            isinstance(linkage[field], str) and linkage[field] for field in fields[:-1]
        )
        or not isinstance(linkage["retry"], int)
        or isinstance(linkage["retry"], bool)
        or linkage["retry"] < 0
    ):
        raise AuthorityServiceError("authority recovery linkage is incomplete")
    return hashlib.sha256(canonical_json(linkage)).hexdigest()


def _completion(
    request: dict[str, object],
    *,
    environment: str,
    issue_receipt: Callable[
        [dict[str, object], dict[str, object], str], dict[str, object]
    ],
) -> dict[str, object]:
    try:
        response = asyncio.run(_provider_response(request, environment=environment))
    except ProviderTransportError as exc:
        return {
            "type": "provider_error",
            "error_type": exc.error_type,
            "normalized": exc.normalized,
            "receipt": None,
            "recovery_accounting": request["recovery"],
        }
    except _TestProviderError as exc:
        return {
            "type": "provider_error",
            "error_type": exc.error_type,
            "normalized": _normalize_provider_error(exc.response),
            "receipt": None,
            "recovery_accounting": request["recovery"],
        }
    try:
        normalized = _normalize_response(response)
    except AuthorityServiceError as exc:
        # A provider return can include exact metering but an unsafe request ID.
        # Return only the transient ID needed for the parent ledger sanitizer,
        # never a receipt, signature, or persisted provider content.
        raw_id = _get_field(response, "id")
        if str(
            exc
        ) == "provider response omitted a safe request identity" and isinstance(
            raw_id, str
        ):
            return {
                "type": "provider_error",
                "error_type": "UnsafeRequestId",
                "normalized": {
                    "text": "",
                    "usage": _usage(response),
                    "cost": _cost(response),
                    "finish_reason": None,
                    "tool_calls": [],
                    "provider_request_id": raw_id,
                },
                "receipt": None,
                "recovery_accounting": request["recovery"],
            }
        raise
    routing = request["routing"]
    assert isinstance(routing, dict)
    normalized["model"] = routing["model"]
    normalized["response_commitment"] = response_commitment(normalized)
    context = request["context"]
    receipt = (
        issue_receipt(context, normalized, _ledger_linkage(request))
        if isinstance(context, dict)
        else None
    )
    return {
        "type": "result",
        "normalized": normalized,
        "receipt": receipt,
        "recovery_accounting": request["recovery"],
    }


def _send(connection: Connection, payload: dict[str, object]) -> None:
    encoded = canonical_json(payload)
    if len(encoded) > MAX_IPC_BYTES:
        raise AuthorityServiceError("authority IPC response exceeds its bound")
    connection.send_bytes(encoded)


def _run_authority(connection: Connection, root_text: str) -> None:
    """Run the child-only provider transport and Ed25519 receipt authority."""
    try:
        root = Path(root_text)
        private_key = Ed25519PrivateKey.generate()
        key_id, environment, root_id = _require_pinned_key(root, private_key)
        _send(
            connection,
            {
                "type": "ready",
                "key_id": key_id,
                "environment": environment,
                "root_id": root_id,
            },
        )
    except Exception as exc:
        try:
            _send(
                connection, {"type": "startup_error", "error_type": type(exc).__name__}
            )
        finally:
            connection.close()
        return

    def issue_receipt(
        context: dict[str, object],
        normalized: dict[str, object],
        ledger_linkage: str,
    ) -> dict[str, object]:
        """Sign only child-derived claims after provider normalization."""
        candidate_fingerprint = context["candidate_fingerprint"]
        gate = context["gate"]
        call_kind = context["call_kind"]
        assert isinstance(candidate_fingerprint, str)
        assert isinstance(gate, str)
        assert isinstance(call_kind, str)
        claims = {
            "version": 4,
            "call_id": uuid.uuid4().hex,
            "candidate_fingerprint": candidate_fingerprint,
            "gate": gate.strip(),
            "call_kind": call_kind.strip(),
            "model": normalized["model"],
            "usage": normalized["usage"],
            "cost": normalized["cost"],
            "provider_request_id": normalized["provider_request_id"],
            "response_commitment": normalized["response_commitment"],
            "ledger_linkage": ledger_linkage,
            "issuer_key_id": key_id,
            "authority_domain": environment,
            "authority_root_id": root_id,
        }
        signature = base64.b64encode(private_key.sign(canonical_json(claims))).decode(
            "ascii"
        )
        return {**claims, "signature": signature}

    while True:
        try:
            encoded = connection.recv_bytes(MAX_IPC_BYTES + 1)
        except EOFError:
            break
        except Exception:
            break
        if len(encoded) > MAX_IPC_BYTES:
            _send(connection, {"type": "error", "error_type": "IpcBoundsError"})
            continue
        try:
            request = _validate_request(json.loads(encoded.decode("utf-8")))
            result = _completion(
                request,
                environment=environment,
                issue_receipt=issue_receipt,
            )
            _send(connection, result)
        except Exception as exc:
            _send(connection, {"type": "error", "error_type": type(exc).__name__})
    connection.close()


def _stdio_connection() -> tuple[Connection, Connection]:
    """Adapt newline-delimited stdin/stdout to the legacy authority loop."""

    class _Pipe:
        def recv_bytes(self, limit: int) -> bytes:
            line = sys.stdin.buffer.readline(limit)
            if not line:
                raise EOFError
            if not line.endswith(b"\n"):
                return line
            return line[:-1]

        def send_bytes(self, value: bytes) -> None:
            sys.stdout.buffer.write(value + b"\n")
            sys.stdout.buffer.flush()

        def close(self) -> None:
            try:
                sys.stdout.buffer.flush()
            except OSError:
                pass

    pipe = _Pipe()
    return pipe, pipe  # type: ignore[return-value]


def _consume_bootstrap_fd(descriptor: int) -> None:
    """Consume the one-shot private bootstrap capability from an inherited FD."""
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISFIFO(metadata.st_mode):
            raise AuthorityServiceError("authority bootstrap is unavailable")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 32 - sum(map(len, chunks)) + 1)
            if not chunk:
                break
            chunks.append(chunk)
            if sum(map(len, chunks)) > 32:
                raise AuthorityServiceError("authority bootstrap is malformed")
    except OSError as exc:
        raise AuthorityServiceError("authority bootstrap is unavailable") from exc
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
    token = b"".join(chunks)
    if len(token) != 32:
        raise AuthorityServiceError("authority bootstrap is malformed")
    return None


# The service loop exists only in the executable module instance. Importing this
# module cannot recover a callable loop or bootstrap production authority state.
if __name__ != "__main__":
    del _run_authority


def main(argv: list[str] | None = None) -> int:
    """Executable-only authority bootstrap used by the parent subprocess."""
    if __name__ != "__main__":
        return 4
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--domain", choices=tuple(TRUST_DOMAINS), required=True)
    parser.add_argument("--bootstrap-fd", type=int, required=True)
    args = parser.parse_args(argv)
    if args.bootstrap_fd < 3:
        return 2
    try:
        root = Path(args.root).absolute()
        if args.domain == "production" and root != Path(
            "/var/lib/swe_forge/teacher-receipt-authority"
        ):
            return 3
        if args.domain == "test":
            root_fd = _open_root(root, create=False)
            try:
                if _root_environment(root_fd) != "test":
                    return 3
            finally:
                os.close(root_fd)
        _consume_bootstrap_fd(args.bootstrap_fd)
        read_write, _ = _stdio_connection()
        _run_authority(read_write, str(root))
        return 0
    except Exception:
        return 1


def parse_message(encoded: bytes) -> dict[str, object]:
    if len(encoded) > MAX_IPC_BYTES:
        raise AuthorityServiceError("authority IPC message exceeds its bound")
    try:
        decoded = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorityServiceError("authority IPC message is malformed") from exc
    if not isinstance(decoded, dict):
        raise AuthorityServiceError("authority IPC message is malformed")
    return decoded


if __name__ == "__main__":
    raise SystemExit(main())
