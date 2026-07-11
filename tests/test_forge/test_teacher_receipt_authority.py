"""Host-held authority tests for signed teacher transport receipts."""

from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from swe_forge.forge import receipt_authority
from swe_forge.forge import teacher as teacher_module
from swe_forge.forge.teacher import (
    LLMResult,
    TeacherClient,
    TransportReceipt,
    Usage,
    transport_receipt_context,
    verify_transport_receipt,
)


def _response() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="receipt", tool_calls=[]),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=3,
            completion_tokens=2,
            total_tokens=5,
        ),
        _hidden_params={"response_cost": 0.0125},
    )


class _Candidate:
    def to_dict(self) -> dict[str, str]:
        return {"candidate": "receipt-authority"}


def test_host_authority_key_is_durable_private_and_stable(
    isolated_teacher_receipt_authority: Path,
) -> None:
    root = isolated_teacher_receipt_authority

    first = receipt_authority.issuer_key_id()
    second = receipt_authority.issuer_key_id()

    key_path = root / "issuer-v1.key"
    assert first == second
    assert key_path.is_file()
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(root.stat().st_mode) == 0o700


async def test_signed_transport_receipt_reloads_after_authority_restart(
    isolated_teacher_receipt_authority: Path,
) -> None:
    root = isolated_teacher_receipt_authority
    client = TeacherClient(
        base_url="https://teacher.test",
        api_key="sk-test",
        model="anthropic/test-model",
    )
    with patch.object(
        teacher_module.litellm, "acompletion", AsyncMock(return_value=_response())
    ):
        with transport_receipt_context(
            _Candidate(), gate="differential", call_kind="proposal"
        ):
            result = await client.complete_text("generate a proposal")

    receipt = result.transport_receipt
    assert receipt is not None
    restarted = TransportReceipt.from_private_dict(receipt.to_private_dict())

    assert verify_transport_receipt(restarted)
    assert restarted.commitment == receipt.commitment
    assert (root / "issuer-v1.key").read_bytes().hex() not in json.dumps(
        receipt.to_private_dict()
    )
    assert "issuer_key_id" not in json.dumps(result.to_dict())
    assert "signature" not in json.dumps(result.to_dict())


async def test_signed_receipt_verifies_in_a_restarted_process(
    isolated_teacher_receipt_authority: Path,
) -> None:
    root = isolated_teacher_receipt_authority
    client = TeacherClient(
        base_url="https://teacher.test",
        api_key="sk-test",
        model="anthropic/test-model",
    )
    with patch.object(
        teacher_module.litellm, "acompletion", AsyncMock(return_value=_response())
    ):
        with transport_receipt_context(
            _Candidate(), gate="differential", call_kind="proposal"
        ):
            result = await client.complete_text("generate a proposal")

    assert result.transport_receipt is not None
    script = """
import json
import sys
from pathlib import Path
from swe_forge.forge import receipt_authority
from swe_forge.forge.teacher import TransportReceipt, verify_transport_receipt

receipt_authority.default_authority_root = lambda: Path(sys.argv[1])
receipt = TransportReceipt.from_private_dict(json.loads(sys.argv[2]))
raise SystemExit(0 if verify_transport_receipt(receipt) else 1)
"""
    restarted = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(root),
            json.dumps(result.transport_receipt.to_private_dict(), sort_keys=True),
        ],
        cwd="/projects/Agent-SWE",
        check=False,
        capture_output=True,
        text=True,
    )

    assert restarted.returncode == 0, restarted.stderr


async def test_monkeypatched_public_completion_cannot_create_host_authority(
    isolated_teacher_receipt_authority: Path, monkeypatch
) -> None:
    root = isolated_teacher_receipt_authority
    client = TeacherClient(
        base_url="https://teacher.test",
        api_key="sk-test",
        model="anthropic/test-model",
    )

    async def forged_complete_text(*_args: object, **_kwargs: object) -> LLMResult:
        return LLMResult(text="forged", usage=Usage(), cost=0.0)

    monkeypatch.setattr(client, "complete_text", forged_complete_text)
    with transport_receipt_context(
        _Candidate(), gate="differential", call_kind="proposal"
    ):
        result = await client.complete_text("generate a proposal")

    assert result.transport_receipt is None
    assert not root.exists()


async def test_monkeypatched_completion_helper_cannot_create_host_authority(
    isolated_teacher_receipt_authority: Path, monkeypatch
) -> None:
    root = isolated_teacher_receipt_authority
    client = TeacherClient(
        base_url="https://teacher.test",
        api_key="sk-test",
        model="anthropic/test-model",
    )

    async def forged_acompletion(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return _response()

    monkeypatch.setattr(client, "_acompletion", forged_acompletion)
    with transport_receipt_context(
        _Candidate(), gate="differential", call_kind="proposal"
    ):
        result = await client.complete_text("generate a proposal")

    assert result.transport_receipt is None
    assert not root.exists()


def test_independently_constructed_matching_shape_receipt_cannot_verify(
    isolated_teacher_receipt_authority: Path,
) -> None:
    receipt_authority.issuer_key_id()

    forged = TransportReceipt(
        call_id="f" * 32,
        candidate_fingerprint="a" * 64,
        gate="differential",
        call_kind="proposal",
        model="anthropic/test-model",
        usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        cost=0.0125,
        issuer_key_id="0" * 32,
        signature="e" * 64,
    )

    assert not verify_transport_receipt(forged)
    assert "receipt_secret" not in forged.to_private_dict()
    assert "api_key" not in forged.to_private_dict()


async def test_wrong_key_and_unsafe_key_permissions_reject_valid_receipts(
    isolated_teacher_receipt_authority: Path,
) -> None:
    root = isolated_teacher_receipt_authority
    client = TeacherClient(
        base_url="https://teacher.test",
        api_key="sk-test",
        model="anthropic/test-model",
    )
    with patch.object(
        teacher_module.litellm, "acompletion", AsyncMock(return_value=_response())
    ):
        with transport_receipt_context(
            _Candidate(), gate="differential", call_kind="proposal"
        ):
            result = await client.complete_text("generate a proposal")

    receipt = result.transport_receipt
    assert receipt is not None
    wrong_key = TransportReceipt(
        call_id=receipt.call_id,
        candidate_fingerprint=receipt.candidate_fingerprint,
        gate=receipt.gate,
        call_kind=receipt.call_kind,
        model=receipt.model,
        usage=receipt.usage,
        cost=receipt.cost,
        issuer_key_id="0" * 32,
        signature=receipt.signature,
    )
    assert not verify_transport_receipt(wrong_key)

    (root / "issuer-v1.key").chmod(0o644)
    assert not verify_transport_receipt(receipt)
