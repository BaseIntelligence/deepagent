"""Test-only fixtures issued through an isolated child authority."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from swe_forge.forge import receipt_authority
from swe_forge.forge.models import OracleTestFile
from swe_forge.forge.oracle.mutation import final_suite_fingerprint
from swe_forge.forge.teacher import TransportReceipt, Usage

_authority: receipt_authority.ReceiptAuthorityClient | None = None
_fallback_root: Path | None = None


def configure_test_authority() -> None:
    """Start a test-root authority session for this test's fixture receipts."""
    global _authority
    close_test_authority()
    _authority = receipt_authority.ReceiptAuthorityClient()


def close_test_authority() -> None:
    """Reap the test authority, if a test configured one."""
    global _authority
    if _authority is not None:
        _authority.close()
    _authority = None


def signed_transport_receipt(
    *,
    call_id: str,
    candidate_fingerprint: str,
    gate: str,
    call_kind: str,
    model: str,
    usage: Usage,
    cost: float,
    recovery: dict[str, object] | None = None,
) -> TransportReceipt:
    """Ask the isolated test authority to transport, derive, and sign a receipt."""
    if _authority is None:
        # Crash-consistency tests run fixture builders in a fresh Python
        # process, outside pytest's fixture lifecycle.  Their fallback remains
        # a process-local test root, never the production root.
        global _fallback_root
        _fallback_root = Path(tempfile.mkdtemp(prefix="forge-test-authority-"))
        receipt_authority.initialize_test_authority_root(_fallback_root)
        receipt_authority.default_authority_root = lambda: _fallback_root  # type: ignore[assignment]
        configure_test_authority()
    assert _authority is not None
    response = _authority.complete(
        {
            "type": "complete",
            "routing": {
                "model": model,
                "api_base": "https://test.invalid",
                "api_key": "test-only",
                "num_retries": 0,
                "timeout": 1.0,
            },
            "messages": [],
            "max_tokens": 1,
            "tools": None,
            "tool_choice": None,
            "response_format": None,
            "context": {
                "candidate_fingerprint": candidate_fingerprint,
                "gate": gate,
                "call_kind": call_kind,
            },
            "recovery": recovery,
            "test_provider_response": {
                "text": "test authority fixture",
                "finish_reason": "stop",
                "usage": usage.to_dict(),
                "cost": cost,
                "request_id": f"fixture-{call_id}",
            },
        },
        timeout=1.0,
    )
    raw_receipt = response.get("receipt")
    receipt = TransportReceipt.from_private_dict(raw_receipt)
    # The authority controls identifiers. Callers may use their requested value
    # only as a deterministic provider fixture label, never as a signed claim.
    assert receipt.candidate_fingerprint == candidate_fingerprint
    return receipt


def protected_alt_correct_audit(
    test_files: list[OracleTestFile],
    test_ids: list[str],
    patches: list[tuple[str, str]],
    *,
    alternative_id: str = "alt_1",
) -> dict[str, object]:
    """Build a schema-v2, non-relaxed protected alt-correct fixture audit."""
    identities = sorted(test_ids)
    files = [
        {
            "path": test_file.path,
            "content_sha256": hashlib.sha256(
                (
                    test_file.content
                    if test_file.content.endswith("\n")
                    else test_file.content + "\n"
                ).encode("utf-8")
            ).hexdigest(),
        }
        for test_file in test_files
        if test_file.content
    ]
    suite = {
        "identities": identities,
        "identity_count": len(identities),
        "identity_sha256": hashlib.sha256(
            "".join(f"{identity}\n" for identity in identities).encode("utf-8")
        ).hexdigest(),
        "suite_fingerprint": final_suite_fingerprint(test_files),
        "files": files,
    }
    proposal_digest = hashlib.sha256()
    for path, content in sorted(patches):
        proposal_digest.update(path.encode("utf-8"))
        proposal_digest.update(b"\0")
        proposal_digest.update(content.encode("utf-8"))
        proposal_digest.update(b"\0")
    hidden = [{"test_id": test_id, "exit_code": 0} for test_id in identities]
    return {
        "version": 2,
        "original_public_suite_sha256": "a" * 64,
        "pre_relax_suite": dict(suite),
        "final_suite": dict(suite),
        "gold": {
            "public": {"passed": True, "exit_code": 0},
            "filtered_p2p": {"passed": True, "exit_code": 0},
            "hidden": hidden,
        },
        "alternatives": {
            alternative_id: {
                "proposal_sha256": proposal_digest.hexdigest(),
                "patches": [
                    {"path": path, "content": content} for path, content in patches
                ],
                "public": {"passed": True, "exit_code": 0},
                "filtered_p2p": {"passed": True, "exit_code": 0},
                "hidden": list(hidden),
            }
        },
    }


def protected_alt_correct_summary(
    test_files: list[OracleTestFile],
) -> dict[str, object]:
    """Build the public source-free summary matching a fixture audit."""
    suite_fingerprint = final_suite_fingerprint(test_files)
    return {
        "public_suite_sha256": "a" * 64,
        "gold_public_suite_passed": True,
        "public_valid_alternatives": 1,
        "invalid_teacher_proposals": [],
        "pre_relax_suite_fingerprint": suite_fingerprint,
        "final_suite_fingerprint": suite_fingerprint,
    }
