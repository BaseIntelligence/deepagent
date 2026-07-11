"""Private helpers for building signed fixture receipts without live transport."""

from __future__ import annotations

import hashlib
import json

from swe_forge.forge import receipt_authority
from swe_forge.forge.models import OracleTestFile
from swe_forge.forge.oracle.mutation import final_suite_fingerprint
from swe_forge.forge.teacher import TransportReceipt, Usage


def signed_transport_receipt(
    *,
    call_id: str,
    candidate_fingerprint: str,
    gate: str,
    call_kind: str,
    model: str,
    usage: Usage,
    cost: float,
) -> TransportReceipt:
    """Build a valid source-free sidecar fixture using the isolated host issuer."""
    key_id = receipt_authority.issuer_key_id()
    claims = {
        "version": 2,
        "call_id": call_id,
        "candidate_fingerprint": candidate_fingerprint,
        "gate": gate,
        "call_kind": call_kind,
        "model": model,
        "usage": usage.to_dict(),
        "cost": cost,
        "issuer_key_id": key_id,
    }
    signed_key_id, signature = receipt_authority._sign_claims(
        json.dumps(claims, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    assert signed_key_id == key_id
    return TransportReceipt(
        call_id=call_id,
        candidate_fingerprint=candidate_fingerprint,
        gate=gate,
        call_kind=call_kind,
        model=model,
        usage=usage,
        cost=cost,
        issuer_key_id=key_id,
        signature=signature,
    )


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
