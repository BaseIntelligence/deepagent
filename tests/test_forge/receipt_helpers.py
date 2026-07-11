"""Private helpers for building signed fixture receipts without live transport."""

from __future__ import annotations

import json

from swe_forge.forge import receipt_authority
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
