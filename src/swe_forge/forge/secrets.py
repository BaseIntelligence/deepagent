"""Non-reversible key fingerprinting for safe endpoint-inheritance checks.

The raw API key must never be emitted on any CLI path (success, error, dry-run,
report, provenance). To still let callers verify endpoint inheritance/override -
the same key yields the same fingerprint, a different key yields a different one -
this exposes a stable, non-reversible ``sha256`` prefix fingerprint instead of
the secret itself. Equal fingerprint + equal ``base_url`` proves inheritance
without leaking the key.
"""

from __future__ import annotations

import hashlib

FINGERPRINT_PREFIX = "sha256:"
DEFAULT_FINGERPRINT_LENGTH = 12
UNSET_FINGERPRINT = ""


def key_fingerprint(
    secret: str | None, *, length: int = DEFAULT_FINGERPRINT_LENGTH
) -> str:
    """Return a stable, non-reversible fingerprint of ``secret``.

    An empty/missing secret yields :data:`UNSET_FINGERPRINT` (so an unset key is
    distinguishable from a set one). A non-empty secret yields
    ``sha256:<hex-prefix>`` which is identical across processes for the same key
    and cannot be reversed to recover it. The full digest length is fixed
    regardless of the key length, so the fingerprint never leaks the key size.
    """
    if not secret:
        return UNSET_FINGERPRINT
    digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    return f"{FINGERPRINT_PREFIX}{digest[:length]}"
