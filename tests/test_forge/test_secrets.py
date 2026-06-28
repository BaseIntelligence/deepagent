"""Unit tests for the non-reversible key-fingerprint helper (offline, no network).

The fingerprint lets endpoint inheritance/override be verified without ever
emitting the raw key. It must be stable, non-reversible, and never contain the
secret itself.
"""

from __future__ import annotations

import hashlib

from swe_forge.forge.secrets import (
    FINGERPRINT_PREFIX,
    UNSET_FINGERPRINT,
    key_fingerprint,
)

SECRET = "sk-super-secret-do-not-print"


class TestKeyFingerprint:
    def test_empty_secret_yields_unset_fingerprint(self) -> None:
        assert key_fingerprint("") == UNSET_FINGERPRINT
        assert key_fingerprint(None) == UNSET_FINGERPRINT

    def test_is_stable_for_same_key(self) -> None:
        assert key_fingerprint(SECRET) == key_fingerprint(SECRET)

    def test_differs_for_different_keys(self) -> None:
        assert key_fingerprint("sk-teacher") != key_fingerprint("sk-panel")

    def test_never_contains_the_raw_secret(self) -> None:
        fp = key_fingerprint(SECRET)
        assert SECRET not in fp
        assert fp.startswith(FINGERPRINT_PREFIX)

    def test_matches_sha256_prefix(self) -> None:
        digest = hashlib.sha256(SECRET.encode("utf-8")).hexdigest()
        assert key_fingerprint(SECRET) == f"{FINGERPRINT_PREFIX}{digest[:12]}"

    def test_length_is_fixed_regardless_of_key_length(self) -> None:
        # A short and a long key produce the same-length fingerprint (no size leak).
        assert len(key_fingerprint("a")) == len(key_fingerprint("a" * 4096))

    def test_custom_length_is_honored(self) -> None:
        fp = key_fingerprint(SECRET, length=8)
        assert (
            fp
            == f"{FINGERPRINT_PREFIX}{hashlib.sha256(SECRET.encode()).hexdigest()[:8]}"
        )
