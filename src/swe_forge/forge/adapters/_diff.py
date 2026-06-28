"""Git-compatible unified-diff generation and application.

Mutation generators must emit patches that apply cleanly with ``git apply`` in
either direction and round-trip byte-for-byte. Rather than hand-roll unified
diffs (and the ``\\ No newline at end of file`` edge cases), this module drives
``git`` itself in an isolated, throwaway temp directory so the produced diff is
exactly what ``git apply`` expects.

Determinism: ``git`` is invoked with global/system config neutralised
(``GIT_CONFIG_GLOBAL``/``GIT_CONFIG_SYSTEM`` -> ``/dev/null``) and CRLF
translation disabled, so identical inputs yield byte-identical diffs regardless
of the host's git configuration.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

_GIT_TIMEOUT = 30.0


class PatchError(RuntimeError):
    """Raised when generating or applying a patch via ``git`` fails."""


def _git_env() -> dict[str, str]:
    """Return an environment that isolates ``git`` from host configuration."""
    return {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
    }


def _git(cwd: Path, *args: str) -> str:
    """Run ``git`` in ``cwd`` with deterministic config; return stdout."""
    try:
        result = subprocess.run(
            ["git", "-c", "core.autocrlf=false", "-c", "core.safecrlf=false", *args],
            cwd=str(cwd),
            env=_git_env(),
            capture_output=True,
            timeout=_GIT_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PatchError(f"git {' '.join(args)} failed to run: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", "replace").strip()
        raise PatchError(f"git {' '.join(args)} exited {result.returncode}: {detail}")
    return result.stdout.decode("utf-8", "replace")


def _normalize_rel(rel_path: str) -> str:
    """Return ``rel_path`` as a clean, repo-relative POSIX path."""
    rel = Path(str(rel_path).lstrip("/")).as_posix()
    if not rel or rel.startswith("../"):
        raise PatchError(f"patch path must be repo-relative; got {rel_path!r}")
    return rel


def make_patch(rel_path: str, original: bytes, modified: bytes) -> str:
    """Return a ``git apply``-compatible diff turning ``original`` into ``modified``.

    The diff is anchored at ``rel_path`` (header paths ``a/<rel>`` / ``b/<rel>``)
    so it applies at a repo root. Returns ``""`` when the two byte strings are
    identical.
    """
    if original == modified:
        return ""
    rel = _normalize_rel(rel_path)
    with tempfile.TemporaryDirectory(prefix="forge-diff-") as tmp:
        root = Path(tmp)
        _git(root, "init", "-q")
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original)
        _git(root, "add", "--", rel)
        target.write_bytes(modified)
        return _git(root, "diff", "--no-color", "--no-ext-diff", "--", rel)


def apply_patch(original: bytes, diff: str, rel_path: str) -> bytes:
    """Apply ``diff`` to ``original`` (the content of ``rel_path``); return the result.

    Raises :class:`PatchError` if the patch does not apply cleanly, mirroring how
    a validator's ``git apply`` would reject it.
    """
    rel = _normalize_rel(rel_path)
    with tempfile.TemporaryDirectory(prefix="forge-apply-") as tmp:
        root = Path(tmp)
        _git(root, "init", "-q")
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original)
        patch_file = root / "_forge.patch"
        patch_file.write_text(diff, encoding="utf-8")
        _git(root, "apply", "--whitespace=nowarn", "--", str(patch_file))
        return target.read_bytes()


def make_multi_patch(changes: Sequence[tuple[str, bytes, bytes]]) -> str:
    """Return one ``git apply``-compatible diff spanning several files.

    ``changes`` is a sequence of ``(rel_path, original, modified)`` triples. The
    per-file diffs are emitted in a stable order (sorted by repo-relative path)
    so the combined patch is byte-identical for identical inputs; files whose two
    byte strings are equal contribute nothing. Returns ``""`` when nothing
    changed.
    """
    sections: list[str] = []
    for rel_path, original, modified in sorted(
        changes, key=lambda change: _normalize_rel(change[0])
    ):
        section = make_patch(rel_path, original, modified)
        if section:
            sections.append(section)
    return "".join(sections)


def apply_multi_patch(originals: Mapping[str, bytes], diff: str) -> dict[str, bytes]:
    """Apply a multi-file ``diff`` over ``originals`` (rel_path -> bytes).

    Lays every original file down in one throwaway git repo, applies the combined
    diff once (so cross-file patches are validated together exactly as a
    validator's ``git apply`` would), and returns the resulting bytes for each
    input path. Raises :class:`PatchError` if the patch does not apply cleanly.
    """
    rels = {rel: _normalize_rel(rel) for rel in originals}
    with tempfile.TemporaryDirectory(prefix="forge-apply-multi-") as tmp:
        root = Path(tmp)
        _git(root, "init", "-q")
        for rel, normalized in sorted(rels.items()):
            target = root / normalized
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(originals[rel])
            _git(root, "add", "--", normalized)
        patch_file = root / "_forge.patch"
        patch_file.write_text(diff, encoding="utf-8")
        _git(root, "apply", "--whitespace=nowarn", "--", str(patch_file))
        return {
            rel: (root / normalized).read_bytes() for rel, normalized in rels.items()
        }
