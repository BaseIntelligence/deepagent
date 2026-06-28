"""Go symbol extraction via a cached ``go/ast`` helper binary.

Go AST parsing must go through the real ``go/ast`` parser (the contract requires
receiver methods and exact spans), so a tiny Go program (``_goparse/goparse.go``)
is compiled once with the host ``go`` toolchain and cached under the user cache
dir; subsequent parses invoke the cached binary directly (no per-call ``go run``
compile). The binary prints a JSON array of symbols on stdout, exits 1 with a
clean parse error on malformed source, and exits 2 on a usage/IO error.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from swe_forge.forge.adapters.base import AdapterError, ParseError, Symbol

_HELPER_DIR = Path(__file__).resolve().parent / "_goparse"
_HELPER_SOURCE = _HELPER_DIR / "goparse.go"
_BUILD_TIMEOUT = 120.0
_PARSE_TIMEOUT = 30.0

# Common install locations probed when ``go`` is not already on ``PATH``.
_GO_FALLBACK_PATHS = (
    "/usr/local/go/bin/go",
    "/usr/lib/go/bin/go",
    str(Path.home() / ".local" / "bin" / "go"),
    str(Path.home() / "go" / "bin" / "go"),
)


class GoToolchainError(AdapterError):
    """Raised when the Go toolchain needed to parse Go source is unavailable."""


def _find_go() -> str:
    found = shutil.which("go")
    if found:
        return found
    for candidate in _GO_FALLBACK_PATHS:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise GoToolchainError(
        "the 'go' toolchain is required to parse Go source but was not found on "
        "PATH; install Go or add it to PATH"
    )


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    path = Path(base) / "swe_forge" / "goparse"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _binary_name() -> str:
    """Cache-key the binary by source digest so source edits rebuild it."""
    digest = hashlib.sha256(_HELPER_SOURCE.read_bytes()).hexdigest()[:16]
    suffix = ".exe" if sys.platform == "win32" else ""
    return f"goparse-{digest}{suffix}"


def _build_helper() -> Path:
    binary = _cache_dir() / _binary_name()
    if binary.exists():
        return binary
    go = _find_go()
    env = {**os.environ, "GOFLAGS": "-mod=mod", "CGO_ENABLED": "0"}
    try:
        result = subprocess.run(
            [go, "build", "-o", str(binary), "."],
            cwd=str(_HELPER_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=_BUILD_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GoToolchainError(f"failed to build the Go parse helper: {exc}") from exc
    if result.returncode != 0 or not binary.exists():
        raise GoToolchainError(
            "failed to build the Go parse helper: "
            f"{(result.stderr or result.stdout).strip()}"
        )
    return binary


def parse_go_symbols(file: str | Path) -> list[Symbol]:
    """Parse a Go file via ``go/ast`` and return its function/method symbols.

    Raises :class:`ParseError` if the source is malformed and
    :class:`GoToolchainError` if the Go toolchain is unavailable.
    """
    path = Path(file)
    binary = _build_helper()
    try:
        result = subprocess.run(
            [str(binary), str(path)],
            capture_output=True,
            text=True,
            timeout=_PARSE_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GoToolchainError(f"failed to run the Go parse helper: {exc}") from exc

    if result.returncode == 1:
        message = (result.stderr or "syntax error in source").strip()
        message = message.replace("parse error: ", "", 1)
        raise ParseError(f"failed to parse Go source {path}: {message}")
    if result.returncode != 0:
        raise GoToolchainError(
            f"Go parse helper failed on {path}: {(result.stderr or '').strip()}"
        )

    records = json.loads(result.stdout)
    return [
        Symbol(
            name=record["name"],
            kind=record["kind"],
            file=str(path),
            start_line=record["start_line"],
            end_line=record["end_line"],
            signature=record.get("signature") or None,
        )
        for record in records
    ]
