"""Docker hygiene for envbuild / DeepAgent agent images (VAL-ENVR-006, VAL-ENVR-008).

Owned container/image name prefixes:
- ``sdf-`` / ``sdf-env-`` (default envbuild)
- ``deepagent-`` / ``deepagent-env-`` (DeepAgent product path)
- ``harbor-sdf-`` (Harbor pack agent/verifier images)

Off-limits names are *never* stopped, removed, or renamed:
- ``mission-test-pg``
- ``challenge-prism*``
- ``acproxy``

Also provides:
- leftover container sweeps
- owned image prune helpers
- free-disk fail-closed gate (before builds)
- concurrency ceiling constant (architecture ≤16–24)
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from typing import Protocol

# Concurrent Pier-ish / envbuild jobs ceiling (architecture band ≤16–24).
MAX_CONCURRENT_ENVBUILD_JOBS = 16
MAX_CONCURRENT_PIER_JOBS = 16
CONCURRENCY_HINT = (
    "Recommended concurrent envbuild/Pier jobs: ≤16 (hard ceiling 24). "
    f"Code default MAX_CONCURRENT_ENVBUILD_JOBS={MAX_CONCURRENT_ENVBUILD_JOBS}."
)

# Minimum free disk (bytes) before starting a dual-build or image bake.
DEFAULT_MIN_FREE_DISK_BYTES = 5 * 1024 * 1024 * 1024  # 5 GiB

_OWNED_CONTAINER_PREFIXES = ("sdf-", "deepagent-", "harbor-sdf-")
_OWNED_IMAGE_PREFIXES = ("sdf-", "sdf-env-", "deepagent-", "deepagent-env-", "harbor-sdf-")

# Match bare or prefixed off-limits names including challenge-prism-* variants.
_OFF_LIMITS_RE = re.compile(
    r"(?:^|/)(?:mission-test-pg(?:$|[^a-z0-9])|challenge-prism(?:$|[^a-z0-9])|acproxy(?:$|[^a-z0-9]))",
    re.IGNORECASE,
)


class HygieneError(RuntimeError):
    """Raised when an operation would violate hygiene / disk / name policy."""


class ContainerLister(Protocol):
    def list_containers(self, *, all_containers: bool = True) -> list[str]: ...

    def remove_container(self, ref: str) -> None: ...


class ImagePruner(Protocol):
    def remove_image(self, ref: str) -> None: ...

    def list_images(self) -> list[str]: ...


def is_off_limits_name(name: str) -> bool:
    """True when *name* must never be touched by factory docker ops."""
    n = (name or "").strip().lower()
    if not n:
        return False
    if n == "acproxy" or n.startswith("acproxy.") or n.startswith("acproxy-"):
        return True
    if n == "mission-test-pg" or n.startswith("mission-test-pg."):
        return True
    if n.startswith("challenge-prism"):
        return True
    return bool(_OFF_LIMITS_RE.search(n))


def is_owned_container_name(name: str) -> bool:
    """True when *name* is under an owned factory prefix and not off-limits."""
    if not name or is_off_limits_name(name):
        return False
    return any(name.startswith(p) for p in _OWNED_CONTAINER_PREFIXES)


def is_owned_image_ref(ref: str) -> bool:
    """True when a docker image ref uses an owned prefix (tag or bare name)."""
    if not ref or is_off_limits_name(ref):
        return False
    # Strip digest suffix if present (name@sha256:…)
    base = ref.split("@", 1)[0]
    # Drop registry host if path-like — still match tag basename prefixes.
    name = base.split("/")[-1]
    return any(name.startswith(p) or base.startswith(p) for p in _OWNED_IMAGE_PREFIXES)


def assert_safe_container_name(name: str) -> None:
    """Refuse off-limits and non-owned container names."""
    if is_off_limits_name(name):
        raise HygieneError(f"refusing to operate on off-limits container {name!r}")
    if not is_owned_container_name(name):
        raise HygieneError(
            f"mission containers must start with one of {_OWNED_CONTAINER_PREFIXES}; got {name!r}"
        )


def free_disk_bytes(path: str = "/") -> int:
    """Return free bytes on the filesystem hosting *path*."""
    usage = shutil.disk_usage(path)
    return int(usage.free)


@dataclass(frozen=True, slots=True)
class DiskGateResult:
    ok: bool
    free_bytes: int
    required_bytes: int
    path: str
    reason: str = ""


def check_disk_for_envbuild(
    *,
    min_free_bytes: int = DEFAULT_MIN_FREE_DISK_BYTES,
    path: str = "/",
) -> DiskGateResult:
    """Fail-closed disk gate before docker layer growth (VAL-ENVR-008)."""
    free = free_disk_bytes(path)
    if free < min_free_bytes:
        return DiskGateResult(
            ok=False,
            free_bytes=free,
            required_bytes=min_free_bytes,
            path=path,
            reason=(
                f"insufficient free disk at {path}: free={free} bytes "
                f"(need ≥ {min_free_bytes}); refusing envbuild to avoid orphan layers"
            ),
        )
    return DiskGateResult(
        ok=True,
        free_bytes=free,
        required_bytes=min_free_bytes,
        path=path,
    )


def require_disk_for_envbuild(
    *,
    min_free_bytes: int = DEFAULT_MIN_FREE_DISK_BYTES,
    path: str = "/",
) -> DiskGateResult:
    """Raise :class:`HygieneError` when free disk is below threshold."""
    result = check_disk_for_envbuild(min_free_bytes=min_free_bytes, path=path)
    if not result.ok:
        raise HygieneError(result.reason)
    return result


def remove_leftover_owned_containers(
    docker: ContainerLister,
    *,
    prefixes: tuple[str, ...] = _OWNED_CONTAINER_PREFIXES,
) -> list[str]:
    """Remove leftover owned containers; never touch off-limits names."""
    removed: list[str] = []
    for name in docker.list_containers(all_containers=True):
        if is_off_limits_name(name):
            continue
        if not any(name.startswith(p) for p in prefixes):
            continue
        try:
            assert_safe_container_name(name)
            docker.remove_container(name)
            removed.append(name)
        except Exception:
            continue
    return removed


def list_owned_images(docker: ImagePruner) -> list[str]:
    """Return image refs whose names start with an owned prefix."""
    try:
        refs = docker.list_images()
    except Exception:
        return []
    return [r for r in refs if is_owned_image_ref(r)]


def prune_owned_images(
    docker: ImagePruner,
    *,
    image_refs: list[str] | None = None,
    keep: frozenset[str] | None = None,
) -> list[str]:
    """Best-effort remove owned images (VAL-ENVR-008 reclamation).

    Only removes refs that pass :func:`is_owned_image_ref`. Off-limits and
    foreign images are never passed to docker rmi.
    """
    keep = keep or frozenset()
    candidates = image_refs if image_refs is not None else list_owned_images(docker)
    pruned: list[str] = []
    for ref in candidates:
        if ref in keep:
            continue
        if not is_owned_image_ref(ref):
            continue
        if is_off_limits_name(ref):
            continue
        try:
            docker.remove_image(ref)
            pruned.append(ref)
        except Exception:
            continue
    return pruned


__all__ = [
    "CONCURRENCY_HINT",
    "DEFAULT_MIN_FREE_DISK_BYTES",
    "DiskGateResult",
    "HygieneError",
    "MAX_CONCURRENT_ENVBUILD_JOBS",
    "MAX_CONCURRENT_PIER_JOBS",
    "assert_safe_container_name",
    "check_disk_for_envbuild",
    "free_disk_bytes",
    "is_off_limits_name",
    "is_owned_container_name",
    "is_owned_image_ref",
    "list_owned_images",
    "prune_owned_images",
    "remove_leftover_owned_containers",
    "require_disk_for_envbuild",
]
