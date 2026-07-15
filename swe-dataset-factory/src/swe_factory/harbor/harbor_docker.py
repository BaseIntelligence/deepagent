"""Agent + separate-verifier Docker context builders for Harbor packs.

Contract (VAL-HARBOR-005 + VAL-HARBOR-003/004 plumbing):
- Agent build context is only ``environment/`` and must never contain
  ``solution/`` or ``tests/test.patch``.
- Verifier Dockerfile is separate; its build context is ``tests/`` with the
  held-out ``test.patch``, ``grader.py``, ``config.json``, and ``test.sh``.

M15 pier preflight: product packs historically ship ``FROM deepswe-agent:local``
in ``tests/Dockerfile``. :func:`ensure_deepswe_agent_local` now mints a
**pack-scoped** agent image tag derived from the environment content digest
(``deepswe-agent-<digest>:local``), rewrites the pack ``tests/Dockerfile``
FROM to that tag, and only short-circuits rebuilds when *that* digest-keyed
image already exists — never reusing another pack's environment under a shared
global mintag.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

AGENT_CONTEXT_FORBIDDEN: tuple[str, ...] = (
    "solution",
    "solution.patch",
    "gold.patch",
    "test.patch",
    "tests/test.patch",
    "grader.py",  # grader is verifier-only
)

# Legacy global mintag (historic product default). New ensure paths mint pack-scoped tags.
DEEPSWE_AGENT_LOCAL_TAG = "deepswe-agent:local"
_ENV_AGENT_IMAGE_KEYS = ("SWE_FACTORY_AGENT_IMAGE", "FACTORY_AGENT_IMAGE")
_PACK_AGENT_TAG_PREFIX = "deepswe-agent-"
_PACK_AGENT_TAG_SUFFIX = ":local"
_DEFAULT_DIGEST_LEN = 12


class HarborDockerError(RuntimeError):
    """Docker context / image build failure for Harbor packs."""


@dataclass(frozen=True, slots=True)
class AgentContextResult:
    """Staged agent Docker build context (environment only)."""

    context_dir: Path
    dockerfile: Path
    forbidden_hits: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TestsContextResult:
    """Staged separate-verifier Docker build context (tests only)."""

    context_dir: Path
    dockerfile: Path
    required_files: tuple[str, ...] = (
        "Dockerfile",
        "test.sh",
        "grader.py",
        "config.json",
        "test.patch",
    )


@dataclass(frozen=True, slots=True)
class HarborImagePair:
    """Tagged agent + verifier images for one pack."""

    agent_image: str
    tests_image: str
    agent_context: Path
    tests_context: Path


def pack_root_from_path(pack_dir: Path | str) -> Path:
    root = Path(pack_dir)
    if not root.is_dir():
        raise HarborDockerError(f"pack directory not found: {root}")
    return root


def list_agent_context_paths(context_dir: Path | str) -> list[str]:
    """Return sorted relative file paths under an agent build context."""
    root = Path(context_dir)
    if not root.is_dir():
        return []
    out: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out.append(str(path.relative_to(root)).replace("\\", "/"))
    return out


def scan_agent_context_forbidden(context_dir: Path | str) -> list[str]:
    """Return isolation violations if solution/held-out tests leak into agent context.

    VAL-HARBOR-005: agent env build must not ship solution/ or tests/test.patch.
    """
    root = Path(context_dir)
    if not root.is_dir():
        return [f"agent context missing: {root}"]
    hits: list[str] = []
    # Whole directories that must never land in agent context
    for forbidden_dir in ("solution", "tests"):
        if (root / forbidden_dir).exists():
            hits.append(f"forbidden agent path: {forbidden_dir}/")
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        name = path.name
        lower = rel.lower()
        if name in {"solution.patch", "gold.patch", "test.patch"} or (
            name == "solve.sh" and "solution" in lower
        ):
            hits.append(f"forbidden agent file: {rel}")
        elif lower.endswith("/test.patch") or lower == "test.patch":
            hits.append(f"forbidden held-out test.patch in agent context: {rel}")
        # Guard: agent context should never include the grader that reads held-out config
        if name == "grader.py" and "tests" in lower:
            hits.append(f"forbidden verifier grader in agent context: {rel}")
    return hits


def stage_agent_context(
    pack_dir: Path | str,
    dest: Path | str,
    *,
    overwrite: bool = True,
) -> AgentContextResult:
    """Copy pack ``environment/`` into ``dest`` for agent image builds.

    Explicitly excludes solution/ and tests/* (including test.patch).
    """
    pack = pack_root_from_path(pack_dir)
    env_src = pack / "environment"
    if not env_src.is_dir():
        raise HarborDockerError(f"pack missing environment/: {env_src}")
    dockerfile = env_src / "Dockerfile"
    if not dockerfile.is_file():
        raise HarborDockerError(f"pack missing environment/Dockerfile: {dockerfile}")

    out = Path(dest)
    if out.exists():
        if not overwrite:
            raise HarborDockerError(f"agent context already exists: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # Copy only environment tree contents into context root
    shutil.copytree(
        env_src,
        out,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            "solution",
            "solution.patch",
            "gold.patch",
            "test.patch",
            "__pycache__",
            "*.pyc",
            ".git",
            ".venv",
            "node_modules",
        ),
    )
    # Defense in depth: never allow sibling pack dirs to sneak in
    for name in ("solution", "tests", "instruction.md", "task.toml"):
        stray = out / name
        if stray.exists():
            if stray.is_dir():
                shutil.rmtree(stray)
            else:
                stray.unlink()

    hits = scan_agent_context_forbidden(out)
    return AgentContextResult(
        context_dir=out,
        dockerfile=out / "Dockerfile",
        forbidden_hits=tuple(hits),
    )


def stage_tests_context(
    pack_dir: Path | str,
    dest: Path | str,
    *,
    overwrite: bool = True,
) -> TestsContextResult:
    """Copy pack ``tests/`` into ``dest`` for separate-verifier image builds."""
    pack = pack_root_from_path(pack_dir)
    tests_src = pack / "tests"
    if not tests_src.is_dir():
        raise HarborDockerError(f"pack missing tests/: {tests_src}")
    required = ("Dockerfile", "test.sh", "grader.py", "config.json", "test.patch")
    missing = [name for name in required if not (tests_src / name).is_file()]
    if missing:
        raise HarborDockerError(f"pack tests/ missing required files: {missing}")

    out = Path(dest)
    if out.exists():
        if not overwrite:
            raise HarborDockerError(f"tests context already exists: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(tests_src, out, dirs_exist_ok=True)
    return TestsContextResult(context_dir=out, dockerfile=out / "Dockerfile")


def assert_certified_tests_config(config_path: Path | str) -> dict[str, object]:
    """Enforce VAL-HARBOR-006: non-empty f2p_node_ids (+ readable config)."""
    path = Path(config_path)
    if not path.is_file():
        raise HarborDockerError(f"tests/config.json missing: {path}")
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HarborDockerError(f"tests/config.json invalid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise HarborDockerError("tests/config.json must be an object")
    f2p = raw.get("f2p_node_ids") or []
    if not isinstance(f2p, list) or not [x for x in f2p if str(x).strip()]:
        raise HarborDockerError("tests/config.json requires non-empty f2p_node_ids")
    base = str(raw.get("base_commit") or "").strip()
    if not base:
        raise HarborDockerError("tests/config.json requires base_commit")
    return dict(raw)


def assert_certified_test_patch(test_patch_path: Path | str) -> str:
    """Enforce VAL-HARBOR-006: non-empty held-out test.patch for certified packs."""
    path = Path(test_patch_path)
    if not path.is_file():
        raise HarborDockerError(f"tests/test.patch missing: {path}")
    body = path.read_text(encoding="utf-8")
    if not body.strip():
        raise HarborDockerError("tests/test.patch must be non-empty for certified packs")
    return body


def rewrite_tests_dockerfile_from(
    tests_dockerfile: str,
    *,
    agent_image: str,
) -> str:
    """Point verifier Dockerfile FROM at the just-built agent image."""
    lines = tests_dockerfile.splitlines()
    out: list[str] = []
    replaced = False
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("FROM ") and not replaced:
            out.append(f"FROM {agent_image}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.insert(0, f"FROM {agent_image}")
    text = "\n".join(out)
    return text if text.endswith("\n") else text + "\n"


def resolve_pier_agent_image_tag(
    agent_image: str | None = None,
    *,
    env: dict[str, str] | None = None,
    pack_dir: Path | str | None = None,
) -> str:
    """Resolve the agent image mintag for pier / Oracle paths.

    Precedence:
    1. explicit non-empty ``agent_image``
    2. env ``SWE_FACTORY_AGENT_IMAGE`` / ``FACTORY_AGENT_IMAGE``
    3. pack-scoped tag from ``environment/`` content digest when ``pack_dir`` given
    4. legacy :data:`DEEPSWE_AGENT_LOCAL_TAG` (``deepswe-agent:local``) when no pack
    """
    if agent_image is not None and str(agent_image).strip():
        return str(agent_image).strip()
    environ = env if env is not None else os.environ
    for key in _ENV_AGENT_IMAGE_KEYS:
        raw = (environ.get(key) or "").strip()
        if raw:
            return raw
    if pack_dir is not None:
        return pack_scoped_agent_image_tag(pack_dir)
    return DEEPSWE_AGENT_LOCAL_TAG


def environment_content_digest(
    pack_dir: Path | str,
    *,
    digest_len: int = _DEFAULT_DIGEST_LEN,
) -> str:
    """Content digest of a pack's ``environment/`` tree (Dockerfile + files).

    Used to mint pack-scoped agent image tags so multi-pack pier prep never
    short-circuits on a shared global ``deepswe-agent:local`` built from a
    different pack's Dockerfile.
    """
    pack = pack_root_from_path(pack_dir)
    env_src = pack / "environment"
    if not env_src.is_dir():
        raise HarborDockerError(f"pack missing environment/: {env_src}")
    hasher = hashlib.sha256()
    files: list[Path] = []
    for path in env_src.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name in {"__pycache__", ".DS_Store"} or name.endswith((".pyc", ".pyo")):
            continue
        if any(part in {".git", "__pycache__", ".venv", "node_modules"} for part in path.parts):
            continue
        files.append(path)
    for path in sorted(files, key=lambda p: str(p.relative_to(env_src)).replace("\\", "/")):
        rel = str(path.relative_to(env_src)).replace("\\", "/")
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        try:
            hasher.update(path.read_bytes())
        except OSError as exc:
            raise HarborDockerError(f"cannot read environment file {rel}: {exc}") from exc
        hasher.update(b"\0")
    length = max(8, min(int(digest_len), 64))
    return hasher.hexdigest()[:length]


def pack_scoped_agent_image_tag(
    pack_dir: Path | str,
    *,
    digest: str | None = None,
    digest_len: int = _DEFAULT_DIGEST_LEN,
) -> str:
    """Return ``deepswe-agent-<digest>:local`` for this pack's environment.

    Different environment Dockerfiles (content digests) always produce distinct
    tags so multi-pack pier never reuses the wrong agent base image.
    """
    dig = (digest or environment_content_digest(pack_dir, digest_len=digest_len)).strip().lower()
    if not dig or not re.fullmatch(r"[0-9a-f]{8,64}", dig):
        raise HarborDockerError(f"invalid environment digest for pack-scoped mintag: {dig!r}")
    return f"{_PACK_AGENT_TAG_PREFIX}{dig}{_PACK_AGENT_TAG_SUFFIX}"


def paint_tests_dockerfile_from(
    pack_dir: Path | str,
    *,
    agent_image: str,
    in_place: bool = True,
) -> str:
    """Rewrite pack ``tests/Dockerfile`` first FROM line to ``agent_image``.

    Returns rewritten text. When ``in_place`` is True (default), also persists
    the rewrite onto the pack so pier force-build resolves the pack-scoped base.
    """
    pack = pack_root_from_path(pack_dir)
    tests_df = pack / "tests" / "Dockerfile"
    if not tests_df.is_file():
        raise HarborDockerError(f"pack missing tests/Dockerfile: {tests_df}")
    tag = str(agent_image).strip()
    if not tag:
        raise HarborDockerError("agent_image required to paint tests/Dockerfile FROM")
    original = tests_df.read_text(encoding="utf-8")
    rewritten = rewrite_tests_dockerfile_from(original, agent_image=tag)
    if in_place and rewritten != original:
        tests_df.write_text(rewritten, encoding="utf-8")
    return rewritten


def docker_image_exists(tag: str, *, binary: str = "docker") -> bool:
    """Return True when ``docker image inspect <tag>`` succeeds."""
    if not tag or not str(tag).strip():
        return False
    try:
        completed = subprocess.run(
            [binary, "image", "inspect", str(tag).strip()],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    return completed.returncode == 0


def ensure_deepswe_agent_local(
    pack_dir: Path | str,
    *,
    work_dir: Path | str,
    agent_image: str | None = None,
    binary: str = "docker",
    build_timeout: float = 1800.0,
    stage_only: bool = False,
    force_rebuild: bool = False,
    paint_tests_from: bool = True,
) -> str:
    """Ensure a pack-scoped pier agent mintag exists for this pack's env image.

    Product packs historically ship ``tests/Dockerfile`` with
    ``FROM deepswe-agent:local``. Sharing that global tag across packs is
    unsafe: the first successful build short-circuits later packs onto the
    wrong environment Dockerfile.

    This helper:
    1. Digests the pack ``environment/`` tree
    2. Mints ``deepswe-agent-<digest>:local`` (or uses an explicit override)
    3. Optionally rewrites ``tests/Dockerfile`` FROM to that pack-scoped tag
    4. Builds only when the **pack-scoped** image is missing (or force_rebuild)

    When ``stage_only`` is True, only the context is staged / Dockerfile
    painted and the resolved tag is returned without invoking docker.
    """
    pack = pack_root_from_path(pack_dir)
    # Explicit agent_image wins; else env override; else pack-scoped digest tag.
    # Never default to the legacy global DEEPSWE_AGENT_LOCAL_TAG for multi-pack
    # safety (cross-pack short-circuit poison).
    if agent_image is not None and str(agent_image).strip():
        tag = str(agent_image).strip()
    else:
        env_override = ""
        for key in _ENV_AGENT_IMAGE_KEYS:
            raw = (os.environ.get(key) or "").strip()
            if raw:
                env_override = raw
                break
        tag = env_override or pack_scoped_agent_image_tag(pack)

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    agent_ctx = stage_agent_context(pack, work / "agent_context")
    if agent_ctx.forbidden_hits:
        raise HarborDockerError(
            f"agent context isolation violations: {list(agent_ctx.forbidden_hits)}"
        )
    if not agent_ctx.dockerfile.is_file():
        raise HarborDockerError(f"pack missing environment/Dockerfile under {pack}")

    # Always paint tests/Dockerfile FROM so pier force-build uses pack-scoped base
    if paint_tests_from and (pack / "tests" / "Dockerfile").is_file():
        paint_tests_dockerfile_from(pack, agent_image=tag, in_place=True)

    if stage_only:
        return tag

    # Short-circuit only when *this* pack-scoped tag already exists. Never skip
    # build based on the legacy global deepswe-agent:local (cross-pack poison).
    if not force_rebuild and docker_image_exists(tag, binary=binary):
        return tag

    docker_build(
        context_dir=agent_ctx.context_dir,
        tag=tag,
        binary=binary,
        timeout=build_timeout,
    )
    return tag


def docker_build(
    *,
    context_dir: Path | str,
    tag: str,
    dockerfile_name: str = "Dockerfile",
    binary: str = "docker",
    timeout: float = 600.0,
) -> None:
    """Build an image from a local context directory (docker CLI)."""
    import subprocess

    ctx = Path(context_dir)
    dockerfile = ctx / dockerfile_name
    if not dockerfile.is_file():
        raise HarborDockerError(f"Dockerfile not found: {dockerfile}")
    # Isolation check for agent tags
    if "agent" in tag.lower():
        hits = scan_agent_context_forbidden(ctx)
        if hits:
            raise HarborDockerError(f"agent build context isolation failed: {hits}")

    try:
        completed = subprocess.run(
            [
                binary,
                "build",
                "-t",
                tag,
                "-f",
                str(dockerfile),
                str(ctx),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise HarborDockerError("docker binary not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise HarborDockerError(f"docker build timed out for {tag!r}") from exc
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "")[-4000:]
        raise HarborDockerError(f"docker build failed for {tag!r}: {tail}")


def build_agent_and_tests_images(
    pack_dir: Path | str,
    *,
    work_dir: Path | str,
    agent_tag: str,
    tests_tag: str,
    binary: str = "docker",
    build_timeout: float = 600.0,
    stage_only: bool = False,
) -> HarborImagePair:
    """Stage contexts and optionally build agent + separate-verifier images.

    When ``stage_only`` is True, no docker binary is invoked (unit/offline path).
    """
    pack = pack_root_from_path(pack_dir)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    # Enforce certified pack artifacts before context staging
    assert_certified_tests_config(pack / "tests" / "config.json")
    assert_certified_test_patch(pack / "tests" / "test.patch")
    sol = pack / "solution" / "solution.patch"
    if not sol.is_file() or not sol.read_text(encoding="utf-8").strip():
        raise HarborDockerError("solution/solution.patch must be non-empty")

    agent_ctx = stage_agent_context(pack, work / "agent_context")
    if agent_ctx.forbidden_hits:
        raise HarborDockerError(
            f"agent context isolation violations: {list(agent_ctx.forbidden_hits)}"
        )
    tests_ctx = stage_tests_context(pack, work / "tests_context")

    # Rewrite verifier FROM to the agent tag we are about to build
    original_df = (tests_ctx.context_dir / "Dockerfile").read_text(encoding="utf-8")
    (tests_ctx.context_dir / "Dockerfile").write_text(
        rewrite_tests_dockerfile_from(original_df, agent_image=agent_tag),
        encoding="utf-8",
    )

    if not stage_only:
        docker_build(
            context_dir=agent_ctx.context_dir,
            tag=agent_tag,
            binary=binary,
            timeout=build_timeout,
        )
        docker_build(
            context_dir=tests_ctx.context_dir,
            tag=tests_tag,
            binary=binary,
            timeout=build_timeout,
        )

    return HarborImagePair(
        agent_image=agent_tag,
        tests_image=tests_tag,
        agent_context=agent_ctx.context_dir,
        tests_context=tests_ctx.context_dir,
    )


def summarize_agent_context(context_dir: Path | str) -> dict[str, object]:
    """Inspect helper used by CLI/tests for isolation evidence."""
    paths = list_agent_context_paths(context_dir)
    hits = scan_agent_context_forbidden(context_dir)
    return {
        "files": paths,
        "file_count": len(paths),
        "forbidden_hits": hits,
        "isolated": len(hits) == 0,
        "has_dockerfile": (Path(context_dir) / "Dockerfile").is_file(),
        "has_solution_dir": (Path(context_dir) / "solution").exists(),
        "has_test_patch": any(p.endswith("test.patch") for p in paths),
    }


@dataclass
class FakeHarborDocker:
    """In-memory docker build recorder for offline unit tests."""

    builds: list[dict[str, str]] = field(default_factory=list)
    fail_tags: set[str] = field(default_factory=set)

    def build(self, *, context_dir: Path | str, tag: str, **_: object) -> None:
        if tag in self.fail_tags:
            raise HarborDockerError(f"fake build fail: {tag}")
        # Honor agent isolation invariant even in fakes
        if "agent" in tag.lower():
            hits = scan_agent_context_forbidden(context_dir)
            if hits:
                raise HarborDockerError(f"agent build context isolation failed: {hits}")
        self.builds.append({"context": str(context_dir), "tag": tag})


def remove_images(tags: Iterable[str], *, binary: str = "docker") -> None:
    """Best-effort rmi for mission Harbor images."""
    import subprocess

    for tag in tags:
        if not tag:
            continue
        # Only allow mission prefixes
        if not (
            tag.startswith("harbor-sdf-") or tag.startswith("sdf-") or tag.startswith("harbor-sdf")
        ):
            continue
        subprocess.run(
            [binary, "rmi", "-f", tag],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )


__all__ = [
    "AGENT_CONTEXT_FORBIDDEN",
    "DEEPSWE_AGENT_LOCAL_TAG",
    "AgentContextResult",
    "FakeHarborDocker",
    "HarborDockerError",
    "HarborImagePair",
    "TestsContextResult",
    "assert_certified_test_patch",
    "assert_certified_tests_config",
    "build_agent_and_tests_images",
    "docker_build",
    "docker_image_exists",
    "ensure_deepswe_agent_local",
    "environment_content_digest",
    "list_agent_context_paths",
    "pack_root_from_path",
    "pack_scoped_agent_image_tag",
    "paint_tests_dockerfile_from",
    "remove_images",
    "resolve_pier_agent_image_tag",
    "rewrite_tests_dockerfile_from",
    "scan_agent_context_forbidden",
    "stage_agent_context",
    "stage_tests_context",
    "summarize_agent_context",
]
