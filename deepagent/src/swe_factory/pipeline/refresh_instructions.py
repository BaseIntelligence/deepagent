"""Instruction-only re-export for certified real_pr Harbor packs (VAL-DPRMPT-004).

Rewrites ``instruction.md`` in place using the full DeepSWE-style builder from
:mod:`swe_factory.pipeline.ship_real_pr` without re-running dual-run / oracle.
Optionally re-fetches PR ``body`` from GitHub REST and persists it onto live
materials ``meta.json`` so subsequent generate/load paths stay rich.

Never pads fixtures into product trees. Always re-runs
:func:`scan_instruction_gold_leak` per pack.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from swe_factory.harbor.real_pack import scan_instruction_gold_leak
from swe_factory.pipeline.deepswe_prompt import (
    PROMPT_STYLE_DEEPSWE_V1,
    has_deepswe_footer,
    has_provenance_fingerprints,
    style_ok,
)
from swe_factory.pipeline.ship_real_pr import (
    RealPrMaterial,
    build_real_pr_agent_instruction,
    sanitize_pr_body_for_prompt,
)

# M18 DeepSWE-true style is product default; keep FULL_V1 as historical alias name.
PROMPT_STYLE_FULL_V1 = PROMPT_STYLE_DEEPSWE_V1
PROMPT_STYLE_DEEPSWE = PROMPT_STYLE_DEEPSWE_V1
_AGENT_INSTRUCTION_MIN_CHARS = 200
_STUB_OPENING = re.compile(r"^Merged PR #\d+\s+on\s+", re.MULTILINE)


class RefreshInstructionsError(RuntimeError):
    """Instruction-only re-export failed fail-closed."""


@dataclass(frozen=True, slots=True)
class PackInstructionRefresh:
    """Per-pack refresh result."""

    task_id: str
    pack_dir: str
    ok: bool
    chars_before: int
    chars_after: int
    body_chars: int
    body_source: str
    sections_ok: bool
    leak_hits: tuple[str, ...] = ()
    error: str = ""
    prompt_style: str = PROMPT_STYLE_FULL_V1

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "pack_dir": self.pack_dir,
            "ok": self.ok,
            "chars_before": self.chars_before,
            "chars_after": self.chars_after,
            "body_chars": self.body_chars,
            "body_source": self.body_source,
            "sections_ok": self.sections_ok,
            "leak_hits": list(self.leak_hits),
            "error": self.error,
            "prompt_style": self.prompt_style,
        }


@dataclass(frozen=True, slots=True)
class RefreshInstructionsResult:
    """Wave-level instruction refresh summary."""

    root: str
    ok: bool
    refreshed: int
    failed: int
    packs: tuple[PackInstructionRefresh, ...]
    prompt_style: str = PROMPT_STYLE_FULL_V1
    materials_root: str = ""
    fetched_bodies: int = 0
    stamped_manifest: bool = False
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "ok": self.ok,
            "refreshed": self.refreshed,
            "failed": self.failed,
            "prompt_style": self.prompt_style,
            "materials_root": self.materials_root,
            "fetched_bodies": self.fetched_bodies,
            "stamped_manifest": self.stamped_manifest,
            "message": self.message,
            "packs": [p.to_dict() for p in self.packs],
        }


@dataclass(slots=True)
class _ResolvedBody:
    text: str
    source: str  # meta | pack | github | empty


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_task_meta(pack_dir: Path) -> dict[str, Any]:
    tom_path = pack_dir / "task.toml"
    if not tom_path.is_file():
        return {}
    try:
        blob = tomllib.loads(tom_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError):
        return {}
    meta = blob.get("metadata") if isinstance(blob, dict) else None
    return dict(meta) if isinstance(meta, dict) else {}


def _read_solution_patch(pack_dir: Path) -> str:
    for cand in (
        pack_dir / "solution" / "solution.patch",
        pack_dir / "solution.patch",
    ):
        if cand.is_file():
            return cand.read_text(encoding="utf-8", errors="replace")
    return ""


def _read_test_patch(pack_dir: Path) -> str:
    for cand in (
        pack_dir / "tests" / "test.patch",
        pack_dir / "test.patch",
    ):
        if cand.is_file():
            return cand.read_text(encoding="utf-8", errors="replace")
    return ""


def _parse_repo_slug(repository_url: str) -> str:
    """Normalize clone/html URL → ``owner/name`` slug for GitHub REST."""
    raw = (repository_url or "").strip()
    if not raw:
        return ""
    raw = raw.rstrip("/")
    if raw.endswith(".git"):
        raw = raw[: -len(".git")]
    # https://github.com/owner/name  |  git@github.com:owner/name
    m = re.search(r"github\.com[:/]+([^/]+)/([^/]+)$", raw, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", raw):
        return raw
    return ""


def _pr_number_from_task_id(task_id: str) -> int | None:
    # task ids look like realpr-<slug>-<pr>
    tail = task_id.rsplit("-", 1)
    if len(tail) != 2:
        return None
    try:
        return int(tail[1])
    except ValueError:
        return None


def _load_meta_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _persist_body_to_materials(meta_path: Path, body: str) -> None:
    """Best-effort write body/pr_body onto materials meta.json (VAL-DPRMPT-001)."""
    if not meta_path.is_file() or not body.strip():
        return
    meta = _load_meta_json(meta_path)
    if not meta:
        return
    meta["body"] = body
    meta["pr_body"] = body
    meta["body_refreshed_at"] = _now_iso()
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fetch_pr_body_via_github(
    repo_slug: str,
    pr_number: int,
    *,
    github_client: Any | None = None,
    get_pull: Callable[[str, int], Mapping[str, Any]] | None = None,
) -> str:
    """Fetch PR body from GitHub REST (live) or injectable get_pull."""
    if get_pull is not None:
        payload = get_pull(repo_slug, pr_number)
        return str(payload.get("body") or "").strip()
    client = github_client
    if client is None:
        from swe_factory.sources.github import GitHubClient

        client = GitHubClient.from_env()
    pull = client.get_pull(repo_slug, int(pr_number))
    if not isinstance(pull, dict):
        return ""
    return str(pull.get("body") or "").strip()


def _resolve_body(
    *,
    materials_meta: Mapping[str, Any],
    materials_meta_path: Path | None,
    pack_meta: Mapping[str, Any],
    repository_url: str,
    pr_number: int | None,
    fetch_github: bool,
    github_client: Any | None,
    get_pull: Callable[[str, int], Mapping[str, Any]] | None,
) -> _ResolvedBody:
    # 1) live materials
    body = str(materials_meta.get("body") or materials_meta.get("pr_body") or "").strip()
    if body:
        return _ResolvedBody(text=body, source="meta")
    # 2) pack-local notes (rare)
    pack_body = str(pack_meta.get("body") or pack_meta.get("pr_body") or "").strip()
    if pack_body:
        return _ResolvedBody(text=pack_body, source="pack")
    # 3) live GitHub
    if fetch_github and pr_number is not None:
        slug = _parse_repo_slug(repository_url)
        if slug:
            try:
                fetched = _fetch_pr_body_via_github(
                    slug,
                    pr_number,
                    github_client=github_client,
                    get_pull=get_pull,
                )
            except Exception as exc:  # noqa: BLE001 — fail soft to title-only body
                return _ResolvedBody(text="", source=f"github_error:{type(exc).__name__}")
            if fetched:
                if materials_meta_path is not None:
                    _persist_body_to_materials(materials_meta_path, fetched)
                return _ResolvedBody(text=fetched, source="github")
            return _ResolvedBody(text="", source="github_empty")
    return _ResolvedBody(text="", source="empty")


def _sources_from_pack_and_materials(
    pack_dir: Path,
    materials_meta: Mapping[str, Any],
    solution_patch: str,
) -> tuple[str, ...]:
    src = materials_meta.get("src") or materials_meta.get("source_files") or []
    if isinstance(src, list) and src:
        return tuple(str(p) for p in src if str(p).strip())
    # Derive from solution.patch file paths
    paths: list[str] = []
    for line in solution_patch.splitlines():
        if line.startswith("diff --git "):
            # diff --git a/path b/path
            parts = line.split()
            if len(parts) >= 4:
                b = parts[3]
                if b.startswith("b/"):
                    b = b[2:]
                if b and b not in paths:
                    paths.append(b)
    if paths:
        return tuple(paths)
    # Fallback: tests config sometimes lists nothing useful
    _ = pack_dir
    return ()


def _tests_from_materials(materials_meta: Mapping[str, Any], test_patch: str) -> tuple[str, ...]:
    tests = materials_meta.get("tests") or materials_meta.get("test_files") or []
    if isinstance(tests, list) and tests:
        return tuple(str(p) for p in tests if str(p).strip())
    paths: list[str] = []
    for line in test_patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                b = parts[3]
                if b.startswith("b/"):
                    b = b[2:]
                if b and b not in paths:
                    paths.append(b)
    return tuple(paths)


def _sections_ok(text: str) -> bool:
    """Accept DeepSWE-true register (M18) or legacy full multi-section (compat)."""
    body = text or ""
    lower = body.lower()
    long_enough = len(body.strip()) >= _AGENT_INSTRUCTION_MIN_CHARS
    not_pure_stub = not (
        body.lstrip().startswith("Merged PR #") and "## " not in body and "# " not in body[:20]
    )
    if _STUB_OPENING.search(body) and not has_deepswe_footer(body):
        return False
    # Preferred: DeepSWE-true (no provenance fingerprints + footer + outcomes)
    if (
        long_enough
        and not_pure_stub
        and not has_provenance_fingerprints(body)
        and has_deepswe_footer(body)
        and style_ok(body)
    ):
        return True
    # Legacy M17 multi-section (still greppable for older packs mid-refresh)
    has_context = "context" in lower
    has_desc = "pr description" in lower or ("description" in lower and "##" in body)
    has_behav = "behavioural" in lower or "behavioral" in lower
    has_deliver = "deliverable" in lower
    return bool(
        has_context and has_desc and has_behav and has_deliver and long_enough and not_pure_stub
    )


def material_from_pack(
    pack_dir: Path,
    *,
    materials_root: Path | None = None,
    fetch_github: bool = True,
    github_client: Any | None = None,
    get_pull: Callable[[str, int], Mapping[str, Any]] | None = None,
) -> tuple[RealPrMaterial, _ResolvedBody]:
    """Assemble a :class:`RealPrMaterial` from pack tree + optional live materials."""
    pack_dir = Path(pack_dir)
    if not pack_dir.is_dir():
        raise RefreshInstructionsError(f"pack dir missing: {pack_dir}")
    task_id = pack_dir.name
    pack_meta = _read_task_meta(pack_dir)
    repository_url = str(pack_meta.get("repository_url") or pack_meta.get("repo_url") or "").strip()
    base_commit = str(
        pack_meta.get("base_commit_hash") or pack_meta.get("base_commit") or ""
    ).strip()
    language = str(pack_meta.get("language") or "python").strip().lower() or "python"
    license_name = str(pack_meta.get("license") or "MIT").strip() or "MIT"
    pr_number: int | None
    raw_pr = pack_meta.get("pr") or pack_meta.get("pr_number")
    if raw_pr is not None:
        try:
            pr_number = int(raw_pr)
        except (TypeError, ValueError):
            pr_number = _pr_number_from_task_id(task_id)
    else:
        pr_number = _pr_number_from_task_id(task_id)

    materials_meta: dict[str, Any] = {}
    materials_meta_path: Path | None = None
    materials_dir = ""
    if materials_root is not None:
        cand = Path(materials_root) / task_id
        if cand.is_dir():
            materials_dir = str(cand)
            materials_meta_path = cand / "meta.json"
            materials_meta = _load_meta_json(materials_meta_path)
            if not repository_url:
                repository_url = str(
                    materials_meta.get("url") or materials_meta.get("repository_url") or ""
                ).strip()
                if not repository_url and materials_meta.get("repo"):
                    repository_url = f"https://github.com/{materials_meta['repo']}.git"
            if not base_commit:
                base_commit = str(
                    materials_meta.get("base") or materials_meta.get("base_commit") or ""
                ).strip()
            if pr_number is None and materials_meta.get("pr") is not None:
                try:
                    pr_number = int(materials_meta["pr"])
                except (TypeError, ValueError):
                    pr_number = pr_number
            if not language or language == "python":
                language = str(materials_meta.get("language") or language).strip().lower()

    title = str(materials_meta.get("title") or pack_meta.get("title") or task_id).strip() or task_id
    solution = _read_solution_patch(pack_dir)
    test_patch = _read_test_patch(pack_dir)
    if materials_dir:
        sol_alt = Path(materials_dir) / "solution.patch"
        test_alt = Path(materials_dir) / "test.patch"
        if not solution.strip() and sol_alt.is_file():
            solution = sol_alt.read_text(encoding="utf-8", errors="replace")
        if not test_patch.strip() and test_alt.is_file():
            test_patch = test_alt.read_text(encoding="utf-8", errors="replace")

    body_res = _resolve_body(
        materials_meta=materials_meta,
        materials_meta_path=materials_meta_path,
        pack_meta=pack_meta,
        repository_url=repository_url,
        pr_number=pr_number,
        fetch_github=fetch_github,
        github_client=github_client,
        get_pull=get_pull,
    )

    sources = _sources_from_pack_and_materials(pack_dir, materials_meta, solution)
    tests = _tests_from_materials(materials_meta, test_patch)
    hunk_raw = materials_meta.get("source_hunk_count")
    try:
        hunk_n = int(hunk_raw) if hunk_raw is not None else None
    except (TypeError, ValueError):
        hunk_n = None
    if hunk_n is None and solution:
        counted = sum(1 for line in solution.splitlines() if line.startswith("@@"))
        hunk_n = counted if counted > 0 else None

    agent_instr = str(materials_meta.get("agent_instruction") or "").strip()
    material = RealPrMaterial(
        task_id=task_id,
        repository_url=repository_url,
        base_commit=base_commit,
        language=language,
        license=license_name,
        pr_number=pr_number,
        title=title,
        body=body_res.text,
        source_files=sources,
        test_files=tests,
        solution_patch=solution if solution.endswith("\n") or not solution else solution + "\n",
        test_patch=test_patch if test_patch.endswith("\n") or not test_patch else test_patch + "\n",
        materials_dir=materials_dir,
        discovery_path=str(materials_meta.get("discovery_path") or ""),
        source_hunk_count=hunk_n,
        agent_instruction=agent_instr,
    )
    return material, body_res


def refresh_pack_instruction(
    pack_dir: Path | str,
    *,
    materials_root: Path | str | None = None,
    fetch_github: bool = True,
    dry_run: bool = False,
    github_client: Any | None = None,
    get_pull: Callable[[str, int], Mapping[str, Any]] | None = None,
    force_offline: bool | None = None,
    use_llm: bool | None = None,
) -> PackInstructionRefresh:
    """Rewrite one pack's instruction.md with DeepSWE-style builder + leak scan.

    *force_offline* / *use_llm* control OpenRouter rewrite. When neither is set,
    live LLM is used iff ``OPENROUTER_API_KEY`` is present (product default).
    Unit tests should pass ``force_offline=True``.
    """
    pack = Path(pack_dir)
    task_id = pack.name
    instr_path = pack / "instruction.md"
    before = ""
    if instr_path.is_file():
        before = instr_path.read_text(encoding="utf-8", errors="replace")
    chars_before = len(before)

    # Resolve offline policy: use_llm=False or force_offline=True ⇒ no network.
    offline_flag = force_offline
    if use_llm is False:
        offline_flag = True
    elif use_llm is True:
        offline_flag = False

    try:
        material, body_res = material_from_pack(
            pack,
            materials_root=Path(materials_root) if materials_root else None,
            fetch_github=fetch_github,
            github_client=github_client,
            get_pull=get_pull,
        )
        if not material.repository_url or len(material.base_commit) != 40:
            raise RefreshInstructionsError(
                f"{task_id}: incomplete pack meta "
                f"(url={material.repository_url!r}, base={material.base_commit!r})"
            )
        # Even empty solution is rare on certified packs; still require for leak-safe.
        instruction = build_real_pr_agent_instruction(
            material,
            force_offline=offline_flag,
        )
        # Defense-in-depth: re-sanitize embedded body (builder already sanitizes).
        _ = sanitize_pr_body_for_prompt(material.body)
        sol_hits = scan_instruction_gold_leak(instruction, material.solution_patch)
        test_hits = scan_instruction_gold_leak(instruction, material.test_patch)
        leak_hits = tuple(dict.fromkeys([*sol_hits, *test_hits]))
        # Hard fail if agent-visible text still carries mining fingerprints.
        fingerprints = (
            ()
            if not has_provenance_fingerprints(instruction)
            else tuple(h for h in ("provenance",))
        )
        sections = _sections_ok(instruction) and not has_provenance_fingerprints(instruction)
        if leak_hits:
            return PackInstructionRefresh(
                task_id=task_id,
                pack_dir=str(pack),
                ok=False,
                chars_before=chars_before,
                chars_after=len(instruction),
                body_chars=len(material.body or ""),
                body_source=body_res.source,
                sections_ok=sections,
                leak_hits=leak_hits,
                error=f"gold leak scan failed: {list(leak_hits)}",
            )
        if fingerprints or has_provenance_fingerprints(instruction):
            return PackInstructionRefresh(
                task_id=task_id,
                pack_dir=str(pack),
                ok=False,
                chars_before=chars_before,
                chars_after=len(instruction),
                body_chars=len(material.body or ""),
                body_source=body_res.source,
                sections_ok=False,
                error="instruction still contains provenance fingerprints (VAL-DSTYLE-001)",
            )
        if not sections:
            return PackInstructionRefresh(
                task_id=task_id,
                pack_dir=str(pack),
                ok=False,
                chars_before=chars_before,
                chars_after=len(instruction),
                body_chars=len(material.body or ""),
                body_source=body_res.source,
                sections_ok=False,
                error="instruction missing required DeepSWE-style sections or length floor",
            )
        if not dry_run:
            instr_path.write_text(instruction, encoding="utf-8")
        return PackInstructionRefresh(
            task_id=task_id,
            pack_dir=str(pack),
            ok=True,
            chars_before=chars_before,
            chars_after=len(instruction),
            body_chars=len(material.body or ""),
            body_source=body_res.source,
            sections_ok=True,
            leak_hits=(),
            prompt_style=PROMPT_STYLE_DEEPSWE_V1,
        )
    except RefreshInstructionsError as exc:
        return PackInstructionRefresh(
            task_id=task_id,
            pack_dir=str(pack),
            ok=False,
            chars_before=chars_before,
            chars_after=chars_before,
            body_chars=0,
            body_source="error",
            sections_ok=False,
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 — surface as pack-level failure
        return PackInstructionRefresh(
            task_id=task_id,
            pack_dir=str(pack),
            ok=False,
            chars_before=chars_before,
            chars_after=chars_before,
            body_chars=0,
            body_source="error",
            sections_ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def _discover_pack_dirs(root: Path) -> list[Path]:
    root = Path(root)
    tasks = root / "tasks"
    base = tasks if tasks.is_dir() else root
    packs: list[Path] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if (child / "instruction.md").is_file() or (child / "task.toml").is_file():
            packs.append(child)
    return packs


def stamp_pack_manifest_prompt_style(
    root: Path,
    *,
    prompt_style: str = PROMPT_STYLE_FULL_V1,
    pack_results: Sequence[PackInstructionRefresh] | None = None,
) -> bool:
    """Optionally note prompt_style=deepagent_full_v1 on under-root pack_manifest.json."""
    manifest_path = Path(root) / "pack_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    data["prompt_style"] = prompt_style
    data["prompt_style_refreshed_at"] = _now_iso()
    if pack_results is not None:
        by_id = {p.task_id: p for p in pack_results if p.ok}
        packs = data.get("packs")
        if isinstance(packs, list):
            for entry in packs:
                if not isinstance(entry, dict):
                    continue
                tid = str(entry.get("task_id") or "")
                if tid in by_id:
                    entry["prompt_style"] = prompt_style
    manifest_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return True


def refresh_product_instructions(
    root: Path | str,
    *,
    materials_root: Path | str | None = None,
    fetch_github: bool = True,
    dry_run: bool = False,
    stamp_manifest: bool = True,
    github_client: Any | None = None,
    get_pull: Callable[[str, int], Mapping[str, Any]] | None = None,
    task_ids: Sequence[str] | None = None,
    force_offline: bool | None = None,
    use_llm: bool | None = None,
) -> RefreshInstructionsResult:
    """Refresh all (or selected) pack ``instruction.md`` under a product root.

    Prefer materials + optional GitHub body re-fetch; never mutates dual-run /
    oracle evidence; never fixture-pads product trees.
    """
    root_p = Path(root)
    if not root_p.is_dir():
        raise RefreshInstructionsError(f"product root missing: {root_p}")
    mats = Path(materials_root) if materials_root else None
    if mats is not None and not mats.is_dir():
        # Soft: continue without materials (pack tree + GH still work)
        mats = None

    packs = _discover_pack_dirs(root_p)
    if task_ids is not None:
        wanted = {str(t).strip() for t in task_ids if str(t).strip()}
        packs = [p for p in packs if p.name in wanted]

    results: list[PackInstructionRefresh] = []
    fetched = 0
    for pack in packs:
        one = refresh_pack_instruction(
            pack,
            materials_root=mats,
            fetch_github=fetch_github,
            dry_run=dry_run,
            github_client=github_client,
            get_pull=get_pull,
            force_offline=force_offline,
            use_llm=use_llm,
        )
        if one.body_source == "github" and one.ok:
            fetched += 1
        results.append(one)

    refreshed = sum(1 for r in results if r.ok)
    failed = sum(1 for r in results if not r.ok)
    stamped = False
    if stamp_manifest and not dry_run and refreshed > 0 and failed == 0:
        stamped = stamp_pack_manifest_prompt_style(
            root_p,
            prompt_style=PROMPT_STYLE_FULL_V1,
            pack_results=results,
        )

    ok = refreshed > 0 and failed == 0
    msg = (
        f"refreshed {refreshed}/{len(results)} packs prompt_style={PROMPT_STYLE_FULL_V1}"
        if ok
        else f"refresh incomplete: ok={refreshed} failed={failed} total={len(results)}"
    )
    return RefreshInstructionsResult(
        root=str(root_p),
        ok=ok,
        refreshed=refreshed,
        failed=failed,
        packs=tuple(results),
        prompt_style=PROMPT_STYLE_FULL_V1,
        materials_root=str(mats) if mats else "",
        fetched_bodies=fetched,
        stamped_manifest=stamped,
        message=msg,
    )


__all__ = [
    "PROMPT_STYLE_DEEPSWE",
    "PROMPT_STYLE_DEEPSWE_V1",
    "PROMPT_STYLE_FULL_V1",
    "PackInstructionRefresh",
    "RefreshInstructionsError",
    "RefreshInstructionsResult",
    "material_from_pack",
    "refresh_pack_instruction",
    "refresh_product_instructions",
    "stamp_pack_manifest_prompt_style",
]
