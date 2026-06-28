"""The ``pr_mirror`` generator: invert a real merged PR's change in current code.

A real merged pull request encodes a behavior-bearing change (``base -> head``).
This generator manufactures a bug by *reverting* that change in the current
checkout: the forward ``mutation_patch`` undoes the PR (``head -> base``) and the
gold ``oracle_patch`` reinstates it (``base -> head``). Because current code is
the PR's post-merge state, reverse-applying the PR diff yields the exact pre-PR
content, so the mutation is the deterministic semantic inverse of the PR diff and
round-trips byte-for-byte.

The teacher proposes the inversion (it is asked to undo the PR's change in the
current file); deterministic execution disposes. A proposal that does not
reproduce the true (reverse-applied) pre-PR content is rejected - never shipped -
and the emitted patch is always the deterministic inverse. Provenance records the
source PR number/sha plus the teacher's usage/cost (no caching).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from swe_forge.forge.adapters._diff import (
    PatchError,
    apply_multi_patch,
    apply_patch,
    make_multi_patch,
)
from swe_forge.forge.adapters.base import LanguageAdapter
from swe_forge.forge.generators._llm import (
    extract_code_field,
    merge_usage,
    run_sync,
    strip_trivia,
    teacher_usage_details,
)
from swe_forge.forge.generators._normalize import is_behavior_changing
from swe_forge.forge.generators._targeting import SOURCE_EXTENSIONS, sha256_bytes
from swe_forge.forge.generators.base import (
    BugGenerator,
    GenerationError,
    GenerationRequest,
)
from swe_forge.forge.models import (
    Candidate,
    CandidateTarget,
    Provenance,
    require_green_baseline,
)
from swe_forge.forge.teacher import TeacherClient

_GITHUB_API = "https://api.github.com"
_HTTP_TIMEOUT = 30.0

_SYSTEM_PROMPT = (
    "You are reverting a merged pull request to manufacture a regression. Given a "
    "source file (its CURRENT, post-merge content) and the unified diff the PR "
    "applied, return the file with the PR's change UNDONE - i.e. the exact pre-PR "
    "content. Change nothing else; preserve all other lines and formatting. Return "
    "only the full reverted file content."
)

_REVERT_SCHEMA = {
    "type": "object",
    "properties": {"reverted_file": {"type": "string"}},
    "required": ["reverted_file"],
}

_REVERT_KEYS = ("reverted_file", "source", "content", "file", "reverted", "code")


@dataclass(frozen=True)
class PrFileChange:
    """One file a PR modified: its repo-relative path and the full git diff."""

    path: str
    patch: str
    status: str = "modified"


@dataclass(frozen=True)
class MergedPullRequest:
    """A resolved, merged pull request the generator mirrors (inverts)."""

    number: int
    sha: str
    repo: str
    files: list[PrFileChange]
    url: str = ""
    title: str = ""


class PullRequestResolver(Protocol):
    """Resolves a real merged PR for the repo under ``repo_root``."""

    def __call__(
        self, repo_root: Path, params: dict[str, object]
    ) -> MergedPullRequest: ...


@dataclass(frozen=True)
class PrInversionContext:
    """Inputs handed to a :class:`PrInverter`: the PR and current file contents."""

    pr: MergedPullRequest
    files: list[tuple[str, str]]


@dataclass(frozen=True)
class InversionProposal:
    """A teacher-proposed reversion: pre-PR content per file plus usage/cost."""

    reverted: dict[str, str]
    model: str = ""
    usage: list[dict[str, object]] = field(default_factory=list)


class PrInverter(Protocol):
    """Proposes the pre-PR (reverted) content for the PR's files (sync surface)."""

    def __call__(self, ctx: PrInversionContext) -> InversionProposal: ...


class TeacherPrInverter:
    """Default :class:`PrInverter` backed by the LiteLLM teacher client."""

    def __init__(self, client: TeacherClient, *, max_tokens: int = 2048) -> None:
        self._client = client
        self._max_tokens = max_tokens

    @classmethod
    def from_settings(cls, *, max_tokens: int = 2048) -> "TeacherPrInverter":
        return cls(TeacherClient.from_settings(), max_tokens=max_tokens)

    def __call__(self, ctx: PrInversionContext) -> InversionProposal:
        reverted: dict[str, str] = {}
        usage: list[dict[str, object]] = []
        patches = {f.path: f.patch for f in ctx.pr.files}
        for rel, current in ctx.files:
            prompt = (
                f"File: {rel}\n\n=== CURRENT CONTENT ===\n{current}\n\n"
                f"=== PR DIFF (applied by the PR) ===\n{patches.get(rel, '')}"
            )
            result = run_sync(
                self._client.complete_json(
                    prompt,
                    _REVERT_SCHEMA,
                    system=_SYSTEM_PROMPT,
                    schema_name="reverted_file",
                    max_tokens=self._max_tokens,
                )
            )
            reverted[rel] = extract_code_field(result.text, _REVERT_KEYS)
            usage.append(teacher_usage_details(result, self._client.model))
        return InversionProposal(
            reverted=reverted, model=self._client.model, usage=usage
        )


class GithubPullRequestResolver:
    """Resolve a merged PR via the GitHub REST API (stdlib HTTP, no extra deps).

    Reads ``repo`` (``owner/name``) and ``pr_number`` from the request params
    (``repo`` defaults to the ``origin`` remote of the checkout). An optional
    ``GITHUB_TOKEN`` env var lifts the unauthenticated rate limit; it is sent in
    the Authorization header and never logged.
    """

    def __call__(self, repo_root: Path, params: dict[str, object]) -> MergedPullRequest:
        repo = _resolve_repo_slug(repo_root, params)
        number = _resolve_pr_number(params)
        owner, name = repo.split("/", 1)

        meta = self._get_json(f"{_GITHUB_API}/repos/{owner}/{name}/pulls/{number}")
        if not isinstance(meta, dict) or not meta.get("merged_at"):
            raise GenerationError(
                f"pr_mirror: PR {repo}#{number} is not a resolvable merged PR"
            )
        sha = str(meta.get("merge_commit_sha") or "").strip()
        if not sha:
            raise GenerationError(
                f"pr_mirror: PR {repo}#{number} has no merge_commit_sha"
            )

        files_payload = self._get_json(
            f"{_GITHUB_API}/repos/{owner}/{name}/pulls/{number}/files?per_page=100"
        )
        changes = _parse_pr_files(files_payload)
        if not changes:
            raise GenerationError(
                f"pr_mirror: PR {repo}#{number} has no usable modified text files"
            )
        return MergedPullRequest(
            number=int(number),
            sha=sha,
            repo=repo,
            files=changes,
            url=str(meta.get("html_url") or ""),
            title=str(meta.get("title") or ""),
        )

    def _get_json(self, url: str) -> object:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "swe-forge-pr-mirror",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(url, headers=headers)  # noqa: S310 (https only)
        try:
            with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise GenerationError(f"pr_mirror: GitHub request failed: {exc}") from exc


def _parse_pr_files(payload: object) -> list[PrFileChange]:
    changes: list[PrFileChange] = []
    if not isinstance(payload, list):
        return changes
    for item in payload:
        if not isinstance(item, dict):
            continue
        if item.get("status") != "modified":
            continue
        path = str(item.get("filename") or "")
        hunks = item.get("patch")
        if not path or not isinstance(hunks, str) or not hunks.strip():
            continue
        changes.append(
            PrFileChange(
                path=path, patch=_wrap_file_diff(path, hunks), status="modified"
            )
        )
    return changes


def _wrap_file_diff(path: str, hunks: str) -> str:
    """Wrap GitHub's header-less per-file hunks into a git-applyable diff."""
    body = hunks if hunks.endswith("\n") else hunks + "\n"
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n{body}"


def _resolve_repo_slug(repo_root: Path, params: dict[str, object]) -> str:
    repo = params.get("repo")
    if isinstance(repo, str) and "/" in repo:
        return repo.strip()
    detected = _detect_origin_slug(repo_root)
    if detected:
        return detected
    raise GenerationError(
        "pr_mirror: no repo slug; pass params['repo']='owner/name' "
        "or run against a checkout with a github origin remote"
    )


def _detect_origin_slug(repo_root: Path) -> str | None:
    config = repo_root / ".git" / "config"
    if not config.is_file():
        return None
    text = config.read_text(encoding="utf-8", errors="replace")
    match = _ORIGIN_RE.search(text)
    if not match:
        return None
    return match.group("slug").removesuffix(".git")


_ORIGIN_RE = re.compile(r"github\.com[:/](?P<slug>[^/\s]+/[^/\s]+?)(?:\.git)?\s")


def _resolve_pr_number(params: dict[str, object]) -> int:
    value = params.get("pr_number")
    if isinstance(value, bool):
        value = None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    raise GenerationError(
        "pr_mirror: missing/invalid params['pr_number'] (a positive PR number)"
    )


class PrMirrorGenerator(BugGenerator):
    """Mirror (invert) a real merged PR's change in the current checkout."""

    name = "pr_mirror"

    def __init__(
        self,
        resolver: PullRequestResolver | None = None,
        inverter: PrInverter | None = None,
    ) -> None:
        self._resolver = resolver
        self._inverter = inverter

    def _resolve_resolver(self) -> PullRequestResolver:
        return (
            self._resolver
            if self._resolver is not None
            else GithubPullRequestResolver()
        )

    def _resolve_inverter(self) -> PrInverter:
        return (
            self._inverter
            if self._inverter is not None
            else TeacherPrInverter.from_settings()
        )

    def generate(
        self, request: GenerationRequest, adapter: LanguageAdapter
    ) -> Candidate:
        if request.env_image is not None:
            require_green_baseline(request.env_image)

        repo_root = Path(request.repo_root).resolve()
        if not repo_root.is_dir():
            raise GenerationError(f"repo path is not a directory: {repo_root}")

        pr = self._resolve_resolver()(repo_root, request.params)

        with contextlib.chdir(repo_root):
            inverses = self._compute_inverses(adapter, pr)
            if not inverses:
                raise GenerationError(
                    f"pr_mirror: PR {pr.repo}#{pr.number} did not reverse-apply to "
                    f"any current non-test source file (drift or no source change)"
                )

            proposal = self._resolve_inverter()(
                PrInversionContext(
                    pr=pr,
                    files=[
                        (rel, current.decode("utf-8")) for rel, current, _ in inverses
                    ],
                )
            )
            accepted = self._accept_inverted(inverses, proposal)
            if not accepted:
                raise GenerationError(
                    f"pr_mirror: the teacher did not reproduce the pre-PR content "
                    f"for any file of {pr.repo}#{pr.number} (non-inverting output)"
                )

            mutation, oracle = self._build_patches(accepted)
            return self._build_candidate(
                request, adapter, pr, accepted, proposal, mutation, oracle
            )

    def _compute_inverses(
        self, adapter: LanguageAdapter, pr: MergedPullRequest
    ) -> list[tuple[str, bytes, bytes]]:
        extensions = SOURCE_EXTENSIONS.get(adapter.name, frozenset())
        inverses: list[tuple[str, bytes, bytes]] = []
        for change in pr.files:
            rel = change.path
            path = Path(rel)
            if path.suffix.lower() not in extensions:
                continue
            if adapter.is_test_file(rel) or not path.is_file():
                continue
            current = path.read_bytes()
            try:
                base = apply_patch(current, change.patch, rel, reverse=True)
            except PatchError:
                continue
            if base == current:
                continue
            # Behavior gate: a PR whose revert only touches trivia/imports does
            # not change behavior and must never become a Candidate.
            if not is_behavior_changing(
                current.decode("utf-8", "replace"), base.decode("utf-8", "replace")
            ):
                continue
            inverses.append((rel, current, base))
        return inverses

    def _accept_inverted(
        self,
        inverses: list[tuple[str, bytes, bytes]],
        proposal: InversionProposal,
    ) -> list[tuple[str, bytes, bytes]]:
        accepted: list[tuple[str, bytes, bytes]] = []
        for rel, current, base in inverses:
            proposed = proposal.reverted.get(rel)
            if proposed is None:
                continue
            # The teacher must semantically reproduce the true (reverse-applied)
            # pre-PR content; a proposal that does not invert is rejected.
            if strip_trivia(proposed) != strip_trivia(base.decode("utf-8")):
                continue
            accepted.append((rel, current, base))
        return accepted

    def _build_patches(
        self, accepted: list[tuple[str, bytes, bytes]]
    ) -> tuple[str, str]:
        mutation = make_multi_patch(
            [(rel, current, base) for rel, current, base in accepted]
        )
        oracle = make_multi_patch(
            [(rel, base, current) for rel, current, base in accepted]
        )
        if not mutation.strip() or not oracle.strip():
            raise GenerationError("pr_mirror: inversion produced an empty patch")

        originals = {rel: current for rel, current, _ in accepted}
        targets = {rel: base for rel, _, base in accepted}
        try:
            applied = apply_multi_patch(originals, mutation)
            restored = apply_multi_patch(applied, oracle)
        except PatchError as exc:
            raise GenerationError(
                f"pr_mirror: inversion patch did not apply cleanly: {exc}"
            ) from exc
        for rel in originals:
            if sha256_bytes(applied[rel]) != sha256_bytes(targets[rel]):
                raise GenerationError(f"pr_mirror: mutation did not revert {rel}")
            if sha256_bytes(restored[rel]) != sha256_bytes(originals[rel]):
                raise GenerationError(f"pr_mirror: oracle did not restore {rel}")
        return mutation, oracle

    def _build_candidate(
        self,
        request: GenerationRequest,
        adapter: LanguageAdapter,
        pr: MergedPullRequest,
        accepted: list[tuple[str, bytes, bytes]],
        proposal: InversionProposal,
        mutation: str,
        oracle: str,
    ) -> Candidate:
        rels = tuple(rel for rel, _, _ in accepted)
        provenance = Provenance(
            generator=self.name,
            seed=request.seed,
            language=adapter.name,
            tool_versions={},
            details={
                "operation": "pr_mirror",
                "pr_number": pr.number,
                "pr_sha": pr.sha,
                "repo": pr.repo,
                "pr_url": pr.url,
                "pr_title": pr.title,
                "files": list(rels),
                "teacher": merge_usage(proposal.usage),
            },
        )
        return Candidate(
            language=adapter.name,
            generator=self.name,
            target=CandidateTarget(files=rels),
            mutation_patch=mutation,
            oracle_patch=oracle,
            difficulty_hint="high",
            provenance=provenance,
        )


__all__ = [
    "GithubPullRequestResolver",
    "InversionProposal",
    "MergedPullRequest",
    "PrFileChange",
    "PrInversionContext",
    "PrInverter",
    "PrMirrorGenerator",
    "PullRequestResolver",
    "TeacherPrInverter",
]
