"""DeepSWE-true agent instruction rewrites (M18 / VAL-DSTYLE).

Real DeepSWE (datacurve-ai/deep-swe) agent prompts are:

- Behavior-first natural developer language (problem + expected outcomes +
  constraints)
- **No provenance**: no PR#, issues, GitHub URLs, base SHA, repo clone URL,
  "Merged PR", source_track
- Shortish (~800–3500 chars preferred, tolerant floor)
- Footer: work on a new branch from main/base and commit when done

Private materials may still store PR meta; agent-visible ``instruction.md`` must
not leak mining fingerprints. Live path uses OpenRouter teacher rewrite; offline
path uses a deterministic sanitizer/fallback so unit tests never need the
network.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

PROMPT_STYLE_DEEPSWE_V1 = "deepswe_style_v1"

DEEPSWE_IMPORTANT_FOOTER = (
    "IMPORTANT: Please work on this in a new branch from main and commit "
    "everything when you are done."
)

# Soft length targets inspired by deep-swe samples (mean ~2k).
_INSTRUCTION_MIN_CHARS = 200
_INSTRUCTION_SOFT_MAX_CHARS = 4500
_PRIVATE_BODY_MAX_CHARS = 3500
_MAX_SOURCE_FILES_IN_PRIVATE = 16

# ---------------------------------------------------------------------------
# Antifingerprint ban list (VAL-DSTYLE-001)
# ---------------------------------------------------------------------------

# Named regexes used by unit tests and post-scan. Keep patterns strict so plain
# technical language can still mention "pull" without matching GitHub trails.
PROVENANCE_FINGERPRINT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("merged_pr", re.compile(r"(?i)\bmerged\s+pr\b")),
    ("pr_hash_number", re.compile(r"(?i)\bPR\s*#\s*\d+\b")),
    ("pull_request_phrase", re.compile(r"(?i)\bpull\s+request\b")),
    (
        "github_pull_or_issue_url",
        re.compile(r"(?i)github\.com/[^\s)>\]]+/(?:pull|issues)/\d+"),
    ),
    (
        "github_url",
        re.compile(r"(?i)https?://(?:www\.)?github\.com/[^\s)>\]]+"),
    ),
    (
        "source_track",
        re.compile(r"(?i)\bsource[_\s-]*track\b\s*[=:]\s*real_pr\b|\bsource[_\s-]*track\b"),
    ),
    ("base_commit_sha", re.compile(r"\b[0-9a-fA-F]{40}\b")),
    ("repository_url_label", re.compile(r"(?i)\brepository\s+url\b")),
    ("base_commit_label", re.compile(r"(?i)\bbase\s+commit\b")),
    # M17 scaffolding headings that listed mining provenance
    ("context_heading", re.compile(r"(?im)^#{1,3}\s*context\s*$")),
    ("pr_description_heading", re.compile(r"(?im)^#{1,3}\s*pr\s+description\s*$")),
    ("git_clone_url", re.compile(r"(?i)\bgit@github\.com:[^\s]+\.git\b")),
    ("raw_git_https", re.compile(r"(?i)https?://[^\s]+?\.git\b")),
)


def find_provenance_fingerprints(text: str) -> list[str]:
    """Return deterministic names of provenance ban-list hits (VAL-DSTYLE-001)."""
    body = text if isinstance(text, str) else str(text or "")
    hits: list[str] = []
    for name, pattern in PROVENANCE_FINGERPRINT_PATTERNS:
        if pattern.search(body):
            hits.append(name)
    return hits


def has_provenance_fingerprints(text: str) -> bool:
    return bool(find_provenance_fingerprints(text))


def has_deepswe_footer(text: str) -> bool:
    """VAL-DSTYLE-004: DeepSWE-shape deliverable footer present."""
    lower = (text or "").lower()
    if "important:" not in lower:
        return False
    has_branch = "new branch" in lower
    has_commit = "commit" in lower and ("done" in lower or "everything" in lower)
    return has_branch and has_commit


def style_ok(text: str) -> bool:
    """Cheap DeepSWE register check for refresh gates (VAL-DSTYLE-002/004)."""
    body = (text or "").strip()
    if len(body) < _INSTRUCTION_MIN_CHARS:
        return False
    if has_provenance_fingerprints(body):
        return False
    if not has_deepswe_footer(body):
        return False
    lower = body.lower()
    # Prefer Expected outcomes / Constraints, tolerate mild variants.
    has_outcomes = (
        "expected outcomes" in lower
        or "expected outcome" in lower
        or re.search(r"(?m)^\s*\d+\.\s+\S", body) is not None
    )
    has_constraints = "constraint" in lower or "implementation notes" in lower
    # Behavior-first: not a raw PR dump heading
    if "pr description" in lower and body.lstrip().lower().startswith("#"):
        # still ok if not used as scaffolding keyword with provenance
        pass
    return bool(has_outcomes or has_constraints)


# ---------------------------------------------------------------------------
# Sanitizers for private → public distillation
# ---------------------------------------------------------------------------

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BARE_URL_RE = re.compile(r"https?://[^\s)>\]]+")
_PR_MARKER_RE = re.compile(
    r"(?i)\b(?:merged\s+)?(?:pull\s+request|pr)\s*#?\s*\d+\b|\bPR\s*#\s*\d+\b"
)
_ISSUE_MARKER_RE = re.compile(r"(?i)\b(?:closes|fixes|resolves)\s+#\d+\b|\bissues?/\d+\b")
_SHA_RE = re.compile(r"\b[0-9a-fA-F]{40}\b")
_CHECKBOX_RE = re.compile(r"(?m)^\s*[-*]\s*\[[ xX]\]\s*")
_BULLET_RE = re.compile(r"(?m)^\s*(?:[-*]|\d+[.)])\s+(.+\S)\s*$")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def sanitize_title_for_prompt(title: str | None) -> str:
    """Strip PR numbers / provenance tokens from a PR title (private→public)."""
    text = (title or "").strip()
    if not text:
        return "Restore multi-module product behaviour"
    text = _PR_MARKER_RE.sub("", text)
    text = _ISSUE_MARKER_RE.sub("", text)
    text = _SHA_RE.sub("", text)
    text = _BARE_URL_RE.sub("", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" -–—:|")
    return text or "Restore multi-module product behaviour"


def sanitize_body_for_deepswe(body: str | None, *, max_chars: int = _PRIVATE_BODY_MAX_CHARS) -> str:
    """Strip links, PR markers, SHAs, and diff junk from a PR body."""
    text = body if isinstance(body, str) else str(body or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HTML_COMMENT_RE.sub("", text)
    # Keep link labels, drop targets
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _BARE_URL_RE.sub("", text)
    text = _PR_MARKER_RE.sub("", text)
    text = _ISSUE_MARKER_RE.sub("", text)
    text = _SHA_RE.sub("", text)
    text = _CHECKBOX_RE.sub("- ", text)
    # Drop unified-diff / code-fence patch blobs
    text = re.sub(
        r"```(?:diff|patch)?\n.*?```",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    lines: list[str] = []
    for ln in text.splitlines():
        stripped = ln.lstrip()
        if stripped.startswith(("diff --git ", "@@ ", "--- a/", "+++ b/")):
            continue
        if re.match(r"(?i)^(co-authored-by|signed-off-by):", stripped):
            continue
        lines.append(ln)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text


def _extract_bullets(body: str, *, limit: int = 8) -> list[str]:
    bullets: list[str] = []
    for match in _BULLET_RE.finditer(body or ""):
        item = match.group(1).strip()
        item = re.sub(r"\s+", " ", item)
        if len(item) < 8:
            continue
        if has_provenance_fingerprints(item):
            item = sanitize_body_for_deepswe(item, max_chars=400)
        if item and item not in bullets:
            bullets.append(item)
        if len(bullets) >= limit:
            break
    return bullets


def _source_hint(source_files: Sequence[str] | None) -> str:
    sources = [str(s).strip() for s in (source_files or ()) if str(s).strip()]
    if not sources:
        return ""
    shown = sources[:8]
    listing = ", ".join(f"`{s}`" for s in shown)
    extra = f" (+{len(sources) - 8} more)" if len(sources) > 8 else ""
    return f"Affected product modules include {listing}{extra}."


def ensure_footer(text: str) -> str:
    body = (text or "").rstrip()
    if has_deepswe_footer(body):
        # Normalize to the canonical DeepSWE line when a looser variant is present.
        if DEEPSWE_IMPORTANT_FOOTER not in body and not body.lower().rstrip().endswith(
            "when you are done."
        ):
            body = body.rstrip() + "\n\n" + DEEPSWE_IMPORTANT_FOOTER
        return body.strip() + "\n"
    return body + "\n\n" + DEEPSWE_IMPORTANT_FOOTER + "\n"


def strip_residual_fingerprints(text: str) -> str:
    """Best-effort scrub if a model/echo still smuggled provenance tokens."""
    body = text if isinstance(text, str) else str(text or "")
    body = _MD_LINK_RE.sub(r"\1", body)
    body = _BARE_URL_RE.sub("", body)
    body = _PR_MARKER_RE.sub("", body)
    body = _ISSUE_MARKER_RE.sub("", body)
    body = _SHA_RE.sub("", body)
    # Drop M17 provenance scaffolding sections if still present.
    body = re.sub(
        r"(?im)^#{1,3}\s*context\s*$[\s\S]*?(?=^#{1,3}\s|\Z)",
        "",
        body,
    )
    body = re.sub(
        r"(?im)^#{1,3}\s*pr\s+description\s*$[\s\S]*?(?=^#{1,3}\s|\Z)",
        "",
        body,
    )
    body = re.sub(r"(?i)\bsource[_\s-]*track\b\s*[=:]\s*real_pr\b", "", body)
    body = re.sub(r"(?i)\brepository\s+url\s*:\s*`?[^`\n]+`?", "", body)
    body = re.sub(r"(?i)\bbase\s+commit(?:\s*\([^)]*\))?\s*:\s*`?[^`\n]+`?", "", body)
    body = re.sub(r"(?i)\bmerged\s+pr\b\s*:?\s*`?#?\d*`?", "", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


# ---------------------------------------------------------------------------
# Offline deterministic rewriter (VAL-DSTYLE-003)
# ---------------------------------------------------------------------------


def build_offline_deepswe_instruction(
    *,
    title: str,
    body: str = "",
    source_files: Sequence[str] | None = None,
    language: str = "python",
) -> str:
    """Deterministic DeepSWE-like instruction with zero provenance fingerprints.

    Use when OpenRouter is unavailable (unit tests / offline) or when live
    rewrite returns dirty text.
    """
    clean_title = sanitize_title_for_prompt(title)
    clean_body = sanitize_body_for_deepswe(body)
    bullets = _extract_bullets(clean_body)
    lang = (language or "python").strip() or "python"
    source_line = _source_hint(source_files)

    problem_bits: list[str] = [clean_title.rstrip(".") + "."]
    # First 1–2 non-bullet prose sentences from body for context.
    prose = re.sub(r"(?m)^\s*(?:[-*]|\d+[.)])\s+.*$", "", clean_body)
    prose = re.sub(r"\n{2,}", "\n", prose).strip()
    if prose:
        # Flatten to a short paragraph
        prose_flat = re.sub(r"\s+", " ", prose)
        if len(prose_flat) > 700:
            prose_flat = prose_flat[:700].rstrip() + "…"
        # Avoid restating the title verbatim when body opens with it
        if prose_flat.lower() not in clean_title.lower():
            problem_bits.append(prose_flat)
    if source_line:
        problem_bits.append(source_line)

    problem = " ".join(problem_bits).strip()
    problem = strip_residual_fingerprints(problem)

    if bullets:
        outcome_lines = [f"{i}. {b}" for i, b in enumerate(bullets, start=1)]
    else:
        outcome_lines = [
            (
                "1. Restore the intended multi-module contracts so product behaviour "
                "matches the described problem across the affected source modules."
            ),
            (
                "2. Keep unrelated public APIs and pass behaviour intact; do not remove, "
                "skip, or rewrite tests to force a green run."
            ),
            (
                "3. Prefer a focused multi-file source change that addresses the "
                f"behaviour in pure {lang} without inventing secrets or credentials."
            ),
        ]

    constraints = [
        "- Do not embed or rely on held-out verifier sources; implement production behaviour only.",
        "- Prefer minimal multi-file edits under the repository root.",
        f"- Language focus: {lang}.",
        "- Do not invent secrets, API keys, or vendor credentials.",
    ]

    text = (
        f"{problem}\n\n"
        f"Expected outcomes\n"
        + "\n".join(outcome_lines)
        + "\n\nConstraints\n"
        + "\n".join(constraints)
        + "\n\n"
        + DEEPSWE_IMPORTANT_FOOTER
        + "\n"
    )
    cleaned = strip_residual_fingerprints(text)
    cleaned = ensure_footer(cleaned)
    # Final hard gate: if anything still fingerprints, collapse to minimal safe form.
    if has_provenance_fingerprints(cleaned):
        cleaned = (
            f"{clean_title.rstrip('.')}.\n\n"
            "Expected outcomes\n"
            "1. Restore the intended product behaviour across the affected modules.\n"
            "2. Keep unrelated pass behaviour intact and leave held-out tests alone.\n\n"
            "Constraints\n"
            "- Implement production source changes only; no secrets.\n\n"
            f"{DEEPSWE_IMPORTANT_FOOTER}\n"
        )
    return cleaned if cleaned.endswith("\n") else cleaned + "\n"


# ---------------------------------------------------------------------------
# Live OpenRouter rewrite path
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    """\
You rewrite internal pull-request notes into agent-facing software-engineering \
task instructions that match the DeepSWE / Harbor style.

Output ONLY the agent-visible instruction (markdown). Requirements:
1. Behavior-first developer prose: state the problem, then "Expected outcomes" \
(numbered) and "Constraints" (bullets). Optionally short "Implementation notes".
2. NEVER echo provenance: no PR numbers, no "Merged PR", no "pull request", no \
GitHub URLs, no issue links, no commit SHAs (especially 40-char hex), no repo \
clone URLs, no "source track", no "Repository URL", no "Base commit", no \
"## Context" / "## PR description" mining scaffolding.
3. Do not invent gold solutions or paste unified diffs / patch bodies.
4. Prefer ~800–3500 characters. Tight, concrete behaviours — not checklists of CI checkboxes.
5. End exactly with this footer line (alone on its line):\n"""
    + DEEPSWE_IMPORTANT_FOOTER
)


def _user_prompt(
    *,
    title: str,
    body: str,
    source_files: Sequence[str],
    language: str,
) -> str:
    sources = [str(s).strip() for s in source_files if str(s).strip()][
        :_MAX_SOURCE_FILES_IN_PRIVATE
    ]
    src_block = "\n".join(f"- {s}" for s in sources) if sources else "(unspecified modules)"
    # Private inputs may still include raw body; the model is instructed not to echo markers.
    return (
        "Rewrite the following PRIVATE PR notes into one public agent instruction.\n"
        f"Language: {language or 'python'}\n"
        f"Title (private): {title or '(none)'}\n"
        f"Source files (private):\n{src_block}\n\n"
        f"Body (private):\n{body or '(empty)'}\n"
    )


class _ChatComplete(Protocol):
    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Any: ...


def _openrouter_available(env: Mapping[str, str] | None = None) -> bool:
    env_map = env if env is not None else os.environ
    key = (env_map.get("OPENROUTER_API_KEY") or "").strip()
    return bool(key)


def _teacher_model(env: Mapping[str, str] | None = None) -> str:
    env_map = env if env is not None else os.environ
    model = (env_map.get("FACTORY_TEACHER_MODEL") or "").strip()
    if model:
        return model
    # Prefer cheap/fast panel ids when teacher unset — still overrideable.
    return "x-ai/grok-4.5"


def _live_rewrite(
    *,
    title: str,
    body: str,
    source_files: Sequence[str],
    language: str,
    client: _ChatComplete | None = None,
    model: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Call OpenRouter teacher; return cleaned text or None on any failure."""
    try:
        if client is None:
            from swe_factory.openrouter import OpenRouterClient

            client = OpenRouterClient.from_settings()
        model_id = (model or _teacher_model(env)).strip()
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _user_prompt(
                    title=title,
                    body=body,
                    source_files=list(source_files),
                    language=language,
                ),
            },
        ]
        result = client.complete(
            model=model_id,
            messages=messages,
            max_tokens=1800,
            temperature=0.2,
        )
        raw = getattr(result, "text", None)
        if not isinstance(raw, str) or not raw.strip():
            return None
        cleaned = strip_residual_fingerprints(raw)
        cleaned = ensure_footer(cleaned)
        if has_provenance_fingerprints(cleaned):
            return None
        if len(cleaned.strip()) < _INSTRUCTION_MIN_CHARS:
            return None
        if len(cleaned) > _INSTRUCTION_SOFT_MAX_CHARS + 1500:
            cleaned = (
                cleaned[:_INSTRUCTION_SOFT_MAX_CHARS].rstrip()
                + "\n\n"
                + DEEPSWE_IMPORTANT_FOOTER
                + "\n"
            )
            cleaned = ensure_footer(strip_residual_fingerprints(cleaned))
        return cleaned if cleaned.endswith("\n") else cleaned + "\n"
    except Exception:
        # Broad by design: missing key / network / schema → offline fallback.
        return None


# ---------------------------------------------------------------------------
# Cache on materials meta (optional agent_instruction)
# ---------------------------------------------------------------------------


def read_cached_agent_instruction(materials_dir: str | Path | None) -> str:
    """Load optional agent_instruction from materials meta.json when clean."""
    if not materials_dir:
        return ""
    meta_path = Path(materials_dir) / "meta.json"
    if not meta_path.is_file():
        return ""
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return ""
    if not isinstance(raw, dict):
        return ""
    text = str(raw.get("agent_instruction") or "").strip()
    if not text:
        return ""
    if has_provenance_fingerprints(text) or not has_deepswe_footer(text):
        return ""
    if len(text) < _INSTRUCTION_MIN_CHARS:
        return ""
    return text if text.endswith("\n") else text + "\n"


def persist_agent_instruction(materials_dir: str | Path | None, instruction: str) -> bool:
    """Best-effort write agent_instruction into materials meta.json (cache)."""
    if not materials_dir:
        return False
    text = (instruction or "").strip()
    if not text or has_provenance_fingerprints(text):
        return False
    meta_path = Path(materials_dir) / "meta.json"
    if not meta_path.is_file():
        return False
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return False
        raw["agent_instruction"] = text if text.endswith("\n") else text + "\n"
        raw["prompt_style"] = PROMPT_STYLE_DEEPSWE_V1
        meta_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return True
    except (OSError, TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Public product builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeepSWEInstructionResult:
    text: str
    source: str  # cache | live_llm | offline
    fingerprints: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "chars": len(self.text),
            "fingerprints": list(self.fingerprints),
        }


def build_deepswe_style_instruction(
    *,
    title: str,
    body: str = "",
    source_files: Sequence[str] | None = None,
    language: str = "python",
    materials_dir: str | Path | None = None,
    cached_instruction: str | None = None,
    force_offline: bool | None = None,
    client: _ChatComplete | None = None,
    model: str | None = None,
    persist_cache: bool = True,
    env: Mapping[str, str] | None = None,
) -> str:
    """Build agent-visible DeepSWE-style instruction (VAL-DSTYLE-001..005).

    Priority:
      1. Clean ``cached_instruction`` / materials ``meta.agent_instruction``
      2. Live OpenRouter rewrite when key present and not force_offline
      3. Deterministic offline sanitizer fallback

    Never embeds solution/test patches (caller still should leak-scan).
    """
    sources = tuple(str(s).strip() for s in (source_files or ()) if str(s).strip())
    lang = (language or "python").strip() or "python"

    # 1) cache
    cached = (cached_instruction or "").strip()
    if not cached:
        cached = read_cached_agent_instruction(materials_dir).strip()
    if cached:
        cleaned_cache = ensure_footer(strip_residual_fingerprints(cached))
        if (
            not has_provenance_fingerprints(cleaned_cache)
            and len(cleaned_cache) >= _INSTRUCTION_MIN_CHARS
        ):
            return cleaned_cache if cleaned_cache.endswith("\n") else cleaned_cache + "\n"

    offline = force_offline
    if offline is None:
        offline = not _openrouter_available(env)

    # 2) live
    if not offline:
        live = _live_rewrite(
            title=title or "",
            body=sanitize_body_for_deepswe(body, max_chars=_PRIVATE_BODY_MAX_CHARS),
            source_files=sources,
            language=lang,
            client=client,
            model=model,
            env=env,
        )
        if live and not has_provenance_fingerprints(live):
            if persist_cache:
                persist_agent_instruction(materials_dir, live)
            return live

    # 3) offline fallback
    offline_text = build_offline_deepswe_instruction(
        title=title or "",
        body=body or "",
        source_files=sources,
        language=lang,
    )
    if persist_cache and materials_dir:
        persist_agent_instruction(materials_dir, offline_text)
    return offline_text


def build_deepswe_style_instruction_detailed(
    **kwargs: Any,
) -> DeepSWEInstructionResult:
    """Same as :func:`build_deepswe_style_instruction` with source metadata."""
    # Force a pure offline build when measuring without env private key path.
    text = build_deepswe_style_instruction(**kwargs)
    source = "offline"
    if kwargs.get("cached_instruction") or read_cached_agent_instruction(
        kwargs.get("materials_dir")
    ):
        # Re-detect: if kwargs forced cache path used
        cached = (kwargs.get("cached_instruction") or "").strip() or read_cached_agent_instruction(
            kwargs.get("materials_dir")
        )
        if cached and not has_provenance_fingerprints(
            ensure_footer(strip_residual_fingerprints(cached))
        ):
            source = "cache"
    elif kwargs.get("force_offline") is not True and _openrouter_available(kwargs.get("env")):
        # We cannot know retroactively if live succeeded without re-call; best-effort:
        # mark live only when caller injected a client (tests) or key present and not offline.
        if kwargs.get("client") is not None:
            source = "live_llm"
    fps = tuple(find_provenance_fingerprints(text))
    return DeepSWEInstructionResult(text=text, source=source, fingerprints=fps)


__all__ = [
    "DEEPSWE_IMPORTANT_FOOTER",
    "PROMPT_STYLE_DEEPSWE_V1",
    "PROVENANCE_FINGERPRINT_PATTERNS",
    "DeepSWEInstructionResult",
    "build_deepswe_style_instruction",
    "build_deepswe_style_instruction_detailed",
    "build_offline_deepswe_instruction",
    "ensure_footer",
    "find_provenance_fingerprints",
    "has_deepswe_footer",
    "has_provenance_fingerprints",
    "persist_agent_instruction",
    "read_cached_agent_instruction",
    "sanitize_body_for_deepswe",
    "sanitize_title_for_prompt",
    "strip_residual_fingerprints",
    "style_ok",
]
