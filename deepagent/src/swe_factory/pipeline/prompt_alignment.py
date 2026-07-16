"""Prompt–verifier alignment gate (M21 / VAL-DHARD-001).

Fail-closed product ship rescue: refuse export/cert when agent-visible
``instruction.md`` contradicts the held-out F2P / gold behavioural delta.

Canonical misalign class (more-itertools-1136):

* instruction claims version/export-only work and/or "do not change runtime"
* test.patch / gold notably change runtime behaviour (windowed invalid-n,
  unique_everseen unhashable contracts, …)

Principles
----------
* Never invent gold into the instruction.
* Prefer high-level behaviours derived from test_patch function names /
  docstrings and assert patterns — not dumping private F2P node IDs into
  agent-visible text.
* Soft signal scoring; only refuse on clear contradictions for product /
  live_generate dests (honesty paths). Offline engineering dests skip hard
  refuse unless ``force=True``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Stable reason codes (gate_audit / drip / raise messages)
# ---------------------------------------------------------------------------

REASON_PROMPT_ALIGN_OK = "prompt_align_ok"
REASON_PROMPT_VERIFIER_MISALIGN = "prompt_verifier_misalign"
REASON_PROMPT_NO_RUNTIME_CLAIM = "prompt_no_runtime_claim_vs_runtime_f2p"
REASON_PROMPT_VERSION_ONLY = "prompt_version_only_vs_behavioral_f2p"
REASON_PROMPT_EMPTY_BEHAVIOR = "prompt_empty_behavior_ask_vs_f2p"
REASON_PROMPT_ALIGN_SKIPPED = "prompt_align_skipped_non_product"

# Product honesty markers — keep in sync with ship_real_pr live generate dests.
_LIVE_GENERATE_DEST_MARKERS = ("test_n10", "prod_hard_keep")


# ---------------------------------------------------------------------------
# Instruction claim patterns
# ---------------------------------------------------------------------------

_NO_RUNTIME_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)do\s+not\s+change\s+(the\s+)?runtime\b"),
    re.compile(r"(?i)without\s+(altering|changing|modifying)\s+(the\s+)?runtime\b"),
    re.compile(r"(?i)do\s+not\s+(alter|modify|change)\s+(the\s+)?(runtime\s+)?behaviou?r\b"),
    re.compile(r"(?i)runtime\s+behaviou?r\s+of\s+existing\b"),
    re.compile(r"(?i)without\s+altering\s+runtime\s+behaviou?r\b"),
    re.compile(r"(?i)keep\s+(all\s+)?other\s+.*\bbehaviou?r\s+unchanged\b"),
    re.compile(r"(?i)do\s+not\s+change\s+the\s+runtime\s+behaviou?r\b"),
)

_VERSION_EXPORT_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b__version__\b"),
    re.compile(r"(?i)\bversion\s+string\b"),
    re.compile(r"(?i)\bversion\s+metadata\b"),
    re.compile(r"(?i)\bbump(?:ing)?\s+(the\s+)?(?:package\s+)?version\b"),
    re.compile(r"(?i)\bpackage\s+version\b"),
    re.compile(r"(?i)\b__all__\b"),
    re.compile(r"(?i)\bpublic\s+API\s+surface\b"),
    re.compile(r"(?i)\bexported\s+via\s+__all__\b"),
    re.compile(r"(?i)\bexports?\s+(are|via|named|public)\b"),
    re.compile(r"(?i)\bnext\s+major\s+release\b"),
    re.compile(r"(?i)\bprepare\s+the\s+codebase\s+for\s+the\s+next\s+(major\s+)?release\b"),
    re.compile(r"(?i)\bmajor\s+release\b"),
)

# Outcomes/bullets that stay in version/export/docs/API-list land (not algorithmic).
_VERSION_OUTCOME_BULLET = re.compile(
    r"(?i)(\bversion\b|\b__version__\b|\b__all__\b|\bexport\b|\bpublic\s+api\b|"
    r"\brelease\b|\bdocs?\b|\bdocstring\b|\bmetadata\b|\bstale\s+entr|\bimporting\s+the\s+package\b)"
)

# Outcome bullets that look algorithmic / multi-behavior product work.
_BEHAVIOR_OUTCOME_BULLET = re.compile(
    r"(?i)(\braise\b|\bvalueerror\b|\btypeerror\b|\bassert\b|\breturn\b|\bwindow\b|"
    r"\biterat|\buniqu|\bhash|\bschema\b|\bjson\b|\bparse\b|\bhandle\b|"
    r"\breject\b|\baccept\b|\binvalid\b|\bvalid\b|\bencoding\b|\bdecode\b|"
    r"\bqueue\b|\bcache\b|\btime\s*out\b|\bretr|\bfail(?:s|ure)?\b|"
    r"\bmust\s+(?:raise|return|accept|reject|produce|emit)\b|"
    r"\b(?:when|if)\s+[a-z0-9_]+\s+(?:is|==|equals|receives)\b)"
)

_OUTCOME_LINE = re.compile(r"(?m)^\s*(?:[-*]|\d+[.)])\s+(\S.*)$")


# ---------------------------------------------------------------------------
# Verifier / gold behavioural signals (from patches — not node-ID dumps)
# ---------------------------------------------------------------------------

_RUNTIME_ASSERT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?i)assertRaises(?:Regex)?\s*\(\s*(ValueError|TypeError|RuntimeError|KeyError|IndexError)"
    ),
    re.compile(
        r"(?i)pytest\.raises\s*\(\s*(ValueError|TypeError|RuntimeError|KeyError|IndexError)"
    ),
    re.compile(r"(?i)with\s+self\.assertRaises"),
    re.compile(r"(?i)self\.assert(?:Equal|CountEqual|NotEqual|True|False|Is|In|ListEqual)\b"),
    re.compile(r"(?i)^\+\s*assert\s+\S", re.M),
)

_BEHAVIOR_TEST_DEF = re.compile(r"(?m)^\+\s*def\s+(test_\w+)\s*\(")
_BEHAVIOR_TEST_RENAME = re.compile(r"(?m)^-\s*def\s+(test_\w+)\s*\(")
_TEST_DEF_CONTEXT = re.compile(r"(?m)^\s*def\s+(test_\w+|test[A-Z]\w*)\s*\(")

# Gold side: runtime body edits (not pure version/export/meta).
_GOLD_VERSION_LINES = re.compile(r"(?m)^[+\-]\s*(?:__version__|version\s*=|__all__)\b")
_GOLD_RUNTIME_BODY = re.compile(
    r"(?m)^[+\-]\s*(?:if |for |while |return |raise |yield |def |\w+\s*=\s*(?!['\"]?\d+\.\d))"
)
_GOLD_RAISE_DELTA = re.compile(r"(?m)^[+\-]\s*.*\braise\s+(?:ValueError|TypeError|RuntimeError)\b")

# High-level behaviour class hints (token presence in patches / names).
_BEHAVIOR_CLASS_HINTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "windowed_invalid_n",
        re.compile(
            r"(?i)\bwindowed\b.*\b(?:0|negative|invalid_?n|n\s*<=?\s*0)\b"
            r"|\btest_invalid_n\b|\bn is zero\b"
        ),
    ),
    (
        "unique_everseen_unhashable",
        re.compile(
            r"(?i)\bunique_everseen\b.*\b(?:unhashable|key\s*=\s*str|TypeError)\b"
            r"|\btest_unhashable_(?:lists|sets|dicts)\b"
        ),
    ),
    (
        "unique_everseen_key",
        re.compile(r"(?i)\bunique_everseen\b.*\bkey\b|\btest_.*key\b.*unique"),
    ),
    (
        "windowed_fillvalue_step",
        re.compile(r"(?i)\bwindowed\b.*\b(?:fillvalue|step)\b|\btest_fillvalue_step\b"),
    ),
    ("distinct_permutations_key", re.compile(r"(?i)\bdistinct_permutations\b.*\bkey\b")),
    ("version_metadata", re.compile(r"(?i)\b__version__\b|\bversion\s*==\s*['\"]?\d")),
    ("export_all", re.compile(r"(?i)\b__all__\b")),
    (
        "schema_generation",
        re.compile(r"(?i)\bjson\s*schema\b|\bget_json_schema\b|\bproperties\b"),
    ),
    ("error_contract", re.compile(r"(?i)\b(?:ValueError|TypeError|RuntimeError)\b")),
    (
        "api_removal",
        re.compile(
            r"(?i)\b(?:remove|remov(?:ed|al)|delete)\b"
            r".*\b(?:function|callable|api|export)\b|\bno longer\b"
        ),
    ),
)


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InstructionClaims:
    """High-level claims derived only from agent-visible instruction text."""

    no_runtime_claim: bool = False
    version_export_claim: bool = False
    version_export_only: bool = False
    behavior_outcome_count: int = 0
    version_outcome_count: int = 0
    total_outcome_count: int = 0
    multi_behavior_ask: bool = False
    matched_no_runtime: tuple[str, ...] = ()
    matched_version_export: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VerifierSignals:
    """High-level behavioural signals from test.patch (+ optional gold).

    Deliberately omits private F2P node-ID inventory — only coarse class
    tokens and counts for alignment scoring.
    """

    runtime_behavior_changes: bool = False
    version_meta_changes: bool = False
    version_only: bool = False
    behavior_class_count: int = 0
    behavior_classes: tuple[str, ...] = ()
    test_functions_added: tuple[str, ...] = ()
    runtime_assert_hits: int = 0
    gold_runtime_delta: bool = False
    gold_version_only: bool = False


@dataclass(frozen=True, slots=True)
class PromptAlignmentResult:
    """Outcome of prompt↔verifier alignment check (VAL-DHARD-001)."""

    ok: bool
    reason_code: str
    detail: str = ""
    claims: InstructionClaims = field(default_factory=InstructionClaims)
    signals: VerifierSignals = field(default_factory=VerifierSignals)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


class PromptVerifierMisalignRejected(Exception):
    """Product/live-generate export refused for prompt↔verifier misalignment."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = REASON_PROMPT_VERIFIER_MISALIGN,
        result: PromptAlignmentResult | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.result = result


# ---------------------------------------------------------------------------
# Dest helpers (local copies so module is independent of ship_real_pr)
# ---------------------------------------------------------------------------


def is_alignment_enforced_dest(dest: Path | str | None, *, offline_only: bool = False) -> bool:
    """True when product / live_generate dest must enforce the alignment gate."""
    if offline_only or dest is None:
        return False
    text = str(dest).replace("\\", "/").lower().rstrip("/")
    parts = [p for p in text.split("/") if p]
    if "deepagent_v1" in parts:
        leaf = parts[-1] if parts else ""
        if not (leaf.startswith("deepagent") and leaf != "deepagent_v1"):
            return True
    return any(part in _LIVE_GENERATE_DEST_MARKERS for part in parts)


# ---------------------------------------------------------------------------
# Instruction analysis
# ---------------------------------------------------------------------------


def analyze_instruction_claims(instruction: str | None) -> InstructionClaims:
    """Extract coarse claims from agent-visible instruction (no gold/test body)."""
    text = instruction or ""
    lower = text.lower()

    no_hits: list[str] = []
    for pat in _NO_RUNTIME_CLAIM_PATTERNS:
        m = pat.search(text)
        if m:
            no_hits.append(m.group(0)[:80])

    ver_hits: list[str] = []
    for pat in _VERSION_EXPORT_CLAIM_PATTERNS:
        m = pat.search(text)
        if m:
            snippet = m.group(0)[:80]
            if snippet not in ver_hits:
                ver_hits.append(snippet)

    outcomes = [m.group(1).strip() for m in _OUTCOME_LINE.finditer(text)]
    # Also accept prose under "## Expected outcomes" fenced as numbered later.
    if not outcomes:
        # loose bullets without ordered list markers still present?
        for line in text.splitlines():
            s = line.strip()
            if s.startswith(("-", "*")) and len(s) > 4:
                outcomes.append(s.lstrip("-* ").strip())

    version_outcomes = 0
    behavior_outcomes = 0
    for bullet in outcomes:
        if _VERSION_OUTCOME_BULLET.search(bullet):
            version_outcomes += 1
        if _BEHAVIOR_OUTCOME_BULLET.search(bullet):
            behavior_outcomes += 1

    total = len(outcomes)
    no_runtime = bool(no_hits)
    version_export = bool(ver_hits) or version_outcomes > 0

    # Version/export-only: version markers dominant, no substantive multi-behavior.
    # 1136-class: outcomes nearly all version/__all__/exports and constraints
    # forbid runtime changes.
    version_only = bool(
        (version_export and behavior_outcomes == 0)
        or (version_export and no_runtime and behavior_outcomes <= 1 and version_outcomes >= 1)
        or (
            version_export
            and total > 0
            and version_outcomes >= max(1, total - 1)
            and behavior_outcomes == 0
        )
        # Explicit version-prep narrative with no-runtime constraint is almost
        # always the 1136 misalign class even if one bullet says tests still pass.
        or (no_runtime and version_export and behavior_outcomes <= 1)
    )

    # Soft multi-behavior ask: ≥2 behavioural outcome bullets and no
    # hard "do not change runtime" contradiction.
    multi_behavior = behavior_outcomes >= 2 and not no_runtime

    # length floor: empty / stub instruction cannot claim multi-behavior.
    if len(text.strip()) < 80:
        multi_behavior = False

    # Title-level "version only" often appears even without bullets.
    if (
        re.search(r"(?i)\b(prepare|bump).{0,40}\bversion\b", lower)
        and re.search(r"(?i)\bpublic\s+api\b|\b__all__\b|\bexport", lower)
        and behavior_outcomes == 0
    ):
        version_only = True
        version_export = True

    return InstructionClaims(
        no_runtime_claim=no_runtime,
        version_export_claim=version_export,
        version_export_only=version_only,
        behavior_outcome_count=behavior_outcomes,
        version_outcome_count=version_outcomes,
        total_outcome_count=total,
        multi_behavior_ask=multi_behavior,
        matched_no_runtime=tuple(no_hits),
        matched_version_export=tuple(ver_hits[:6]),
    )


# ---------------------------------------------------------------------------
# Verifier / gold analysis
# ---------------------------------------------------------------------------


def _collect_behavior_classes(*texts: str) -> list[str]:
    joined = "\n".join(t for t in texts if t)
    found: list[str] = []
    for name, pat in _BEHAVIOR_CLASS_HINTS:
        if pat.search(joined):
            found.append(name)
    return found


def _patch_added_test_funcs(test_patch: str) -> list[str]:
    names = [m.group(1) for m in _BEHAVIOR_TEST_DEF.finditer(test_patch or "")]
    # renames: capture both -def and +def uniqueness
    removed = {m.group(1) for m in _BEHAVIOR_TEST_RENAME.finditer(test_patch or "")}
    # keep added and also context defs near runtime asserts
    out: list[str] = []
    for n in names:
        if n not in out:
            out.append(n)
    for n in removed:
        # renamed/removed functions still signal surface changes
        if n not in out and n.startswith("test_"):
            out.append(n)
    return out[:24]


def analyze_verifier_signals(
    test_patch: str | None,
    *,
    solution_patch: str | None = None,
) -> VerifierSignals:
    """Derive high-level F2P behavioural signals from patches.

    Preferred isolation-safe path: function names + assert patterns from
    test_patch, with optional gold corroboration. Does **not** return
    private node IDs.
    """
    tp = test_patch or ""
    sol = solution_patch or ""

    runtime_assert_hits = 0
    for pat in _RUNTIME_ASSERT_PATTERNS:
        runtime_assert_hits += len(pat.findall(tp))

    added_tests = _patch_added_test_funcs(tp)
    classes = _collect_behavior_classes(tp, sol)
    # Drop pure version/export class when stronger runtime classes present.
    runtime_like = [c for c in classes if c not in {"version_metadata", "export_all"}]

    version_meta = bool(
        re.search(r"(?i)\b__version__\b|\bversion\s*==\s*", tp + "\n" + sol)
        or re.search(r"(?i)\b__all__\b", tp + "\n" + sol)
    )

    gold_runtime = False
    gold_version_only = False
    if sol.strip():
        ver_lines = len(_GOLD_VERSION_LINES.findall(sol))
        raise_delta = bool(_GOLD_RAISE_DELTA.search(sol))
        body_delta = bool(_GOLD_RUNTIME_BODY.search(sol))
        # Count non-version +/- content lines.
        code_lines = [
            ln
            for ln in sol.splitlines()
            if (ln.startswith("+") or ln.startswith("-"))
            and not ln.startswith(("+++", "---"))
            and not re.search(r"\b__version__\b|\b__all__\b", ln)
            and not re.search(r"(?i)\bdocstring\b|\bdocs/", ln)
        ]
        nontrivial = [ln for ln in code_lines if len(ln.strip()) > 2]
        gold_runtime = raise_delta or (body_delta and len(nontrivial) >= 3)
        gold_version_only = ver_lines > 0 and not gold_runtime and len(nontrivial) <= 4

    # Runtime behaviour changes in test surface:
    runtime_behavior = False
    if runtime_assert_hits >= 2:
        runtime_behavior = True
    if runtime_like:
        runtime_behavior = True
    if any(
        re.search(r"(?i)(invalid|unhashable|raises?|error|contract|windowed|unique)", n)
        for n in added_tests
    ):
        runtime_behavior = True
    # Gold corroborates runtime when test side is light.
    if gold_runtime and (runtime_assert_hits >= 1 or bool(runtime_like)):
        runtime_behavior = True
    if gold_runtime and not version_meta:
        runtime_behavior = True

    # Pure version test edits?
    version_only = False
    if version_meta and not runtime_behavior and not gold_runtime:
        version_only = True
    if gold_version_only and not runtime_behavior:
        version_only = True

    return VerifierSignals(
        runtime_behavior_changes=runtime_behavior,
        version_meta_changes=version_meta,
        version_only=version_only,
        behavior_class_count=len(runtime_like) if runtime_like else (1 if runtime_behavior else 0),
        behavior_classes=tuple(runtime_like or classes),
        test_functions_added=tuple(added_tests),
        runtime_assert_hits=runtime_assert_hits,
        gold_runtime_delta=gold_runtime,
        gold_version_only=gold_version_only,
    )


# ---------------------------------------------------------------------------
# Core check
# ---------------------------------------------------------------------------


def check_prompt_verifier_alignment(
    instruction: str | None,
    *,
    test_patch: str | None = None,
    solution_patch: str | None = None,
    f2p_node_ids: Sequence[str] | None = None,
) -> PromptAlignmentResult:
    """Mechanical prompt↔verifier alignment check (VAL-DHARD-001).

    *f2p_node_ids* may inform multi-behavior emptiness only (count / coarse
    token presence). Prefer high-level classes from *test_patch*; never embed
    node IDs into instruction (caller responsibility).
    """
    claims = analyze_instruction_claims(instruction)
    signals = analyze_verifier_signals(test_patch, solution_patch=solution_patch)

    f2p = [str(x).strip() for x in (f2p_node_ids or ()) if str(x).strip()]
    # Coarse multi-behavior F2P mark without leaking node names into result detail.
    multi_f2p = (
        len(f2p) >= 3
        or signals.behavior_class_count >= 2
        or (signals.runtime_behavior_changes and signals.runtime_assert_hits >= 2)
    )

    # --- Fail-closed contradictions (order: specific → general) ---

    # 1) "do not change runtime" while verifier asserts runtime deltas.
    if claims.no_runtime_claim and signals.runtime_behavior_changes:
        return PromptAlignmentResult(
            ok=False,
            reason_code=REASON_PROMPT_NO_RUNTIME_CLAIM,
            detail=(
                "instruction claims do-not-change-runtime while test.patch/gold "
                f"exercises runtime behavioural asserts "
                f"(classes={list(signals.behavior_classes)[:5]})"
            ),
            claims=claims,
            signals=signals,
        )

    # 2) Version/export-only narrative vs behavioural F2P (1136-class).
    if claims.version_export_only and signals.runtime_behavior_changes:
        return PromptAlignmentResult(
            ok=False,
            reason_code=REASON_PROMPT_VERSION_ONLY,
            detail=(
                "instruction is version/export-only narrative while F2P/gold "
                f"change runtime behaviour (classes={list(signals.behavior_classes)[:5]})"
            ),
            claims=claims,
            signals=signals,
        )

    # Version-export narrative with no-runtime claim + gold runtime — catch soft
    # version_export_only misses still.
    if (
        claims.version_export_claim
        and claims.no_runtime_claim
        and (signals.gold_runtime_delta or signals.runtime_behavior_changes)
    ):
        return PromptAlignmentResult(
            ok=False,
            reason_code=REASON_PROMPT_VERSION_ONLY,
            detail=(
                "instruction version/export + no-runtime claims conflict with "
                f"runtime gold/F2P delta (classes={list(signals.behavior_classes)[:5]})"
            ),
            claims=claims,
            signals=signals,
        )

    # 3) Empty multi-behavior ask vs non-empty multi-behavior F2P.
    empty_behavior_ask = (
        claims.behavior_outcome_count == 0
        and not claims.multi_behavior_ask
        and len((instruction or "").strip()) >= 40
    )
    # Do not refuse pure version-matched pairs (both sides version-only).
    if (
        empty_behavior_ask
        and multi_f2p
        and signals.runtime_behavior_changes
        and not signals.version_only
    ):
        return PromptAlignmentResult(
            ok=False,
            reason_code=REASON_PROMPT_EMPTY_BEHAVIOR,
            detail=(
                "instruction lacks multi-behavior ask while F2P presents non-empty "
                f"behavioural delta (behavior_classes={signals.behavior_class_count}, "
                f"f2p_count={len(f2p)})"
            ),
            claims=claims,
            signals=signals,
        )

    # Aligned controls / non-contradictions.
    return PromptAlignmentResult(
        ok=True,
        reason_code=REASON_PROMPT_ALIGN_OK,
        detail="instruction claims consistent with high-level F2P/gold behaviours",
        claims=claims,
        signals=signals,
    )


def refuse_prompt_verifier_misalign(
    instruction: str | None,
    *,
    test_patch: str | None = None,
    solution_patch: str | None = None,
    f2p_node_ids: Sequence[str] | None = None,
    dest: Path | str | None = None,
    offline_only: bool = False,
    force: bool = False,
    task_id: str | None = None,
) -> PromptAlignmentResult:
    """Fail-closed refuse when product/live_generate pack is misaligned.

    Returns the green result when aligned or when the dest is not under product
    honesty (unless *force*). Raises
    :class:`PromptVerifierMisalignRejected` with a stable ``reason_code`` on
    refuse.
    """
    enforce = force or is_alignment_enforced_dest(dest, offline_only=offline_only)
    result = check_prompt_verifier_alignment(
        instruction,
        test_patch=test_patch,
        solution_patch=solution_patch,
        f2p_node_ids=f2p_node_ids,
    )
    if not enforce:
        return PromptAlignmentResult(
            ok=True,
            reason_code=REASON_PROMPT_ALIGN_SKIPPED,
            detail=f"alignment hard refuse skipped (non-product dest={dest!r})",
            claims=result.claims,
            signals=result.signals,
        )
    if result.ok:
        return result

    label = task_id or "pack"
    raise PromptVerifierMisalignRejected(
        f"product prompt–verifier alignment refuse for {label}: "
        f"{result.reason_code}: {result.detail} "
        f"(VAL-DHARD-001; dest={dest})",
        reason_code=result.reason_code,
        result=result,
    )


def summarize_test_intent(
    test_patch: str | None,
    *,
    max_items: int = 6,
) -> list[str]:
    """Cautious high-level test intent summaries (function names only).

    Safe for internal gates / optional teacher hints — **not** a dump of
    private tester bodies or full node-ID inventories into agent prompts.
    """
    names = _patch_added_test_funcs(test_patch or "")
    classes = _collect_behavior_classes(test_patch or "")
    items: list[str] = []
    for c in classes:
        if c in {"version_metadata", "export_all"}:
            continue
        items.append(f"behaviour_class:{c}")
        if len(items) >= max_items:
            return items
    for n in names:
        items.append(f"test_fn:{n}")
        if len(items) >= max_items:
            break
    return items


def alignment_result_from_pack_dir(pack_dir: Path | str) -> PromptAlignmentResult:
    """Convenience: read instruction.md + tests/test.patch + solution patch."""
    root = Path(pack_dir)
    instruction = ""
    instr_path = root / "instruction.md"
    if instr_path.is_file():
        instruction = instr_path.read_text(encoding="utf-8", errors="replace")
    test_patch = ""
    tp = root / "tests" / "test.patch"
    if tp.is_file():
        test_patch = tp.read_text(encoding="utf-8", errors="replace")
    sol = ""
    for rel in ("solution/solution.patch", "solution.patch"):
        sp = root / rel
        if sp.is_file():
            sol = sp.read_text(encoding="utf-8", errors="replace")
            break
    f2p: list[str] = []
    cfg = root / "tests" / "config.json"
    if cfg.is_file():
        try:
            import json

            blob = json.loads(cfg.read_text(encoding="utf-8"))
            raw = blob.get("f2p_node_ids") or blob.get("fail_to_pass") or []
            if isinstance(raw, Sequence):
                f2p = [str(x) for x in raw]
        except Exception:  # noqa: BLE001 — best-effort meta only
            f2p = []
    return check_prompt_verifier_alignment(
        instruction,
        test_patch=test_patch,
        solution_patch=sol,
        f2p_node_ids=f2p,
    )


__all__ = [
    "REASON_PROMPT_ALIGN_OK",
    "REASON_PROMPT_ALIGN_SKIPPED",
    "REASON_PROMPT_EMPTY_BEHAVIOR",
    "REASON_PROMPT_NO_RUNTIME_CLAIM",
    "REASON_PROMPT_VERIFIER_MISALIGN",
    "REASON_PROMPT_VERSION_ONLY",
    "InstructionClaims",
    "PromptAlignmentResult",
    "PromptVerifierMisalignRejected",
    "VerifierSignals",
    "alignment_result_from_pack_dir",
    "analyze_instruction_claims",
    "analyze_verifier_signals",
    "check_prompt_verifier_alignment",
    "is_alignment_enforced_dest",
    "refuse_prompt_verifier_misalign",
    "summarize_test_intent",
]
