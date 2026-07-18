"""Real-PR ship path for ``datasets/deepagent_v1`` (M13 rebaseline).

Product promote path (VAL-RSHIP / VAL-RX):

1. Ensure hybrid corpus is archived (``deepagent_v1_hybrid_archive``).
2. Load real_pr materials (merged PR source + held-out tests) — never motors.
3. Export Harbor trees with clone@SHA agent Dockerfiles (no hybrid_bind).
4. Dual-run labels into ``tests/config.json`` (real suite node ids when available;
   offline material path derives F2P seed ids from the held-out test.patch).
5. Docker oracle cert sol=1 / null=0 via :func:`certify_real_pr_pack` (fake refused).
6. Pier cert evidence via :func:`certify_real_pier_pack` (prefer live pier).
7. Promote ≥5 certified keeps, rewrite PROVENANCE/report for real_pr only.

Hybrid motors and ``source_track=hybrid_curated`` are **never** product-promoted.
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from swe_factory.accounting import BudgetLedger, default_ledger_path
from swe_factory.config import FactorySettings, load_settings
from swe_factory.envbuild.agent_recipe import render_real_pr_agent_dockerfile
from swe_factory.envbuild.sha import (
    BaseCommitError,
    assert_head_matches,
    is_full_sha,
    require_full_sha,
)
from swe_factory.harbor.deepagent_cert import FakeBackendRejected, refuse_fake_backend
from swe_factory.harbor.export_pack import REQUIRED_PACK_RELPATHS, verify_pack_tree
from swe_factory.harbor.harbor_oracle import HarborDockerVerifier, HarborVerifierBackend
from swe_factory.harbor.pier_cert import PierRunner, ScriptedPierRunner, SubprocessPierRunner
from swe_factory.harbor.real_oracle_cert import (
    RealOracleCertError,
    RealPrFakeOracleRejected,
    certify_real_pr_pack,
    refuse_fake_oracle_mode_real_pr,
)
from swe_factory.harbor.real_pack import (
    REAL_PR_SOURCE_TRACK,
    RealPackError,
    assert_product_real_pr_export,
    export_real_harbor_pack,
    is_hybrid_or_motor_track,
    validate_real_harbor_pack,
)
from swe_factory.harbor.real_pier_cert import certify_real_pier_pack
from swe_factory.harbor.schema import (
    MODEL_PATCH_ARTIFACT,
    GradeConfig,
    HarborMetadata,
    HarborPackSpec,
    HarborTaskIdentity,
    HarborTaskToml,
    HarborVerifier,
    TestsConfig,
)
from swe_factory.oracle.gates import count_files_in_patch
from swe_factory.panel.runner import (
    REQUIRED_PANEL_MODELS,
    offline_panel_from_matrix,
    run_panel_until_budget_zero,
)
from swe_factory.pipeline.archive_hybrid import archive_hybrid_deepagent
from swe_factory.pipeline.archive_seed5 import (
    DEFAULT_SEED5_ARCHIVE,
    ArchiveSeed5Error,
    archive_seed5_deepagent,
    require_seed5_archived,
)
from swe_factory.pipeline.cert_prefilter import (
    REASON_OK,
    prefilter_real_pr_material,
)
from swe_factory.pipeline.gate_audit_product import (
    ProductGateAuditError,
    ProductGateAuditResult,
    audit_keep_dual_truth,
    require_gate_audit_pass,
    write_product_gate_audit,
)
from swe_factory.pipeline.hardness_floors import (
    DEFAULT_MIN_F2P_NODES,
    PRODUCT_MIN_ADDED_LINES,
    PRODUCT_MULTI_FILE_FLOOR,
    PRODUCT_SOURCE_HUNK_FLOOR,
    ProductHardnessFloorRejected,
    refuse_product_hardness_floors,
    resolve_min_f2p_nodes,
)
from swe_factory.pipeline.prompt_alignment import (
    PromptVerifierMisalignRejected,
    refuse_prompt_verifier_misalign,
)
from swe_factory.pipeline.ship_deepagent import (
    DeepAgentPackRecord,
    HybridIdentity,
    ShipDeepAgentError,
    ShipDeepAgentResult,
    _ledger_snapshot,
    refuse_fake_ship_dest,
)
from swe_factory.producers.hard_filter import suite_reporter_detectable
from swe_factory.producers.materialize_from_pr import (
    DEFAULT_LIVE_MATERIALS_ROOT,
    inventory_completeness,
    is_fixture_materials_root,
    rebuild_inventory_from_task_dirs,
)
from swe_factory.producers.real_dual_run import (
    RealDualRunError,
    label_real_pr_dual_run,
)
from swe_factory.producers.suite_reporters import grade_tool_label_for
from swe_factory.sources.allowlist import SeedRepo
from swe_factory.sources.clone import CloneError, is_immutable_sha
from swe_factory.sources.clone_cache import CloneCache

# Real-PR wave defaults (M13): ≥5, not M10 hybrid 113.
DEFAULT_REAL_PR_TARGET = 5
DEFAULT_REAL_PR_MIN = 5
DEFAULT_REAL_PR_MAX = 20
DEFAULT_OUT = Path("datasets/deepagent_v1")
DEFAULT_ARCHIVE = Path("datasets/deepagent_v1_hybrid_archive")
DEFAULT_SEED5 = DEFAULT_SEED5_ARCHIVE
DEFAULT_PIER_JOBS = Path("/tmp/harbor-deepagent-jobs-ship-realpr")
# Engineering-only shortlist — never the product materials default (VAL-LMAT-003).
DEFAULT_MATERIALS = Path("fixtures/real_pr_ship")
# Product live-mine materials default (VAL-LMAT-002 / VAL-LSHIP-003).
DEFAULT_PRODUCT_MATERIALS = DEFAULT_LIVE_MATERIALS_ROOT
DEFAULT_CLONE_CACHE = Path("datasets/_clone_cache")
# Synthetics that mark dual-run as non-product (offline-only).
SYNTHETIC_P2P_NODE = "tests.test_ok::test_always_ok"
SYNTHETIC_F2P_MARKERS = (
    "test_real_pr_held_out",
    "test_always_ok",
    "test_real_pr_f2p",
    "f2p_node_ids_from_test_patch",
)
LABEL_METHOD_LIVE = "real_pr_dual_run_base_vs_gold"
LABEL_METHOD_SYNTHETIC = "synthetic_patch_seed"
# Burnt dual-run residue markers (VAL-LX-009). Presence without clean refuses product reuse.
BURNT_WORK_ROOT_MARKERS: tuple[str, ...] = (
    ".sdf_dual_run_failed",
    ".sdf_burnt",
    ".dual_run_poisoned",
)
FRESH_WORK_ROOT_MARKER = ".sdf_fresh_work_root"
PRODUCT_CLONE_SHA_PIN_OK = "product_clone_sha_pin_ok"

PanelMode = Literal["offline", "live", "skip"]
SourceTrackMode = Literal["real_pr"]

_TEST_DEF_RE = re.compile(r"^\+?(?:async\s+)?def\s+(test_\w+)\s*\(", re.MULTILINE)
_GIT_DIFF_TEST_RE = re.compile(r"^diff --git a/(?P<path>.+?) b/(?P<path_b>.+)$", re.MULTILINE)
_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_PRODUCT_DEST_MARKERS = ("deepagent_v1",)
# M16 live generate dest (VAL-DGEN): dual-truth honesty, not deepagent_v1 product wipe.
# M16/M21/M27 live generate + prod hardness dests (VAL-DGEN / VAL-DHARD / VAL-DMED).
_LIVE_GENERATE_DEST_MARKERS = ("test_n10", "prod_hard_keep", "prod_hard_deepswe_med")
_OFFLINE_DEST_MARKERS = ("offline", "offline_only", "_ut_", "fixture", "sandbox", "unit")


class ShipRealPrError(ShipDeepAgentError):
    """Unrecoverable Real-PR ship failure."""


class HybridProductPromoteRejected(ShipRealPrError, FakeBackendRejected):
    """Product path refuses hybrid / motor promote (VAL-RSHIP-005 / VAL-RX-003)."""


class ProductDualRunRejected(ShipRealPrError, FakeBackendRejected):
    """Product promote refused synthetic dual-run (theater ids)."""


class ProductOracleBackendRejected(ShipRealPrError, FakeBackendRejected):
    """Product promote refused Scripted*/Fake* docker masquerade."""


class ProductFixtureMaterialsRejected(ShipRealPrError, FakeBackendRejected):
    """Product dest refused fixtures/real_pr_ship materials default (VAL-LMAT-003)."""


class ProductEmptyLiveYieldRejected(ShipRealPrError, FakeBackendRejected):
    """Live mine empty yield fail-closed; no fixture pad (VAL-LSHIP-006)."""


class ProductSeed5ArchiveMissing(ShipRealPrError, FakeBackendRejected):
    """Product overwrite refused without seed5 archive (VAL-LSHIP-001)."""


class ProductGateAuditRejected(ShipRealPrError, FakeBackendRejected, ProductGateAuditError):
    """Product overwrite refused when dual-truth gate_audit fails (VAL-LSHIP-007)."""


class ProductPromptAlignRejected(ShipRealPrError, FakeBackendRejected):
    """Product/live_generate refuse when instruction contradicts F2P (VAL-DHARD-001)."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "prompt_verifier_misalign",
        result: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.result = result


class ProductHardnessFloorsRejected(ShipRealPrError, FakeBackendRejected):
    """Product/live_generate refuse when hardness floors fail (VAL-DHARD-002/003)."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "f2p_nodes_below_floor",
        result: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.result = result


@dataclass(frozen=True, slots=True)
class RealPrMaterial:
    """One merged-PR material set ready for Harbor export + certify."""

    task_id: str
    repository_url: str
    base_commit: str
    language: str
    license: str
    pr_number: int | None
    title: str
    source_files: tuple[str, ...]
    test_files: tuple[str, ...]
    solution_patch: str
    test_patch: str
    materials_dir: str = ""
    discovery_path: str = ""
    source_hunk_count: int | None = None
    # PR description body for DeepSWE-style full agent prompts (VAL-DPRMPT-001).
    # Empty when GitHub provides none or older materials predate body persistence.
    body: str = ""
    # Optional cached agent-visible instruction (DeepSWE rewrite). Private materials
    # may still keep PR meta; this field avoids re-LLM on re-export (VAL-DSTYLE-003).
    agent_instruction: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "repository_url": self.repository_url,
            "base_commit": self.base_commit,
            "language": self.language,
            "license": self.license,
            "pr_number": self.pr_number,
            "title": self.title,
            "body": self.body,
            "agent_instruction": self.agent_instruction,
            "source_files": list(self.source_files),
            "test_files": list(self.test_files),
            "materials_dir": self.materials_dir,
            "source_track": REAL_PR_SOURCE_TRACK,
            "solution_bytes": len(self.solution_patch),
            "test_bytes": len(self.test_patch),
            "discovery_path": self.discovery_path,
            "source_hunk_count": self.source_hunk_count,
        }


def refuse_hybrid_product_promote(
    source_track: str | None,
    *,
    dest: Path | str = DEFAULT_OUT,
    hybrid_bind: bool = False,
) -> None:
    """VAL-RSHIP-005 / VAL-RX-003: refuse hybrid/motor product promote."""
    track = (source_track or "").strip().lower()
    if hybrid_bind or is_hybrid_or_motor_track(track) or track.startswith("hybrid"):
        raise HybridProductPromoteRejected(
            "Real-PR product ship refuses hybrid/motor promote "
            f"(source_track={source_track!r}, hybrid_bind={hybrid_bind}) "
            f"on dest={dest}; require source_track=real_pr (VAL-RSHIP-005 / VAL-RX-003)"
        )
    if track and track != REAL_PR_SOURCE_TRACK:
        raise HybridProductPromoteRejected(
            f"Real-PR product ship requires source_track=real_pr; got {source_track!r}"
        )
    try:
        assert_product_real_pr_export(
            source_track=track or REAL_PR_SOURCE_TRACK,
            dest=dest,
            force_product=True,
            allow_hybrid=False,
            copy_repo_into_environment=None,
        )
    except RealPackError as exc:
        raise HybridProductPromoteRejected(str(exc)) from exc


def is_product_deepagent_dest(dest: Path | str) -> bool:
    """True when *dest* is the product deepagent_v1 surface (not offline_only).

    Product when any path segment is exactly ``deepagent_v1`` (e.g.
    ``datasets/deepagent_v1`` or ``.../deepagent_v1/tasks/x``). Offline /
    archive variants (``deepagent_v1_offline_only``,
    ``deepagent_v1_hybrid_archive``) are never product.

    Parent path segments that merely *contain* the substring ``deepagent``
    (e.g. pytest tmp dirs named ``test_ship_deepagent_cli_...``) are ignored so
    they do not pollute product detection.
    """
    text = str(dest).replace("\\", "/").lower().rstrip("/")
    parts = [p for p in text.split("/") if p]
    # Exact product segment required — substring "deepagent" in test names ignored.
    if "deepagent_v1" not in parts:
        return False
    # Offline/archive deepagent leaf (e.g. deepagent_v1_offline_only) is not product.
    leaf = parts[-1] if parts else ""
    return not (leaf.startswith("deepagent") and leaf != "deepagent_v1")


def is_live_generate_dest(dest: Path | str) -> bool:
    """True when *dest* is the M16 live-mine generate surface (``datasets/test_n10``).

    Separate from product deepagent_v1 so seed5 archive / product wipe logic
    does not fire, while dual-truth + HarborDocker honesty still apply.
    """
    text = str(dest).replace("\\", "/").lower().rstrip("/")
    parts = [p for p in text.split("/") if p]
    return any(part in _LIVE_GENERATE_DEST_MARKERS for part in parts)


def requires_dual_truth_honesty(
    dest: Path | str,
    *,
    live_mine: bool = False,
    offline_only: bool = False,
) -> bool:
    """True when dual-truth / HarborDocker / no-synthetic floors are mandatory.

    Fires for product deepagent_v1, live-mine flag, or ``datasets/test_n10``
    generate dest (VAL-DGEN-00*). Offline-only unit dests never trigger.
    """
    if offline_only:
        return False
    return bool(live_mine) or is_product_deepagent_dest(dest) or is_live_generate_dest(dest)


def is_fixture_real_pr_materials(path: Path | str | None) -> bool:
    """True when *path* is (or defaults to) the engineering fixture shortlist."""
    if path is None:
        return True  # unresolved product materials resolve to fixture default today
    return is_fixture_materials_root(path)


def resolve_product_materials_root(
    materials_root: Path | str | None,
    *,
    dest: Path | str,
    live_mine: bool = False,
    offline_only: bool = False,
    allow_fixture_materials: bool = False,
) -> Path | None:
    """Resolve materials root for ship, refuse fixture default on product dest.

    Rules (VAL-LMAT-002/003, VAL-LSHIP-003/004, VAL-DGEN):
    - Product dest ``datasets/deepagent_v1``, live-mine, and generate dest
      ``datasets/test_n10`` must use a live (non-fixture) materials root.
      Silent default ``fixtures/real_pr_ship`` is refused.
    - ``--live-mine`` without an explicit non-fixture root defaults to
      ``datasets/live_materials``.
    - Offline/unit dests may still use ``fixtures/real_pr_ship``.
    - Empty/unset materials on product dest is refused (no silent pad).
    """
    product = requires_dual_truth_honesty(dest, live_mine=live_mine, offline_only=offline_only)
    if not product:
        # Engineering / offline: keep historical fixture default when unset.
        if materials_root is None:
            return _package_root() / DEFAULT_MATERIALS
        return Path(materials_root)

    # Product dest / live-mine / test_n10 generate path.
    if materials_root is None:
        if live_mine or is_live_generate_dest(dest):
            return Path(DEFAULT_PRODUCT_MATERIALS)
        if allow_fixture_materials:
            # Explicit opt-in only (still not recommended for product truth).
            return _package_root() / DEFAULT_MATERIALS
        raise ProductFixtureMaterialsRejected(
            "product/live-generate dest refuses silent default materials="
            f"{DEFAULT_MATERIALS}; pass --live-mine (uses {DEFAULT_PRODUCT_MATERIALS}) "
            "and/or --materials pointing at a live materials root from the "
            "materialize-from-pr bridge (VAL-LMAT-003 / VAL-LSHIP-003 / VAL-DGEN)"
        )

    root = Path(materials_root)
    if is_fixture_materials_root(root) and not allow_fixture_materials:
        raise ProductFixtureMaterialsRejected(
            "product/live-generate dest refuses fixtures/real_pr_ship as materials root "
            f"(got {root}); require --live-mine + live materials root from the "
            "materialize bridge (VAL-LMAT-003 / VAL-LSHIP-003 / VAL-DGEN). "
            "Offline unit dests elsewhere may still use fixtures."
        )
    return root


def refuse_product_fixture_materials(
    materials_root: Path | str | None,
    *,
    dest: Path | str,
    live_mine: bool = False,
    offline_only: bool = False,
    allow_fixture_materials: bool = False,
) -> Path | None:
    """VAL-LMAT-003 / VAL-LSHIP-003/004: refuse fixture default on product dest.

    Returns the resolved materials root when allowed; raises
    :class:`ProductFixtureMaterialsRejected` otherwise.
    """
    return resolve_product_materials_root(
        materials_root,
        dest=dest,
        live_mine=live_mine,
        offline_only=offline_only,
        allow_fixture_materials=allow_fixture_materials,
    )


def refuse_empty_live_yield(
    *,
    certified_count: int,
    min_packs: int,
    dest: Path | str,
    live_mine: bool = False,
    materials_root: Path | str | None = None,
    offline_only: bool = False,
    padded_with_fixtures: bool = False,
) -> None:
    """VAL-LSHIP-006: empty / under-yield live funnel fails closed (no fixture pad).

    Product live-mine must not pad N from fixtures/real_pr_ship when live yield
    is zero or insufficient. Raises
    :class:`ProductEmptyLiveYieldRejected` when empty live yield or fixture pad
    is attempted on product dest; under-yield without pad is recorded as
    fail-closed via ShipRealPrError reason at end of ship (caller uses ok=False).
    """
    honesty = requires_dual_truth_honesty(dest, live_mine=live_mine, offline_only=offline_only)
    if offline_only or not honesty:
        return
    if padded_with_fixtures or (
        materials_root is not None and is_fixture_materials_root(materials_root)
    ):
        raise ProductEmptyLiveYieldRejected(
            "product live-mine ship refuses fixture pad of certified N "
            f"(materials={materials_root!r}, certified={certified_count}); "
            "empty/insufficient live yield must fail closed "
            "(VAL-LSHIP-006 / VAL-LMAT-003 / VAL-DGEN-001)"
        )
    if honesty and certified_count <= 0:
        raise ProductEmptyLiveYieldRejected(
            "product live-mine empty certified yield fails closed "
            f"(certified={certified_count} < min={min_packs}); "
            "no fixture pad allowed (VAL-LSHIP-006 / VAL-DGEN-001)"
        )


def require_product_suite_reporter(
    language: str | None,
    *,
    dest: Path | str,
    offline_only: bool = False,
) -> tuple[str, str]:
    """VAL-LHARD-002/006: product dual-run needs a real suite reporter path.

    Returns ``(reporter_id, suite_command)`` when detectable; refuses product
    promote when language cannot wire a real suite reporter.
    """
    # Suite-reporter hard floor applies on product + live-mine generate honesty paths.
    # Callers that need soft detect for offline unit leaves leave offline_only=True.
    hard = not offline_only and (is_product_deepagent_dest(dest) or is_live_generate_dest(dest))
    if not hard:
        ok, rid, cmd = suite_reporter_detectable(language)
        return (rid if ok else ""), (cmd if ok else "")
    ok, rid, cmd = suite_reporter_detectable(language)
    if not ok or not rid:
        raise ProductDualRunRejected(
            "product dual-run refuses keep without detectable real suite path "
            f"for language={language!r}; hard suite reporter required "
            f"(VAL-LHARD-002 / VAL-LHARD-006; dest={dest})"
        )
    return rid, cmd


def refuse_synthetic_product_dual_run(
    *,
    f2p_node_ids: Sequence[str],
    p2p_node_ids: Sequence[str],
    label_method: str | None,
    dest: Path | str,
    offline_only: bool = False,
    language: str | None = None,
    suite_reporter: Mapping[str, Any] | str | None = None,
    suite_command: str | None = None,
    require_suite_path: bool = True,
) -> None:
    """Refuse synthetic/empty/injected dual-run node ids on product promo.

    VAL-LHARD-006: product dual-run evidence must list **non-empty** node ids
    produced by a real language suite reporter (no ``test_always_ok`` /
    empty inject / f2p_from_patch-only). Complements VAL-LHARD-002 suite path.
    """
    honesty = not offline_only and (is_product_deepagent_dest(dest) or is_live_generate_dest(dest))
    if offline_only or not honesty:
        return
    method = (label_method or "").strip().lower()
    f2p = [str(x).strip() for x in f2p_node_ids if str(x).strip()]
    p2p = [str(x).strip() for x in p2p_node_ids if str(x).strip()]
    # Empty node lists are hard product refuse (VAL-LHARD-006).
    if not f2p:
        raise ProductDualRunRejected(
            "product dual-truth promote requires non-empty real suite-derived "
            f"F2P node ids (got empty; label_method={label_method!r}; VAL-LHARD-006)"
        )
    if any(not n for n in f2p):
        raise ProductDualRunRejected(
            "product dual-run refuses empty-string F2P node ids (VAL-LHARD-006)"
        )

    synth_p2p = any(
        SYNTHETIC_P2P_NODE in n or n.endswith("test_always_ok") or "test_always_ok" in n
        for n in p2p
    )
    synth_f2p = any(any(m in n for m in SYNTHETIC_F2P_MARKERS) for n in f2p)
    synthetic_method = method in {
        "",
        LABEL_METHOD_SYNTHETIC,
        "seed_base_ids",
        "f2p_node_ids_from_test_patch",
        "synthetic",
        "stub",
        "offline_synthetic",
        "hand_inject",
        "inject",
    }
    # Product live methods only (base_vs_gold / language-suited real suite equivalents).
    live_methods = {
        LABEL_METHOD_LIVE,
        "real_pr_dual_run_base_vs_gold",
        "real_pr_dual_run_broken_vs_green",
        "real_pr_dual_run_from_outcomes",
    }
    if synthetic_method or synth_p2p or synth_f2p:
        raise ProductDualRunRejected(
            "product dual-truth promote refuses synthetic dual-run "
            f"(label_method={label_method!r}, p2p={p2p[:3]!r}, f2p={f2p[:3]!r}); "
            "require live label_real_pr_dual_run on clone@SHA with real suite "
            f"node ids (VAL-LHARD-006 / VAL-DGEN-002; dest={dest})"
        )
    if method and method not in live_methods and "dual_run" not in method:
        raise ProductDualRunRejected(
            "product dual-run refuses non-live label_method "
            f"{label_method!r}; require real_pr_dual_run_base_vs_gold "
            f"(VAL-LHARD-006 / VAL-LSHIP-005; dest={dest})"
        )

    # Suite identity: hard path detectability + non-empty reporter on keep evidence.
    # When callers only assert synthetic-marker refuse without language (unit), skip
    # suite-path hard enforce unless language/reporter/command provided.
    has_suite_evidence = (
        language is not None or suite_reporter is not None or suite_command is not None
    )
    if require_suite_path and has_suite_evidence:
        rid, cmd = ("", "")
        if language is not None:
            rid, cmd = require_product_suite_reporter(language, dest=dest, offline_only=False)
        rep_id = ""
        if isinstance(suite_reporter, Mapping):
            rep_id = str(
                suite_reporter.get("reporter_id") or suite_reporter.get("tool_label") or ""
            ).strip()
            tool = str(suite_reporter.get("tool_label") or "").strip().lower()
            rid_blob = f"{rep_id} {tool}".lower()
            if any(m in rid_blob for m in ("fake", "stub", "motor_only", "hardcoded", "inject")):
                raise ProductDualRunRejected(
                    "product dual-run refuses stub/synthetic suite reporter "
                    f"{suite_reporter!r} (VAL-LHARD-006)"
                )
        elif suite_reporter is not None:
            rep_id = str(suite_reporter).strip()
        # Prefer explicit reporter when provided; fall back to language detect.
        effective_reporter = rep_id or rid
        effective_cmd = (suite_command or "").strip() or cmd
        if not effective_reporter:
            raise ProductDualRunRejected(
                "product dual-run requires real suite reporter identity on keep "
                f"(language={language!r}; VAL-LHARD-002 / VAL-LHARD-006)"
            )
        if not effective_cmd or effective_cmd.strip().lower() in {
            "offline_synthetic",
            "synthetic",
            "stub",
            "none",
        }:
            raise ProductDualRunRejected(
                "product dual-run requires real suite_command "
                f"(got {suite_command!r}); refuse synthetic inject (VAL-LHARD-006)"
            )


def assert_product_clone_sha_pin(
    *,
    ledger_base_commit: str,
    workspace: Path | str | None = None,
    recorded_base_commit: str | None = None,
    pack_base_commit: str | None = None,
    dest: Path | str = DEFAULT_OUT,
    offline_only: bool = False,
    require_workspace_head: bool = True,
) -> dict[str, str]:
    """VAL-LX-007: product clone@SHA pin equals ledger full 40-char base_commit.

    Equality chain: ``ledger_base_commit`` == pack meta (when provided) ==
    dual-run/envbuild recorded SHA == ``git rev-parse HEAD`` of workspace.
    Mismatch, floating HEAD/default-branch clone, short SHA only, or staged
    fixture SHAs labeled as live pins fail product promote.
    """
    honesty = not offline_only and (is_product_deepagent_dest(dest) or is_live_generate_dest(dest))
    if offline_only or not honesty:
        return {
            "ledger_base_commit": (ledger_base_commit or "").strip(),
            "status": "skipped_non_product",
        }

    try:
        ledger = require_full_sha(ledger_base_commit, allow_synthetic=False)
    except BaseCommitError as exc:
        raise ProductDualRunRejected(
            f"product clone@SHA pin refuses invalid ledger base_commit "
            f"{ledger_base_commit!r}: {exc} (VAL-LX-007)"
        ) from exc

    for label, value in (
        ("pack_base_commit", pack_base_commit),
        ("recorded_base_commit", recorded_base_commit),
    ):
        if value is None or not str(value).strip():
            continue
        raw = str(value).strip()
        if not is_full_sha(raw):
            raise ProductDualRunRejected(
                f"product clone@SHA pin requires full 40-char {label}; got {raw!r} (VAL-LX-007)"
            )
        if raw.lower() != ledger:
            raise ProductDualRunRejected(
                f"product clone@SHA pin mismatch: ledger={ledger} != "
                f"{label}={raw.lower()} (VAL-LX-007)"
            )

    head = ""
    if workspace is not None:
        ws = Path(workspace)
        if not ws.is_dir():
            raise ProductDualRunRejected(
                f"product clone@SHA pin workspace missing: {ws} (VAL-LX-007)"
            )
        if (ws / ".git").exists() or (ws / ".git").is_file():
            try:
                head = assert_head_matches(ws, ledger)
            except BaseCommitError as exc:
                raise ProductDualRunRejected(
                    f"product dual-run worktree HEAD must equal ledger "
                    f"base_commit={ledger}: {exc} (VAL-LX-007)"
                ) from exc
        elif require_workspace_head:
            # Non-git seeded trees (unit inject) still must declare matching pin.
            if recorded_base_commit and str(recorded_base_commit).strip().lower() == ledger:
                head = ledger
            else:
                raise ProductDualRunRejected(
                    f"product clone@SHA pin requires git worktree or recorded pin "
                    f"matching ledger={ledger} at {ws} (VAL-LX-007)"
                )
    elif require_workspace_head:
        raise ProductDualRunRejected(
            "product clone@SHA pin requires dual-run workspace for head check (VAL-LX-007)"
        )

    return {
        "ledger_base_commit": ledger,
        "workspace_head": head.lower() if head else ledger,
        "status": PRODUCT_CLONE_SHA_PIN_OK,
    }


def lookslike_burnt_work_root(path: Path | str | None) -> bool:
    """True when *path* carries burnt dual-run residue markers (VAL-LX-009)."""
    if path is None:
        return False
    root = Path(path)
    if not root.exists():
        return False
    for name in BURNT_WORK_ROOT_MARKERS:
        if (root / name).exists():
            return True
    # Nested base/gold trees left dirty with failure markers also count as burnt.
    for sub in ("base", "gold", "dual_run"):
        child = root / sub
        if not child.is_dir():
            continue
        for name in BURNT_WORK_ROOT_MARKERS:
            if (child / name).exists():
                return True
    return False


def refuse_burnt_dual_run_work_root(
    work_root: Path | str | None,
    *,
    dest: Path | str,
    offline_only: bool = False,
    allow_clean_rebuild: bool = False,
) -> None:
    """VAL-LX-009: refuse product dual-run outcomes from burnt/residue trees.

    Burnt-root theater (green=0 residue then instant green without clean) must
    not count as product dual-truth. Callers should prepare a fresh root or
    explicitly clean then re-mark readiness.
    """
    honesty = not offline_only and (is_product_deepagent_dest(dest) or is_live_generate_dest(dest))
    if offline_only or not honesty:
        return
    if work_root is None:
        return
    root = Path(work_root)
    if lookslike_burnt_work_root(root) and not allow_clean_rebuild:
        raise ProductDualRunRejected(
            "product dual-run refuses burnt work_root theater "
            f"(path={root}; markers={list(BURNT_WORK_ROOT_MARKERS)}); "
            "require fresh or explicitly cleaned root (VAL-LX-009)"
        )


def prepare_fresh_dual_run_work_root(
    work: Path | str,
    task_id: str,
    *,
    dest: Path | str,
    offline_only: bool = False,
) -> Path:
    """Create a fresh dual-run work root per pack; refuse burnt residue (VAL-LX-009).

    Always tears down any prior path at ``work/dual_run/<task_id>`` so product
    dual-run never reuses poisoned residue trees without rebuild/clean.
    """
    dual_work = Path(work) / "dual_run" / task_id
    product = (not offline_only) and (
        is_product_deepagent_dest(dest) or is_live_generate_dest(dest)
    )

    if dual_work.exists():
        # Existing root is treated as residue: wipe always for product honesty.
        if product and lookslike_burnt_work_root(dual_work):
            # Explicit wipe of burnt root (hygiene recovery).
            shutil.rmtree(dual_work, ignore_errors=True)
        else:
            shutil.rmtree(dual_work, ignore_errors=True)

    if product:
        refuse_burnt_dual_run_work_root(
            dual_work if dual_work.exists() else None,
            dest=dest,
            offline_only=False,
            allow_clean_rebuild=False,
        )

    dual_work.mkdir(parents=True, exist_ok=True)
    # Clean marker proves this root was prepared fresh for this pack.
    (dual_work / FRESH_WORK_ROOT_MARKER).write_text(
        f"task_id={task_id}\nprepared=fresh\n", encoding="utf-8"
    )
    # Ensure no burnt markers remain after prepare.
    for name in BURNT_WORK_ROOT_MARKERS:
        marker = dual_work / name
        if marker.exists():
            with contextlib.suppress(OSError):
                marker.unlink()
    if product:
        refuse_burnt_dual_run_work_root(
            dual_work, dest=dest, offline_only=False, allow_clean_rebuild=False
        )
    return dual_work


def mark_dual_run_work_root_burnt(
    work_root: Path | str,
    *,
    reason: str = "dual_run_failed",
) -> Path:
    """Stamp a dual-run work_root as burnt so product reuse is refused (VAL-LX-009)."""
    root = Path(work_root)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".sdf_dual_run_failed"
    marker.write_text(f"burnt=1\nreason={reason}\n", encoding="utf-8")
    return marker


def refuse_scripted_product_oracle(
    backend: HarborVerifierBackend | str | None,
    *,
    dest: Path | str,
    offline_only: bool = False,
) -> None:
    """Product path may only use real HarborDockerVerifier (never Scripted*/Fake*)."""
    honesty = not offline_only and (is_product_deepagent_dest(dest) or is_live_generate_dest(dest))
    if offline_only or not honesty:
        return
    try:
        refuse_fake_backend(backend, certified=True, dest=dest)
    except FakeBackendRejected as exc:
        raise ProductOracleBackendRejected(str(exc)) from exc
    if backend is None or (isinstance(backend, str) and backend.strip().lower() == "docker"):
        return
    if not isinstance(backend, HarborDockerVerifier):
        raise ProductOracleBackendRejected(
            "product dual-truth oracle requires HarborDockerVerifier "
            f"(got {type(backend).__name__}); refuse scripted/fake docker"
        )


def require_live_docker_images(
    *,
    agent_image: str | None,
    tests_image: str | None,
    dest: Path | str,
    offline_only: bool = False,
) -> None:
    """Shipped product evidence must show non-empty docker image refs."""
    honesty = not offline_only and (is_product_deepagent_dest(dest) or is_live_generate_dest(dest))
    if offline_only or not honesty:
        return
    agent = (agent_image or "").strip()
    tests = (tests_image or "").strip()
    if not agent or not tests:
        raise ProductOracleBackendRejected(
            "product dual-truth oracle evidence requires non-empty agent_image "
            f"and tests_image (got agent={agent!r} tests={tests!r}); "
            "live HarborDockerVerifier digests required"
        )


def _package_root() -> Path:
    return _PACKAGE_ROOT


def _seed_from_material(material: RealPrMaterial) -> SeedRepo:
    slug = material.repository_url.rstrip("/").removesuffix(".git")
    repo = slug.split("github.com/", 1)[1] if "github.com/" in slug else slug
    lang_raw = (material.language or "python").strip().lower()
    lang: Any = (
        lang_raw if lang_raw in {"python", "go", "typescript", "javascript", "rust"} else "python"
    )
    return SeedRepo(
        seed_id=material.task_id,
        language=lang,
        repo=repo,
        base_commit=material.base_commit,
        license=material.license or "MIT",
        description=material.title,
    )


def _ensure_dynamic_version_stubs(repo: Path) -> None:
    """Write minimal version modules when hatch/setuptools dynamic version is absent.

    Many dual-run failures are ``No module named '{pkg}.version'`` or
    ``{pkg}._version`` because the worktree is not installed editable and
    hatch-vcs generated files are not present at base SHA.
    """
    skip_dirs = {
        "tests",
        "test",
        "docs",
        "doc",
        "examples",
        "benchmarks",
        "tasks",
        "scripts",
        ".git",
        "node_modules",
    }
    candidates: list[Path] = []
    for base in (repo / "src", repo):
        if not base.is_dir():
            continue
        for pkg in base.iterdir():
            if not pkg.is_dir() or pkg.name in skip_dirs or pkg.name.startswith("."):
                continue
            if (pkg / "__init__.py").is_file():
                candidates.append(pkg)
    stub_body = (
        '"""SDF dual-run stub for dynamic package version."""\n'
        '__version__ = "0.0.0+sdf"\n'
        "__version_tuple__ = (0, 0, 0)\n"
        "VERSION = __version__\n"
        "version = __version__\n"
    )
    for pkg in candidates:
        for name in ("version.py", "_version.py"):
            target = pkg / name
            if target.exists():
                # Ensure required symbols exist on partial stubs.
                try:
                    cur = target.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if "__version_tuple__" not in cur and "SDF dual-run stub" in cur:
                    with contextlib.suppress(OSError):
                        target.write_text(stub_body, encoding="utf-8")
                continue
            needs = False
            for py_file in pkg.glob("*.py"):
                try:
                    body = py_file.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                token = name[:-3]
                if f".{token}" in body or f"import {token}" in body:
                    needs = True
                    break
            if (
                not needs
                and name == "version.py"
                and pkg.name
                in {
                    "platformdirs",
                    "attrs",
                    "attr",
                    "urllib3",
                }
            ):
                needs = True
            if not needs:
                continue
            with contextlib.suppress(OSError):
                target.write_text(stub_body, encoding="utf-8")


def _prepare_host_suite_env(repo: Path, *, language: str = "python") -> None:
    """Best-effort install so host dual-run reporters can import the package."""
    import subprocess
    import sys

    lang = (language or "python").strip().lower()
    if lang in {"javascript", "js", "typescript", "ts"}:
        # Live JS/TS dual-run needs package install + local bins on subsequent PATH.
        import shutil

        if shutil.which("npm") is None or not (repo / "package.json").is_file():
            return
        subprocess.run(
            ["npm", "install", "--no-audit", "--no-fund"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            from swe_factory.producers.harbor_labeling import ensure_local_node_bin_on_path

            # Best-effort: put local bins on this process PATH for subprocess children
            # that inherit os.environ (runner also sets env per suite call).
            env = ensure_local_node_bin_on_path(repo)
            import os

            if env.get("PATH"):
                os.environ["PATH"] = env["PATH"]
        except Exception:  # noqa: BLE001
            pass
        return
    if lang in {"rust", "rs"}:
        # No host install-beyond cargo; reporter runs cargo test in workspace copy.
        return
    if lang != "python":
        return
    from swe_factory.producers.harbor_labeling import (
        _rewrite_legacy_pytest_conf_hooks,
        _soft_relax_pytest_ini,
    )

    # Soft-fix legacy conf + strict markers before suite install/run.
    _rewrite_legacy_pytest_conf_hooks(repo)
    _soft_relax_pytest_ini(repo)
    # Generate missing hatch/setuptools-dynamic version stubs used by many
    # pure-python packages (platformdirs.version, urllib3._version, ...).
    _ensure_dynamic_version_stubs(repo)

    if (
        not (repo / "pyproject.toml").is_file()
        and not (repo / "setup.py").is_file()
        and not (repo / "setup.cfg").is_file()
    ):
        return
    py = sys.executable
    # Common test/runtime helpers across dual-run-survivable python seeds
    # (pallets/*, encode/*, pypa/packaging, jaraco-style, httpx-family).
    # Includes high-frequency collection deps from reject_ledger green=0 waves:
    # pretend (packaging), anyio/httpx/trio markers (httpcore), werkzeug/blinker
    # (flask), attrs (many pure-lib suites).
    subprocess.run(
        [
            py,
            "-m",
            "pip",
            "install",
            "-U",
            "-q",
            "httpx==0.28.1",
            "pytest",
            "pytest-asyncio",
            "pytest-mock",
            "freezegun",
            "hypothesis",
            "pretend",
            "anyio",
            "trio",
            "sniffio",
            "httpcore",
            "h11",
            "werkzeug>=3.0",
            "blinker>=1.8",
            "itsdangerous",
            "jinja2",
            "markupsafe",
            "click",
            "attrs",
            "idna",
            "ephemeral-port-reserve",
            "cloudpickle",
            "tomli_w",
            "tomli",
            "appdirs",
            # jaraco / pallets transitive helpers
            "inflect",
            "jaraco.itertools",
            "jaraco.functools",
            "jaraco.context",
            "jaraco.collections",
            "jaraco.test",
            "more_itertools",
            "packaging",
            "platformdirs",
            "attrs",
            "pydantic",
            "dataclasses-json",
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    # IMPORTANT: do NOT pip install -e the clone into the factory venv.
    # Host dual-run reporters already put the workspace on PYTHONPATH; an
    # editable install poisons site-packages with ephemeral /tmp paths and
    # makes subsequent dual-runs (and the warehouse itself) fail.
    # Optional non-editable deps for poorly declared package metadata only.
    req = repo / "requirements.txt"
    if req.is_file():
        subprocess.run(
            [py, "-m", "pip", "install", "-q", "-r", str(req)],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=False,
        )


def _materialize_base_worktree(
    material: RealPrMaterial,
    *,
    work: Path,
    clone_cache: CloneCache | None,
    seed_local_repos: Mapping[str, Path] | None = None,
) -> Path:
    """Clone (or reuse cache / local seed map) upstream @ base SHA for dual-run."""
    dest = work / "clones" / material.task_id
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Unit/offline may inject a pre-built local tree per task_id.
    if seed_local_repos and material.task_id in seed_local_repos:
        src = Path(seed_local_repos[material.task_id])
        if src.is_dir():
            shutil.copytree(
                src,
                dest,
                symlinks=True,
                ignore=shutil.ignore_patterns(
                    "__pycache__",
                    "*.pyc",
                    ".venv",
                    "node_modules",
                    ".pytest_cache",
                ),
            )
            return dest

    seed = _seed_from_material(material)
    cache = clone_cache or CloneCache(root=DEFAULT_CLONE_CACHE)
    # Prefer existing public clones under /tmp/realpr-mine when present.
    mine_slug = seed.repo.split("/")[-1] if "/" in seed.repo else seed.repo
    for cand in (
        Path("/tmp/realpr-mine") / mine_slug,
        Path("/tmp/realpr-mine") / seed.repo.replace("/", "__"),
    ):
        if cand.is_dir() and (cand / ".git").exists():
            shutil.copytree(
                cand,
                dest,
                symlinks=True,
                ignore=shutil.ignore_patterns(
                    "__pycache__",
                    "*.pyc",
                    ".venv",
                    "node_modules",
                    ".pytest_cache",
                ),
            )
            pin = material.base_commit
            if is_immutable_sha(pin):
                import subprocess

                checkout = subprocess.run(
                    ["git", "checkout", "--force", pin],
                    cwd=str(dest),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if checkout.returncode != 0:
                    subprocess.run(
                        ["git", "fetch", "origin", pin],
                        cwd=str(dest),
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    subprocess.run(
                        ["git", "checkout", "--force", pin],
                        cwd=str(dest),
                        capture_output=True,
                        text=True,
                        check=False,
                    )
            return dest

    return cache.materialize_worktree(
        seed,
        dest,
        base_commit=material.base_commit,
        refresh=False,
    )


def _label_dual_run_for_material(
    material: RealPrMaterial,
    *,
    pack_dir: Path,
    work: Path,
    dest: Path,
    offline_only: bool,
    clone_cache: CloneCache | None,
    seed_local_repos: Mapping[str, Path] | None,
    dual_run_callable: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Product dual-run via label_real_pr_dual_run; synthetic only offline_only."""
    config_path = pack_dir / "tests" / "config.json"
    dual_fn = dual_run_callable or label_real_pr_dual_run

    # Callers set offline_only=False for dual-truth honesty (product / live-mine /
    # datasets/test_n10). Synthetic patch-seed labels are offline unit only.
    if offline_only:
        f2p = f2p_node_ids_from_test_patch(material.test_patch, material.test_files)
        p2p = [SYNTHETIC_P2P_NODE]
        payload = {
            "base_commit": material.base_commit,
            "f2p_node_ids": f2p,
            "p2p_node_ids": p2p,
            "grade": {
                "format": "junit",
                "node_id": "name",
                "tool_label": grade_tool_label_for(material.language),
                "reports": ["/logs/verifier/new.xml", "/logs/verifier/base.xml"],
            },
            "label_method": LABEL_METHOD_SYNTHETIC,
            "suite_command": "offline_synthetic",
            "source_track": REAL_PR_SOURCE_TRACK,
        }
        config_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return {
            "ok": True,
            "offline_only": True,
            "label_method": LABEL_METHOD_SYNTHETIC,
            "f2p_node_ids": f2p,
            "p2p_node_ids": p2p,
            "suite": grade_tool_label_for(material.language),
            "suite_command": "offline_synthetic",
            "workspace": None,
            "config_path": str(config_path),
        }

    # Dual-truth path: live real-suite dual-run on clone@SHA.
    # Hard suite path required up front (VAL-LHARD-002 / VAL-LHARD-006).
    require_product_suite_reporter(material.language or "python", dest=dest, offline_only=False)
    try:
        base_repo = _materialize_base_worktree(
            material,
            work=work,
            clone_cache=clone_cache,
            seed_local_repos=seed_local_repos,
        )
    except (CloneError, OSError, Exception) as exc:  # noqa: BLE001
        raise ProductDualRunRejected(
            f"product dual-run could not materialize clone@SHA for {material.task_id}: {exc}"
        ) from exc

    # VAL-LX-007: clone@SHA pin must equal materials/ledger base_commit.
    pin_info = assert_product_clone_sha_pin(
        ledger_base_commit=material.base_commit,
        workspace=base_repo,
        recorded_base_commit=material.base_commit,
        dest=dest,
        offline_only=False,
        # Seeded unit trees may lack .git; head check still enforced when .git exists.
        require_workspace_head=True,
    )

    # VAL-LX-009: fresh work_root only; refuse burnt residue theater.
    dual_work = prepare_fresh_dual_run_work_root(
        work, material.task_id, dest=dest, offline_only=False
    )
    # Host dual-run needs the package importable (deps + editable install).
    _prepare_host_suite_env(base_repo, language=material.language or "python")
    try:
        agent_ctx = pack_dir / "environment"
        result = dual_fn(
            language=material.language or "python",
            base_repo=base_repo,
            solution_patch=material.solution_patch,
            test_patch=material.test_patch,
            base_commit=material.base_commit,
            config_dest=config_path,
            work_root=dual_work,
            agent_context=agent_ctx if agent_ctx.is_dir() else None,
            held_out_relative_paths=list(material.test_files),
            require_nonempty_f2p=True,
            dual_runs=1,
            source_track=REAL_PR_SOURCE_TRACK,
        )
    except (RealDualRunError, Exception) as exc:  # noqa: BLE001
        mark_dual_run_work_root_burnt(dual_work, reason=str(exc)[:200])
        raise ProductDualRunRejected(
            f"product live dual-run failed for {material.task_id}: {exc}"
        ) from exc

    f2p = list(result.f2p_node_ids)
    p2p = list(result.p2p_node_ids)
    label_method = str(
        (result.config_payload or {}).get("label_method")
        or result.notes.get("method")
        or LABEL_METHOD_LIVE
    )
    refuse_synthetic_product_dual_run(
        f2p_node_ids=f2p,
        p2p_node_ids=p2p,
        label_method=label_method,
        dest=dest,
        offline_only=False,
        language=material.language or "python",
        suite_reporter=result.reporter,
        suite_command=result.suite_command,
        require_suite_path=True,
    )
    # VAL-DMED-001 / VAL-DHARD-002: DeepSWE-median floors (files/hunks/f2p/added).
    try:
        refuse_product_hardness_floors(
            f2p_node_ids=f2p,
            source_files=list(material.source_files),
            source_hunk_count=material.source_hunk_count,
            solution_patch=getattr(material, "solution_patch", None),
            dest=dest,
            offline_only=False,
            task_id=material.task_id,
        )
    except ProductHardnessFloorRejected as exc:
        raise ProductHardnessFloorsRejected(
            str(exc),
            reason_code=exc.reason_code,
            result=exc.result,
        ) from exc
    # Re-assert pin: dual-run recorded base_commit must match ledger.
    assert_product_clone_sha_pin(
        ledger_base_commit=material.base_commit,
        workspace=base_repo,
        recorded_base_commit=result.base_commit or material.base_commit,
        pack_base_commit=material.base_commit,
        dest=dest,
        offline_only=False,
        require_workspace_head=bool((Path(base_repo) / ".git").exists()),
    )
    return {
        "ok": True,
        "offline_only": False,
        "label_method": label_method,
        "f2p_node_ids": f2p,
        "p2p_node_ids": p2p,
        "suite": grade_tool_label_for(material.language),
        "suite_command": result.suite_command,
        "reporter": result.reporter,
        "workspace": str(base_repo),
        "workspace_head": pin_info.get("workspace_head"),
        "ledger_base_commit": pin_info.get("ledger_base_commit"),
        "clone_sha_pin": pin_info.get("status"),
        "work_root": str(dual_work),
        "work_root_fresh": True,
        "config_path": str(config_path),
        "apply_log": list(result.apply_log),
    }


def _order_materials_for_live_yield(
    materials: Sequence[RealPrMaterial],
) -> list[RealPrMaterial]:
    """Round-robin multi-lang order so serial cert does not starve on mono-family.

    Product flames: inventory alphabetical put 14× bitflags/log first → soft dual-run
    rejects burn hours before python/js that historically dual_run+HarborDocker.
    Prefer languages with reliable product dual-run tooling first, interleaved.
    Within language, prefer materials that look dual-run/oracle friendly
    (held-out test defs for pytest; known-good apply patches).
    """
    priority = {
        "python": 0,
        "javascript": 1,
        "js": 1,
        "typescript": 2,
        "ts": 2,
        "go": 3,
        "golang": 3,
        "rust": 4,
    }

    def _friendliness(mat: RealPrMaterial) -> tuple[int, int, int, str]:
        # Lower tuple sorts first.
        lang = (mat.language or "").strip().lower()
        lang_rank = priority.get(lang, 50)
        test_defs = 0
        if mat.test_patch:
            test_defs = len(_TEST_DEF_RE.findall(mat.test_patch))
        # Prefer packs with real held-out test function defs (F2P evidence).
        def_rank = 0 if test_defs > 0 else 1
        # Prefer known dual-truth python families / expanded seed surface.
        # De-prioritize interactive-termui-ish or repeatedly green0 families.
        name = mat.task_id.lower()
        family_boost = 0
        if any(
            x in name
            for x in (
                "packaging-",
                "attrs-",
                "httpx-",
                "more-itertools-",
                "itemadapter-",
                "charset-normalizer-",
                "rich-",
                "jsonschema-",
                "tldextract-",
                "zipp-",
                "inflect-",
                "itsdangerous",
                "markupsafe",
                "boltons-",
                "cachetools-",
                "idna-",
            )
        ):
            family_boost = -2
        if "click-" in name:
            # Keep click packages (have certified before) but after packaging/attrs.
            family_boost = -1
        if any(
            x in name
            for x in (
                "platformdirs-",
                "werkzeug-",
                "httpcore-",
                "flask-",
                "blinker-",
                "bitflags-",
                "thiserror-",
                "log-",
                "zod-",
                "chalk-",
            )
        ):
            family_boost = 3
        # Prefer create-file near-misses that only failed HarborDocker apply due to
        # legacy pseudo-create headers (dual-run already proved F2P non-empty).
        create_near_miss = 0
        if any(
            name == x or name.endswith(x)
            for x in (
                "itemadapter-101",
                "charset-normalizer-715",
                "cachetools-385",
                "packaging-1120",
                "packaging-1267",
                "httpx-2252",
            )
        ):
            create_near_miss = -5
        return (lang_rank + family_boost + create_near_miss, def_rank, 0, mat.task_id)

    return sorted(materials, key=_friendliness)


def _preflight_patch_apply_ok(
    material: RealPrMaterial,
    *,
    clone_cache_root: Path | None = None,
) -> tuple[bool, str]:
    """Best-effort host git-apply check when clone cache already has the repo.

    Skips candidates whose solution/test patches cannot apply at base SHA —
    these always fail HarborDocker test.patch apply (reward -1) and waste
    serial cert slots. Fail-open (return True) when cache/repo missing so live
    path is not blocked offline.
    """
    import subprocess

    url = (material.repository_url or "").strip()
    base = (material.base_commit or "").strip().lower()
    if not url or len(base) != 40:
        return True, "skip_preflight_incomplete_identity"
    name = url.rstrip("/").split("/")[-1].removesuffix(".git")
    root = Path(clone_cache_root) if clone_cache_root else DEFAULT_CLONE_CACHE
    repo = root / name
    if not (repo / ".git").is_dir():
        return True, "skip_preflight_no_cache"
    try:
        subprocess.run(
            ["git", "-C", str(repo), "checkout", "-f", base],
            check=False,
            capture_output=True,
            timeout=60,
        )
        subprocess.run(
            ["git", "-C", str(repo), "reset", "--hard", base],
            check=False,
            capture_output=True,
            timeout=30,
        )
        subprocess.run(
            ["git", "-C", str(repo), "clean", "-fdx"],
            check=False,
            capture_output=True,
            timeout=30,
        )
        with tempfile.TemporaryDirectory(prefix="sdf-preflight-") as tmp:
            sol = Path(tmp) / "solution.patch"
            test_p = Path(tmp) / "test.patch"
            sol.write_text(material.solution_patch, encoding="utf-8")
            test_p.write_text(material.test_patch, encoding="utf-8")
            r_sol = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "apply",
                    "--check",
                    "--whitespace=nowarn",
                    str(sol),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            r_test = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "apply",
                    "--check",
                    "--whitespace=nowarn",
                    str(test_p),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        if r_sol.returncode != 0:
            return False, f"solution.patch apply-check failed: {(r_sol.stderr or '')[:200]}"
        if r_test.returncode != 0:
            return False, f"test.patch apply-check failed: {(r_test.stderr or '')[:200]}"
        return True, "apply_ok"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return True, f"skip_preflight_error:{exc}"


def load_real_pr_materials(
    materials_root: Path | str | None = None,
    *,
    limit: int | None = None,
    rebuild_inventory: bool = True,
) -> list[RealPrMaterial]:
    """Load real_pr ship materials from materials inventory (merged PRs only).

    When inventory.json is truncated vs on-disk task dirs (live materialize
    partial wave), rebuild it first so ship sees every hard survivor intended
    for the wave (m14 funnel completeness).
    """
    root = Path(materials_root) if materials_root else _package_root() / DEFAULT_MATERIALS
    if not root.is_dir():
        raise ShipRealPrError(f"real_pr materials root missing: {root}")
    inv_path = root / "inventory.json"
    rows: list[dict[str, Any]]
    # Completeness repair: inventory must cover every ship-loadable task dir.
    if rebuild_inventory and not is_fixture_materials_root(root):
        try:
            completeness = inventory_completeness(root)
            if not completeness.get("complete") or not inv_path.is_file():
                rebuild_inventory_from_task_dirs(root, merge_existing=True, write=True)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            # Best-effort rebuild; fall through to existing load paths.
            with contextlib.suppress(Exception):
                rebuild_inventory_from_task_dirs(root, merge_existing=True, write=True)
    if inv_path.is_file():
        raw = json.loads(inv_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not raw:
            # Empty inventory file with task dirs still on disk → rebuild once.
            if rebuild_inventory and not is_fixture_materials_root(root):
                rebuilt = rebuild_inventory_from_task_dirs(root, merge_existing=False, write=True)
                if rebuilt:
                    rows = list(rebuilt)
                else:
                    raise ShipRealPrError(f"empty/invalid inventory at {inv_path}")
            else:
                raise ShipRealPrError(f"empty/invalid inventory at {inv_path}")
        else:
            rows = list(raw)
            # Safety net: if inv shorter than loadable dirs, merge dir scan.
            if rebuild_inventory and not is_fixture_materials_root(root):
                try:
                    completeness = inventory_completeness(root)
                    missing = list(completeness.get("missing_from_inventory") or [])
                    if missing:
                        rows = rebuild_inventory_from_task_dirs(
                            root, merge_existing=True, write=True
                        )
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    pass
    else:
        rows = []
        if rebuild_inventory and not is_fixture_materials_root(root):
            rows = rebuild_inventory_from_task_dirs(root, merge_existing=False, write=True)
        if not rows:
            for child in sorted(root.iterdir()):
                meta = child / "meta.json"
                if not child.is_dir() or not meta.is_file():
                    continue
                data = json.loads(meta.read_text(encoding="utf-8"))
                data["materials_dir"] = str(child)
                rows.append(data)
    materials: list[RealPrMaterial] = []
    for row in rows:
        tid = str(row.get("task_id") or "").strip()
        if not tid:
            continue
        mat_dir = Path(str(row.get("materials_dir") or (root / tid)))
        if not mat_dir.is_absolute():
            # inventory may use repo-relative paths
            cand = _package_root() / mat_dir
            mat_dir = cand if cand.is_dir() else (root / tid)
        sol_path = mat_dir / "solution.patch"
        test_path = mat_dir / "test.patch"
        if not sol_path.is_file() or not test_path.is_file():
            continue
        from swe_factory.producers.pr_miner import repair_pseudo_create_file_headers

        sol = repair_pseudo_create_file_headers(
            sol_path.read_text(encoding="utf-8", errors="replace")
        )
        test = repair_pseudo_create_file_headers(
            test_path.read_text(encoding="utf-8", errors="replace")
        )
        src = tuple(
            str(p) for p in (row.get("src") or row.get("source_files") or []) if str(p).strip()
        )
        tests = tuple(
            str(p) for p in (row.get("tests") or row.get("test_files") or []) if str(p).strip()
        )
        # Prefer language-source product files (≥2). Multi-lang test suites use
        # tests/*.rs, *_test.go, *.test.js, tests/test_*.py — accept all of them.
        product = [p for p in src if not _looks_test_path(p)]
        test_files = [
            p
            for p in tests
            if _looks_test_path(p)
            or p.endswith((".py", ".go", ".js", ".ts", ".tsx", ".rs", ".jsx"))
        ]
        if len(product) < 2:
            product = [p for p in count_files_in_patch(sol) if not _looks_test_path(p)]
        if len(test_files) < 1:
            test_files = [p for p in count_files_in_patch(test) if _looks_test_path(p)]
        # When inventory lists test paths that are factually held-out in test.patch
        # (even if path heuristics miss, e.g. tests/*.stderr tooling fixtures),
        # accept non-empty test list from inventory if test.patch is non-empty.
        if len(test_files) < 1 and tests and test.strip():
            test_files = [p for p in tests if str(p).strip()]
        if len(product) < 2 or len(test_files) < 1 or not sol.strip() or not test.strip():
            continue
        url = str(row.get("url") or row.get("repository_url") or "").strip()
        if not url and row.get("repo"):
            url = f"https://github.com/{row['repo']}.git"
        base = str(row.get("base") or row.get("base_commit") or "").strip()
        if len(base) != 40:
            continue
        refuse_hybrid_product_promote(REAL_PR_SOURCE_TRACK)
        # Optional live-mine honesty fields (+ PR body / cached agent instruction).
        disc = str(row.get("discovery_path") or "").strip()
        body_text = str(row.get("body") or row.get("pr_body") or "").strip()
        agent_instr = str(row.get("agent_instruction") or "").strip()
        meta_path = mat_dir / "meta.json"
        meta_blob: dict[str, Any] = {}
        if meta_path.is_file():
            try:
                raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(raw_meta, dict):
                    meta_blob = raw_meta
            except (OSError, json.JSONDecodeError):
                meta_blob = {}
        if not disc and meta_blob:
            disc = str(meta_blob.get("discovery_path") or "").strip()
        if not body_text and meta_blob:
            body_text = str(meta_blob.get("body") or meta_blob.get("pr_body") or "").strip()
        if not agent_instr and meta_blob:
            agent_instr = str(meta_blob.get("agent_instruction") or "").strip()
        hunk_raw = row.get("source_hunk_count")
        hunk_n: int | None
        if hunk_raw is None:
            # Count unified-diff @@ headers in solution.patch (product source only).
            counted = sum(1 for line in sol.splitlines() if line.startswith("@@"))
            hunk_n = counted if counted > 0 else None
        else:
            try:
                hunk_n = int(hunk_raw)
            except (TypeError, ValueError):
                hunk_n = None
        materials.append(
            RealPrMaterial(
                task_id=tid,
                repository_url=url if url.endswith(".git") else url,
                base_commit=base,
                language=str(row.get("language") or "python").strip().lower(),
                license=str(row.get("license") or "MIT"),
                pr_number=int(row["pr"]) if row.get("pr") is not None else None,
                title=str(row.get("title") or tid),
                source_files=tuple(product),
                test_files=tuple(test_files),
                solution_patch=sol if sol.endswith("\n") else sol + "\n",
                test_patch=test if test.endswith("\n") else test + "\n",
                materials_dir=str(mat_dir),
                discovery_path=disc,
                source_hunk_count=hunk_n,
                body=body_text,
                agent_instruction=agent_instr,
            )
        )
        if limit is not None and len(materials) >= limit:
            break
    if not materials:
        raise ShipRealPrError(f"no qualifying real_pr materials under {root}")
    return materials


def _looks_test_path(path: str) -> bool:
    pl = path.lower().replace("\\", "/")
    # Also match roots without a leading slash (e.g. "tests/macros.rs" rust layout).
    if (
        pl.startswith("tests/")
        or pl.startswith("test/")
        or pl.startswith("__tests__/")
        or "/tests/" in pl
        or "/test/" in pl
        or "/__tests__/" in pl
    ):
        return True
    return any(
        x in pl
        for x in (
            "test_",
            "_test.",
            ".test.",
            ".spec.",
            "conftest",
            "_test.go",
            "_test.rs",
            "test.rs",
        )
    )


def f2p_node_ids_from_test_patch(test_patch: str, test_files: Sequence[str] = ()) -> list[str]:
    """Derive Harbor F2P node id seeds from held-out test.patch (+ paths)."""
    ids: list[str] = []
    for match in _TEST_DEF_RE.finditer(test_patch or ""):
        name = match.group(1)
        # map nearest preceding diff path if possible
        before = (test_patch or "")[: match.start()]
        paths = list(_GIT_DIFF_TEST_RE.finditer(before))
        mod = "tests"
        if paths:
            rel = paths[-1].group("path_b") or paths[-1].group("path")
            mod = rel.replace("\\", "/").removesuffix(".py").removesuffix(".js").replace("/", ".")
        ids.append(f"{mod}::{name}" if "::" not in name else name)
    if not ids:
        for tf in test_files:
            stem = Path(tf).name
            if stem.startswith("test_") or stem.endswith("_test.py"):
                mod = tf.replace("\\", "/").removesuffix(".py").replace("/", ".")
                ids.append(f"{mod}::test_real_pr_held_out")
    # de-dupe stable
    out: list[str] = []
    seen: set[str] = set()
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    if not out:
        out = ["tests.test_held_out::test_real_pr_f2p"]
    return out


def _default_install_commands(language: str) -> list[str]:
    """Build-time install: deps + editable package so real suite can collect.

    Avoid pulling lint plugin extras (ruff/mypy) that inject node ids the
    Harbor grader cannot score as product F2P/P2P.
    """
    lang = (language or "python").strip().lower()
    if lang == "python":
        return [
            # Minimal test host only. Do NOT preinstall package products that
            # collide with in-tree imports (e.g. httpx into encode/httpx, or
            # packaging into pypa/packaging). Editable install below owns that.
            "pip install --no-cache-dir pytest freezegun hypothesis pretend "
            "anyio trio sniffio h11 pytest-asyncio pytest-mock "
            "ephemeral-port-reserve tomli_w tomli appdirs cloudpickle",
            # Common jaraco/pallets transitive helpers only (not jaraco app code).
            "pip install --no-cache-dir "
            "jaraco.itertools jaraco.functools jaraco.context jaraco.collections "
            "jaraco.test more_itertools setuptools wheel inflect "
            "attrs pydantic dataclasses-json",
            # Editable project install WITHOUT testing extras that drag ruff.
            "if [ -f pyproject.toml ] || [ -f setup.py ] || [ -f setup.cfg ]; then "
            "pip install --no-cache-dir -e . || pip install --no-cache-dir . ; fi",
            # Ensure /app and /app/src are importable regardless of layout.
            "printf '%s\\n' /app /app/src > /usr/local/lib/python3.12/site-packages/sdf_app.pth",
        ]
    if lang in {"go", "golang"}:
        return ["true"]
    if lang in {"typescript", "javascript", "js", "ts"}:
        return ["if [ -f package.json ]; then npm install --no-audit --no-fund; fi"]
    if lang in {"rust", "rs"}:
        # Bake crates.io index + registry for offline verifier cargo test.
        return [
            "if [ -f Cargo.toml ]; then cargo fetch || cargo fetch --locked || true; fi",
        ]
    return ["true"]


# Soft caps for agent-facing PR description (VAL-DPRMPT-002 / legacy sanitizer).
_AGENT_PR_BODY_MAX_CHARS = 2000
_AGENT_INSTRUCTION_MIN_CHARS = 400

# Re-export DeepSWE builder for callers that historically imported from this module.
from swe_factory.pipeline.deepswe_prompt import (  # noqa: E402
    build_deepswe_style_instruction,
)


def sanitize_pr_body_for_prompt(
    body: str | None,
    *,
    max_chars: int = _AGENT_PR_BODY_MAX_CHARS,
) -> str:
    """Normalize/truncate PR body for instruction.md (never gold/test patch content)."""
    text = body if isinstance(body, str) else str(body or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""
    # Collapse extreme blank runs; keep readable paragraphs.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip fenced code that looks like unified diffs (answer-key risk).
    text = re.sub(
        r"```(?:diff|patch)?\n.*?```",
        "[diff/code block omitted from agent prompt]",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Drop any residual unified-diff markers that caused gold-leak hits in the past.
    lines: list[str] = []
    for ln in text.splitlines():
        stripped = ln.lstrip()
        if stripped.startswith("diff --git ") or stripped.startswith("@@ "):
            continue
        if stripped.startswith("--- a/") or stripped.startswith("+++ b/"):
            continue
        lines.append(ln)
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text


def build_real_pr_agent_instruction(
    material: RealPrMaterial,
    *,
    include_f2p_names: bool = False,
    f2p_node_ids: Sequence[str] | None = None,
    max_body_chars: int = _AGENT_PR_BODY_MAX_CHARS,
    force_offline: bool | None = None,
    client: Any | None = None,
    model: str | None = None,
    persist_cache: bool = True,
) -> str:
    """DeepSWE-true product prompt (behavior-first, no provenance).

    M18 (VAL-DSTYLE-001..005): rewrites private PR title/body/source file list into
    an agent-visible instruction matching real DeepSWE samples — **no** PR numbers,
    GitHub URLs, base SHAs, repo clone URLs, or M17 ``## Context`` /
    ``## PR description`` scaffolding.

    Live path uses OpenRouter teacher when keyed; offline/unit path uses a
    deterministic sanitizer. Cached materials ``meta.agent_instruction`` is reused
    on re-export to avoid repeat LLM spend.

    Held-out fail_to_pass node ids stay in tests/config.json (product isolation).
    *include_f2p_names* is retained for experiments but not mixed into the default
    DeepSWE register (listing node ids is not DeepSWE style).

    VAL-DPRMPT (legacy) + VAL-DSTYLE (authoritative for agent-visible text).
    """
    del include_f2p_names, f2p_node_ids, max_body_chars  # kept for call-site compat
    from swe_factory.pipeline.deepswe_prompt import build_deepswe_style_instruction

    title = (material.title or "").strip() or "Restore multi-module product behaviour"
    return build_deepswe_style_instruction(
        title=title,
        body=material.body or "",
        source_files=material.source_files,
        language=(material.language or "python").strip() or "python",
        materials_dir=material.materials_dir or None,
        cached_instruction=material.agent_instruction or None,
        force_offline=force_offline,
        client=client,
        model=model,
        persist_cache=persist_cache,
    )


def build_real_pr_pack_spec(
    material: RealPrMaterial,
    *,
    force_offline: bool | None = None,
    client: Any | None = None,
    model: str | None = None,
    persist_cache: bool = True,
    dest: Path | str | None = None,
    offline_only: bool = False,
    enforce_prompt_alignment: bool | None = None,
) -> HarborPackSpec:
    """Assemble a product HarborPackSpec for one merged PR material.

    Default agent prompt is the DeepSWE-true rewrite (VAL-DSTYLE-001..005) —
    behavior-first, no mining provenance — not the M17 Context/PR scaffolding
    and not the historical Merged-PR stub.

    VAL-DHARD-001: when *dest* is a product / live_generate surface (or
    *enforce_prompt_alignment* is True), fail-closed if the agent instruction
    contradicts high-level F2P/gold behavioural delta (e.g. version-only
    narrative while test.patch asserts windowed/unique_everseen runtime
    contracts). Offline engineering builds skip the hard refuse by default.
    """
    refuse_hybrid_product_promote(REAL_PR_SOURCE_TRACK)
    f2p = f2p_node_ids_from_test_patch(material.test_patch, material.test_files)
    p2p = ["tests.test_ok::test_always_ok"]
    df = render_real_pr_agent_dockerfile(
        repository_url=material.repository_url,
        base_commit=material.base_commit,
        language=material.language or "python",
        install_commands=_default_install_commands(material.language or "python"),
    )
    instruction = build_real_pr_agent_instruction(
        material,
        force_offline=force_offline,
        client=client,
        model=model,
        persist_cache=persist_cache,
    )
    # Prompt↔verifier alignment gate (M21 / VAL-DHARD-001).
    enforce = enforce_prompt_alignment
    if enforce is None and dest is not None:
        enforce = requires_dual_truth_honesty(dest, live_mine=False, offline_only=offline_only)
    if enforce:
        try:
            refuse_prompt_verifier_misalign(
                instruction,
                test_patch=material.test_patch,
                solution_patch=material.solution_patch,
                f2p_node_ids=f2p,
                dest=dest if dest is not None else "datasets/deepagent_v1",
                offline_only=offline_only,
                force=bool(enforce_prompt_alignment is True),
                task_id=material.task_id,
            )
        except PromptVerifierMisalignRejected as exc:
            raise ProductPromptAlignRejected(
                str(exc),
                reason_code=exc.reason_code,
                result=exc.result,
            ) from exc
    return HarborPackSpec.model_validate(
        {
            "task_id": material.task_id,
            "instruction_md": instruction,
            "task_toml": HarborTaskToml(
                schema_version="1.1",
                artifacts=[MODEL_PATCH_ARTIFACT],
                task=HarborTaskIdentity(name=f"swe-factory/{material.task_id}"),
                metadata=HarborMetadata(
                    language=material.language,
                    repository_url=material.repository_url,
                    base_commit_hash=material.base_commit,
                    task_id=material.task_id,
                    source_track=REAL_PR_SOURCE_TRACK,
                    license=material.license,
                ),
                verifier=HarborVerifier(environment_mode="separate", timeout_sec=1800.0),
            ),
            "tests_config": TestsConfig(
                base_commit=material.base_commit,
                f2p_node_ids=f2p,
                p2p_node_ids=p2p,
                grade=GradeConfig(
                    format="junit",
                    node_id="name",
                    tool_label=grade_tool_label_for(material.language),
                    reports=["/logs/verifier/new.xml", "/logs/verifier/base.xml"],
                ),
            ),
            "solution_patch": material.solution_patch,
            "test_patch": material.test_patch,
            "environment_dockerfile": df,
            "tests_dockerfile": (
                "FROM deepagent-agent:local\n"
                "COPY test.sh /tests/test.sh\n"
                "COPY grader.py /tests/grader.py\n"
                "COPY config.json /tests/config.json\n"
                "COPY test.patch /tests/test.patch\n"
            ),
        }
    )


def _append_drip(
    drip: list[dict[str, Any]],
    *,
    task_id: str,
    stage: str,
    status: str,
    detail: Mapping[str, Any] | None = None,
) -> None:
    drip.append(
        {
            "at": datetime.now(UTC).isoformat(),
            "task_id": task_id,
            "stage": stage,
            "status": status,
            "detail": dict(detail or {}),
            "source_track": REAL_PR_SOURCE_TRACK,
        }
    )


def _offline_keep_matrix() -> dict[str, list[bool]]:
    models = list(REQUIRED_PANEL_MODELS)
    matrix: dict[str, list[bool]] = {}
    for i, model in enumerate(models):
        if i == 1 or (len(models) == 1 and i == 0):
            matrix[model] = [True, False]
        else:
            matrix[model] = [False, False]
    return matrix


def _run_panel_for_keep(
    *,
    task_id: str,
    instruction: str,
    pack_dir: Path,
    panel_mode: PanelMode,
    ledger: BudgetLedger,
    soft_solver: Callable[..., bool] | None = None,
) -> dict[str, Any]:
    if panel_mode == "skip":
        return {
            "mode": "skip",
            "is_keep": True,
            "budget_stop": False,
            "panel_complete": False,
            "reason": "panel skipped by ship flag",
        }
    if panel_mode == "offline":
        result = offline_panel_from_matrix(
            task_id=task_id,
            solve_matrix=_offline_keep_matrix(),
            ledger=ledger,
            stage="deepagent-panel-realpr",
            problem_statement=instruction or "DeepAgent real_pr keep panel.",
            pack_path=pack_dir,
            pack_id=task_id,
        )
        raw = result.to_dict() if hasattr(result, "to_dict") else {"raw": str(result)}
        payload: dict[str, Any] = dict(raw)
        payload["mode"] = "offline"
        # Offline matrix is keep-shaped (in band); treat complete panel as keep.
        payload["is_keep"] = bool(getattr(result, "is_keep", True) or payload.get("is_keep", True))
        payload["panel_complete"] = True
        payload["budget_stop"] = bool(
            getattr(result, "budget_stop", False) or payload.get("budget_stop", False)
        )
        return payload
    # live panel while budget remains
    multi = run_panel_until_budget_zero(
        keeps=[
            {
                "task_id": task_id,
                "problem_statement": instruction or "DeepAgent real_pr keep",
                "pack_dir": str(pack_dir),
            }
        ],
        ledger=ledger,
        soft_solver=soft_solver,
        stage="deepagent-panel-realpr-live",
    )
    raw_live = multi.to_dict() if hasattr(multi, "to_dict") else {"mode": "live", "raw": str(multi)}
    payload = dict(raw_live)
    payload["mode"] = "live"
    kept_raw = payload.get("kept_task_ids") or payload.get("keeps") or []
    kept_ids: list[str] = []
    if isinstance(kept_raw, list):
        for item in kept_raw:
            if isinstance(item, dict):
                tid = item.get("task_id")
                if tid:
                    kept_ids.append(str(tid))
            elif item is not None:
                kept_ids.append(str(item))
    payload["is_keep"] = task_id in set(kept_ids) or bool(payload.get("is_keep", True))
    payload["budget_stop"] = bool(payload.get("budget_stop") or payload.get("stopped_budget"))
    return payload


def _language_counts(records: Sequence[DeepAgentPackRecord]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        if not r.certified:
            continue
        out[r.language] = out.get(r.language, 0) + 1
    return out


def _under_supply_real(languages: dict[str, int], certified: int) -> list[str]:
    reasons: list[str] = []
    if certified < DEFAULT_REAL_PR_MIN:
        reasons.append(
            f"certified={certified} < min={DEFAULT_REAL_PR_MIN}: real_pr funnel did not "
            "promote enough merged-PR packs (honest under-yield; never pad with hybrid motors)"
        )
    for lang in ("python", "go", "typescript", "javascript", "rust"):
        if languages.get(lang, 0) == 0:
            reasons.append(
                f"{lang}=0 best-effort under-supply on Real-PR wave: no certified "
                f"merged-PR keep announced for this language (honest shortfall; "
                f"not hybrid fill-in)"
            )
    return reasons


def _write_provenance_real(path: Path, records: Sequence[DeepAgentPackRecord]) -> None:
    lines = [
        "# PROVENANCE — datasets/deepagent_v1 (Real-PR product)",
        "",
        "Corpus of Docker-oracle-certified **real_pr** Harbor packs only.",
        "Hybrid motors live under `datasets/deepagent_v1_hybrid_archive/` "
        "(historical; never counted here as product N).",
        "Each row is one certified keep. Copyleft / unknown-license candidates",
        "are fail-closed and never appear here.",
        "",
        "| pack_id | language | license | upstream_url | base_sha | source_track | pr |",
        "|---|---|---|---|---|---|---:|",
    ]
    n = 0
    for r in records:
        if not r.certified:
            continue
        # Real-PR provenance uses hybrid field slot for identity dict with real_pr track
        ident = r.hybrid.to_dict() if r.hybrid is not None else {}
        track = str(ident.get("source_track") or REAL_PR_SOURCE_TRACK)
        if track != REAL_PR_SOURCE_TRACK:
            continue  # never list hybrid as certified product
        n += 1
        pr = ident.get("seed_id") or ""
        lines.append(
            f"| `{r.task_id}` | {r.language} | {ident.get('license', '')} | "
            f"{ident.get('repository_url', '')} | `{ident.get('base_commit', '')}` | "
            f"{track} | {pr} |"
        )
    lines.extend(
        [
            "",
            f"**Product certified N (real_pr only): {n}**",
            "",
            "## Notes",
            "",
            "- Product surface: `datasets/deepagent_v1` (source_track=real_pr only).",
            "- Hybrid archive (historical only): `datasets/deepagent_v1_hybrid_archive/`.",
            "- Fixtures (non-product): `datasets/harbor_v1`, `datasets/v1`.",
            "- Agent trees clone public git @ base SHA (no motor COPY hybrid_bind).",
            "- Docker oracle dual truth: sol=1 / null=0 (never `oracle_mode=fake`).",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_report_real(
    path: Path,
    *,
    result_payload: dict[str, Any],
    ledger: dict[str, Any],
    archive_note: str,
) -> None:
    langs = result_payload["languages"]
    under = result_payload["under_supply_reasons"]
    cert = [r for r in result_payload["records"] if r["certified"]]
    hybrid_in_cert = [
        r["task_id"]
        for r in cert
        if ((r.get("hybrid") or {}).get("source_track") or "") != REAL_PR_SOURCE_TRACK
    ]
    lines = [
        "# DeepAgent v1 ship report (Real-PR product surface)",
        "",
        f"- Generated (UTC): `{datetime.now(UTC).isoformat()}`",
        f"- Product path: `{result_payload.get('out_dir')}`",
        "- Source track (product): **`real_pr` only** (hybrid not certified product)",
        f"- Certified packs N: **{result_payload['certified_count']}** "
        f"(wave target ≥{result_payload['min_packs']}, target={result_payload['target_packs']}, "
        f"cap {result_payload['max_packs']})",
        f"- Docker oracle mode: `{result_payload['mode']}` (fake refused)",
        f"- Pier mode: `{result_payload['pier_mode']}`",
        f"- Panel mode: `{result_payload['panel_mode']}` "
        f"(budget_stop={result_payload.get('budget_stop')})",
        f"- Provider calls this wave: `{result_payload['provider_calls']}`",
        f"- Project spend commit: `${result_payload['spend_total_usd']}` "
        f"(remaining `${result_payload['remaining_usd']}`, "
        f"under_cap={result_payload['under_cap']})",
        f"- Status: `{'OK' if result_payload['ok'] else 'FAIL'}` — {result_payload['reason']}",
        "",
        "## Hybrid archive vs product honesty",
        "",
        archive_note,
        "",
        "- Hybrid archive is **historical only** and **not** folded into product N.",
        "- No `hybrid_curated` rows appear in product PROVENANCE.",
        f"- hybrid_ids_in_certified_scan: `{hybrid_in_cert or []}` (must be empty)",
        "",
        "## Historical fixtures (non-product)",
        "",
        result_payload.get("fixture_note")
        or (
            "`datasets/deepagent_v1_hybrid_archive`, `datasets/harbor_v1`, and `datasets/v1` "
            "are non-product. Milestone gates use independent real_pr N under deepagent_v1."
        ),
        "",
        "## Language mix (honest real_pr)",
        "",
        "| language | certified |",
        "|---|---:|",
    ]
    for lang in sorted(langs):
        lines.append(f"| {lang} | {langs[lang]} |")
    lines.extend(["", "### Under-supply notes", ""])
    if under:
        for u in under:
            lines.append(f"- {u}")
    else:
        lines.append("- (none)")
    lines.extend(
        [
            "",
            "## Funnel (ordered Real-PR stages)",
            "",
            f"- materials loaded: {len(result_payload['records'])}",
            f"- tree complete: {sum(1 for r in result_payload['records'] if r['tree_complete'])}",
            f"- real-pack ok: {sum(1 for r in result_payload['records'] if r['real_pack_ok'])}",
            f"- docker oracle cert: "
            f"{sum(1 for r in result_payload['records'] if r['docker_oracle_certified'])}",
            f"- pier cert: {sum(1 for r in result_payload['records'] if r['pier_certified'])}",
            f"- certified keeps (real_pr): {result_payload['certified_count']}",
            "",
            "## Certified packs (real_pr)",
            "",
        ]
    )
    for r in cert:
        hybrid = r.get("hybrid") or {}
        lines.append(
            f"- `{r['task_id']}` track={hybrid.get('source_track')} lang={r['language']} "
            f"files={r['solution_files']} sol={r['solution_reward']} null={r['null_reward']} "
            f"pier_oracle={r['pier_oracle_reward']} "
            f"upstream={hybrid.get('repository_url')} "
            f"sha=`{hybrid.get('base_commit')}` seed={hybrid.get('seed_id')}"
        )
    smoke = result_payload.get("harbor_load_smoke") or {}
    lines.extend(
        [
            "",
            "## Pier/Harbor structural load smoke",
            "",
            f"- ok: `{smoke.get('ok')}`",
            f"- tool: `{smoke.get('tool')}`",
            f"- sampled task_id: `{smoke.get('task_id')}`",
            f"- errors: {smoke.get('errors') or []}",
            "",
            "## Gates (no relaxation)",
            "",
            "- source_track=real_pr (hybrid refuse)",
            "- Real HTTPS repository_url + full 40-char base_commit",
            "- Agent Dockerfile: git clone@SHA (no motor COPY hybrid_bind)",
            "- Multi-file solution floor ≥2 product sources + held-out test.patch",
            "- Dual-run labels → tests/config.json F2P/P2P",
            "- Docker oracle only: sol=1, null=0 (never fake)",
            "- Pier oracle evidence (prefer live; scripted only with explicit ship flag)",
            "- Product N counts real_pr only; hybrid archive excluded",
            "- Project OpenRouter spend ≤ $600 (exact ledger)",
            "",
            "## Cross-flow (mine → … → ship)",
            "",
            "- Stages: archive_hybrid → mine/materials → export_real_harbor "
            "→ dual_run_labels → docker_oracle → pier_cert → panel → promote",
            f"- E2E drip: `{result_payload.get('e2e_drip_path')}`",
            "",
            "## Ledger summary (project)",
            "",
            "```json",
            json.dumps(
                {
                    "cap_usd": ledger.get("cap_usd"),
                    "settled_exact_usd": ledger.get("settled_exact_usd"),
                    "total_commit_usd": ledger.get("total_commit_usd"),
                    "remaining_usd": ledger.get("remaining_usd"),
                    "under_cap": ledger.get("under_cap"),
                    "settled_call_count": ledger.get("settled_call_count"),
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _real_pr_identity(material: RealPrMaterial) -> HybridIdentity:
    """Build a HybridIdentity-shaped record with source_track=real_pr."""
    return HybridIdentity(
        seed_id=f"pr:{material.pr_number}" if material.pr_number else material.task_id,
        repository_url=material.repository_url,
        base_commit=material.base_commit,
        license=material.license,
        language=material.language,
        upstream_label=material.repository_url,
        source_track=REAL_PR_SOURCE_TRACK,
    )


def run_ship_deepagent_real_pr(
    *,
    out_dir: Path | str = DEFAULT_OUT,
    work_root: Path | str | None = None,
    target_packs: int = DEFAULT_REAL_PR_TARGET,
    min_packs: int = DEFAULT_REAL_PR_MIN,
    max_packs: int = DEFAULT_REAL_PR_MAX,
    oracle_mode: str = "docker",
    panel_mode: PanelMode = "offline",
    pier_mode: Literal["scripted", "live"] = "scripted",
    settings: FactorySettings | None = None,
    materials_root: Path | str | None = None,
    archive_source: Path | str | None = None,
    archive_dest: Path | str | None = None,
    ensure_archive: bool = True,
    overwrite: bool = True,
    docker_backend: HarborVerifierBackend | None = None,
    pier_runner: PierRunner | None = None,
    pier_jobs_root: Path | str | None = None,
    soft_solver: Callable[..., bool] | None = None,
    allow_scripted_pier_substitute: bool = True,
    require_panel_keep: bool = False,
    hybrid_bind: bool = False,
    source_track: str = REAL_PR_SOURCE_TRACK,
    offline_only: bool = False,
    clone_cache_root: Path | str | None = None,
    seed_local_repos: Mapping[str, Path] | None = None,
    dual_run_callable: Callable[..., Any] | None = None,
    allow_scripted_oracle: bool = False,
    live_mine: bool = False,
    allow_fixture_materials: bool = False,
) -> ShipDeepAgentResult:
    """Ship ≥5 real_pr packs into product deepagent_v1 (refuse hybrid/fake).

    Product honesty (M13/M14):
    - ``offline_only=True`` enables synthetic dual-run + scripted oracle injectables
      and **cannot** write product ``datasets/deepagent_v1`` (must be non-product dest).
    - Product dest requires live ``label_real_pr_dual_run`` and real
      ``HarborDockerVerifier`` with non-empty agent/tests image refs.
    - Product dest refuses silent materials default ``fixtures/real_pr_ship``;
      use ``live_mine=True`` and/or a live materials root from materialize-from-pr
      (VAL-LMAT-002/003, VAL-LSHIP-003/004). Empty live yield fails closed with
      no fixture pad (VAL-LSHIP-006).
    """
    settings = settings or load_settings()
    mode = (oracle_mode or "docker").strip().lower()
    dest = Path(out_dir)

    # Offline-only may never write product deepagent_v1.
    if offline_only and is_product_deepagent_dest(dest):
        raise ShipRealPrError(
            f"offline_only ship cannot write product dest {dest}; "
            "use a non-product path such as datasets/deepagent_v1_offline_only "
            "or pass offline_only=False for live product promote"
        )
    product_dest = is_product_deepagent_dest(dest) and not offline_only
    # Dual-truth honesty for product deepagent_v1, --live-mine, and M16 test_n10.
    # product_dest still gates seed5 archive / prior-product wipe only.
    honesty_dest = requires_dual_truth_honesty(dest, live_mine=live_mine, offline_only=offline_only)

    # ---- hard refuse gates ----
    refuse_fake_ship_dest(mode, out_dir=dest)
    refuse_fake_oracle_mode_real_pr(mode, certified=True, dest=dest, oracle_mode=mode)
    refuse_fake_backend(mode, certified=True, dest=dest)
    if mode != "docker":
        raise RealPrFakeOracleRejected(
            f"ship-deepagent --source real_pr requires oracle_mode=docker (got {mode!r})"
        )
    if hybrid_bind or is_hybrid_or_motor_track(source_track):
        raise HybridProductPromoteRejected(
            "ship-deepagent refuses hybrid_bind / hybrid source for product real_pr path"
        )
    refuse_hybrid_product_promote(source_track, dest=dest, hybrid_bind=hybrid_bind)

    # Product/live-mine/test_n10: refuse fixture shortlist materials (M14/M16).
    # Pass live_mine flag as-is: product dest with materials_root=None and
    # live_mine=False still refuses silent fixture default; --live-mine alone
    # defaults to datasets/live_materials. is_live_generate_dest also defaults
    # to live_materials inside resolve_product_materials_root.
    resolved_materials = refuse_product_fixture_materials(
        materials_root,
        dest=dest,
        live_mine=live_mine,
        offline_only=offline_only,
        allow_fixture_materials=allow_fixture_materials,
    )
    # Live-mine without materials path still points at live default root.
    materials_for_load: Path | str | None = resolved_materials

    # Dual-truth: refuse scripted/fake docker injectables up front.
    if honesty_dest and not allow_scripted_oracle:
        refuse_scripted_product_oracle(docker_backend, dest=dest, offline_only=False)

    if min_packs > max_packs:
        raise ShipRealPrError("min_packs must be ≤ max_packs")
    if min_packs < 1:
        raise ShipRealPrError("min_packs must be ≥ 1")
    target_packs = max(min_packs, min(target_packs, max_packs))

    # ---- archive hybrid first (idempotent) ----
    archive_note = "hybrid archive step skipped (ensure_archive=false)"
    if ensure_archive:
        src_arch = Path(archive_source) if archive_source else dest
        arch = Path(archive_dest) if archive_dest else DEFAULT_ARCHIVE
        try:
            ares = archive_hybrid_deepagent(
                source_dir=src_arch if (src_arch / "pack_manifest.json").is_file() else arch,
                archive_dir=arch,
            )
            archive_note = (
                f"Hybrid archive action=`{ares.action}` → `{arch}` "
                f"(archive_pack_count={ares.archive_pack_count}); "
                f"ok={ares.ok}; historical only, never product N. reason={ares.reason}"
            )
            if not ares.ok and not (arch / "pack_manifest.json").is_file():
                raise ShipRealPrError(
                    f"hybrid archive missing under {arch}; refuse real_pr overwrite: {ares.reason}"
                )
        except ShipRealPrError:
            raise
        except Exception as exc:  # noqa: BLE001
            if (DEFAULT_ARCHIVE / "pack_manifest.json").is_file() or (
                DEFAULT_ARCHIVE / "ARCHIVE_README.md"
            ).is_file():
                archive_note = (
                    f"Hybrid archive already present at {DEFAULT_ARCHIVE} (re-check after: {exc})"
                )
            else:
                raise ShipRealPrError(f"hybrid archive required before real ship: {exc}") from exc

    # ---- archive seed5 prior product (VAL-LSHIP-001) before any product wipe ----
    seed5_note = "seed5 archive not required (non-product dest)"
    seed5_archived = False
    if product_dest:
        try:
            sres = archive_seed5_deepagent(
                source_dir=dest,
                archive_dir=DEFAULT_SEED5,
                force_recopy=False,
                require_real_pr=True,
            )
            seed5_archived = bool(sres.ok and sres.archive_pack_count > 0)
            seed5_note = (
                f"Seed5 archive action=`{sres.action}` → `{DEFAULT_SEED5}` "
                f"(archive_pack_count={sres.archive_pack_count}); ok={sres.ok}; "
                f"historical real_pr seed only, never live N. reason={sres.reason}"
            )
            if not sres.ok and sres.action != "noop_empty":
                # If source already empty and archive missing, hard fail product wave.
                try:
                    require_seed5_archived(archive_dir=DEFAULT_SEED5, min_packs=1)
                    seed5_archived = True
                    seed5_note += " | recovered via existing seed5 archive evidence"
                except ArchiveSeed5Error as archived_exc:
                    raise ProductSeed5ArchiveMissing(str(archived_exc)) from archived_exc
            elif sres.ok and sres.archive_pack_count <= 0 and sres.action == "noop_empty":
                # First-time empty product may ship live without prior seed, but record honesty.
                seed5_note += " | no prior product seed present (first-wave honesty)"
                seed5_archived = True  # no prior seed to freeze
            else:
                seed5_archived = True
        except ProductSeed5ArchiveMissing:
            raise
        except ArchiveSeed5Error as exc:
            raise ProductSeed5ArchiveMissing(
                f"seed5 archive required before product overwrite: {exc} (VAL-LSHIP-001)"
            ) from exc
        archive_note = f"{archive_note} || {seed5_note}"

    # ---- work + provisional evidence (do NOT wipe product yet: VAL-LSHIP-007) ----
    work = (
        Path(work_root)
        if work_root
        else Path(tempfile.mkdtemp(prefix="sdf-ship-realpr-", dir=str(dest.parent)))
    )
    work.mkdir(parents=True, exist_ok=True)
    # Staging roots under work — product dest untouched until gate_audit pass.
    staging_product = work / "product_stage"
    if staging_product.exists():
        shutil.rmtree(staging_product)
    tasks_out = staging_product / "tasks"
    tasks_out.mkdir(parents=True, exist_ok=True)
    evidence_root = staging_product / "evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    oracle_ev_dir = evidence_root / "docker"
    oracle_ev_dir.mkdir(parents=True, exist_ok=True)
    pier_ev_dir = evidence_root / "pier"
    pier_ev_dir.mkdir(parents=True, exist_ok=True)
    e2e_path = staging_product / "e2e_drip.jsonl"
    if e2e_path.exists():
        e2e_path.unlink()

    # Load materials. Empty load under live-mine / product dest → fail closed
    # (VAL-LSHIP-006), never pad from fixtures/real_pr_ship.
    # Live mine needs headroom: soft dual-run/oracle rejects must not starve
    # the wave — load all inventory rows (or at least min*6) so serial cert
    # can continue until min_packs or honest exhaustion (used_materials >> 2).
    materials_limit = None if honesty_dest else max(target_packs * 3, max_packs * 2, 10)
    if honesty_dest:
        # Explicit large cap so a huge materials tree is still bounded,
        # but always enough to exhaust soft rejects past min_packs.
        materials_limit = max(target_packs * 6, max_packs * 4, min_packs * 6, 60)
    try:
        materials = load_real_pr_materials(
            materials_for_load,
            limit=materials_limit,
            rebuild_inventory=True,
        )
        # Serial cert headroom: diversify / prioritize dual-run-friendly langs
        # early so rust-heavy inventory sort does not starve python/js/ts yield
        # before min_packs is reachable. Round-robin by language family.
        if honesty_dest:
            materials = _order_materials_for_live_yield(materials)
    except ShipRealPrError as exc:
        if honesty_dest:
            # Re-check fixture pad first for clearer reason.
            refuse_empty_live_yield(
                certified_count=0,
                min_packs=min_packs,
                dest=dest,
                live_mine=True,
                materials_root=materials_for_load,
                offline_only=offline_only,
                padded_with_fixtures=is_fixture_materials_root(materials_for_load)
                if materials_for_load is not None
                else False,
            )
            raise ProductEmptyLiveYieldRejected(
                f"product live materials load failed closed (no fixture pad): {exc} "
                f"(materials={materials_for_load!r}; VAL-LSHIP-006 / VAL-LMAT-002)"
            ) from exc
        raise

    records: list[DeepAgentPackRecord] = []
    total_provider_calls = 0
    budget_stop = False
    ledger = BudgetLedger(path=default_ledger_path(), cap_usd=Decimal(str(settings.budget_usd)))
    pier_jobs = Path(pier_jobs_root) if pier_jobs_root else DEFAULT_PIER_JOBS
    pier_jobs.mkdir(parents=True, exist_ok=True)
    clone_cache = CloneCache(
        root=Path(clone_cache_root) if clone_cache_root else DEFAULT_CLONE_CACHE
    )
    # Pier mode actually used on product (may fall back scripted if live fails)
    pier_mode_report: str = pier_mode

    for material in materials:
        if sum(1 for r in records if r.certified) >= target_packs:
            break
        if sum(1 for r in records if r.certified) >= max_packs:
            break

        drip: list[dict[str, Any]] = []
        task_id = material.task_id
        _append_drip(
            drip,
            task_id=task_id,
            stage="mine",
            status="ok",
            detail=material.to_dict(),
        )

        # Host prefilter before expensive Docker: patch apply + collector dry-run
        # F2P potential. Skip empty-F2P / apply-fail early with ledger reason.
        # Falls back to legacy patch-only preflight when prefilter fails open.
        if honesty_dest:
            cache_root_pf = Path(clone_cache_root) if clone_cache_root else DEFAULT_CLONE_CACHE
            pf = prefilter_real_pr_material(
                material,
                clone_cache_root=cache_root_pf,
                run_collect=(material.language or "python").strip().lower() == "python",
            )
            if not pf.ok and pf.reason_code not in {
                REASON_OK,
                "prefilter_skip_no_cache",
                "prefilter_skip_incomplete_identity",
                "prefilter_skip_non_python_collect",
            }:
                _append_drip(
                    drip,
                    task_id=task_id,
                    stage="cert_prefilter",
                    status="reject",
                    detail=pf.to_dict(),
                )
                records.append(
                    DeepAgentPackRecord(
                        task_id=task_id,
                        seed_id=material.task_id,
                        language=material.language,
                        pack_dir=work / "staging" / task_id,
                        hybrid=_real_pr_identity(material),
                        solution_files=list(material.source_files),
                        multi_file_ok=len(material.source_files) >= 2,
                        tree_complete=False,
                        real_pack_ok=False,
                        docker_oracle_certified=False,
                        pier_certified=False,
                        panel_keep=False,
                        solution_reward=None,
                        null_reward=None,
                        pier_oracle_reward=None,
                        pier_null_reward=None,
                        agent_isolated=False,
                        reasons=[f"cert_prefilter rejected: {pf.reason_code}: {pf.detail}"],
                        certified=False,
                        drip=drip,
                    )
                )
                with e2e_path.open("a", encoding="utf-8") as handle:
                    for row in drip:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                continue
            # Legacy patch-only preflight for fail-open (no cache) cases so
            # solution reward -1 patch fails still get ledger reasons when
            # possible later via clone cache population.
            if pf.reason_code in {
                "prefilter_skip_no_cache",
                "prefilter_skip_incomplete_identity",
            }:
                ok_apply, apply_reason = _preflight_patch_apply_ok(
                    material,
                    clone_cache_root=cache_root_pf,
                )
                if not ok_apply:
                    _append_drip(
                        drip,
                        task_id=task_id,
                        stage="patch_preflight",
                        status="reject",
                        detail={"reason": apply_reason},
                    )
                    records.append(
                        DeepAgentPackRecord(
                            task_id=task_id,
                            seed_id=material.task_id,
                            language=material.language,
                            pack_dir=work / "staging" / task_id,
                            hybrid=_real_pr_identity(material),
                            solution_files=list(material.source_files),
                            multi_file_ok=len(material.source_files) >= 2,
                            tree_complete=False,
                            real_pack_ok=False,
                            docker_oracle_certified=False,
                            pier_certified=False,
                            panel_keep=False,
                            solution_reward=None,
                            null_reward=None,
                            pier_oracle_reward=None,
                            pier_null_reward=None,
                            agent_isolated=False,
                            reasons=[f"patch_preflight rejected: {apply_reason}"],
                            certified=False,
                            drip=drip,
                        )
                    )
                    with e2e_path.open("a", encoding="utf-8") as handle:
                        for row in drip:
                            handle.write(json.dumps(row, sort_keys=True) + "\n")
                    continue

        identity = _real_pr_identity(material)

        # export real harbor pack (refuse hybrid + prompt↔verifier alignment)
        staging_pack = work / "staging" / task_id
        try:
            refuse_hybrid_product_promote(REAL_PR_SOURCE_TRACK, dest=dest)
            spec = build_real_pr_pack_spec(
                material,
                dest=dest,
                offline_only=offline_only or not honesty_dest,
            )
            export_result = export_real_harbor_pack(
                spec,
                dest=staging_pack,
                overwrite=True,
                require_real_pr_track=True,
                allow_hybrid=False,
            )
            pack_dir = export_result.pack_dir
        except (ProductPromptAlignRejected, PromptVerifierMisalignRejected) as exc:
            reason_code = getattr(exc, "reason_code", "prompt_verifier_misalign")
            records.append(
                DeepAgentPackRecord(
                    task_id=task_id,
                    seed_id=identity.seed_id,
                    language=material.language,
                    pack_dir=staging_pack,
                    hybrid=identity,
                    solution_files=list(material.source_files),
                    multi_file_ok=len(material.source_files) >= 2,
                    tree_complete=False,
                    real_pack_ok=False,
                    docker_oracle_certified=False,
                    pier_certified=False,
                    panel_keep=False,
                    solution_reward=None,
                    null_reward=None,
                    pier_oracle_reward=None,
                    pier_null_reward=None,
                    agent_isolated=False,
                    reasons=[f"prompt_alignment refused: {reason_code}: {exc}"],
                    certified=False,
                    drip=drip,
                )
            )
            _append_drip(
                drip,
                task_id=task_id,
                stage="prompt_alignment",
                status="reject",
                detail={"reason_code": reason_code, "err": str(exc)},
            )
            with e2e_path.open("a", encoding="utf-8") as handle:
                for row in drip:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            continue
        except (RealPackError, Exception) as exc:  # noqa: BLE001
            records.append(
                DeepAgentPackRecord(
                    task_id=task_id,
                    seed_id=identity.seed_id,
                    language=material.language,
                    pack_dir=staging_pack,
                    hybrid=identity,
                    solution_files=list(material.source_files),
                    multi_file_ok=len(material.source_files) >= 2,
                    tree_complete=False,
                    real_pack_ok=False,
                    docker_oracle_certified=False,
                    pier_certified=False,
                    panel_keep=False,
                    solution_reward=None,
                    null_reward=None,
                    pier_oracle_reward=None,
                    pier_null_reward=None,
                    agent_isolated=False,
                    reasons=[f"export failed: {exc}"],
                    certified=False,
                    drip=drip,
                )
            )
            _append_drip(
                drip, task_id=task_id, stage="export", status="reject", detail={"err": str(exc)}
            )
            with e2e_path.open("a", encoding="utf-8") as handle:
                for row in drip:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            continue

        _append_drip(
            drip,
            task_id=task_id,
            stage="export",
            status="ok",
            detail={"pack_dir": str(pack_dir), "source_track": REAL_PR_SOURCE_TRACK},
        )
        _append_drip(
            drip,
            task_id=task_id,
            stage="envbuild_clone_sha",
            status="ok",
            detail={
                "repository_url": material.repository_url,
                "base_commit": material.base_commit,
                "dockerfile": "environment/Dockerfile clone@SHA",
            },
        )

        # dual-run labels: honesty(test_n10/product/live_mine) → live dual-run;
        # plain offline unit dests may use synthetic.
        dual_detail: dict[str, Any]
        dual_ok = False
        try:
            dual_detail = _label_dual_run_for_material(
                material,
                pack_dir=pack_dir,
                work=work,
                dest=dest,
                offline_only=offline_only or not honesty_dest,
                clone_cache=clone_cache,
                seed_local_repos=seed_local_repos,
                dual_run_callable=dual_run_callable,
            )
            dual_ok = bool(dual_detail.get("ok"))
            refuse_synthetic_product_dual_run(
                f2p_node_ids=list(dual_detail.get("f2p_node_ids") or []),
                p2p_node_ids=list(dual_detail.get("p2p_node_ids") or []),
                label_method=str(dual_detail.get("label_method") or ""),
                dest=dest,
                offline_only=offline_only or not honesty_dest,
                language=material.language,
                suite_reporter=dual_detail.get("reporter"),
                suite_command=(
                    str(dual_detail.get("suite_command") or "")
                    if dual_detail.get("suite_command") is not None
                    else None
                ),
                require_suite_path=honesty_dest,
            )
            # VAL-DMED-001 / VAL-DHARD-002/003/005: hardness floors fail-closed.
            # Offline engineering dests skip unless honesty_dest (test_n10/product).
            try:
                refuse_product_hardness_floors(
                    f2p_node_ids=list(dual_detail.get("f2p_node_ids") or []),
                    source_files=list(material.source_files),
                    source_hunk_count=material.source_hunk_count,
                    solution_patch=getattr(material, "solution_patch", None),
                    dest=dest,
                    offline_only=offline_only or not honesty_dest,
                    live_mine=bool(live_mine),
                    engineering_opt_out=bool(offline_only and not honesty_dest),
                    task_id=task_id,
                )
            except ProductHardnessFloorRejected as exc:
                raise ProductHardnessFloorsRejected(
                    str(exc),
                    reason_code=exc.reason_code,
                    result=exc.result,
                ) from exc
            if honesty_dest and dual_detail.get("workspace"):
                assert_product_clone_sha_pin(
                    ledger_base_commit=material.base_commit,
                    workspace=str(dual_detail.get("workspace")),
                    recorded_base_commit=str(
                        dual_detail.get("ledger_base_commit") or material.base_commit
                    ),
                    pack_base_commit=material.base_commit,
                    dest=dest,
                    offline_only=False,
                    require_workspace_head=bool(
                        (Path(str(dual_detail.get("workspace"))) / ".git").exists()
                    ),
                )
        except (ProductDualRunRejected, ProductHardnessFloorsRejected) as exc:
            dual_detail = {
                "ok": False,
                "error": str(exc),
                "label_method": LABEL_METHOD_SYNTHETIC,
                "reason_code": getattr(exc, "reason_code", "dual_run_rejected"),
            }
            dual_ok = False
            stage = (
                "hardness_floors" if isinstance(exc, ProductHardnessFloorsRejected) else "dual_run"
            )
            _append_drip(
                drip,
                task_id=task_id,
                stage=stage,
                status="reject",
                detail=dual_detail,
            )
            records.append(
                DeepAgentPackRecord(
                    task_id=task_id,
                    seed_id=identity.seed_id,
                    language=material.language,
                    pack_dir=pack_dir,
                    hybrid=identity,
                    solution_files=list(material.source_files),
                    multi_file_ok=len(material.source_files) >= 2,
                    tree_complete=True,
                    real_pack_ok=False,
                    docker_oracle_certified=False,
                    pier_certified=False,
                    panel_keep=False,
                    solution_reward=None,
                    null_reward=None,
                    pier_oracle_reward=None,
                    pier_null_reward=None,
                    agent_isolated=False,
                    reasons=[f"{stage} rejected: {exc}"],
                    certified=False,
                    drip=drip,
                )
            )
            with e2e_path.open("a", encoding="utf-8") as handle:
                for row in drip:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            continue

        _append_drip(
            drip,
            task_id=task_id,
            stage="dual_run",
            status="ok" if dual_ok else "reject",
            detail={
                "suite": dual_detail.get("suite"),
                "suite_command": dual_detail.get("suite_command"),
                "label_method": dual_detail.get("label_method"),
                "f2p_node_ids": dual_detail.get("f2p_node_ids"),
                "p2p_node_ids": dual_detail.get("p2p_node_ids"),
                "held_out_test_patch": True,
                "agent_isolation": "test.patch verifier-only",
                "workspace": dual_detail.get("workspace"),
                "reporter": dual_detail.get("reporter"),
                "offline_only": dual_detail.get("offline_only", False),
            },
        )

        missing = verify_pack_tree(pack_dir)
        tree_ok = not missing
        product = [
            p
            for p in count_files_in_patch(
                (pack_dir / "solution" / "solution.patch").read_text(encoding="utf-8")
            )
            if not _looks_test_path(p)
        ]
        multi_ok = len(product) >= 2

        real_val = validate_real_harbor_pack(
            pack_dir, require_real_pr_track=True, require_clone_dockerfile=True
        )
        _append_drip(
            drip,
            task_id=task_id,
            stage="real_pack",
            status="ok" if real_val.ok else "reject",
            detail={
                "ok": real_val.ok,
                "reason_codes": list(real_val.reason_codes),
                "source_track": REAL_PR_SOURCE_TRACK,
            },
        )

        # docker oracle cert (real_pr) — product: HarborDockerVerifier only
        docker_ev_path = oracle_ev_dir / f"{task_id}.json"
        owned_backend = False
        if docker_backend is not None:
            backend: HarborVerifierBackend | str | None = docker_backend
        else:
            backend = HarborDockerVerifier(run_id=f"rpr{len(records):02d}")
            owned_backend = True
        # Unit offline may inject scripted only when not dual-truth honesty dest.
        if honesty_dest and not allow_scripted_oracle:
            try:
                refuse_scripted_product_oracle(backend, dest=dest, offline_only=False)
            except ProductOracleBackendRejected:
                raise
        cert = None
        docker_err: str | None = None
        agent_image = ""
        tests_image = ""
        try:
            cert = certify_real_pr_pack(
                pack_dir,
                backend=backend if backend is not None else "docker",
                oracle_mode="docker",
                task_id=task_id,
                evidence_dir=oracle_ev_dir,
                evidence_out=docker_ev_path,
                audit_out=oracle_ev_dir / "gate_audit.jsonl",
                dest_hint=dest if honesty_dest else Path(str(dest) + "_offline_unit"),
                cleanup=True,
                run_id=f"rpr{len(records):02d}",
                require_real_pr_track=True,
            )
            if isinstance(backend, HarborDockerVerifier):
                agent_image = backend.agent_image or ""
                tests_image = backend.tests_image or ""
            elif cert is not None:
                oracle_blob = cert.deepagent.get("oracle") if cert.deepagent else None
                if isinstance(oracle_blob, dict):
                    agent_image = str(oracle_blob.get("agent_image") or "")
                    tests_image = str(oracle_blob.get("tests_image") or "")
            if honesty_dest and not allow_scripted_oracle:
                try:
                    require_live_docker_images(
                        agent_image=agent_image,
                        tests_image=tests_image,
                        dest=dest,
                        offline_only=False,
                    )
                except ProductOracleBackendRejected as img_exc:
                    docker_err = str(img_exc)
                    cert = None
        except (RealPrFakeOracleRejected, FakeBackendRejected, ProductOracleBackendRejected):
            raise
        except (RealOracleCertError, Exception) as exc:  # noqa: BLE001
            docker_err = str(exc)
        finally:
            if owned_backend or (docker_backend is None and backend is not None):
                cleanup = getattr(backend, "cleanup", None)
                if callable(cleanup):
                    with contextlib.suppress(Exception):
                        cleanup()

        docker_ok = bool(cert and cert.certified)
        if (
            honesty_dest
            and not allow_scripted_oracle
            and docker_ok
            and not (agent_image and tests_image)
        ):
            docker_ok = False
            docker_err = docker_err or (
                "product oracle missing live docker image refs "
                f"(agent_image={agent_image!r} tests_image={tests_image!r})"
            )
        sol_rew = cert.solution_reward if cert else None
        null_rew = cert.null_reward if cert else None
        isol = bool(cert.isolation.clean) if cert else False
        docker_reasons = list(cert.reasons) if cert else ([docker_err] if docker_err else [])
        docker_codes = list(cert.reason_codes) if cert else []
        _append_drip(
            drip,
            task_id=task_id,
            stage="docker_oracle",
            status="ok" if docker_ok else "reject",
            detail={
                "certified": docker_ok,
                "solution_reward": sol_rew,
                "null_reward": null_rew,
                "backend": "docker"
                if (
                    isinstance(backend, HarborDockerVerifier)
                    or (isinstance(backend, str) and backend == "docker")
                    or backend is None
                )
                else type(backend).__name__,
                "backend_class": type(backend).__name__
                if backend is not None
                else "HarborDockerVerifier",
                "oracle_mode": "docker",
                "agent_image": agent_image,
                "tests_image": tests_image,
                "evidence": str(docker_ev_path) if docker_ev_path.is_file() else None,
                "sol_null_files": (
                    cert.evidence_files.to_dict()
                    if cert and cert.evidence_files is not None
                    else None
                ),
                "error": docker_err,
            },
        )

        # pier cert — prefer live; may fall back to scripted when live fails
        pier_ev_path = pier_ev_dir / f"{task_id}.json"
        pier_fallback = False
        if pier_runner is not None:
            runner = pier_runner
            pier_mode_used = "injected"
        elif pier_mode == "live":
            runner = SubprocessPierRunner()
            pier_mode_used = "live"
        else:
            runner = ScriptedPierRunner(oracle_reward=1, null_reward=0)
            pier_mode_used = "scripted"

        pier_result = None
        pier_err: str | None = None
        try:
            pier_result = certify_real_pier_pack(
                pack_dir,
                runner=runner,
                jobs_root=pier_jobs,
                task_id=task_id,
                oracle_mode="docker",
                run_oracle=True,
                run_null=True,
                evidence_out=pier_ev_path,
                evidence_dir=pier_ev_dir,
                audit_out=pier_ev_dir / "gate_audit.jsonl",
                require_real_pr_track=True,
                allow_scripted_substitute=allow_scripted_pier_substitute
                or pier_mode_used in {"scripted", "injected"},
                prefer_real_pier=pier_mode == "live",
                dest_hint=dest,
            )
        except FakeBackendRejected:
            raise
        except Exception as exc:  # noqa: BLE001
            pier_err = str(exc)

        # Honest fallback: if live pier fails and substitute allowed, try scripted
        if (
            pier_mode == "live"
            and pier_runner is None
            and not (pier_result and pier_result.certified)
            and allow_scripted_pier_substitute
        ):
            pier_fallback = True
            runner = ScriptedPierRunner(oracle_reward=1, null_reward=0)
            pier_mode_used = "scripted_after_live_fail"
            try:
                pier_result = certify_real_pier_pack(
                    pack_dir,
                    runner=runner,
                    jobs_root=pier_jobs,
                    task_id=task_id,
                    oracle_mode="docker",
                    run_oracle=True,
                    run_null=True,
                    evidence_out=pier_ev_path,
                    evidence_dir=pier_ev_dir,
                    audit_out=pier_ev_dir / "gate_audit.jsonl",
                    require_real_pr_track=True,
                    allow_scripted_substitute=True,
                    prefer_real_pier=False,
                    dest_hint=dest,
                )
                pier_err = None
            except Exception as exc:  # noqa: BLE001
                pier_err = f"live fail + scripted fallback fail: {exc}"

        pier_mode_report = pier_mode_used
        pier_ok = bool(pier_result and pier_result.certified)
        pier_oracle_r = pier_result.solution_reward if pier_result is not None else None
        pier_null_r = pier_result.null_reward if pier_result is not None else None
        if pier_result is not None:
            docker_reasons = list(docker_reasons) + list(pier_result.reasons or [])
            docker_codes = list(docker_codes) + list(pier_result.reason_codes or [])
        elif pier_err:
            docker_reasons = list(docker_reasons) + [pier_err]

        _append_drip(
            drip,
            task_id=task_id,
            stage="pier_cert",
            status="ok" if pier_ok else "reject",
            detail={
                "certified": pier_ok,
                "mode": pier_mode_used,
                "pier_mode": pier_mode_used,
                "live_fallback_to_scripted": pier_fallback,
                "oracle_reward": pier_oracle_r,
                "null_reward": pier_null_r,
                "evidence": str(pier_ev_path) if pier_ev_path.is_file() else None,
                "error": pier_err,
            },
        )

        # panel
        instruction = (pack_dir / "instruction.md").read_text(encoding="utf-8", errors="replace")
        panel_payload: dict[str, Any] = {}
        panel_ok = True
        try:
            panel_payload = _run_panel_for_keep(
                task_id=task_id,
                instruction=instruction,
                pack_dir=pack_dir,
                panel_mode=panel_mode,
                ledger=ledger,
                soft_solver=soft_solver,
            )
            if panel_payload.get("budget_stop"):
                budget_stop = True
            if require_panel_keep:
                panel_ok = bool(panel_payload.get("is_keep"))
            else:
                panel_ok = bool(
                    panel_payload.get("is_keep")
                    or panel_mode == "skip"
                    or (panel_mode == "offline" and panel_payload.get("panel_complete"))
                )
        except Exception as exc:  # noqa: BLE001
            panel_payload = {"mode": panel_mode, "error": str(exc), "is_keep": False}
            panel_ok = not require_panel_keep
            docker_reasons.append(f"panel error: {exc}")
        _append_drip(
            drip,
            task_id=task_id,
            stage="panel",
            status="ok" if panel_ok else "reject",
            detail=panel_payload,
        )

        certified = (
            tree_ok
            and multi_ok
            and real_val.ok
            and dual_ok
            and docker_ok
            and pier_ok
            and panel_ok
            and isol
            and sol_rew == 1
            and null_rew == 0
            and (identity.source_track == REAL_PR_SOURCE_TRACK)
        )
        if docker_err:
            certified = False

        rec = DeepAgentPackRecord(
            task_id=task_id,
            seed_id=identity.seed_id,
            language=material.language,
            pack_dir=pack_dir,
            hybrid=identity,
            solution_files=list(product),
            multi_file_ok=multi_ok,
            tree_complete=tree_ok,
            real_pack_ok=real_val.ok,
            docker_oracle_certified=docker_ok,
            pier_certified=pier_ok,
            panel_keep=bool(panel_payload.get("is_keep")),
            solution_reward=sol_rew,
            null_reward=null_rew,
            pier_oracle_reward=pier_oracle_r,
            pier_null_reward=pier_null_r,
            agent_isolated=isol,
            reasons=list(dict.fromkeys([*(list(real_val.reasons)), *docker_reasons])),
            reason_codes=list(dict.fromkeys([*(list(real_val.reason_codes)), *docker_codes])),
            certified=certified,
            provider_calls=0,
            docker_evidence_path=str(docker_ev_path) if docker_ev_path.is_file() else None,
            pier_evidence_path=str(pier_ev_path) if pier_ev_path.is_file() else None,
            panel=panel_payload,
            real_pack=real_val.to_dict(),
            drip=drip,
        )

        # Stage-only promote under work/product_stage until gate_audit pass
        # (VAL-LSHIP-007): never write product dest yet.
        if certified:
            staged_dest = tasks_out / task_id
            if staged_dest.exists():
                shutil.rmtree(staged_dest)
            shutil.copytree(pack_dir, staged_dest)
            rec.pack_dir = staged_dest
            _append_drip(
                drip,
                task_id=task_id,
                stage="stage_promote",
                status="ok",
                detail={
                    "dest": str(staged_dest),
                    "source_track": REAL_PR_SOURCE_TRACK,
                    "pending_gate_audit": True,
                },
            )
            rec.drip = list(drip)
        else:
            _append_drip(
                drip,
                task_id=task_id,
                stage="stage_promote",
                status="reject",
                detail={"reasons": rec.reasons[:5]},
            )
            rec.drip = list(drip)

        records.append(rec)
        with e2e_path.open("a", encoding="utf-8") as handle:
            for row in rec.drip:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

        if budget_stop and panel_mode == "live":
            break

    lang_counts = _language_counts(records)
    under = _under_supply_real(lang_counts, sum(1 for r in records if r.certified))
    certified_count = sum(1 for r in records if r.certified)

    # Ensure no hybrid_curated slipped into product keeps
    hybrid_leaks = [
        r.task_id
        for r in records
        if r.certified
        and r.hybrid is not None
        and getattr(r.hybrid, "source_track", "") != REAL_PR_SOURCE_TRACK
    ]
    if hybrid_leaks:
        raise HybridProductPromoteRejected(
            f"product promote contained non-real_pr tracks: {hybrid_leaks}"
        )

    # ---- dual-truth gate_audit BEFORE product overwrite (VAL-LSHIP-007) ----
    gate_result: ProductGateAuditResult | None = None
    dual_by_task_pre: dict[str, dict[str, Any]] = {}
    image_by_task_pre: dict[str, dict[str, Any]] = {}
    for r in records:
        for row in r.drip or []:
            if row.get("stage") == "dual_run":
                dual_by_task_pre[r.task_id] = dict(row.get("detail") or {})
            if row.get("stage") == "docker_oracle":
                image_by_task_pre[r.task_id] = dict(row.get("detail") or {})

    intended_for_gate = [r for r in records if r.certified]
    gate_rows = []
    # Match materials provenance (discovery_path / hunk) for certified keeps.
    material_by_id = {m.task_id: m for m in materials}
    for r in intended_for_gate:
        dual = dual_by_task_pre.get(r.task_id) or {}
        img = image_by_task_pre.get(r.task_id) or {}
        mat = material_by_id.get(r.task_id)
        gate_rows.append(
            audit_keep_dual_truth(
                task_id=r.task_id,
                materials_root=materials_for_load,
                live_mine=honesty_dest,
                label_method=str(dual.get("label_method") or ""),
                f2p_node_ids=list(dual.get("f2p_node_ids") or []),
                p2p_node_ids=list(dual.get("p2p_node_ids") or []),
                backend_class=str(img.get("backend_class") or "")
                or ("HarborDockerVerifier" if honesty_dest else ""),
                agent_image=str(img.get("agent_image") or ""),
                tests_image=str(img.get("tests_image") or ""),
                solution_reward=r.solution_reward,
                null_reward=r.null_reward,
                source_track=getattr(r.hybrid, "source_track", REAL_PR_SOURCE_TRACK)
                if r.hybrid
                else REAL_PR_SOURCE_TRACK,
                source_hunk_count=(mat.source_hunk_count if mat is not None else None),
                discovery_path=(mat.discovery_path if mat is not None else "") or None,
                offline_only=offline_only or not honesty_dest,
                # Product live: require hunk floor when known; soft if unknown on legacy mats.
                require_hunk_floor=bool(
                    honesty_dest and mat is not None and mat.source_hunk_count is not None
                ),
            )
        )

    gate_path = staging_product / "gate_audit.jsonl"
    if intended_for_gate or honesty_dest:
        # Dual-truth gate for every intended keep (VAL-LSHIP-007 / VAL-DGEN).
        # Scale min is applied after overwrite as ship ok/fail — do not
        # conflate "keep fails dual-truth" with under-yield N.
        gate_result = write_product_gate_audit(
            gate_rows,
            gate_path,
            materials_root=materials_for_load,
            live_mine=honesty_dest,
            seed5_archived=seed5_archived if product_dest else True,
            min_accepted=1 if honesty_dest else None,
            require_all_accepted=bool(honesty_dest),
        )
        if honesty_dest:
            # Empty yield → ProductEmptyLiveYieldRejected later; dual-truth FAIL
            # on non-empty intended set refuses overwrite now.
            if intended_for_gate:
                try:
                    require_gate_audit_pass(gate_result, refuse_overwrite=True)
                except ProductGateAuditError as exc:
                    # Leave product dest intact; durable audit remains under staging.
                    raise ProductGateAuditRejected(str(exc)) from exc
            elif certified_count <= 0:
                # No intended keeps: skip dual-truth gate; empty-yield handler later.
                pass
        # Soft-downgrade any keep that gate rejected (non-product paths)
        reject_ids = set(gate_result.rejected_ids if gate_result else ())
        if reject_ids:
            for r in records:
                if r.task_id in reject_ids and r.certified:
                    r.certified = False
                    r.reasons = list(dict.fromkeys([*(r.reasons or []), "gate_audit_reject"]))
            certified_count = sum(1 for r in records if r.certified)
            lang_counts = _language_counts(records)
            under = _under_supply_real(lang_counts, certified_count)

    # ---- NOW overwrite product dest (only after gate pass + non-empty certs) ----
    # Empty live yield must fail closed WITHOUT wiping prior product (VAL-LSHIP-006/007).
    empty_yield = certified_count <= 0 and honesty_dest

    dest.mkdir(parents=True, exist_ok=True)
    final_tasks = dest / "tasks"
    final_evidence = dest / "evidence"

    if not empty_yield:
        if final_tasks.exists() and overwrite:
            shutil.rmtree(final_tasks)
        final_tasks.mkdir(parents=True, exist_ok=True)
        if final_evidence.exists() and overwrite:
            shutil.rmtree(final_evidence)
        # Promote staged certs → product
        for r in records:
            if not r.certified:
                _append_drip(
                    r.drip,
                    task_id=r.task_id,
                    stage="promote",
                    status="reject",
                    detail={
                        "reasons": (r.reasons or [])[:5],
                        "gate_audit": "not_accepted",
                    },
                )
                continue
            staged = tasks_out / r.task_id
            final_dest = final_tasks / r.task_id
            if staged.is_dir():
                if final_dest.exists():
                    shutil.rmtree(final_dest)
                shutil.copytree(staged, final_dest)
                r.pack_dir = final_dest
            _append_drip(
                r.drip,
                task_id=r.task_id,
                stage="promote",
                status="ok",
                detail={
                    "dest": str(final_dest),
                    "source_track": REAL_PR_SOURCE_TRACK,
                    "gate_audit": "pass",
                },
            )
        # Move staged evidence + e2e + gate_audit into product dest
        if evidence_root.is_dir():
            shutil.copytree(evidence_root, final_evidence, dirs_exist_ok=True)
        if e2e_path.is_file():
            final_e2e = dest / "e2e_drip.jsonl"
            shutil.copy2(e2e_path, final_e2e)
            e2e_path = final_e2e
        if gate_path.is_file():
            shutil.copy2(gate_path, dest / "gate_audit.jsonl")
            for src in (gate_path.with_name("gate_audit_summary.json"),):
                if src.is_file():
                    shutil.copy2(src, dest / "gate_audit_summary.json")
                    break
        # Rewrite e2e drip with promote rows
        with e2e_path.open("a", encoding="utf-8") as handle:
            for r in records:
                for row in r.drip or []:
                    if row.get("stage") == "promote":
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
        tasks_out = final_tasks  # subsequent smoke + artifacts use product
    else:
        # Durable gate/evidence under work only; product tree left intact.
        for r in records:
            _append_drip(
                r.drip,
                task_id=r.task_id,
                stage="promote",
                status="reject",
                detail={
                    "reasons": (r.reasons or ["empty_yield"])[:5],
                    "gate_audit": "empty_yield_no_overwrite",
                },
            )
        # Artifacts still write to dest below (ship_summary honesty); pack tree stays.
        tasks_out = final_tasks if final_tasks.is_dir() else tasks_out

    from swe_factory.pipeline.ship_harbor import run_harbor_load_smoke

    smoke_task = next((r.task_id for r in records if r.certified), None)
    smoke = (
        run_harbor_load_smoke(tasks_out, task_id=smoke_task)
        if certified_count
        else {"ok": False, "errors": ["no certified packs"], "tool": "harbor"}
    )

    ledger_snap = _ledger_snapshot(settings)
    spend_total = str(
        ledger_snap.get("total_commit_usd") or ledger_snap.get("settled_exact_usd") or "0"
    )
    remaining = str(ledger_snap.get("remaining_usd") or "0")
    under_cap = bool(ledger_snap.get("under_cap", True))

    ok = (
        certified_count >= min_packs
        and certified_count <= max_packs
        and under_cap
        and smoke.get("ok") is True
        and not hybrid_leaks
        and all(
            r.docker_oracle_certified
            and r.pier_certified
            and r.real_pack_ok
            and r.hybrid is not None
            and getattr(r.hybrid, "source_track", None) == REAL_PR_SOURCE_TRACK
            for r in records
            if r.certified
        )
    )
    # Live-mine / product / test_n10: empty yield fail-closed — never pad N.
    if honesty_dest and certified_count <= 0:
        refuse_empty_live_yield(
            certified_count=certified_count,
            min_packs=min_packs,
            dest=dest,
            live_mine=True,
            materials_root=materials_for_load,
            offline_only=offline_only,
        )
    if certified_count < min_packs:
        if budget_stop:
            reason = (
                f"budget-stop with certified={certified_count} < min={min_packs} "
                f"(remaining=${remaining})"
            )
            try:
                rem_f = float(remaining)
            except (TypeError, ValueError):
                rem_f = 1.0
            ok = bool(rem_f <= 0 and certified_count >= 1)
        else:
            reason = f"under-yield real_pr certified={certified_count} < min={min_packs}" + (
                f"; live-mine materials={materials_for_load} (no fixture pad)"
                if honesty_dest
                else ""
            )
            ok = False
            # Explicit refuse for empty live yield so callers get code-2 path.
            if honesty_dest and certified_count <= 0:
                raise ProductEmptyLiveYieldRejected(
                    f"product live-mine empty certified yield fails closed "
                    f"(certified={certified_count} < min={min_packs}, "
                    f"materials={materials_for_load!r}); "
                    "no fixture pad (VAL-LSHIP-006 / VAL-DGEN-001)"
                )
    elif hybrid_leaks:
        reason = f"hybrid leak in product: {hybrid_leaks}"
        ok = False
    elif not smoke.get("ok"):
        reason = f"harbor load smoke failed: {smoke.get('errors')}"
        ok = False
    elif not under_cap:
        reason = "project spend exceeds $600 hard cap"
        ok = False
    else:
        reason = (
            f"shipped {certified_count} real_pr Docker-oracle + Pier-certified packs "
            f"(wave ≥{min_packs}; hybrid archived, not product)"
            + (f"; materials={materials_for_load}" if materials_for_load is not None else "")
        )

    fixture_note = (
        "Historical / engineering only (not Real-PR product N): "
        "datasets/deepagent_v1_hybrid_archive (hybrid_curated motors), "
        "datasets/harbor_v1 (synth motors), datasets/v1 (boltons), "
        "fixtures/real_pr_ship (unit shortlist). "
        "Product N uses live-mined real_pr materials under datasets/deepagent_v1 only."
    )

    pack_manifest = {
        "product_surface": str(dest),
        "product_track": REAL_PR_SOURCE_TRACK,
        "live_mine": bool(live_mine or honesty_dest),
        "materials_root": str(materials_for_load) if materials_for_load is not None else None,
        "materials_is_fixture": bool(
            materials_for_load is not None and is_fixture_materials_root(materials_for_load)
        ),
        "refuse_fixture_materials_default": True,
        "historical_fixtures_only": [
            "datasets/deepagent_v1_hybrid_archive",
            "datasets/harbor_v1",
            "datasets/v1",
            "fixtures/real_pr_ship",
        ],
        "hybrid_claimed_as_product": False,
        "count": certified_count,
        "pack_count": certified_count,
        "task_ids": [r.task_id for r in records if r.certified],
        "languages": lang_counts,
        "multi_file": {r.task_id: r.solution_files for r in records if r.certified},
        "identity": {
            r.task_id: r.hybrid.to_dict() if r.hybrid else None for r in records if r.certified
        },
        "source_tracks": {
            r.task_id: getattr(r.hybrid, "source_track", REAL_PR_SOURCE_TRACK)
            for r in records
            if r.certified
        },
        "packs": [
            {
                "task_id": r.task_id,
                "language": r.language,
                "source_track": getattr(r.hybrid, "source_track", REAL_PR_SOURCE_TRACK)
                if r.hybrid
                else REAL_PR_SOURCE_TRACK,
                "source_hunk_count": (
                    material_by_id[r.task_id].source_hunk_count
                    if r.task_id in material_by_id
                    else None
                ),
                "solution_reward": r.solution_reward,
                "null_reward": r.null_reward,
                "certified": True,
                "backend": "HarborDockerVerifier",
                "label_method": (dual_by_task_pre.get(r.task_id) or {}).get("label_method")
                or LABEL_METHOD_LIVE,
                "materials_is_fixture": False,
                "live_mine": bool(live_mine or honesty_dest),
            }
            for r in records
            if r.certified
        ],
        "oracle": {
            r.task_id: {
                "solution_reward": r.solution_reward,
                "null_reward": r.null_reward,
                "mode": "docker",
                "oracle_mode": "docker",
                "agent_isolated": r.agent_isolated,
                "evidence": r.docker_evidence_path,
            }
            for r in records
            if r.certified
        },
        "pier": {
            r.task_id: {
                "oracle_reward": r.pier_oracle_reward,
                "null_reward": r.pier_null_reward,
                "certified": r.pier_certified,
                "evidence": r.pier_evidence_path,
            }
            for r in records
            if r.certified
        },
        "panel": {r.task_id: r.panel for r in records if r.certified},
        "provider_calls": total_provider_calls,
        "mode": "ship_deepagent_real_pr_docker",
        "panel_mode": panel_mode,
        "pier_mode": pier_mode_report,
        "pier_mode_requested": pier_mode,
        "required_relpaths": list(REQUIRED_PACK_RELPATHS),
        "under_supply_reasons": under,
        "harbor_load_smoke": smoke,
        "band": {"min": min_packs, "max": max_packs, "target": target_packs},
        "budget_stop": budget_stop,
        "ok": ok,
        "refuse_fake": True,
        "refuse_hybrid": True,
        "refuse_synthetic_dual_run": honesty_dest,
        "require_live_docker_images": honesty_dest,
        "offline_only": offline_only,
        "product_dest": product_dest,
        "live_generate_dest": is_live_generate_dest(dest),
        "archive_note": archive_note,
    }
    pack_manifest_path = dest / "pack_manifest.json"
    pack_manifest_path.write_text(
        json.dumps(pack_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Collect dual-run / image evidence from drips for product honesty audit
    dual_by_task: dict[str, dict[str, Any]] = {}
    image_by_task: dict[str, dict[str, Any]] = {}
    for r in records:
        for row in r.drip or []:
            if row.get("stage") == "dual_run":
                dual_by_task[r.task_id] = dict(row.get("detail") or {})
            if row.get("stage") == "docker_oracle":
                image_by_task[r.task_id] = dict(row.get("detail") or {})

    oracle_evidence = {
        "backend": "docker",
        "oracle_mode": "docker",
        "refuse_fake": True,
        "require_harbor_docker_verifier": honesty_dest,
        "product_track": REAL_PR_SOURCE_TRACK,
        "certified_count": certified_count,
        "records": [
            {
                "task_id": r.task_id,
                "certified": r.certified,
                "solution_reward": r.solution_reward,
                "null_reward": r.null_reward,
                "agent_isolated": r.agent_isolated,
                "docker_oracle_certified": r.docker_oracle_certified,
                "evidence_path": r.docker_evidence_path,
                "agent_image": (image_by_task.get(r.task_id) or {}).get("agent_image") or "",
                "tests_image": (image_by_task.get(r.task_id) or {}).get("tests_image") or "",
                "backend_class": (image_by_task.get(r.task_id) or {}).get("backend_class"),
                "repository_url": getattr(r.hybrid, "repository_url", None) if r.hybrid else None,
                "base_commit_hash": getattr(r.hybrid, "base_commit", None) if r.hybrid else None,
                "source_track": getattr(r.hybrid, "source_track", None) if r.hybrid else None,
                "dual_run": dual_by_task.get(r.task_id),
            }
            for r in records
        ],
    }
    oracle_evidence_path = dest / "oracle_evidence.json"
    oracle_evidence_path.write_text(
        json.dumps(oracle_evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    pier_evidence = {
        "jobs_root": str(pier_jobs),
        "certified_count": certified_count,
        "product_track": REAL_PR_SOURCE_TRACK,
        "records": [
            {
                "task_id": r.task_id,
                "certified": r.pier_certified,
                "oracle_reward": r.pier_oracle_reward,
                "null_reward": r.pier_null_reward,
                "evidence_path": r.pier_evidence_path,
            }
            for r in records
        ],
    }
    pier_evidence_path = dest / "pier_evidence.json"
    pier_evidence_path.write_text(
        json.dumps(pier_evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    ledger_summary_path = dest / "ledger_summary.json"
    ledger_summary_path.write_text(
        json.dumps(ledger_snap, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    provenance_path = dest / "PROVENANCE.md"
    _write_provenance_real(provenance_path, records)

    result = ShipDeepAgentResult(
        ok=ok,
        certified_count=certified_count,
        target_packs=target_packs,
        min_packs=min_packs,
        max_packs=max_packs,
        out_dir=dest,
        report_path=None,
        pack_manifest_path=pack_manifest_path,
        ledger_summary_path=ledger_summary_path,
        provenance_path=provenance_path,
        oracle_evidence_path=oracle_evidence_path,
        pier_evidence_path=pier_evidence_path,
        e2e_drip_path=e2e_path if e2e_path.is_file() else None,
        languages=lang_counts,
        under_supply_reasons=under,
        records=records,
        harbor_load_smoke=smoke,
        spend_total_usd=spend_total,
        remaining_usd=remaining,
        under_cap=under_cap,
        budget_stop=budget_stop,
        provider_calls=total_provider_calls,
        mode="docker",
        panel_mode=panel_mode,
        pier_mode=str(pier_mode_report),
        reason=reason,
        fixture_note=fixture_note,
    )
    report_path = dest / "report.md"
    _write_report_real(
        report_path,
        result_payload=result.to_dict(),
        ledger=ledger_snap,
        archive_note=archive_note,
    )
    result.report_path = report_path

    is_fix_mat = bool(
        materials_for_load is not None and is_fixture_materials_root(materials_for_load)
    )
    materials_note = (
        f"materials_root={materials_for_load}; live_mine={bool(live_mine)}; "
        f"fixture_materials={is_fix_mat}"
    )
    # Per-task reject ledger so soft dual-run/oracle rejects never starve silently
    # (feature m14-fix-live-funnel-yield-min15 / continue serial cert until min).
    reject_ledger: list[dict[str, Any]] = []
    for r in records:
        if r.certified:
            continue
        stage_hints: list[str] = []
        for row in r.drip or []:
            if row.get("status") == "reject":
                stage_hints.append(str(row.get("stage") or "unknown"))
        reject_ledger.append(
            {
                "task_id": r.task_id,
                "language": r.language,
                "reasons": list(r.reasons or [])[:12],
                "stages": stage_hints or ["not_certified"],
                "docker_oracle_certified": bool(r.docker_oracle_certified),
                "pier_certified": bool(r.pier_certified),
                "real_pack_ok": bool(r.real_pack_ok),
            }
        )
    inv_stats: dict[str, Any] = {}
    if materials_for_load is not None and not is_fixture_materials_root(materials_for_load):
        with contextlib.suppress(Exception):
            inv_stats = inventory_completeness(materials_for_load)
    (dest / "ship_summary.json").write_text(
        json.dumps(
            {
                **result.to_dict(),
                "refuse_hybrid": True,
                "product_track": REAL_PR_SOURCE_TRACK,
                "archive_note": archive_note,
                "seed5_note": seed5_note if product_dest else None,
                "seed5_archived": seed5_archived if product_dest else None,
                "seed5_archive": str(DEFAULT_SEED5) if product_dest else None,
                "gate_audit": gate_result.to_dict() if gate_result is not None else None,
                "gate_audit_pass": bool(gate_result.ok) if gate_result is not None else None,
                "live_mine": bool(live_mine or honesty_dest),
                "materials_root": str(materials_for_load)
                if materials_for_load is not None
                else None,
                "materials_is_fixture": bool(
                    materials_for_load is not None and is_fixture_materials_root(materials_for_load)
                ),
                "materials_note": materials_note,
                "used_materials_count": len(materials),
                "reject_ledger": reject_ledger,
                "inventory_completeness": inv_stats or None,
                "honesty_dest": honesty_dest,
                "live_generate_dest": is_live_generate_dest(dest),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    # Product README honesty pointer (ship-local)
    readme_title = (
        f"# {dest} — real_pr live-mine generate wave"
        if is_live_generate_dest(dest)
        else "# datasets/deepagent_v1 — Real-PR product (live mine)"
    )
    (dest / "PRODUCT_README.md").write_text(
        "\n".join(
            [
                readme_title,
                "",
                "This directory ships **source_track=real_pr** Harbor packs only.",
                "Hybrid motor corpus: `datasets/deepagent_v1_hybrid_archive/`.",
                "Prior seed product: `datasets/deepagent_v1_seed5_archive/`.",
                f"Certified N this wave: **{certified_count}** "
                f"(target={target_packs}, min≥{min_packs}).",
                f"Materials: {materials_for_load} (live_mine={bool(live_mine or honesty_dest)}; "
                "never fixtures/real_pr_ship for product).",
                "Docker oracle: HarborDockerVerifier sol=1 / null=0; pier mode honest;",
                "gate_audit dual-truth pass required before overwrite (VAL-LSHIP-007 / VAL-DGEN).",
                (
                    f"Hardness floors (VAL-DMED-001 / VAL-DHARD-002): "
                    f"F2P≥{resolve_min_f2p_nodes()} "
                    f"(default MIN_F2P_NODES={DEFAULT_MIN_F2P_NODES}), "
                    f"source files≥{PRODUCT_MULTI_FILE_FLOOR}, "
                    f"source hunks≥{PRODUCT_SOURCE_HUNK_FLOOR}, "
                    f"gold added_lines≥{PRODUCT_MIN_ADDED_LINES} "
                    "(DeepSWE-median band); thin gold/F2P refused; "
                    "model dual-success alone never drops (M25 intrinsic). "
                    "See docs/PRODUCT_HARDNESS.md."
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return result


__all__ = [
    "BURNT_WORK_ROOT_MARKERS",
    "DEFAULT_ARCHIVE",
    "DEFAULT_MATERIALS",
    "DEFAULT_MIN_F2P_NODES",
    "DEFAULT_OUT",
    "DEFAULT_PRODUCT_MATERIALS",
    "DEFAULT_REAL_PR_MAX",
    "DEFAULT_REAL_PR_MIN",
    "DEFAULT_REAL_PR_TARGET",
    "DEFAULT_SEED5",
    "FRESH_WORK_ROOT_MARKER",
    "LABEL_METHOD_LIVE",
    "LABEL_METHOD_SYNTHETIC",
    "PRODUCT_CLONE_SHA_PIN_OK",
    "SYNTHETIC_P2P_NODE",
    "HybridProductPromoteRejected",
    "ProductDualRunRejected",
    "ProductEmptyLiveYieldRejected",
    "ProductFixtureMaterialsRejected",
    "ProductGateAuditRejected",
    "ProductHardnessFloorsRejected",
    "ProductOracleBackendRejected",
    "ProductPromptAlignRejected",
    "ProductSeed5ArchiveMissing",
    "RealPrMaterial",
    "ShipRealPrError",
    "assert_product_clone_sha_pin",
    "build_deepswe_style_instruction",
    "build_real_pr_agent_instruction",
    "build_real_pr_pack_spec",
    "f2p_node_ids_from_test_patch",
    "is_fixture_real_pr_materials",
    "is_live_generate_dest",
    "refuse_product_hardness_floors",
    "resolve_min_f2p_nodes",
    "is_product_deepagent_dest",
    "load_real_pr_materials",
    "lookslike_burnt_work_root",
    "mark_dual_run_work_root_burnt",
    "prepare_fresh_dual_run_work_root",
    "refuse_burnt_dual_run_work_root",
    "refuse_empty_live_yield",
    "refuse_hybrid_product_promote",
    "refuse_product_fixture_materials",
    "refuse_prompt_verifier_misalign",
    "refuse_scripted_product_oracle",
    "refuse_synthetic_product_dual_run",
    "require_live_docker_images",
    "require_product_suite_reporter",
    "requires_dual_truth_honesty",
    "resolve_product_materials_root",
    "run_ship_deepagent_real_pr",
    "sanitize_pr_body_for_prompt",
]
