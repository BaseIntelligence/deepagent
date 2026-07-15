"""Offline Harbor pack fixture (no LLM, no Docker required for tree emit).

VAL-CROSS-006: Offline fixture emits one complete Harbor pack tree.
Local null/solution oracle is implemented in
:mod:`swe_factory.harbor.harbor_oracle` (fake + Docker). This module emits
the structural pack tree offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swe_factory.fixture.offline import default_fixture_root
from swe_factory.harbor.export_pack import (
    HarborExportError,
    HarborPackResult,
    export_harbor_pack,
    verify_pack_tree,
)
from swe_factory.harbor.grader_frame import (
    default_tests_dockerfile,
    offline_environment_dockerfile,
)
from swe_factory.harbor.schema import (
    GradeConfig,
    HarborEnvironment,
    HarborMetadata,
    HarborPackSpec,
    HarborTaskIdentity,
    HarborTaskToml,
    HarborVerifier,
    TestsConfig,
    validate_pack_spec,
)

HARBOR_FIXTURE_TASK_ID = "fixture-tiny-harbor-gate-demo"
FIXTURE_BASE_COMMIT = "fixture00000000000000000000000000000001"
FIXTURE_REPO_URL = "file://fixtures/tiny_offline"


@dataclass(frozen=True, slots=True)
class OfflineHarborResult:
    """Result of offline Harbor pack emit."""

    task_id: str
    pack_dir: Path
    out_dir: Path
    missing: tuple[str, ...]
    provider_calls: int = 0
    pack: HarborPackResult | None = None


def _load_gold(fixture_root: Path) -> str:
    gold = fixture_root / "gold.patch"
    if not gold.is_file():
        raise HarborExportError(f"fixture gold.patch missing: {gold}")
    return gold.read_text(encoding="utf-8")


def _load_meta(fixture_root: Path) -> dict[str, Any]:
    meta_path = fixture_root / "task_meta.json"
    if not meta_path.is_file():
        raise HarborExportError(f"fixture meta missing: {meta_path}")
    raw: object = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise HarborExportError("fixture meta must be object")
    return dict(raw)


def build_fixture_test_patch() -> str:
    """Held-out tests that current fixture F2P nodes exercise.

    For the offline structural fixture the tests already live on the broken
    tree; we still ship a non-empty test.patch that is a no-op header plus
    a marker so config + tree checks stay DeepSWE-complete. Real producers
    later replace this with held-out suite additions.
    """
    return """\
diff --git a/tests/.harbor_held_out_marker b/tests/.harbor_held_out_marker
new file mode 100644
index 0000000..e69de29
--- /dev/null
+++ b/tests/.harbor_held_out_marker
@@ -0,0 +1 @@
+held-out verifier path active
"""


def build_offline_harbor_spec(
    *,
    fixture_root: Path | None = None,
) -> HarborPackSpec:
    """Construct a validated HarborPackSpec from fixtures/tiny_offline."""
    root = fixture_root or default_fixture_root()
    if not root.is_dir():
        raise HarborExportError(f"fixture root not found: {root}")
    meta = _load_meta(root)
    gold = _load_gold(root)
    language = str(meta.get("language") or "python")
    base_commit = str(meta.get("base_commit") or FIXTURE_BASE_COMMIT)
    problem = str(meta.get("problem_statement") or "Restore multi-module behaviour so tests pass.")
    f2p_cmds = meta.get("fail_to_pass") or []
    # Node ids aligned with fixture test functions (DeepSWE style)
    f2p_nodes = [
        "tests.test_math.test_add",
        "tests.test_text.test_reverse_words",
    ]
    p2p_nodes = ["tests.test_ok.test_always_ok"]
    del f2p_cmds

    task_toml = HarborTaskToml(
        schema_version="1.1",
        artifacts=["/logs/artifacts/model.patch"],
        task=HarborTaskIdentity(
            name=f"swe-factory/{HARBOR_FIXTURE_TASK_ID}",
            description="Offline Harbor pack fixture (structural + schema)",
            authors=[],
            keywords=["fixture", "offline", "python"],
        ),
        metadata=HarborMetadata(
            language=language,
            repository_url=FIXTURE_REPO_URL,
            base_commit_hash=base_commit,
            task_id=HARBOR_FIXTURE_TASK_ID,
            ext_id="sdf-offline-harbor-fixture-v1",
            display_title="Tiny offline Harbor gate demo",
            display_description=problem[:200],
            original_title="Fixture multi-file restore",
            category="fixture",
            source_track=str(meta.get("source_track") or "synthetic_grounded"),
            license=str(meta.get("license") or "MIT"),
        ),
        verifier=HarborVerifier(environment_mode="separate", timeout_sec=600.0),
        environment=HarborEnvironment(
            docker_image="harbor-sdf-fixture-agent:offline",
            cpus=1,
            memory_mb=2048,
            storage_mb=4096,
            allow_internet=False,
        ),
    )
    tests_config = TestsConfig(
        base_commit=base_commit,
        f2p_node_ids=f2p_nodes,
        p2p_node_ids=p2p_nodes,
        grade=GradeConfig(
            format="junit",
            node_id="name",
            tool_label="pytest",
            reports=["/logs/verifier/new.xml", "/logs/verifier/base.xml"],
        ),
    )
    instruction = (
        problem.rstrip()
        + "\n\n"
        + "IMPORTANT: Please work on this in a new branch from main and commit "
        + "everything when you are done.\n"
    )
    spec = HarborPackSpec(
        task_id=HARBOR_FIXTURE_TASK_ID,
        instruction_md=instruction,
        task_toml=task_toml,
        tests_config=tests_config,
        solution_patch=gold,
        test_patch=build_fixture_test_patch(),
        environment_dockerfile=offline_environment_dockerfile(),
        tests_dockerfile=default_tests_dockerfile(
            agent_image_ref="harbor-sdf-fixture-agent:offline"
        ),
    )
    return validate_pack_spec(spec)


def run_offline_harbor_fixture(
    *,
    out_dir: Path | str | None = None,
    fixture_root: Path | None = None,
) -> OfflineHarborResult:
    """Emit one complete Harbor pack tree offline (provider_calls=0)."""
    root = fixture_root or default_fixture_root()
    dest_root = Path(out_dir) if out_dir is not None else Path("datasets/harbor_fixture")
    dest_root.mkdir(parents=True, exist_ok=True)
    spec = build_offline_harbor_spec(fixture_root=root)
    pack = export_harbor_pack(
        spec,
        dest=dest_root / "tasks" / spec.task_id,
        overwrite=True,
        copy_repo_into_environment=root / "repo",
    )
    missing = verify_pack_tree(pack.pack_dir)
    # Write a small manifest for inspectability
    (dest_root / "pack_manifest.json").write_text(
        json.dumps(
            {
                "task_id": pack.task_id,
                "pack_dir": str(pack.pack_dir),
                "missing": missing,
                "provider_calls": 0,
                "mode": "offline_harbor_fixture",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return OfflineHarborResult(
        task_id=pack.task_id,
        pack_dir=pack.pack_dir,
        out_dir=dest_root,
        missing=tuple(missing),
        provider_calls=0,
        pack=pack,
    )


__all__ = [
    "FIXTURE_BASE_COMMIT",
    "FIXTURE_REPO_URL",
    "HARBOR_FIXTURE_TASK_ID",
    "OfflineHarborResult",
    "build_fixture_test_patch",
    "build_offline_harbor_spec",
    "run_offline_harbor_fixture",
]
