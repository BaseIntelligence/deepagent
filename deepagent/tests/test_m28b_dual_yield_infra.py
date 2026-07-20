"""m28b dual-yield infra: no SUT-shadow host deps, nodeid strip, collection F2P expand."""

from __future__ import annotations

from swe_factory.pipeline.ship_real_pr import (
    HOST_SUITE_COMMON_DEPS,
    host_suite_pip_env,
    sut_shadow_dists_for_task,
)
from swe_factory.producers.harbor_labeling import SuiteOutcome
from swe_factory.producers.real_dual_run import (
    expand_collection_errors_to_f2p_nodes,
    labels_from_real_suite_outcomes,
)


def test_host_suite_common_deps_exclude_sut_shadows() -> None:
    tokens = {d.split("==")[0].split(">=")[0].split("[")[0].lower() for d in HOST_SUITE_COMMON_DEPS}
    for ban in (
        "werkzeug",
        "click",
        "jinja2",
        "flask",
        "packaging",
        "attrs",
        "httpcore",
        "httpx",
        "oauthlib",
        "wtforms",
        "marshmallow",
        "paramiko",
        "rich",
    ):
        assert ban not in tokens
    assert "pytest-xprocess" in tokens
    assert "simplejson" in tokens
    assert "redis" in tokens
    assert "greenlet" in tokens


def test_host_suite_pip_env_strips_socks_proxy() -> None:
    env = host_suite_pip_env(
        {
            "ALL_PROXY": "socks5://user:pass@host:7777",
            "HTTPS_PROXY": "http://x",
            "OXYLABS_PROXY_URL": "socks5://x",
            "GITHUB_TOKEN": "secret-token",
            "PATH": "/bin",
        }
    )
    assert "ALL_PROXY" not in env
    assert "HTTPS_PROXY" not in env
    assert "OXYLABS_PROXY_URL" not in env
    assert env.get("GITHUB_TOKEN") == "secret-token"
    assert env.get("NO_PROXY") == "*"


def test_sut_shadow_dists_for_task() -> None:
    assert "click" in sut_shadow_dists_for_task("realpr-click-3442")
    assert "oauthlib" in sut_shadow_dists_for_task("realpr-oauthlib-889")
    assert "werkzeug" in sut_shadow_dists_for_task("realpr-werkzeug-3116")


def test_expand_collection_errors_to_green_nodes() -> None:
    broken = {"tests.test_datalist", "tests/test_context.py"}
    green = {
        "tests.test_datalist.TestX.test_a",
        "tests.test_datalist.TestX.test_b",
        "tests.test_context.TestY.test_c",
        "tests.other.test_z",
    }
    exp = expand_collection_errors_to_f2p_nodes(broken, green)
    assert "tests.test_datalist.TestX.test_a" in exp
    assert "tests.test_context.TestY.test_c" in exp
    assert "tests.other.test_z" not in exp


def test_labels_from_collection_errors_produce_f2p() -> None:
    green = SuiteOutcome(
        language="python",
        passed=(
            "tests.mod.test_new_feature",
            "tests.mod.test_other",
            "tests.keep.test_p2p",
        ),
        failed=(),
        errors=(),
        returncode=0,
    )
    broken = SuiteOutcome(
        language="python",
        passed=("tests.keep.test_p2p",),
        failed=(),
        errors=("tests.mod",),
        returncode=2,
    )
    lab = labels_from_real_suite_outcomes(green, broken, require_nonempty_f2p=True, min_f2p_nodes=1)
    assert "tests.mod.test_new_feature" in lab.f2p_node_ids
    assert "tests.mod.test_other" in lab.f2p_node_ids
    assert "tests.keep.test_p2p" in lab.p2p_node_ids
