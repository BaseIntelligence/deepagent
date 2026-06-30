"""Stage 0: contamination-resistant seed-repo registry with per-repo caps.

The registry is a small curated set of real, permissively-licensed, test-bearing
repositories (>=1 per supported language) each pinned to a concrete commit SHA
with its commit date, so the env-build checks out exactly that commit and never a
moving branch tip. Every entry carries a per-repo instance cap; the registry
hands out task instances through :meth:`SourceRegistry.acquire`, which enforces
the cap and tracks usage so a single repo can never seed more than its configured
number of tasks (guarding against repo-memorization / contamination).

The pinned commits below are real commit SHAs resolved from each repository's
upstream history; the env-build fetches and checks out exactly these SHAs.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from swe_forge.forge.models import InstanceGrant, RepoSpec


class SourceError(RuntimeError):
    """Base error for the source registry."""


class UnknownRepoError(SourceError):
    """Raised when a repo id is not present in the registry."""


class SourceRegistry:
    """An ordered, name-unique collection of :class:`RepoSpec` seed entries."""

    def __init__(self, specs: Iterable[RepoSpec]) -> None:
        self._specs: dict[str, RepoSpec] = {}
        for spec in specs:
            if spec.repo_id in self._specs:
                raise SourceError(
                    f"duplicate repo id in source registry: {spec.repo_id!r}"
                )
            self._specs[spec.repo_id] = spec

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Iterator[RepoSpec]:
        return iter(tuple(self._specs.values()))

    def repo_ids(self) -> tuple[str, ...]:
        """Return the registered repo ids in registration order."""
        return tuple(self._specs)

    def specs(self) -> tuple[RepoSpec, ...]:
        """Return the registered :class:`RepoSpec` entries in registration order."""
        return tuple(self._specs.values())

    def languages(self) -> tuple[str, ...]:
        """Return the distinct languages covered by the registry (sorted)."""
        return tuple(sorted({spec.language for spec in self._specs.values()}))

    def by_language(self, language: str) -> list[RepoSpec]:
        """Return every entry whose language matches ``language``."""
        target = language.strip().lower()
        return [spec for spec in self._specs.values() if spec.language == target]

    def has_language(self, language: str) -> bool:
        """Return ``True`` iff at least one entry covers ``language``."""
        return bool(self.by_language(language))

    def get(self, repo_id: str) -> RepoSpec:
        """Return the entry registered under ``repo_id`` or raise."""
        try:
            return self._specs[repo_id]
        except KeyError:
            raise UnknownRepoError(
                f"no repo {repo_id!r} in source registry; "
                f"known: {', '.join(self._specs) or '(none)'}"
            ) from None

    def acquire(self, repo_id: str) -> InstanceGrant:
        """Request one task instance from ``repo_id`` (enforces its cap)."""
        return self.get(repo_id).acquire()

    def reset_usage(self) -> None:
        """Reset usage counters for every entry."""
        for spec in self._specs.values():
            spec.reset_usage()

    def to_list(self) -> list[dict[str, object]]:
        """Return a JSON-serializable list of every entry's metadata."""
        return [spec.to_dict() for spec in self._specs.values()]


def _curated_specs() -> list[RepoSpec]:
    """Build fresh :class:`RepoSpec` instances for the curated seed set.

    Returns new objects on every call so each registry owns independent usage
    counters (acquiring instances from one registry never affects another).

    Three groups make up the pilot Stage-0 supply (see
    ``library/pilot-sources.md``):

    * the three original tiny single-module seeds (kept for the m2 contract);
    * the 15 git-verified **``pr_mirror`` allowlist** entries -- each pinned to
      its own merged-PR ``base_commit`` and carrying the upstream slug +
      ``pr_number`` the ``pr_mirror`` generator reintroduces the isolated fault
      from (an F2P with P2P green BY CONSTRUCTION, which defeats the
      collateral-damage rejection that starved the first pilot run);
    * the diversified **MODULAR** source repos (cachetools/boltons; validator.js/
      qs; cast/mux) where the structural generators can isolate a fault.

    Per-repo install/baseline/P2P-exclusion overrides live on the RepoSpec (never
    in the language-agnostic stage code); ``envbuild`` honors them.
    """
    return [
        *_original_seeds(),
        *_pr_mirror_allowlist(),
        *_modular_sources(),
    ]


# Reusable per-repo install overrides (the bare adapter defaults miss test
# extras / non-default runners; see library/pilot-sources.md per-repo env notes).
_PIP_PYTEST: tuple[str, ...] = ("pip install -e .", "pip install pytest")
_PIP_PYTEST_CRYPTO: tuple[str, ...] = (
    "pip install -e '.[crypto-eth-addresses]'",
    "pip install pytest",
)
_PIP_PYTEST_TESTS: tuple[str, ...] = ("pip install -e '.[tests]'", "pip install pytest")
_NPM_LEGACY: tuple[str, ...] = ("npm install --no-audit --no-fund --legacy-peer-deps",)
# validator.js hardcodes a version constant that lags package.json at non-release
# commits; that self-test is fix-independent and excluded from P2P there.
_VERSION_SELFTEST: tuple[str, ...] = ("should export the version number",)


def _original_seeds() -> list[RepoSpec]:
    """The three tiny single-module seeds (kept for the m2 source contract)."""
    return [
        RepoSpec(
            repo_id="pytest-dev/iniconfig",
            url="https://github.com/pytest-dev/iniconfig.git",
            commit="77db208ab4ae0cd2061d909fe222a1db72867850",
            commit_date="2026-02-25T11:10:21Z",
            language="python",
            license="MIT",
            instance_cap=5,
            default_branch="main",
            description="Brain-dead simple parsing of ini files (pure-Python, pytest).",
        ),
        RepoSpec(
            repo_id="sindresorhus/yocto-queue",
            url="https://github.com/sindresorhus/yocto-queue.git",
            commit="b07eac099753833b29d06c614149904445739776",
            commit_date="2025-11-11T06:30:03Z",
            language="javascript",
            license="MIT",
            instance_cap=4,
            default_branch="main",
            description="Tiny queue data structure (ESM JS/TS, zero runtime deps).",
        ),
        RepoSpec(
            repo_id="golang-jwt/jwt",
            url="https://github.com/golang-jwt/jwt.git",
            commit="e8e5b83ca9a5c5a3f287eda52c7bca78f9a6d176",
            commit_date="2026-05-26T21:30:25Z",
            language="go",
            license="MIT",
            instance_cap=6,
            default_branch="main",
            description="Go implementation of JSON Web Tokens (dependency-free).",
        ),
    ]


def _pr(
    slug: str,
    commit: str,
    commit_date: str,
    pr_number: int,
    language: str,
    license: str,
    *,
    install: tuple[str, ...] = (),
    test: str = "",
    exclude: tuple[str, ...] = (),
    structural: bool = False,
    cap: int = 2,
    description: str = "",
) -> RepoSpec:
    """Build one ``pr_mirror`` allowlist :class:`RepoSpec` (unique id ``slug#pr``)."""
    return RepoSpec(
        repo_id=f"{slug}#{pr_number}",
        url=f"https://github.com/{slug}.git",
        commit=commit,
        commit_date=commit_date,
        language=language,
        license=license,
        instance_cap=cap,
        description=description,
        baseline_install=install,
        baseline_test=test,
        p2p_exclusions=exclude,
        pr_repo=slug,
        pr_number=pr_number,
        pr_generator="pr_mirror",
        structural_source=structural,
    )


def _pr_mirror_allowlist() -> list[RepoSpec]:
    """The 15 git-verified merged-PR entries (>=1 isolated F2P each, P2P green)."""
    return [
        # -- Python (validators + humanize) -------------------------------- #
        _pr(
            "python-validators/validators",
            "25fcef05c293def421fd3f6714403a54fce993cf",
            "2025-03-28T20:52:14Z",
            411,
            "python",
            "MIT",
            install=_PIP_PYTEST_CRYPTO,
            exclude=("test_returns_failed_validation_on_invalid_email",),
            description="Revert email-validation fix (2 invalid-email F2P).",
        ),
        _pr(
            "python-validators/validators",
            "15984e89a96ee8e8ceb9c2148cac6896a85cd71e",
            "2023-10-05T01:07:51Z",
            305,
            "python",
            "MIT",
            install=_PIP_PYTEST,
            exclude=("test_returns_true_on_valid_url",),
            description="Revert URL fragment fix (matrix.to F2P; 0.22-era, no extra).",
        ),
        _pr(
            "python-validators/validators",
            "f83ee1b226d1e8f15d5afaf1724324fd4770beae",
            "2023-09-25T03:58:30Z",
            301,
            "python",
            "MIT",
            install=_PIP_PYTEST,
            exclude=("test_returns_true_on_valid_range",),
            description="Revert between() boundary fix (2 F2P; 0.22-era, no extra).",
        ),
        _pr(
            "python-validators/validators",
            "bcb134204edcfdabf0de0e5fc0a5010105659c36",
            "2024-05-17T02:15:01Z",
            374,
            "python",
            "MIT",
            install=_PIP_PYTEST_CRYPTO,
            exclude=("test_returns_true_on_valid_public_ipv4_address",),
            description="Revert public-IPv4 fix (3 F2P; merge net source diff).",
        ),
        _pr(
            "python-humanize/humanize",
            "0a06a3d4a12113cd5f3d0df0cfbb3e27d92499eb",
            "2026-05-22T05:37:13Z",
            318,
            "python",
            "MIT",
            install=_PIP_PYTEST_TESTS,
            description="Revert empty-list fix in lists.py (IndexError F2P).",
        ),
        # -- JavaScript (validator.js + qs) -------------------------------- #
        _pr(
            "validatorjs/validator.js",
            "a38f15be6931722d295d2c7ad1d13d35036078a5",
            "2026-06-29T14:48:15Z",
            2787,
            "javascript",
            "MIT",
            install=_NPM_LEGACY,
            test="npm test",
            exclude=("should validate ISO 8601 dates",),
            structural=True,
            description="Revert isISO8601 fix (2 F2P; GOLD fully green - strongest).",
        ),
        _pr(
            "validatorjs/validator.js",
            "941db7fac5263cc7e0df0eba37253678f92989b0",
            "2026-04-01T18:10:34Z",
            2693,
            "javascript",
            "MIT",
            install=_NPM_LEGACY,
            test="npm test",
            exclude=_VERSION_SELFTEST,
            description="Revert isSlug fix (1 F2P; exclude version self-test).",
        ),
        _pr(
            "validatorjs/validator.js",
            "cf401458b8733d981a3724d634c795a9d612b516",
            "2025-11-04T09:44:37Z",
            2622,
            "javascript",
            "MIT",
            install=_NPM_LEGACY,
            test="npm test",
            exclude=_VERSION_SELFTEST,
            description="Revert isURL fix (1 F2P; exclude version self-test).",
        ),
        _pr(
            "ljharb/qs",
            "a0a81ea2071acce3eff41a040f719ac8f5c4f64c",
            "2026-04-24T06:17:57Z",
            555,
            "javascript",
            "BSD-3-Clause",
            install=_NPM_LEGACY,
            test="npm run tests-only",
            structural=True,
            description="Revert stringify delimiter fix (1 F2P; GOLD 920/920).",
        ),
        _pr(
            "ljharb/qs",
            "59da434d5de8c3d2564e4d75aeedde2e8af72369",
            "2026-06-07T12:04:18Z",
            559,
            "javascript",
            "BSD-3-Clause",
            install=_NPM_LEGACY,
            test="npm run tests-only",
            description="Revert surrogate-pair fix in utils.js (5 F2P; GOLD 937/937).",
        ),
        # -- Go (jwt + go-humanize + lo + uuid) ---------------------------- #
        _pr(
            "golang-jwt/jwt",
            "e8e5b83ca9a5c5a3f287eda52c7bca78f9a6d176",
            "2026-05-26T21:30:25Z",
            509,
            "go",
            "MIT",
            exclude=("TestMapClaims_GetExpirationTime_ZeroIsExpired",),
            description="Revert MapClaims GetExpirationTime fix (1 F2P).",
        ),
        _pr(
            "golang-jwt/jwt",
            "9a70137d962aa5105727b6f2f31519bfb6ec90b4",
            "2026-05-26T21:29:25Z",
            510,
            "go",
            "MIT",
            exclude=("Test_Validator_verifyExpiresAt",),
            description="Revert validator leeway fix (1 F2P).",
        ),
        _pr(
            "dustin/go-humanize",
            "4d1d9082551ec085912e7d2253a33ae547fca000",
            "2025-11-25T00:15:11Z",
            65,
            "go",
            "MIT",
            exclude=("TestReltimeOffbyone",),
            description="Revert Reltime off-by-one fix in times.go (1 F2P).",
        ),
        _pr(
            "samber/lo",
            "0b4623da1e71d19237c519b79f1852ec7b707961",
            "2026-02-21T18:29:10Z",
            796,
            "go",
            "MIT",
            exclude=("TestEllipsis",),
            description="Revert Ellipsis fix in string.go (1 F2P).",
        ),
        _pr(
            "google/uuid",
            "a2b2b32373ff0b1a312b7fdf6d38a977099698a6",
            "2024-01-11T18:16:31Z",
            150,
            "go",
            "BSD-3-Clause",
            exclude=("TestVersion7Monotonicity",),
            description="Revert v7 monotonicity fix in version7.go (1 F2P).",
        ),
    ]


def _modular(
    slug: str,
    commit: str,
    commit_date: str,
    language: str,
    license: str,
    *,
    install: tuple[str, ...] = (),
    test: str = "",
    cap: int = 6,
    description: str = "",
) -> RepoSpec:
    """Build one diversified MODULAR structural-generator :class:`RepoSpec`."""
    return RepoSpec(
        repo_id=slug,
        url=f"https://github.com/{slug}.git",
        commit=commit,
        commit_date=commit_date,
        language=language,
        license=license,
        instance_cap=cap,
        description=description,
        baseline_install=install,
        baseline_test=test,
        structural_source=True,
    )


def _modular_sources() -> list[RepoSpec]:
    """Diversified multi-module repos where structural generators isolate a fault."""
    return [
        _modular(
            "tkem/cachetools",
            "48284d73d0a8834c9c50f8d41bb99e6f93b2dfed",
            "2026-05-21T22:34:39Z",
            "python",
            "MIT",
            install=_PIP_PYTEST,
            description="Extensible memoizing collections + decorators (modular).",
        ),
        _modular(
            "mahmoud/boltons",
            "979fa9b613fa8c0a455ae16ea6f2ec91c11ecafe",
            "2026-06-19T06:04:59Z",
            "python",
            "BSD-3-Clause",
            install=_PIP_PYTEST,
            description="Pure-Python utility modules (modular; prefer *utils.py).",
        ),
        _modular(
            "spf13/cast",
            "6fd6afdc9df068c662e923a7c3ad04fda4647121",
            "2026-04-06T18:43:42Z",
            "go",
            "MIT",
            description="Safe Go type conversions (modular structural source).",
        ),
        _modular(
            "gorilla/mux",
            "db9d1d0073d27a0a2d9a8c1bc52aa0af4374d265",
            "2024-06-19T23:50:04Z",
            "go",
            "BSD-3-Clause",
            description="Go HTTP request router/dispatcher (modular structural source).",
        ),
    ]


def build_source_registry() -> SourceRegistry:
    """Return a fresh curated source registry (>=1 usable repo per language)."""
    return SourceRegistry(_curated_specs())
