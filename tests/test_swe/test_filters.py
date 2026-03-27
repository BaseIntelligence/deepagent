"""Tests for PR filtering logic."""


from swe_forge.swe.enricher import EnrichedPullRequest
from swe_forge.swe.filters import (
    FilterConfig,
    apply_filters,
    filter_bot,
    filter_org,
    filter_stars,
    filter_language,
    filter_file_count,
    filter_prs,
)


def make_pr(
    *,
    repo: str = "testorg/testrepo",
    number: int = 1,
    user: str = "human",
    is_bot: bool = False,
    stars: int = 100,
    language: str = "python",
    files_changed: int = 5,
) -> EnrichedPullRequest:
    """Create a test PR with sensible defaults."""
    return EnrichedPullRequest(
        id="test-id",
        repo=repo,
        number=number,
        user=user,
        is_bot=is_bot,
        stars=stars,
        language=language,
        files_changed=files_changed,
    )


class TestFilterConfig:
    def test_default_config(self):
        config = FilterConfig()
        assert config.exclude_bots is True
        assert config.allowed_orgs is None
        assert config.min_stars == 0
        assert config.allowed_languages == ["python"]
        assert config.max_files_changed == 50

    def test_custom_config(self):
        config = FilterConfig(
            exclude_bots=False,
            allowed_orgs=["django", "flask"],
            min_stars=100,
            allowed_languages=["python", "rust"],
            max_files_changed=10,
        )
        assert config.exclude_bots is False
        assert config.allowed_orgs == ["django", "flask"]
        assert config.min_stars == 100
        assert config.allowed_languages == ["python", "rust"]
        assert config.max_files_changed == 10


class TestFilterBot:
    def test_passes_when_bot_filter_disabled(self):
        pr = make_pr(is_bot=True)
        assert filter_bot(pr, exclude_bots=False) is True

    def test_passes_human_author(self):
        pr = make_pr(is_bot=False)
        assert filter_bot(pr, exclude_bots=True) is True

    def test_rejects_bot_author(self):
        pr = make_pr(is_bot=True)
        assert filter_bot(pr, exclude_bots=True) is False


class TestFilterOrg:
    def test_passes_when_no_org_filter(self):
        pr = make_pr(repo="anyorg/repo")
        assert filter_org(pr, allowed_orgs=None) is True

    def test_passes_allowed_org(self):
        pr = make_pr(repo="django/django")
        assert filter_org(pr, allowed_orgs=["django", "flask"]) is True

    def test_rejects_wrong_org(self):
        pr = make_pr(repo="unknown/repo")
        assert filter_org(pr, allowed_orgs=["django", "flask"]) is False


class TestFilterStars:
    def test_passes_when_no_star_minimum(self):
        pr = make_pr(stars=0)
        assert filter_stars(pr, min_stars=0) is True

    def test_passes_enough_stars(self):
        pr = make_pr(stars=100)
        assert filter_stars(pr, min_stars=50) is True

    def test_rejects_insufficient_stars(self):
        pr = make_pr(stars=10)
        assert filter_stars(pr, min_stars=50) is False


class TestFilterLanguage:
    def test_passes_allowed_language(self):
        pr = make_pr(language="python")
        assert filter_language(pr, allowed_languages=["python"]) is True

    def test_passes_unknown_language(self):
        pr = make_pr(language="unknown")
        assert filter_language(pr, allowed_languages=["python"]) is True

    def test_passes_empty_language(self):
        pr = make_pr(language="")
        assert filter_language(pr, allowed_languages=["python"]) is True

    def test_passes_null_language(self):
        pr = make_pr(language="null")
        assert filter_language(pr, allowed_languages=["python"]) is True

    def test_rejects_disallowed_language(self):
        pr = make_pr(language="javascript")
        assert filter_language(pr, allowed_languages=["python"]) is False

    def test_case_insensitive_match(self):
        pr = make_pr(language="Python")
        assert filter_language(pr, allowed_languages=["python"]) is True

    def test_empty_allowed_languages_passes_all(self):
        pr = make_pr(language="javascript")
        assert filter_language(pr, allowed_languages=[]) is True


class TestFilterFileCount:
    def test_passes_within_limit(self):
        pr = make_pr(files_changed=10)
        assert filter_file_count(pr, max_files=50) is True

    def test_passes_at_limit(self):
        pr = make_pr(files_changed=50)
        assert filter_file_count(pr, max_files=50) is True

    def test_rejects_over_limit(self):
        pr = make_pr(files_changed=60)
        assert filter_file_count(pr, max_files=50) is False


class TestApplyFilters:
    def test_passes_all_filters(self):
        pr = make_pr(
            is_bot=False,
            repo="django/django",
            stars=1000,
            language="python",
            files_changed=10,
        )
        config = FilterConfig(
            exclude_bots=True,
            allowed_orgs=["django"],
            min_stars=100,
            allowed_languages=["python"],
            max_files_changed=50,
        )
        assert apply_filters(pr, config) is True

    def test_rejected_by_bot_filter(self):
        pr = make_pr(is_bot=True)
        config = FilterConfig()
        assert apply_filters(pr, config) is False

    def test_rejected_by_org_filter(self):
        pr = make_pr(repo="unknown/repo")
        config = FilterConfig(allowed_orgs=["django"])
        assert apply_filters(pr, config) is False

    def test_rejected_by_stars_filter(self):
        pr = make_pr(stars=5)
        config = FilterConfig(min_stars=100)
        assert apply_filters(pr, config) is False

    def test_rejected_by_language_filter(self):
        pr = make_pr(language="go")
        config = FilterConfig(allowed_languages=["python"])
        assert apply_filters(pr, config) is False

    def test_rejected_by_file_count_filter(self):
        pr = make_pr(files_changed=100)
        config = FilterConfig(max_files_changed=50)
        assert apply_filters(pr, config) is False

    def test_all_filters_disabled_passes(self):
        pr = make_pr(
            is_bot=True,
            repo="anyorg/repo",
            stars=0,
            language="cobol",
            files_changed=1000,
        )
        config = FilterConfig(
            exclude_bots=False,
            allowed_orgs=None,
            min_stars=0,
            allowed_languages=[],
            max_files_changed=10000,
        )
        assert apply_filters(pr, config) is True


class TestFilterPrs:
    def test_filters_batch(self):
        prs = [
            make_pr(number=1, is_bot=False, stars=1000),
            make_pr(number=2, is_bot=True),
            make_pr(number=3, is_bot=False, stars=5),
        ]
        config = FilterConfig(exclude_bots=True, min_stars=100)
        passed = filter_prs(prs, config)
        assert len(passed) == 1
        assert passed[0].number == 1

    def test_empty_batch(self):
        assert filter_prs([], FilterConfig()) == []


class TestFilterResultLogging:
    def test_logs_rejection_reason(self, caplog):
        import logging

        caplog.set_level(logging.DEBUG)

        pr = make_pr(is_bot=True)
        config = FilterConfig(exclude_bots=True)
        result = apply_filters(pr, config)

        assert result is False
        assert "bot" in caplog.text.lower()
