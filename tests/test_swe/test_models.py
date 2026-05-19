import pytest

from swe_forge.swe.models import (
    SweTask,
    SweTaskStatus,
    ValidationOutcome,
    validate_git_ref,
    validate_repo_name,
)


class TestValidateGitRef:
    def test_accepts_hex_sha(self):
        assert validate_git_ref("abc123def456") == "abc123def456"

    def test_accepts_branch_names(self):
        assert validate_git_ref("main") == "main"
        assert validate_git_ref("feature/my-branch") == "feature/my-branch"
        assert validate_git_ref("HEAD~1") == "HEAD~1"
        assert validate_git_ref("v1.2.3") == "v1.2.3"

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            validate_git_ref("")

    def test_rejects_shell_injection(self):
        with pytest.raises(ValueError):
            validate_git_ref("abc123; rm -rf /")
        with pytest.raises(ValueError):
            validate_git_ref("$(whoami)")
        with pytest.raises(ValueError):
            validate_git_ref("`id`")

    def test_rejects_double_dot(self):
        with pytest.raises(ValueError, match="path traversal"):
            validate_git_ref("main..HEAD")

    def test_rejects_leading_dash(self):
        with pytest.raises(ValueError, match="flag"):
            validate_git_ref("-n")

    def test_rejects_too_long(self):
        with pytest.raises(ValueError, match="too long"):
            validate_git_ref("a" * 257)


class TestValidateRepoName:
    def test_accepts_valid(self):
        assert validate_repo_name("owner/repo") == "owner/repo"
        assert validate_repo_name("my-org/my-repo") == "my-org/my-repo"
        assert validate_repo_name("user123/project.js") == "user123/project.js"

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            validate_repo_name("")

    def test_rejects_invalid_format(self):
        with pytest.raises(ValueError, match="owner/repo"):
            validate_repo_name("noslash")
        with pytest.raises(ValueError, match="owner/repo"):
            validate_repo_name("too/many/slashes")

    def test_rejects_leading_dot_or_dash(self):
        with pytest.raises(ValueError):
            validate_repo_name(".hidden/repo")
        with pytest.raises(ValueError):
            validate_repo_name("owner/.repo")
        with pytest.raises(ValueError):
            validate_repo_name("-flag/repo")

    def test_rejects_too_long(self):
        long_name = f"{'a' * 128}/{'b' * 128}"
        with pytest.raises(ValueError, match="too long"):
            validate_repo_name(long_name)


class TestSweTask:
    def test_create_task(self):
        task = SweTask(
            id="test-123",
            repo="owner/repo",
            base_commit="abc123",
            merge_commit="def456",
            language="python",
        )
        assert task.id == "test-123"
        assert task.repo == "owner/repo"
        assert task.status == SweTaskStatus.CANDIDATE
        assert task.has_tests() is False

    def test_validates_repo(self):
        task = SweTask(id="test", repo="owner/repo")
        assert task.repo == "owner/repo"

    def test_rejects_invalid_repo(self):
        with pytest.raises(ValueError):
            SweTask(id="test", repo="invalid")

    def test_validates_git_refs(self):
        task = SweTask(
            id="test",
            repo="owner/repo",
            base_commit="abc123",
            merge_commit="def456",
        )
        assert task.base_commit == "abc123"
        assert task.merge_commit == "def456"

    def test_has_tests_true(self):
        task = SweTask(
            id="test",
            repo="owner/repo",
            fail_to_pass=["pytest"],
        )
        assert task.has_tests() is True

    def test_default_values(self):
        task = SweTask(id="test", repo="owner/repo")
        assert task.language == "unknown"
        assert task.source_type == "github_pr"
        assert task.deletion_patch == ""
        assert task.difficulty_score == 1
        assert task.quality_passed is False
        assert task.docker_passed is False


class TestValidationOutcome:
    def test_passed(self):
        outcome = ValidationOutcome(passed=True)
        assert outcome.passed is True
        assert outcome.reason is None

    def test_rejected(self):
        outcome = ValidationOutcome(passed=False, reason="test failed")
        assert outcome.passed is False
        assert outcome.reason == "test failed"
