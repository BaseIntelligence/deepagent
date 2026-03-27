"""Tests for patch extraction functionality."""


from swe_forge.swe.extractor import (
    ExtractedPatch,
    count_line_delta,
    extract_patch,
    is_test_file,
    parse_diff_file_name,
    split_solution_and_tests,
)


class TestIsTestFile:
    """Test test file detection patterns."""

    def test_python_test_underscore_prefix(self):
        assert is_test_file("test_example.py") is True
        assert is_test_file("test_foo_bar.py") is True

    def test_python_test_underscore_suffix(self):
        assert is_test_file("example_test.py") is True
        assert is_test_file("foo_bar_test.py") is True

    def test_rust_test_file(self):
        assert is_test_file("module_test.rs") is True

    def test_typescript_test_file(self):
        assert is_test_file("component.test.ts") is True
        assert is_test_file("utils.spec.ts") is True

    def test_javascript_test_file(self):
        assert is_test_file("app.test.js") is True
        assert is_test_file("helpers.spec.js") is True

    def test_pytest_conftest(self):
        assert is_test_file("conftest.py") is True

    def test_pytest_ini(self):
        assert is_test_file("pytest.ini") is True

    def test_jest_config(self):
        assert is_test_file("jest.config.js") is True
        assert is_test_file("jest.config.ts") is True

    def test_vitest_config(self):
        assert is_test_file("vitest.config.ts") is True

    def test_setup_cfg(self):
        assert is_test_file("setup.cfg") is True

    def test_test_directory_patterns(self):
        assert is_test_file("tests/test_example.py") is True
        assert is_test_file("test/test_foo.py") is True
        assert is_test_file("__tests__/component.js") is True
        assert is_test_file("spec/unit_spec.rb") is True

    def test_non_test_files(self):
        assert is_test_file("main.py") is False
        assert is_test_file("utils.py") is False
        assert is_test_file("app.js") is False
        assert is_test_file("src/index.ts") is False
        assert is_test_file("lib/module.rs") is False

    def test_windows_paths(self):
        assert is_test_file("tests\\test_example.py") is True
        assert is_test_file("src\\main.py") is False

    def test_deeply_nested_paths(self):
        assert is_test_file("src/project/tests/unit/test_module.py") is True
        assert is_test_file("src/project/lib/module.py") is False


class TestParseDiffFileName:
    """Test parsing diff header lines."""

    def test_standard_format(self):
        result = parse_diff_file_name("diff --git a/path/to/file.py b/path/to/file.py")
        assert result == "path/to/file.py"

    def test_nested_path(self):
        result = parse_diff_file_name(
            "diff --git a/src/lib/utils.py b/src/lib/utils.py"
        )
        assert result == "src/lib/utils.py"

    def test_root_file(self):
        result = parse_diff_file_name("diff --git a/README.md b/README.md")
        assert result == "README.md"

    def test_rename_format(self):
        result = parse_diff_file_name("diff --git a/old_name.py a/new_name.py")
        assert result == "new_name.py"

    def test_invalid_format(self):
        assert parse_diff_file_name("not a diff line") is None
        assert parse_diff_file_name("") is None
        assert parse_diff_file_name("diff --git") is None

    def test_special_chars_in_path(self):
        result = parse_diff_file_name(
            "diff --git a/src/my-file_v2.py b/src/my-file_v2.py"
        )
        assert result == "src/my-file_v2.py"


class TestSplitSolutionAndTests:
    """Test splitting diffs into solution and test patches."""

    def test_empty_diff(self):
        solution, test = split_solution_and_tests("")
        assert solution == ""
        assert test == ""

    def test_solution_only(self):
        diff = """diff --git a/src/main.py b/src/main.py
new file mode 100644
index 0000000..abc1234
--- /dev/null
+++ b/src/main.py
@@ -0,0 +1,5 @@
+def main():
+    print("Hello")
+
+if __name__ == "__main__":
+    main()
"""
        solution, test = split_solution_and_tests(diff)
        assert "main.py" in solution
        assert test == ""

    def test_test_only(self):
        diff = """diff --git a/tests/test_main.py b/tests/test_main.py
new file mode 100644
index 0000000..abc1234
--- /dev/null
+++ b/tests/test_main.py
@@ -0,0 +1,3 @@
+def test_main():
+    result = main()
+    assert result is not None
"""
        solution, test = split_solution_and_tests(diff)
        assert solution == ""
        assert "test_main.py" in test

    def test_mixed_solution_and_tests(self):
        diff = """diff --git a/src/utils.py b/src/utils.py
new file mode 100644
index 0000000..abc1234
--- /dev/null
+++ b/src/utils.py
@@ -0,0 +1,3 @@
+def add(a, b):
+    return a + b
diff --git a/tests/test_utils.py b/tests/test_utils.py
new file mode 100644
index 0000000..def5678
--- /dev/null
+++ b/tests/test_utils.py
@@ -0,0 +1,5 @@
+from src.utils import add
+
+def test_add():
+    assert add(1, 2) == 3
"""
        solution, test = split_solution_and_tests(diff)
        assert "diff --git a/src/utils.py" in solution
        assert "diff --git a/tests/test_utils.py" in test
        assert "tests/test_utils.py" not in solution
        assert "src/utils.py" not in test

    def test_multiple_solution_files(self):
        diff = """diff --git a/src/a.py b/src/a.py
new file mode 100644
+++ b/src/a.py
+A content
diff --git a/src/b.py b/src/b.py
new file mode 100644
+++ b/src/b.py
+B content
"""
        solution, test = split_solution_and_tests(diff)
        assert "a.py" in solution
        assert "b.py" in solution
        assert test == ""

    def test_multiple_test_files(self):
        diff = """diff --git a/tests/test_a.py b/tests/test_a.py
+++ b/tests/test_a.py
+test A
diff --git a/tests/test_b.py b/tests/test_b.py
+++ b/tests/test_b.py
+test B
"""
        solution, test = split_solution_and_tests(diff)
        assert solution == ""
        assert "test_a.py" in test
        assert "test_b.py" in test


class TestCountLineDelta:
    """Test counting added and removed lines."""

    def test_empty_diff(self):
        added, removed = count_line_delta("")
        assert added == 0
        assert removed == 0

    def test_additions_only(self):
        diff = """diff --git a/file.py b/file.py
+++ b/file.py
+line 1
+line 2
+line 3
"""
        added, removed = count_line_delta(diff)
        assert added == 3
        assert removed == 0

    def test_removals_only(self):
        diff = """diff --git a/file.py b/file.py
--- a/file.py
-line 1
-line 2
"""
        added, removed = count_line_delta(diff)
        assert added == 0
        assert removed == 2

    def test_mixed_additions_and_removals(self):
        diff = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
-old line
+new line
-another old
+another new
"""
        added, removed = count_line_delta(diff)
        assert added == 2
        assert removed == 2

    def test_skips_headers(self):
        diff = """diff --git a/file.py b/file.py
index abc1234..def5678 100644
--- a/file.py
+++ b/file.py
@@ -1,5 +1,5 @@
-old
+new
"""
        added, removed = count_line_delta(diff)
        assert added == 1
        assert removed == 1

    def test_handles_hunk_headers(self):
        diff = """@@ -0,0 +1,3 @@
+line 1
+line 2
+line 3
"""
        added, removed = count_line_delta(diff)
        assert added == 3
        assert removed == 0


class TestExtractPatch:
    """Test full patch extraction."""

    def test_empty_input(self):
        result = extract_patch("")
        assert result.solution_patch == ""
        assert result.test_patch == ""
        assert result.files_changed == 0
        assert result.added_lines == 0
        assert result.removed_lines == 0

    def test_full_extraction(self):
        diff = """diff --git a/src/main.py b/src/main.py
new file mode 100644
--- /dev/null
+++ b/src/main.py
@@ -0,0 +1,3 @@
+def main():
+    pass
+
diff --git a/tests/test_main.py b/tests/test_main.py
new file mode 100644
--- /dev/null
+++ b/tests/test_main.py
@@ -0,0 +1,2 @@
+def test_main():
+    assert True
"""
        result = extract_patch(diff, summary="Add main module with tests")

        assert "main.py" in result.solution_patch
        assert "test_main.py" in result.test_patch
        assert result.files_changed == 2
        assert result.added_lines == 5
        assert result.removed_lines == 0
        assert result.summary == "Add main module with tests"

    def test_solution_only_extraction(self):
        diff = """diff --git a/lib/utils.py b/lib/utils.py
--- /dev/null
+++ b/lib/utils.py
@@ -0,0 +1,1 @@
+def helper(): pass
"""
        result = extract_patch(diff)

        assert result.solution_patch != ""
        assert result.test_patch == ""
        assert result.files_changed == 1

    def test_test_only_extraction(self):
        diff = """diff --git a/test_example.py b/test_example.py
--- /dev/null
+++ b/test_example.py
@@ -0,0 +1,1 @@
+def test_example(): pass
"""
        result = extract_patch(diff)

        assert result.solution_patch == ""
        assert result.test_patch != ""
        assert result.files_changed == 1


class TestExtractedPatchDataclass:
    """Test ExtractedPatch dataclass."""

    def test_dataclass_creation(self):
        patch = ExtractedPatch(
            solution_patch="solution",
            test_patch="test",
            files_changed=2,
            added_lines=10,
            removed_lines=5,
            summary="test",
        )
        assert patch.solution_patch == "solution"
        assert patch.test_patch == "test"
        assert patch.files_changed == 2
        assert patch.added_lines == 10
        assert patch.removed_lines == 5
        assert patch.summary == "test"

    def test_dataclass_immutability(self):
        patch = ExtractedPatch(
            solution_patch="",
            test_patch="",
            files_changed=0,
            added_lines=0,
            removed_lines=0,
            summary="",
        )
        assert patch.solution_patch == ""
