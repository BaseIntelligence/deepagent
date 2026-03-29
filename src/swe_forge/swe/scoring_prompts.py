"""Example-based prompts for consistent LLM scoring.

This module contains prompts with concrete examples for:
- Prompt quality assessment
- Difficulty evaluation
- Test coverage scoring

Each prompt includes examples for scores 0.9, 0.7, 0.5, and 0.3
to ensure consistent LLM scoring.
"""

# =============================================================================
# Prompt Quality Assessment Example
# =============================================================================

PROMPT_QUALITY_EXAMPLES = {
    "excellent": {
        "score": 0.95,
        "prompt": """In `src/auth/login.py`, the `validate_token()` function fails 
when token contains Unicode characters. The function should accept UTF-8 encoded tokens.

Expected behavior: `validate_token('用户_token')` should return True.

Current behavior: Raises UnicodeDecodeError.

File: src/auth/login.py, line 45

Acceptance criteria:
- UTF-8 tokens work correctly
- Existing ASCII tokens still work
- Tests pass: pytest tests/test_auth.py -v
""",
        "reasoning": "Clear description, provides file location, expected behavior, "
                     "reproduction code, and acceptance criteria."
    },
    
    "good": {
        "score": 0.75,
        "prompt": """The login function crashes with special characters in tokens.

Fix the token validation in the auth module (src/auth/).

Users report getting 500 errors when logging in with non-ASCII usernames.""",
        "reasoning": "Clear description but lacks reproduction code and specific file location."
    },
    
    "acceptable": {
        "score": 0.55,
        "prompt": """Fix the auth bug in the login system.

See PR description for details.""",
        "reasoning": "Vague - requires developer to investigate what the bug is."
    },
    
    "poor": {
        "score": 0.25,
        "prompt": """Fix authentication.""",
        "reasoning": "No details, no context, cannot determine what needs fixing."
    }
}


# =============================================================================
# Difficulty Assessment Examples
# =============================================================================

DIFFICULTY_EXAMPLES = {
    "easy": {
        "score": 0.15,
        "level": "easy",
        "case": """1 file changed, 5 lines changed
- src/utils.py: Fix typo in function name

Reasoning: Single file, minimal lines, cosmetic change.""",
    },
    
    "easy_medium": {
        "score": 0.35,
        "level": "easy",
        "case": """2 files changed, 12 lines changed
- src/api.py: Add parameter validation
- tests/test_api.py: Add validation tests

Reasoning: Small change, straightforward addition, tests included.""",
    },
    
    "medium": {
        "score": 0.50,
        "level": "medium",
        "case": """3 files changed, 25 lines changed
- src/models/user.py: Add new field
- src/api/users.py: Update endpoint
- tests/test_users.py: Add tests

Reasoning: Multiple files, moderate complexity, cross-component change.""",
    },
    
    "medium_hard": {
        "score": 0.65,
        "level": "medium",
        "case": """4 files changed, 40 lines changed
- src/auth/login.py: Refactor authentication
- src/auth/middleware.py: Update middleware
- src/api/auth.py: Update endpoints
- tests/test_auth.py: Update tests

Reasoning: Cross-module refactoring with multiple components.""",
    },
    
    "hard": {
        "score": 0.80,
        "level": "hard",
        "case": """5 files changed, 75 lines changed
- src/db/schema.py: Add new tables
- src/models/entities.py: New entity classes
- src/api/crud.py: Update CRUD operations
- src/api/routes.py: New routes
- tests/test_api.py: Integration tests

Reasoning: Large change, database schema update, multiple components.""",
    },
    
    "very_hard": {
        "score": 0.95,
        "level": "hard",
        "case": """8 files changed, 150 lines changed
- Breaking API change
- Database migration required
- Multiple modules affected
- Security implications

Reasoning: Very large change, breaking changes, high complexity, security-critical.""",
    }
}


# =============================================================================
# Test Coverage Examples
# =============================================================================

TEST_COVERAGE_EXAMPLES = {
    "excellent": {
        "score": 0.92,
        "tests": """FAIL_TO_PASS:
- pytest tests/test_auth.py::test_unicode_token -v
- pytest tests/test_auth.py::test_invalid_token -v
- pytest tests/test_auth.py::test_expired_token -v

PASS_TO_PASS:
- pytest tests/test_auth.py::test_valid_ascii_token -v
- pytest tests/test_auth.py::test_token_refresh -v""",
        "reasoning": "5 tests covering happy path, edge cases, errors, and regression."
    },
    
    "good": {
        "score": 0.75,
        "tests": """FAIL_TO_PASS:
- pytest tests/test_auth.py::test_unicode_token -v

PASS_TO_PASS:
- pytest tests/test_auth.py::test_valid_token -v""",
        "reasoning": "2 tests covering main functionality, missing edge cases."
    },
    
    "acceptable": {
        "score": 0.55,
        "tests": """FAIL_TO_PASS:
- pytest tests/ -v""",
        "reasoning": "Single test command, generic, may miss edge cases."
    },
    
    "poor": {
        "score": 0.20,
        "tests": """No tests specified.""",
        "reasoning": "Cannot verify the fix without tests."
    }
}


# =============================================================================
# Combined Scoring Examples
# =============================================================================

FULL_TASK_EXAMPLES = [
    {
        "name": "High-quality task",
        "prompt": PROMPT_QUALITY_EXAMPLES["excellent"]["prompt"],
        "lines_changed": 15,
        "files_changed": 2,
        "difficulty_score": 0.45,
        "test_score": 0.85,
        "overall": 0.68,
        "passed": True,
    },
    {
        "name": "Acceptable task",
        "prompt": PROMPT_QUALITY_EXAMPLES["acceptable"]["prompt"],
        "lines_changed": 8,
        "files_changed": 1,
        "difficulty_score": 0.25,
        "test_score": 0.40,
        "overall": 0.42,
        "passed": False,  # Below thresholds
    },
    {
        "name": "Hard complex task",
        "prompt": PROMPT_QUALITY_EXAMPLES["good"]["prompt"],
        "lines_changed": 80,
        "files_changed": 5,
        "difficulty_score": 0.90,
        "test_score": 0.70,
        "overall": 0.75,
        "passed": True,
    },
]


# =============================================================================
# Scoring Thresholds
# =============================================================================

SCORING_THRESHOLDS = {
    "min_prompt_quality": 0.50,
    "min_test_coverage": 0.50,
    "min_overall_score": 0.50,
    
    # Difficulty classification
    "easy_max_lines": 10,
    "medium_max_lines": 30,
    
    # Multi-file complexity
    "multi_file_threshold": 2,
    "multi_file_boost": 0.15,
}
