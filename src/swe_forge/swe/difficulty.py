"""Difficulty classifier for SWE tasks using LLM-based scoring.

Two-phase classification:
1. Triage: Quick assessment using PR title + body
2. Full classification: Detailed scoring based on diff stats and quality metrics
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel

from swe_forge.llm import (
    GenerationRequest,
    GenerationResponse,
    LLMClient,
    Message,
    ToolDefinition,
)


class DifficultyLevel(str, Enum):
    """Difficulty level for SWE tasks."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# Score ranges for each difficulty level
SCORE_RANGES = {
    DifficultyLevel.EASY: (0.1, 0.35),
    DifficultyLevel.MEDIUM: (0.4, 0.65),
    DifficultyLevel.HARD: (0.7, 1.0),
}


@dataclass
class TriageResponse:
    """Response from triage classification."""

    difficulty: str  # "easy", "medium", "hard"
    reasoning: str


@dataclass
class ClassifyResponse:
    """Response from full classification."""

    difficulty: str
    score: float
    quality_good: bool
    reasoning: str


# Tool schemas for function calling
TRIAGE_TOOL_SCHEMA: dict[str, Any] = {
    "name": "triage",
    "parameters": {
        "type": "object",
        "properties": {
            "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
            "reasoning": {"type": "string"},
        },
        "required": ["difficulty", "reasoning"],
    },
}

CLASSIFY_TOOL_SCHEMA: dict[str, Any] = {
    "name": "classify_pr",
    "parameters": {
        "type": "object",
        "properties": {
            "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
            "score": {"type": "number"},
            "quality_good": {"type": "boolean"},
            "reasoning": {"type": "string"},
        },
        "required": ["difficulty", "score", "quality_good", "reasoning"],
    },
}

# System prompts from Rust implementation
TRIAGE_SYSTEM_PROMPT = """You are an expert at triaging software engineering tasks.

Your job is to quickly assess the difficulty of a PR/task based on:
- File paths and their complexity (e.g., test files, core modules, docs)
- Line count indicators in the PR body
- Scope of changes (single file vs multiple files)

Focus on identifying:
- Easy: Documentation, typo fixes, small test additions (< 10 lines)
- Medium: Bug fixes, feature additions, moderate refactoring
- Hard: Architecture changes, complex refactoring, security-sensitive code

Provide your assessment using the triage tool."""

CLASSIFY_SYSTEM_PROMPT = """You are an expert at classifying software engineering task difficulty.

Assess difficulty based on these criteria:

EASY (score 0.1-0.35):
- Single file changes, < 20 lines modified
- Documentation or comment-only changes
- Test additions without code changes
- Configuration tweaks

MEDIUM (score 0.4-0.65):
- 20-100 lines modified across 1-3 files
- Bug fixes requiring investigation
- Feature additions following existing patterns
- Minor refactoring

HARD (score 0.7-1.0):
- 100+ lines modified or 4+ files changed
- Architecture or component refactoring
- Security-sensitive code changes
- Breaking changes or API modifications
- Complex algorithms or logic changes

Also assess quality_good based on:
- Clear description of changes
- Tests included or unnecessary
- Follows contribution guidelines
- No obvious issues in the patch

Provide your full classification using the classify_pr tool."""


class PRInfo(BaseModel):
    """PR information for triage."""

    title: str
    body: str

    def truncated_body(self, max_chars: int = 1000) -> str:
        """Return truncated body text."""
        if len(self.body) <= max_chars:
            return self.body
        return self.body[:max_chars] + "..."


class TaskInfo(BaseModel):
    """Full task information for classification."""

    pr_info: PRInfo
    files_changed: int
    lines_added: int
    lines_removed: int
    file_paths: list[str]
    diff_preview: str = ""


def clamp_score(score: float) -> float:
    """Clamp score to valid range [0.0, 1.0]."""
    return max(0.0, min(1.0, score))


def score_to_difficulty(score: float) -> str:
    """Convert numeric score to difficulty level string."""
    score = clamp_score(score)
    if score <= 0.35:
        return "easy"
    elif score <= 0.65:
        return "medium"
    else:
        return "hard"


def get_score_for_difficulty(difficulty: str) -> float:
    """Get a default score for a difficulty level (midpoint of range)."""
    level = DifficultyLevel(difficulty)
    low, high = SCORE_RANGES[level]
    return (low + high) / 2


class DifficultyClassifier:
    """LLM-based difficulty classifier for SWE tasks."""

    def __init__(self, client: LLMClient, model: str = "moonshotai/kimi-k2.5"):
        self._client = client
        self._model = model

    def _build_tool(self, schema: dict[str, Any]) -> ToolDefinition:
        """Build a ToolDefinition from a schema dict."""
        return ToolDefinition.create(
            name=schema["name"],
            description=f"Call this function to provide {schema['name']} results",
            parameters=schema["parameters"],
        )

    def _parse_tool_call(
        self, response: GenerationResponse, tool_name: str
    ) -> dict[str, Any] | None:
        """Parse tool call arguments from response."""
        if not response.choices:
            return None

        message = response.choices[0].message
        if not message.tool_calls:
            return None

        for tool_call in message.tool_calls:
            if tool_call.function.name == tool_name:
                try:
                    return json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    return None
        return None

    def _parse_json_fallback(self, content: str) -> dict[str, Any] | None:
        """Attempt to parse JSON from content as fallback."""
        # Try to find JSON object in the content
        start = content.find("{")
        if start == -1:
            return None
        end = content.rfind("}") + 1
        if end == 0:
            return None

        try:
            return json.loads(content[start:end])
        except json.JSONDecodeError:
            return None

    async def classify_triage(self, pr_info: PRInfo) -> TriageResponse:
        """Perform quick triage using PR title and body.

        Args:
            pr_info: PR information with title and body

        Returns:
            TriageResponse with difficulty and reasoning
        """
        tool = self._build_tool(TRIAGE_TOOL_SCHEMA)

        user_content = f"""PR Title: {pr_info.title}

PR Body (truncated):
{pr_info.truncated_body(1000)}

Please triage this PR."""

        request = GenerationRequest(
            model=self._model,
            messages=[
                Message.system(TRIAGE_SYSTEM_PROMPT),
                Message.user(user_content),
            ],
        ).with_tool(tool)

        try:
            response = await self._client.complete_with_tools(request)
            parsed = self._parse_tool_call(response, "triage")

            if parsed:
                return TriageResponse(
                    difficulty=parsed.get("difficulty", "medium"),
                    reasoning=parsed.get("reasoning", "No reasoning provided"),
                )

            # Try JSON fallback
            content = response.first_content() or ""
            fallback = self._parse_json_fallback(content)
            if fallback:
                return TriageResponse(
                    difficulty=fallback.get("difficulty", "medium"),
                    reasoning=fallback.get("reasoning", "Parsed from fallback"),
                )

            # Default to medium on parse failure
            return TriageResponse(
                difficulty="medium",
                reasoning="Failed to parse LLM response, defaulting to medium",
            )
        except Exception:
            # Default to medium on error
            return TriageResponse(
                difficulty="medium",
                reasoning="Error during classification, defaulting to medium",
            )

    async def classify_full(self, task_info: TaskInfo) -> ClassifyResponse:
        """Perform full classification with diff stats.

        Args:
            task_info: Full task information with diff stats

        Returns:
            ClassifyResponse with difficulty, score, quality_good, and reasoning
        """
        tool = self._build_tool(CLASSIFY_TOOL_SCHEMA)

        files_section = "\n".join(f"- {path}" for path in task_info.file_paths[:20])
        if len(task_info.file_paths) > 20:
            files_section += f"\n... and {len(task_info.file_paths) - 20} more files"

        # Build diff preview section (avoid backslash in f-string for Python 3.11)
        diff_preview_section = ""
        if task_info.diff_preview:
            diff_preview_section = f"Diff preview:\n{task_info.diff_preview[:500]}..."

        user_content = f"""PR Title: {task_info.pr_info.title}

PR Body (truncated):
{task_info.pr_info.truncated_body(1000)}

Diff Statistics:
- Files changed: {task_info.files_changed}
- Lines added: {task_info.lines_added}
- Lines removed: {task_info.lines_removed}

Changed files:
{files_section}

{diff_preview_section}

Please classify this PR."""

        request = GenerationRequest(
            model=self._model,
            messages=[
                Message.system(CLASSIFY_SYSTEM_PROMPT),
                Message.user(user_content),
            ],
        ).with_tool(tool)

        try:
            response = await self._client.complete_with_tools(request)
            parsed = self._parse_tool_call(response, "classify_pr")

            if parsed:
                return ClassifyResponse(
                    difficulty=parsed.get("difficulty", "medium"),
                    score=clamp_score(float(parsed.get("score", 0.5))),
                    quality_good=bool(parsed.get("quality_good", False)),
                    reasoning=parsed.get("reasoning", "No reasoning provided"),
                )

            # Try JSON fallback
            content = response.first_content() or ""
            fallback = self._parse_json_fallback(content)
            if fallback:
                return ClassifyResponse(
                    difficulty=fallback.get("difficulty", "medium"),
                    score=clamp_score(float(fallback.get("score", 0.5))),
                    quality_good=bool(fallback.get("quality_good", False)),
                    reasoning=fallback.get("reasoning", "Parsed from fallback"),
                )

            # Default to medium on parse failure
            return ClassifyResponse(
                difficulty="medium",
                score=0.5,
                quality_good=False,
                reasoning="Failed to parse LLM response, defaulting to medium",
            )
        except Exception:
            # Default to medium on error
            return ClassifyResponse(
                difficulty="medium",
                score=0.5,
                quality_good=False,
                reasoning="Error during classification, defaulting to medium",
            )
