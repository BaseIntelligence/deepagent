"""Multi-dimensional task scoring system for SWE-bench datasets.

This module provides sophisticated scoring based on:
- Prompt quality (clarity, completeness, verifiability)
- Difficulty prediction (lines changed, files changed)
- Test coverage (comprehensive test assessment)
- Patch quality (minimal, focused changes)

Based on SWE-bench research:
- Lines changed is 11x predictive of difficulty
- Multi-file tasks are 4x harder
- 50% of real-world issues are multi-file (vs 14% in SWE-bench Verified)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from swe_forge.llm.client import LLMClient, GenerationRequest


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ScoringConfig:
    """Configuration for task scoring thresholds."""
    
    # Minimum scores for acceptance
    min_prompt_quality: float = 0.5
    min_test_coverage: float = 0.5
    min_overall_score: float = 0.5
    
    # Difficulty thresholds (lines changed)
    easy_threshold: int = 10
    medium_threshold: int = 30
    
    # Multi-file multiplier
    multi_file_boost: float = 0.15
    
    # Scoring weights for overall score
    prompt_quality_weight: float = 0.30
    difficulty_weight: float = 0.25
    test_coverage_weight: float = 0.25
    patch_quality_weight: float = 0.20
    
    # Difficulty score ranges
    easy_score_range: tuple[float, float] = (0.1, 0.35)
    medium_score_range: tuple[float, float] = (0.4, 0.65)
    hard_score_range: tuple[float, float] = (0.7, 1.0)


# =============================================================================
# Task Score
# =============================================================================

@dataclass
class TaskScore:
    """Multi-dimensional task quality assessment."""
    
    # Individual scores (0.0 - 1.0)
    prompt_quality: float = 0.0
    difficulty_score: float = 0.5
    test_coverage: float = 0.0
    patch_quality: float = 0.5
    
    # Composite score
    overall_score: float = 0.0
    
    # Classification
    difficulty_level: str = "medium"  # "easy", "medium", "hard"
    passed: bool = False
    
    # Predictors
    lines_changed: int = 0
    files_changed: int = 0
    is_multi_file: bool = False
    
    # Metadata
    rejection_reasons: list[str] = field(default_factory=list)
    quality_reasoning: str = ""
    difficulty_reasoning: str = ""
    test_reasoning: str = ""
    
    def compute_overall(self, config: ScoringConfig) -> float:
        """Compute weighted overall score."""
        self.overall_score = (
            self.prompt_quality * config.prompt_quality_weight +
            self.difficulty_score * config.difficulty_weight +
            self.test_coverage * config.test_coverage_weight +
            self.patch_quality * config.patch_quality_weight
        )
        return self.overall_score
    
    def check_passed(self, config: ScoringConfig) -> bool:
        """Check if task passes all thresholds."""
        self.rejection_reasons = []
        
        if self.prompt_quality < config.min_prompt_quality:
            self.rejection_reasons.append(
                f"Prompt quality {self.prompt_quality:.2f} < {config.min_prompt_quality}"
            )
        
        if self.test_coverage < config.min_test_coverage:
            self.rejection_reasons.append(
                f"Test coverage {self.test_coverage:.2f} < {config.min_test_coverage}"
            )
        
        if self.overall_score < config.min_overall_score:
            self.rejection_reasons.append(
                f"Overall score {self.overall_score:.2f} < {config.min_overall_score}"
            )
        
        self.passed = len(self.rejection_reasons) == 0
        return self.passed


# =============================================================================
# Difficulty Prediction
# =============================================================================

def predict_difficulty(
    lines_changed: int,
    files_changed: int,
    config: ScoringConfig | None = None,
) -> tuple[float, str, str]:
    """Predict difficulty using lines changed as primary factor.
    
    Based on SWE-bench research:
    - Lines changed scales 11x from easy to hard
    - Multi-file tasks are 4x harder on average
    
    Args:
        lines_changed: Number of lines changed in the PR
        files_changed: Number of files modified
        config: Scoring configuration
        
    Returns:
        Tuple of (difficulty_score, difficulty_level, reasoning)
    """
    if config is None:
        config = ScoringConfig()
    
    # Base difficulty from lines changed (strongest predictor)
    if lines_changed <= config.easy_threshold:
        base_score = config.easy_score_range[0] + (
            (lines_changed / config.easy_threshold) * 
            (config.easy_score_range[1] - config.easy_score_range[0])
        )
        base_level = "easy"
        
    elif lines_changed <= config.medium_threshold:
        # Interpolate between easy and medium
        progress = (lines_changed - config.easy_threshold) / (config.medium_threshold - config.easy_threshold)
        base_score = config.medium_score_range[0] + (
            progress * (config.medium_score_range[1] - config.medium_score_range[0])
        )
        base_level = "medium"
        
    else:
        # Hard: interpolate based on lines beyond threshold
        # Cap at 100 lines for score 1.0
        excess_lines = min(lines_changed - config.medium_threshold, 70)
        progress = excess_lines / 70
        base_score = config.hard_score_range[0] + (
            progress * (config.hard_score_range[1] - config.hard_score_range[0])
        )
        base_level = "hard"
    
    # Multi-file boost
    score = base_score
    if files_changed >= 2:
        score = min(1.0, score + config.multi_file_boost)
        if files_changed >= 4:
            score = min(1.0, score + config.multi_file_boost)
    
    # Determine final level (may upgrade due to multi-file)
    if score >= config.hard_score_range[0]:
        level = "hard"
    elif score >= config.medium_score_range[0]:
        level = "medium"
    else:
        level = "easy"
    
    # Generate reasoning
    reasoning = f"Lines: {lines_changed}, Files: {files_changed}. "
    if lines_changed <= config.easy_threshold:
        reasoning += "Small change → easy"
    elif lines_changed <= config.medium_threshold:
        reasoning += "Medium change"
    else:
        reasoning += "Large change → hard"
    
    if files_changed >= 2:
        reasoning += f" (+{files_changed} files = complexity boost)"
    
    return (round(score, 2), level, reasoning)


# =============================================================================
# Prompt Quality Assessment
# =============================================================================

PROMPT_QUALITY_PROMPT = """You are evaluating SWE-bench task prompts for quality.

Score the prompt on a scale of 0.0 to 1.0 based on these criteria:

## Score 0.9-1.0 (Excellent)
The prompt:
- Clearly describes the bug/feature
- Provides expected behavior
- Contains runnable reproduction code
- Specifies exact file locations
- Has clear acceptance criteria

Example: "In `src/auth/login.py`, the `validate_token()` function fails when token contains Unicode characters. The function should accept UTF-8 encoded tokens. Test with: `validate_token('用户_token')` which should return True."

## Score 0.7-0.85 (Good)
The prompt:
- Describes the issue clearly
- May lack reproduction code
- May have ambiguous requirements
- Missing some context

Example: "The login function crashes with special characters. Fix the token validation in auth module."

## Score 0.5-0.65 (Acceptable)
The prompt:
- Basic description of issue
- Missing key details
- Requires investigation to understand

Example: "Fix the auth bug. See PR description."

## Score 0.0-0.45 (Poor)
The prompt:
- Vague or missing
- No clear objective
- Cannot determine what needs fixing

Example: "Fix stuff" or empty prompt.

Rate this prompt:
{prompt}

Output ONLY a JSON object:
{{"score": 0.XX, "reasoning": "Brief explanation", "missing_elements": ["list", "of", "missing"]}}"""


async def assess_prompt_quality(
    llm: LLMClient,
    prompt: str,
    model: str = "moonshotai/kimi-k2.5",
) -> tuple[float, str, list[str]]:
    """Assess prompt quality using LLM with examples.
    
    Args:
        llm: LLM client for assessment
        prompt: The task prompt to assess
        model: Model to use for assessment
        
    Returns:
        Tuple of (score, reasoning, missing_elements)
    """
    if not prompt or len(prompt.strip()) < 10:
        return (0.1, "Prompt too short or empty", ["content", "clarity", "context"])
    
    try:
        request = GenerationRequest(
            model=model,
            messages=[
                {"role": "user", "content": PROMPT_QUALITY_PROMPT.format(prompt=prompt)}
            ],
            temperature=0.1,  # Low temperature for consistent scoring
            max_tokens=200,
        )
        
        response = await llm.complete(request)
        
        if not response.choices:
            return (0.5, "No response from LLM", [])
        
        content = response.choices[0].message.content
        
        # Parse JSON response
        import json
        # Extract JSON from response
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(content[start:end])
            score = float(data.get("score", 0.5))
            reasoning = data.get("reasoning", "")
            missing = data.get("missing_elements", [])
            return (round(score, 2), reasoning, missing)
        
    except Exception as e:
        pass
    
    # Fallback: simple heuristic scoring
    score = 0.5
    reasoning = []
    
    if len(prompt) > 100:
        score += 0.1
        reasoning.append("Has substantial content")
    
    if "error" in prompt.lower() or "bug" in prompt.lower() or "fix" in prompt.lower():
        score += 0.1
        reasoning.append("Describes issue")
    
    if "test" in prompt.lower():
        score += 0.1
        reasoning.append("Mentions testing")
    
    return (min(1.0, score), "Heuristic scoring (LLM failed)", [])


# =============================================================================
# Test Coverage Assessment
# =============================================================================

TEST_COVERAGE_PROMPT = """Evaluate test coverage quality for a code fix.

Score based on:

## Score 0.9-1.0 (Excellent)
Tests include:
- Happy path (main functionality works)
- Edge cases (boundary conditions)
- Error scenarios (invalid inputs)
- Regression tests (existing functionality unchanged)
- Clear assertions with expected values

Example: 5+ test cases covering all branches.

## Score 0.7-0.85 (Good)
Tests include:
- Happy path
- At least 1 edge case
- Basic error handling
- Some assertions

Example: 3 test cases covering main flows.

## Score 0.5-0.65 (Acceptable)
Tests include:
- Happy path only
- Minimal assertions
- May miss edge cases

Example: 1-2 test cases.

## Score 0.0-0.45 (Poor)
Tests:
- Missing or broken
- No assertions
- Cannot verify fix

Example: No tests or placeholder tests.

Test commands:
{test_commands}

Output ONLY JSON:
{{"score": 0.XX, "reasoning": "Brief explanation", "missing_coverage": ["list"]}}"""


async def assess_test_coverage(
    llm: LLMClient,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    model: str = "moonshotai/kimi-k2.5",
) -> tuple[float, str, list[str]]:
    """Assess test coverage quality.
    
    Args:
        llm: LLM client
        fail_to_pass: Tests that should fail before fix
        pass_to_pass: Tests that should pass before and after
        model: Model to use
        
    Returns:
        Tuple of (score, reasoning, missing_coverage)
    """
    if not fail_to_pass and not pass_to_pass:
        return (0.1, "No tests provided", ["all_tests"])
    
    # Heuristic base score
    total_tests = len(fail_to_pass) + len(pass_to_pass)
    
    # More tests = better coverage
    base_score = min(0.9, 0.3 + (total_tests * 0.1))
    
    # Having both fail_to_pass and pass_to_pass is good
    if fail_to_pass and pass_to_pass:
        base_score += 0.1
    
    reasoning = f"{total_tests} test(s): {len(fail_to_pass)} fail_to_pass, {len(pass_to_pass)} pass_to_pass"
    
    missing = []
    if not fail_to_pass:
        missing.append("fail_to_pass tests")
    if not pass_to_pass:
        missing.append("regression tests")
    if total_tests < 3:
        missing.append("edge case coverage")
    
    try:
        test_commands = "\n".join([
            f"FAIL_TO_PASS: {t}" for t in fail_to_pass
        ] + [
            f"PASS_TO_PASS: {t}" for t in pass_to_pass
        ])
        
        request = GenerationRequest(
            model=model,
            messages=[
                {"role": "user", "content": TEST_COVERAGE_PROMPT.format(test_commands=test_commands)}
            ],
            temperature=0.1,
            max_tokens=200,
        )
        
        response = await llm.complete(request)
        
        if response.choices:
            content = response.choices[0].message.content
            import json
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
                return (
                    round(float(data.get("score", base_score)), 2),
                    data.get("reasoning", reasoning),
                    data.get("missing_coverage", missing)
                )
    except:
        pass
    
    return (round(base_score, 2), reasoning, missing)


# =============================================================================
# TaskScorer
# =============================================================================

class TaskScorer:
    """Multi-dimensional task quality scorer."""
    
    def __init__(
        self,
        llm: LLMClient | None = None,
        model: str = "moonshotai/kimi-k2.5",
        config: ScoringConfig | None = None,
    ):
        self.llm = llm
        self.model = model
        self.config = config or ScoringConfig()
    
    async def score_task(
        self,
        prompt: str,
        lines_changed: int,
        files_changed: int,
        fail_to_pass: list[str],
        pass_to_pass: list[str],
    ) -> TaskScore:
        """Score a task across all dimensions.
        
        Args:
            prompt: Task prompt/description
            lines_changed: Number of lines changed
            files_changed: Number of files modified
            fail_to_pass: Test commands for fail_to_pass
            pass_to_pass: Test commands for pass_to_pass
            
        Returns:
            TaskScore with all assessments
        """
        score = TaskScore(
            lines_changed=lines_changed,
            files_changed=files_changed,
            is_multi_file=files_changed >= 2,
        )
        
        # 1. Difficulty prediction (rule-based, no LLM needed)
        diff_score, diff_level, diff_reason = predict_difficulty(
            lines_changed, files_changed, self.config
        )
        score.difficulty_score = diff_score
        score.difficulty_level = diff_level
        score.difficulty_reasoning = diff_reason
        
        # 2. Prompt quality (LLM-based if available)
        if self.llm:
            pq_score, pq_reason, pq_missing = await assess_prompt_quality(
                self.llm, prompt, self.model
            )
            score.prompt_quality = pq_score
            score.quality_reasoning = pq_reason
            if pq_missing:
                score.rejection_reasons.extend([f"Missing: {m}" for m in pq_missing])
        else:
            # Simple heuristic without LLM
            score.prompt_quality = 0.5 if len(prompt) > 50 else 0.3
            score.quality_reasoning = "No LLM for quality assessment"
        
        # 3. Test coverage (LLM-based if available)
        if self.llm:
            tc_score, tc_reason, tc_missing = await assess_test_coverage(
                self.llm, fail_to_pass, pass_to_pass, self.model
            )
            score.test_coverage = tc_score
            score.test_reasoning = tc_reason
            if tc_missing:
                score.rejection_reasons.extend([f"Missing coverage: {m}" for m in tc_missing])
        else:
            # Simple heuristic
            total_tests = len(fail_to_pass) + len(pass_to_pass)
            score.test_coverage = min(0.9, 0.3 + (total_tests * 0.15))
            score.test_reasoning = f"{total_tests} tests (heuristic score)"
        
        # 4. Patch quality (placeholder - would need actual patch analysis)
        score.patch_quality = 0.5
        
        # Compute overall score
        score.compute_overall(self.config)
        score.check_passed(self.config)
        
        return score
