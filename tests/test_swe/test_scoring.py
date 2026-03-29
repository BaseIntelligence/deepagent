"""Tests for multi-dimensional task scoring."""

import pytest

from swe_forge.swe.scoring import (
    TaskScore,
    ScoringConfig,
    predict_difficulty,
    TaskScorer,
)


class TestPredictDifficulty:
    """Test difficulty prediction based on lines changed."""
    
    def test_easy_single_file_small_change(self):
        """Small changes in single file = easy."""
        score, level, reason = predict_difficulty(
            lines_changed=5,
            files_changed=1,
        )
        assert level == "easy"
        assert score < 0.35
        assert "easy" in reason.lower() or "small" in reason.lower()
    
    def test_easy_boundary(self):
        """At easy threshold."""
        score, level, _ = predict_difficulty(
            lines_changed=10,
            files_changed=1,
        )
        assert level == "easy"
        assert score <= 0.35
    
    def test_medium_range(self):
        """Medium-sized changes."""
        score, level, _ = predict_difficulty(
            lines_changed=20,
            files_changed=1,
        )
        assert level == "medium"
        assert 0.35 < score < 0.7
    
    def test_medium_boundary(self):
        """At medium threshold."""
        score, level, _ = predict_difficulty(
            lines_changed=30,
            files_changed=1,
        )
        assert level == "medium" or level == "hard"
    
    def test_hard_large_change(self):
        """Large changes = hard."""
        score, level, _ = predict_difficulty(
            lines_changed=50,
            files_changed=1,
        )
        assert level == "hard"
        assert score >= 0.7
    
    def test_very_large_change(self):
        """Very large changes approach max score."""
        score, level, _ = predict_difficulty(
            lines_changed=100,
            files_changed=1,
        )
        assert level == "hard"
        assert score >= 0.9
    
    def test_multi_file_boosts_difficulty(self):
        """Multi-file changes boost difficulty."""
        score_single, _, _ = predict_difficulty(
            lines_changed=20,
            files_changed=1,
        )
        score_multi, level_multi, _ = predict_difficulty(
            lines_changed=20,
            files_changed=2,
        )
        assert score_multi > score_single
        assert "2 files" in level_multi or level_multi != "easy"
    
    def test_many_files_higher_boost(self):
        """More files = higher boost."""
        score_2, _, _ = predict_difficulty(lines_changed=30, files_changed=2)
        score_4, _, _ = predict_difficulty(lines_changed=30, files_changed=4)
        assert score_4 > score_2
    
    def test_zero_lines(self):
        """Edge case: no lines changed."""
        score, level, _ = predict_difficulty(
            lines_changed=0,
            files_changed=1,
        )
        assert level == "easy"
        assert score >= 0.1


class TestTaskScore:
    """Test TaskScore dataclass."""
    
    def test_compute_overall(self):
        """Overall score is weighted combination."""
        score = TaskScore(
            prompt_quality=0.8,
            difficulty_score=0.5,
            test_coverage=0.7,
            patch_quality=0.6,
        )
        config = ScoringConfig()
        overall = score.compute_overall(config)
        
        # Expected: 0.8*0.3 + 0.5*0.25 + 0.7*0.25 + 0.6*0.20 = 0.655
        expected = (
            0.8 * 0.30 +
            0.5 * 0.25 +
            0.7 * 0.25 +
            0.6 * 0.20
        )
        assert abs(overall - expected) < 0.01
    
    def test_check_passed_meets_thresholds(self):
        """Task passes when all thresholds met."""
        score = TaskScore(
            prompt_quality=0.8,
            test_coverage=0.7,
            overall_score=0.75,
        )
        config = ScoringConfig(
            min_prompt_quality=0.5,
            min_test_coverage=0.5,
            min_overall_score=0.5,
        )
        passed = score.check_passed(config)
        assert passed is True
        assert len(score.rejection_reasons) == 0
    
    def test_check_passed_fails_prompt_quality(self):
        """Task fails when prompt quality too low."""
        score = TaskScore(
            prompt_quality=0.3,
            test_coverage=0.7,
            overall_score=0.5,
        )
        config = ScoringConfig(min_prompt_quality=0.5)
        passed = score.check_passed(config)
        assert passed is False
        assert any("prompt quality" in r.lower() for r in score.rejection_reasons)
    
    def test_check_passed_fails_test_coverage(self):
        """Task fails when test coverage too low."""
        score = TaskScore(
            prompt_quality=0.8,
            test_coverage=0.3,
            overall_score=0.5,
        )
        config = ScoringConfig(min_test_coverage=0.5)
        passed = score.check_passed(config)
        assert passed is False
        assert any("test coverage" in r.lower() for r in score.rejection_reasons)


class TestScoringConfig:
    """Test scoring configuration."""
    
    def test_default_thresholds(self):
        """Default thresholds are reasonable."""
        config = ScoringConfig()
        assert 0 < config.min_prompt_quality < 1
        assert 0 < config.min_test_coverage < 1
        assert config.easy_threshold < config.medium_threshold
    
    def test_custom_thresholds(self):
        """Can customize thresholds."""
        config = ScoringConfig(
            min_prompt_quality=0.7,
            min_test_coverage=0.6,
            min_overall_score=0.6,
        )
        assert config.min_prompt_quality == 0.7
        assert config.min_test_coverage == 0.6
    
    def test_weights_sum_to_one(self):
        """Scoring weights should sum to approximately 1."""
        config = ScoringConfig()
        total_weight = (
            config.prompt_quality_weight +
            config.difficulty_weight +
            config.test_coverage_weight +
            config.patch_quality_weight
        )
        assert abs(total_weight - 1.0) < 0.01


class TestTaskScorer:
    """Test TaskScorer class."""
    
    def test_score_task_without_llm(self):
        """Can score task without LLM (uses heuristics)."""
        scorer = TaskScorer(llm=None)
        
        # sync wrapper for async
        import asyncio
        score = asyncio.run(scorer.score_task(
            prompt="Fix the bug in auth module",
            lines_changed=15,
            files_changed=2,
            fail_to_pass=["pytest tests/test_auth.py"],
            pass_to_pass=["pytest tests/test_other.py"],
        ))
        
        assert score.lines_changed == 15
        assert score.files_changed == 2
        assert score.is_multi_file is True
        assert 0 <= score.prompt_quality <= 1
        assert 0 <= score.difficulty_score <= 1
        assert 0 <= score.test_coverage <= 1
        assert 0 <= score.overall_score <= 1
        assert score.difficulty_level in ("easy", "medium", "hard")
    
    def test_difficulty_classification_matches_score(self):
        """Difficulty level should match score range."""
        scorer = TaskScorer()
        
        import asyncio
        
        # Easy task
        score_easy = asyncio.run(scorer.score_task(
            prompt="Fix typo",
            lines_changed=3,
            files_changed=1,
            fail_to_pass=["pytest tests/"],
            pass_to_pass=[],
        ))
        assert score_easy.difficulty_level == "easy"
        
        # Hard task
        score_hard = asyncio.run(scorer.score_task(
            prompt="Complex refactoring",
            lines_changed=80,
            files_changed=5,
            fail_to_pass=["pytest tests/"],
            pass_to_pass=[],
        ))
        assert score_hard.difficulty_level == "hard"
