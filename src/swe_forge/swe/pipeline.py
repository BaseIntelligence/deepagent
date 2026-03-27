"""Pipeline orchestrator for SWE mining workflow.

This module wires together all pipeline stages:
GH Archive -> Enricher -> Filter -> LLM Classify -> Extract -> Export

Uses aggressive parallelism with semaphores controlling concurrency at each stage.
"""

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncGenerator

from .difficulty import (
    ClassifyResponse,
    DifficultyClassifier,
    PRInfo,
    TaskInfo,
    TriageResponse,
)
from .enricher import EnrichedPullRequest, enrich_pr
from .filters import FilterConfig, apply_filters
from .gharchive import GhArchiveClient, GhArchiveEvent
from .github_api import GitHubClient
from .models import SweTask, SweTaskStatus
from .test_generator import GeneratedTests

from swe_forge.execution.docker_client import DockerClient
from swe_forge.execution.sandbox import DockerSandbox, SandboxConfig

if TYPE_CHECKING:
    from swe_forge.llm.client import LLMClient

    from .test_generator import TestGenerator

logger = logging.getLogger(__name__)


class SwePipelineEventType(str, Enum):
    """Types of events emitted during pipeline execution."""

    COLLECTION_STARTED = "collection_started"
    CANDIDATE_FILTERED = "candidate_filtered"
    TASK_EXTRACTED = "task_extracted"
    QUALITY_SCORED = "quality_scored"
    PIPELINE_COMPLETED = "pipeline_completed"


@dataclass
class SwePipelineEvent:
    """Event emitted during pipeline execution for progress tracking."""

    event_type: SwePipelineEventType
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class DifficultyTargets:
    """Per-difficulty quotas for multi-target mining mode."""

    targets: dict[str, int] = field(default_factory=dict)

    def all_met(self, counts: dict[str, int]) -> bool:
        """Check if all difficulty quotas are met."""
        return all(
            counts.get(level, 0) >= quota for level, quota in self.targets.items()
        )


@dataclass
class SwePipelineConfig:
    """Configuration for the SWE mining pipeline.

    Attributes:
        min_stars: Minimum GitHub stars required for a repository.
        languages: List of allowed programming languages (empty = all).
        max_candidates: Maximum PR candidates to process.
        max_tasks: Maximum tasks to extract before stopping.
        once: If True, stop after max_tasks; otherwise run continuously.
        concurrency_enrich: Max concurrent enrichment operations.
        concurrency_preclassify: Max concurrent pre-classification LLM calls.
        concurrency_deep: Max concurrent deep processing operations.
        difficulty_filter: Filter for specific difficulty level.
        difficulty_targets: Per-difficulty quotas for multi-target mode.
    """

    min_stars: int = 100
    languages: list[str] = field(default_factory=list)
    max_candidates: int = 50
    max_tasks: int = 1
    once: bool = True
    concurrency_enrich: int = 20
    concurrency_preclassify: int = 25
    concurrency_deep: int = 8
    difficulty_filter: str | None = None
    difficulty_targets: DifficultyTargets | None = None
    llm_client: "LLMClient | None" = None
    difficulty_classifier: DifficultyClassifier | None = None
    test_generator: "TestGenerator | None" = None


@dataclass
class BenchmarkMetrics:
    """Aggregate metrics collected during a full pipeline run."""

    total_raw_events: int = 0
    total_merged_events: int = 0
    total_prefiltered: int = 0
    enriched_count: int = 0
    enrichment_failed: int = 0
    filter_passed: int = 0
    filter_rejected: int = 0
    filter_rejection_reasons: dict[str, int] = field(default_factory=dict)
    preclassify_count: int = 0
    preclassify_easy: int = 0
    preclassify_medium: int = 0
    preclassify_hard: int = 0
    extraction_attempted: int = 0
    extraction_succeeded: int = 0
    extraction_failed: int = 0
    quality_scored: int = 0
    quality_passed: int = 0
    quality_failed: int = 0
    difficulty_easy: int = 0
    difficulty_medium: int = 0
    difficulty_hard: int = 0
    accepted_count: int = 0
    total_processing_time_ms: float = 0.0
    avg_per_pr_time_ms: float = 0.0
    throughput_prs_per_sec: float = 0.0
    avg_quality_score: float = 0.0
    languages: dict[str, int] = field(default_factory=dict)


@dataclass
class SwePipelineRunResult:
    """Result of a pipeline run."""

    tasks: list[SweTask] = field(default_factory=list)
    filtered: int = 0
    extracted: int = 0
    scored: int = 0
    finished_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    benchmark_metrics: BenchmarkMetrics | None = None


class SwePipeline:
    """Orchestrates the SWE mining pipeline stages.

    Pipeline stages (in order):
    1. Enrich: Fetch PR details from GitHub API
    2. Filter: Apply local filters (fast, no external calls)
    3. Pre-classify: LLM-based difficulty classification
    4. Deep: Extraction, test generation, quality scoring

    Uses semaphores to control concurrency at each stage.
    """

    def __init__(
        self,
        gh_client: GitHubClient,
        gh_archive_client: GhArchiveClient | None = None,
        config: SwePipelineConfig | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            gh_client: GitHub API client for enrichment.
            gh_archive_client: GH Archive client (or create default).
            config: Pipeline configuration.
        """
        self.gh_client = gh_client
        self.gh_archive_client = gh_archive_client or GhArchiveClient()
        self.config = config or SwePipelineConfig()
        self._filter_config = self._build_filter_config()
        self._active = False
        self._classifier: DifficultyClassifier | None = None

    def _build_filter_config(self) -> FilterConfig:
        """Build filter config from pipeline config."""
        return FilterConfig(
            min_stars=self.config.min_stars,
            allowed_languages=self.config.languages
            if self.config.languages
            else ["python"],
            max_files_changed=50,
            exclude_bots=True,
        )

    def _get_classifier(self) -> DifficultyClassifier | None:
        """Get or create the difficulty classifier.

        Creates classifier on first use. Returns None if no llm_client available,
        which causes fallback to heuristic-based classification.

        Returns:
            DifficultyClassifier instance or None if llm_client not configured.
        """
        if self._classifier is None:
            if self.config.difficulty_classifier is not None:
                self._classifier = self.config.difficulty_classifier
            elif self.config.llm_client is not None:
                self._classifier = DifficultyClassifier(client=self.config.llm_client)
        return self._classifier

    async def __aenter__(self) -> "SwePipeline":
        """Enter async context manager."""
        if self.gh_archive_client._session is None:
            await self.gh_archive_client._ensure_session()
        self._active = True
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        self._active = False
        await self.gh_archive_client.close()

    async def _create_sandbox(
        self,
        repo_url: str,
        base_commit: str,
    ) -> DockerSandbox:
        """Create a DockerSandbox configured for task execution.

        Factory method for creating sandbox instances that can be used
        for isolated repository operations during task execution.

        Args:
            repo_url: Repository URL to clone (e.g., "https://github.com/owner/repo").
            base_commit: Base commit SHA to checkout.

        Returns:
            DockerSandbox instance configured with ubuntu:24.04 image.
            The sandbox is not yet started - caller must use async context manager.
        """
        client = DockerClient()
        config = SandboxConfig(image="ubuntu:24.04")
        return DockerSandbox(client, config)

    async def _emit_event(
        self, queue: asyncio.Queue[SwePipelineEvent] | None, event: SwePipelineEvent
    ) -> None:
        """Emit an event to the progress queue if available."""
        if queue is not None:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Event queue full, dropping event: %s", event.event_type)

    async def _enrich_stage(
        self,
        event: GhArchiveEvent,
        semaphore: asyncio.Semaphore,
        metrics: BenchmarkMetrics,
    ) -> EnrichedPullRequest | None:
        """Stage 1: Enrich a GH Archive event with GitHub API data."""
        async with semaphore:
            try:
                enriched = await enrich_pr(event, self.gh_client)
                metrics.enriched_count += 1
                return enriched
            except Exception as e:
                logger.warning(
                    "Enrichment failed for %s#%d: %s",
                    event.repository,
                    event.pull_number,
                    e,
                )
                metrics.enrichment_failed += 1
                return None

    def _filter_stage(
        self,
        enriched: EnrichedPullRequest,
        metrics: BenchmarkMetrics,
    ) -> bool:
        """Stage 2: Apply local filters (pure Python, fast)."""
        passed = apply_filters(enriched, self._filter_config)
        if passed:
            metrics.filter_passed += 1
        else:
            metrics.filter_rejected += 1
        return passed

    async def _preclassify_stage(
        self,
        enriched: EnrichedPullRequest,
        semaphore: asyncio.Semaphore,
        metrics: BenchmarkMetrics,
    ) -> str | None:
        """Stage 3: Pre-classify difficulty using LLM or heuristics."""
        async with semaphore:
            metrics.preclassify_count += 1

            classifier = self._get_classifier()
            if classifier is not None:
                pr_info = PRInfo(title=enriched.title, body=enriched.body or "")
                response: TriageResponse = await classifier.classify_triage(pr_info)
                difficulty = response.difficulty
            else:
                if enriched.files_changed <= 2:
                    difficulty = "easy"
                elif enriched.files_changed <= 5:
                    difficulty = "medium"
                else:
                    difficulty = "hard"

            if difficulty == "easy":
                metrics.preclassify_easy += 1
            elif difficulty == "medium":
                metrics.preclassify_medium += 1
            else:
                metrics.preclassify_hard += 1

            logger.debug(
                "Pre-classified %s#%d as %s",
                enriched.repo,
                enriched.number,
                difficulty,
            )
            return difficulty

    async def _deep_stage(
        self,
        enriched: EnrichedPullRequest,
        semaphore: asyncio.Semaphore,
        metrics: BenchmarkMetrics,
        event_queue: asyncio.Queue[SwePipelineEvent] | None,
    ) -> SweTask | None:
        """Stage 4: Deep processing (extraction, quality scoring).

        Extracts patches and generates test commands using TestGenerator when
        configured. Falls back to mock logic when test_generator is None.
        """
        async with semaphore:
            metrics.extraction_attempted += 1

            try:
                patch, test_patch = await self._extract_patch(enriched)

                task = SweTask(
                    id=f"{enriched.repo.replace('/', '-')}-{enriched.number}",
                    repo=enriched.repo,
                    base_commit=enriched.base_commit,
                    merge_commit=enriched.merge_commit,
                    language=enriched.language,
                    prompt=enriched.title,
                    original_pr_body=enriched.body or "",
                    status=SweTaskStatus.CANDIDATE,
                    patch=patch,
                    test_patch=test_patch,
                )

                metrics.extraction_succeeded += 1

                if self.config.test_generator is not None:
                    await self._run_test_generation(
                        enriched, task, metrics, event_queue
                    )
                else:
                    quality_score = 0.75 + (len(enriched.changed_files) * 0.02)
                    quality_score = min(quality_score, 1.0)
                    task.quality_score = quality_score
                    task.quality_passed = quality_score >= 0.7

                    metrics.quality_scored += 1
                    if task.quality_passed:
                        metrics.quality_passed += 1
                    else:
                        metrics.quality_failed += 1

                    classifier = self._get_classifier()
                    if classifier is not None:
                        pr_info = PRInfo(title=enriched.title, body=enriched.body or "")
                        task_info = TaskInfo(
                            pr_info=pr_info,
                            files_changed=enriched.files_changed,
                            lines_added=enriched.additions,
                            lines_removed=enriched.deletions,
                            file_paths=enriched.changed_files,
                        )
                        response: ClassifyResponse = await classifier.classify_full(
                            task_info
                        )
                        task.difficulty_score = max(
                            1, min(10, int(response.score * 10))
                        )
                        if response.difficulty == "easy":
                            metrics.difficulty_easy += 1
                        elif response.difficulty == "medium":
                            metrics.difficulty_medium += 1
                        else:
                            metrics.difficulty_hard += 1
                    else:
                        if enriched.files_changed <= 2:
                            task.difficulty_score = 1
                            metrics.difficulty_easy += 1
                        elif enriched.files_changed <= 5:
                            task.difficulty_score = 2
                            metrics.difficulty_medium += 1
                        else:
                            task.difficulty_score = 3
                            metrics.difficulty_hard += 1

                    await self._emit_event(
                        event_queue,
                        SwePipelineEvent(
                            SwePipelineEventType.QUALITY_SCORED,
                            {
                                "task_id": task.id,
                                "score": quality_score,
                                "passed": task.quality_passed,
                            },
                        ),
                    )

                    logger.info(
                        "Task %s processed (score=%.2f, passed=%s, no test_generator)",
                        task.id,
                        quality_score,
                        task.quality_passed,
                    )

                return task

            except Exception as e:
                metrics.extraction_failed += 1
                logger.warning(
                    "Deep processing failed for %s#%d: %s",
                    enriched.repo,
                    enriched.number,
                    e,
                )
                return None

    async def _run_test_generation(
        self,
        enriched: EnrichedPullRequest,
        task: SweTask,
        metrics: BenchmarkMetrics,
        event_queue: asyncio.Queue[SwePipelineEvent] | None,
    ) -> None:
        """Run test generation for a task using configured TestGenerator.

        Creates a sandbox, runs test generation, and maps results to task fields.
        Marks task as REJECTED if generation fails.

        Args:
            enriched: Enriched PR data for sandbox setup.
            task: SweTask to populate with generated tests.
            metrics: Metrics tracker for quality scoring stats.
            event_queue: Optional event queue for progress updates.
        """
        repo_url = f"https://github.com/{enriched.repo}"
        sandbox: DockerSandbox | None = None

        try:
            sandbox = self._create_sandbox(repo_url, enriched.base_commit)

            async with sandbox:
                await sandbox.setup_workspace(repo_url, enriched.base_commit)

                generated: GeneratedTests = (
                    await self.config.test_generator.generate_tests(  # type: ignore[union-attr]
                        task, sandbox
                    )
                )

                task.fail_to_pass = generated.fail_to_pass
                task.pass_to_pass = generated.pass_to_pass

                if generated.test_files:
                    test_files_content = "\n".join(
                        f"# Test file: {tf.path}\n{tf.content}"
                        for tf in generated.test_files
                    )
                    if task.test_patch:
                        task.test_patch = f"{task.test_patch}\n\n{test_files_content}"
                    else:
                        task.test_patch = test_files_content

                if generated.success:
                    task.quality_score = 1.0
                    task.quality_passed = True
                    task.status = SweTaskStatus.READY
                else:
                    task.quality_score = 0.0
                    task.quality_passed = False
                    task.status = SweTaskStatus.REJECTED

                if generated.install_commands:
                    task.install_config["install_commands"] = generated.install_commands
                    task.install_config["validated"] = True

                metrics.quality_scored += 1
                if task.quality_passed:
                    metrics.quality_passed += 1
                else:
                    metrics.quality_failed += 1

                if generated.turn_count <= 2:
                    task.difficulty_score = 1
                    metrics.difficulty_easy += 1
                elif generated.turn_count <= 5:
                    task.difficulty_score = 2
                    metrics.difficulty_medium += 1
                else:
                    task.difficulty_score = 3
                    metrics.difficulty_hard += 1

                await self._emit_event(
                    event_queue,
                    SwePipelineEvent(
                        SwePipelineEventType.QUALITY_SCORED,
                        {
                            "task_id": task.id,
                            "score": task.quality_score,
                            "passed": task.quality_passed,
                            "turn_count": generated.turn_count,
                        },
                    ),
                )

                logger.info(
                    "Task %s processed with tests (score=%.2f, passed=%s, turns=%d)",
                    task.id,
                    task.quality_score,
                    task.quality_passed,
                    generated.turn_count,
                )

        except Exception as e:
            task.quality_score = 0.0
            task.quality_passed = False
            task.status = SweTaskStatus.REJECTED
            metrics.quality_failed += 1
            logger.warning(
                "Test generation failed for %s#%d: %s",
                enriched.repo,
                enriched.number,
                e,
            )

    def _should_skip_event(self, event: GhArchiveEvent) -> bool:
        """Pre-filter events before enrichment."""
        if event.pull_number == 0:
            return True
        if "[bot]" in event.actor or event.actor == "dependabot":
            return True
        if not event.has_org:
            return True
        return False

    def _is_test_file(self, filename: str) -> bool:
        """Check if a filename is a test file.

        Detects test files by patterns:
        - starts with 'tests/' or 'test/'
        - contains 'test_' in filename
        - ends with '_test.py'
        - contains '/spec/' in path
        """
        filename_lower = filename.lower()
        return (
            filename_lower.startswith("tests/")
            or filename_lower.startswith("test/")
            or "test_" in filename_lower
            or filename_lower.endswith("_test.py")
            or "/spec/" in filename_lower
        )

    async def _extract_patch(self, enriched: EnrichedPullRequest) -> tuple[str, str]:
        """Extract and separate patch into main and test patches.

        Fetches PR diff from GitHub API and separates it into:
        - patch: hunks from non-test files
        - test_patch: hunks from test files

        Args:
            enriched: Enriched PR data containing repo and PR number.

        Returns:
            Tuple of (patch, test_patch) strings. Returns ('', '') if diff is empty.
        """
        # Parse repo format: "owner/repo"
        parts = enriched.repo.split("/", 1)
        if len(parts) != 2:
            logger.warning("Invalid repo format: %s", enriched.repo)
            return "", ""

        owner, repo = parts[0], parts[1]

        try:
            diff = await self.gh_client.get_pr_diff(owner, repo, enriched.number)
        except Exception as e:
            logger.warning(
                "Failed to fetch diff for %s#%d: %s",
                enriched.repo,
                enriched.number,
                e,
            )
            return "", ""

        if not diff:
            return "", ""

        # Parse unified diff format
        patch_parts: list[str] = []
        test_patch_parts: list[str] = []

        lines = diff.split("\n")
        current_file: str | None = None
        current_is_test: bool = False
        current_hunk: list[str] = []

        for line in lines:
            # New file header
            if line.startswith("diff --git "):
                # Save previous hunk
                if current_file is not None and current_hunk:
                    if current_is_test:
                        test_patch_parts.extend(current_hunk)
                    else:
                        patch_parts.extend(current_hunk)
                    current_hunk = []

                # Parse filename from diff --git a/path b/path
                # Format: diff --git a/path/to/file.py b/path/to/file.py
                match_parts = line.split(" ")
                if len(match_parts) >= 3:
                    # Take the 'b/' prefixed path (destination)
                    b_path = match_parts[-1]
                    if b_path.startswith("b/"):
                        current_file = b_path[2:]
                    else:
                        current_file = b_path
                    current_is_test = self._is_test_file(current_file)
                else:
                    current_file = None
                    current_is_test = False

            current_hunk.append(line)

        # Don't forget the last hunk
        if current_file is not None and current_hunk:
            if current_is_test:
                test_patch_parts.extend(current_hunk)
            else:
                patch_parts.extend(current_hunk)

        patch = "\n".join(patch_parts)
        test_patch = "\n".join(test_patch_parts)

        return patch, test_patch

    def _difficulty_matches_filter(
        self, difficulty: str | None, config: SwePipelineConfig
    ) -> bool:
        """Check if difficulty matches the configured filter."""
        if config.difficulty_targets is not None:
            return difficulty in config.difficulty_targets.targets
        if config.difficulty_filter is not None:
            return difficulty == config.difficulty_filter
        return True  # No filter, all pass

    async def run(self) -> SwePipelineRunResult:
        """Run the pipeline and return results.

        This is a convenience method that collects all events.
        For progress tracking, use run_with_progress() instead.
        """
        tasks: list[SweTask] = []
        metrics = BenchmarkMetrics()
        filtered_count = 0

        async for event in self.run_with_progress():
            if event.event_type == SwePipelineEventType.TASK_EXTRACTED:
                task_data = event.data.get("task")
                if task_data and isinstance(task_data, SweTask):
                    tasks.append(task_data)
            elif event.event_type == SwePipelineEventType.CANDIDATE_FILTERED:
                if not event.data.get("accepted", True):
                    filtered_count += 1

        # Calculate final metrics
        if metrics.enriched_count > 0:
            metrics.throughput_prs_per_sec = metrics.enriched_count / max(
                metrics.total_processing_time_ms / 1000.0, 0.001
            )
            metrics.avg_per_pr_time_ms = (
                metrics.total_processing_time_ms / metrics.enriched_count
            )

        return SwePipelineRunResult(
            tasks=tasks,
            filtered=filtered_count,
            extracted=len(tasks),
            scored=metrics.quality_scored,
            benchmark_metrics=metrics,
        )

    async def run_with_progress(
        self,
    ) -> AsyncGenerator[SwePipelineEvent, None]:
        """Run the pipeline yielding progress events.

        Yields events as tasks progress through the pipeline stages.
        """
        start_time = datetime.now(timezone.utc)
        metrics = BenchmarkMetrics()

        # Create semaphores for concurrency control
        enrich_sem = asyncio.Semaphore(self.config.concurrency_enrich)
        preclassify_sem = asyncio.Semaphore(self.config.concurrency_preclassify)
        deep_sem = asyncio.Semaphore(self.config.concurrency_deep)

        # Create event queue for progress tracking
        event_queue: asyncio.Queue[SwePipelineEvent] = asyncio.Queue(maxsize=1000)

        # Emit collection started
        yield SwePipelineEvent(
            SwePipelineEventType.COLLECTION_STARTED,
            {"requested": self.config.max_candidates},
        )

        # Calculate hours back based on candidates
        hours_back = min(max((self.config.max_candidates // 50) + 1, 6), 12)

        # Fetch events from GH Archive
        try:
            # For testing, we might have mock events
            events = await self._fetch_events(hours_back)
        except Exception as e:
            logger.error("Failed to fetch GH Archive events: %s", e)
            yield SwePipelineEvent(
                SwePipelineEventType.PIPELINE_COMPLETED,
                {"emitted": 0, "error": str(e)},
            )
            return

        metrics.total_raw_events = len(events)

        # Filter to merged PRs only
        events = [e for e in events if e.action.lower() == "merged"]
        metrics.total_merged_events = len(events)

        # Pre-filter events
        events = [e for e in events if not self._should_skip_event(e)]
        metrics.total_prefiltered = len(events)

        if not events:
            logger.warning("No merged PRs found in GH Archive data")
            yield SwePipelineEvent(
                SwePipelineEventType.PIPELINE_COMPLETED,
                {"emitted": 0, "error": "No merged PRs found"},
            )
            return

        # Shuffle for diversity
        random.shuffle(events)

        # Truncate to max candidates
        if self.config.max_candidates > 0 and len(events) > self.config.max_candidates:
            events = events[: self.config.max_candidates]

        # Track completed tasks and per-difficulty counts
        completed_count = 0
        per_difficulty_counts: dict[str, int] = {}
        accepted_tasks: list[SweTask] = []

        # Process events through pipeline stages
        pending_tasks: list[asyncio.Task] = []

        async def process_event(event: GhArchiveEvent) -> SweTask | None:
            """Process a single event through all stages."""
            nonlocal completed_count

            # Check if we've reached the target
            if self.config.once:
                if self.config.difficulty_targets is not None:
                    if self.config.difficulty_targets.all_met(per_difficulty_counts):
                        return None
                elif completed_count >= self.config.max_tasks:
                    return None

            # Stage 1: Enrich
            enriched = await self._enrich_stage(event, enrich_sem, metrics)
            if enriched is None:
                return None

            # Basic validation
            if enriched.title == "Untitled change" or not enriched.merge_commit:
                return None

            if enriched.files_changed == 0:
                return None

            # Stage 2: Filter
            if not self._filter_stage(enriched, metrics):
                await self._emit_event(
                    event_queue,
                    SwePipelineEvent(
                        SwePipelineEventType.CANDIDATE_FILTERED,
                        {
                            "event_id": event.id,
                            "accepted": False,
                            "reasons": ["filter_rejected"],
                        },
                    ),
                )
                return None

            # Emit filter passed
            await self._emit_event(
                event_queue,
                SwePipelineEvent(
                    SwePipelineEventType.CANDIDATE_FILTERED,
                    {"event_id": event.id, "accepted": True, "reasons": []},
                ),
            )

            # Check again after filter
            if self.config.once:
                if self.config.difficulty_targets is not None:
                    if self.config.difficulty_targets.all_met(per_difficulty_counts):
                        return None
                elif completed_count >= self.config.max_tasks:
                    return None

            # Stage 3: Pre-classify
            difficulty = await self._preclassify_stage(
                enriched, preclassify_sem, metrics
            )

            # Apply difficulty filter
            if not self._difficulty_matches_filter(difficulty, self.config):
                logger.debug(
                    "Difficulty %s does not match filter for %s#%d",
                    difficulty,
                    enriched.repo,
                    enriched.number,
                )
                return None

            # Check per-difficulty quota
            if self.config.difficulty_targets is not None and difficulty:
                current = per_difficulty_counts.get(difficulty, 0)
                quota = self.config.difficulty_targets.targets.get(difficulty, 0)
                if quota > 0 and current >= quota:
                    return None

            # Stage 4: Deep processing
            task = await self._deep_stage(enriched, deep_sem, metrics, event_queue)
            if task is None:
                return None

            if task.quality_passed:
                # Update task status
                task.status = SweTaskStatus.READY
                metrics.accepted_count += 1

                # Track language
                if task.language:
                    metrics.languages[task.language] = (
                        metrics.languages.get(task.language, 0) + 1
                    )

                # Track per-difficulty
                if difficulty:
                    per_difficulty_counts[difficulty] = (
                        per_difficulty_counts.get(difficulty, 0) + 1
                    )

                completed_count += 1

                return task

            return None

        # Run all events concurrently (semaphores control actual concurrency)
        for event in events:
            task = asyncio.create_task(process_event(event))
            pending_tasks.append(task)

        # Wait for all tasks and yield events
        for coro in asyncio.as_completed(pending_tasks):
            try:
                result = await coro
                if result is not None:
                    accepted_tasks.append(result)
                    yield SwePipelineEvent(
                        SwePipelineEventType.TASK_EXTRACTED,
                        {"task": result, "task_id": result.id},
                    )

                    # Check if we should stop
                    if self.config.once:
                        if self.config.difficulty_targets is not None:
                            if self.config.difficulty_targets.all_met(
                                per_difficulty_counts
                            ):
                                break
                        elif completed_count >= self.config.max_tasks:
                            break
            except Exception as e:
                logger.warning("Event processing failed: %s", e)

        # Calculate timing metrics
        end_time = datetime.now(timezone.utc)
        metrics.total_processing_time_ms = (
            end_time - start_time
        ).total_seconds() * 1000

        if metrics.enriched_count > 0:
            metrics.avg_per_pr_time_ms = (
                metrics.total_processing_time_ms / metrics.enriched_count
            )
            metrics.throughput_prs_per_sec = metrics.enriched_count / max(
                metrics.total_processing_time_ms / 1000.0, 0.001
            )

        # Yield remaining queued events
        while not event_queue.empty():
            yield event_queue.get_nowait()

        # Emit completion
        yield SwePipelineEvent(
            SwePipelineEventType.PIPELINE_COMPLETED,
            {"emitted": len(accepted_tasks), "metrics": metrics},
        )

    async def _fetch_events(self, hours_back: int) -> list[GhArchiveEvent]:
        """Fetch events from GH Archive.

        Subclasses can override this for testing.
        """
        from datetime import timedelta

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(hours=hours_back)

        return await self.gh_archive_client.fetch_merged_prs(
            start_date, end_date, skip_missing=True
        )


async def run_pipeline_once(
    gh_client: GitHubClient,
    config: SwePipelineConfig | None = None,
) -> SwePipelineRunResult:
    """Convenience function to run the pipeline once.

    Args:
        gh_client: GitHub API client.
        config: Pipeline configuration.

    Returns:
        Pipeline run result with extracted tasks.
    """
    async with SwePipeline(gh_client, config=config or SwePipelineConfig()) as pipeline:
        return await pipeline.run()
