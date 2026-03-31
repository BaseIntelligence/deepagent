"""Full Pipeline Orchestrator - End-to-end GitHub to HuggingFace pipeline."""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .dataset_orchestrator import DatasetOrchestrator
from .github_source_orchestrator import GitHubSourceOrchestrator, PreFilterConfig
from .models import OrchestratorStats, OrchestratorTask, TaskState

logger = logging.getLogger(__name__)


class FullPipelineOrchestrator:
    """Full pipeline from GitHub to HuggingFace dataset.

    Orchestrates:
    1. GitHub scanning with pre-filtering
    2. Test generation (LLM agent)
    3. Docker verification (FAIL→PASS)
    4. Repair (if needed)
    5. Scoring
    6. Publishing to HuggingFace

    Usage:
        async with FullPipelineOrchestrator(config) as orchestrator:
            stats = await orchestrator.run("language:python", target_count=100)
    """

    def __init__(
        self,
        github_token: str | None = None,
        openrouter_api_key: str | None = None,
        hf_token: str | None = None,
        pre_filter: PreFilterConfig | None = None,
        parallel_workers: int = 5,
        min_score_threshold: float = 0.5,
        max_repair_attempts: int = 5,
        model: str = "openai/gpt-4o-mini",
        dataset_name: str = "CortexLM/swe-forge",
        skip_generation: bool = False,
        use_existing_image: bool = False,
        progress_file: str | Path | None = None,
    ):
        """Initialize the FullPipelineOrchestrator.

        Args:
            github_token: GitHub API token for repository scanning.
            openrouter_api_key: OpenRouter API key for LLM operations.
            hf_token: HuggingFace API token for dataset publishing.
            pre_filter: Pre-filtering configuration.
            parallel_workers: Number of parallel workers for task processing.
            min_score_threshold: Minimum score threshold for publishing.
            max_repair_attempts: Maximum repair attempts before rejecting.
            model: LLM model to use for test generation and repair.
            dataset_name: Name of the HuggingFace dataset to publish to.
            skip_generation: Whether to skip test generation phase.
            use_existing_image: Whether to use existing Docker images.
            progress_file: Path to the progress file for resume capability.
        """
        self.github_token = github_token
        self.openrouter_api_key = openrouter_api_key
        self.hf_token = hf_token
        self.pre_filter = pre_filter or PreFilterConfig()
        self.parallel_workers = parallel_workers
        self.min_score_threshold = min_score_threshold
        self.max_repair_attempts = max_repair_attempts
        self.model = model
        self.dataset_name = dataset_name
        self.skip_generation = skip_generation
        self.use_existing_image = use_existing_image
        self.progress_file = Path(progress_file) if progress_file else None

        self._stats = OrchestratorStats()
        self._processed_ids: set[str] = set()

    async def __aenter__(self) -> "FullPipelineOrchestrator":
        """Enter async context manager, loading progress if exists."""
        if self.progress_file and self.progress_file.exists():
            await self._load_progress()
        return self

    async def __aexit__(self, *args) -> None:
        """Exit async context manager, saving progress."""
        await self._save_progress()

    async def run_from_github(
        self,
        query: str = "language:python stars:>100",
        target_count: int = 100,
        limit: int = 1000,
    ) -> OrchestratorStats:
        """Run full pipeline from GitHub search.

        Args:
            query: GitHub search query.
            target_count: Target number of valid tasks.
            limit: Maximum GitHub repos to scan.

        Returns:
            OrchestratorStats with results.
        """
        logger.info(f"Starting pipeline: query='{query}', target={target_count}")

        # Phase 1: GitHub scan + pre-filter
        async with GitHubSourceOrchestrator(
            github_token=self.github_token,
            pre_filter=self.pre_filter,
        ) as github_orchestrator:
            tasks = await github_orchestrator.scan_and_filter(
                query=query,
                limit=limit,
            )

        logger.info(f"Pre-filtering complete: {len(tasks)} tasks")

        # Filter out already processed
        tasks = [t for t in tasks if t.task_id not in self._processed_ids]

        # Phase 2: Process tasks
        return await self._process_tasks(tasks, target_count)

    async def run_from_tasks_dir(
        self,
        tasks_dir: str | Path,
        target_count: int = 100,
    ) -> OrchestratorStats:
        """Run pipeline on existing tasks directory.

        Args:
            tasks_dir: Directory with workspace.yaml files.
            target_count: Target number of valid tasks.

        Returns:
            OrchestratorStats with results.
        """
        import yaml

        tasks_dir = Path(tasks_dir)
        tasks = []

        for task_dir in sorted(tasks_dir.iterdir()):
            wp = task_dir / "workspace.yaml"
            if not wp.exists():
                continue

            with open(wp) as f:
                w = yaml.safe_load(f)

            patch_file = task_dir / "patch.diff"
            patch = patch_file.read_text() if patch_file.exists() else ""

            task = OrchestratorTask(
                task_id=w.get("task_id", task_dir.name),
                repo_url=w.get("repo", {}).get("url", ""),
                base_commit=w.get("repo", {}).get("base_commit", ""),
                merge_commit=w.get("repo", {}).get("merge_commit", ""),
                patch=patch,
                language=w.get("language", "python"),
                docker_image=w.get("docker", {}).get("image", ""),
                difficulty_score=float(w.get("difficulty_score", 0)) / 10.0,
                prompt=w.get("prompt", ""),
            )

            tests = w.get("tests", {})
            task.tests = {
                "fail_to_pass": tests.get("fail_to_pass", []),
                "pass_to_pass": tests.get("pass_to_pass", []),
            }
            task.install_commands = w.get("install", {}).get("commands", [])

            if task.task_id not in self._processed_ids:
                tasks.append(task)

        logger.info(f"Loaded {len(tasks)} tasks from {tasks_dir}")

        # Force skip generation for existing tasks
        self.skip_generation = True
        self.use_existing_image = True

        return await self._process_tasks(tasks, target_count)

    async def _process_tasks(
        self,
        tasks: list[OrchestratorTask],
        target_count: int,
    ) -> OrchestratorStats:
        """Process tasks through pipeline with parallel workers."""
        semaphore = asyncio.Semaphore(self.parallel_workers)
        completed_count = 0

        self._stats.total_tasks = len(tasks)

        async def worker(
            task: OrchestratorTask, worker_id: int
        ) -> OrchestratorTask | Exception:
            async with semaphore:
                try:
                    orchestrator = DatasetOrchestrator(
                        orchestrator_id=worker_id,
                        hf_token=self.hf_token,
                        min_score_threshold=self.min_score_threshold,
                        max_repair_attempts=self.max_repair_attempts,
                        model=self.model,
                        skip_generation=self.skip_generation,
                        use_existing_image=self.use_existing_image,
                    )
                    result = await orchestrator.run_pipeline(task)
                    self._processed_ids.add(task.task_id)
                    return result
                except Exception as e:
                    logger.error(f"Worker {worker_id} crashed: {e}")
                    return e

        # Process in batches until target reached
        for batch_start in range(0, len(tasks), self.parallel_workers * 2):
            batch = tasks[batch_start : batch_start + self.parallel_workers * 2]

            results = await asyncio.gather(
                *[worker(task, i) for i, task in enumerate(batch)],
                return_exceptions=False,
            )

            for result in results:
                if isinstance(result, OrchestratorTask):
                    if result.state == TaskState.COMPLETED:
                        completed_count += 1
                        self._stats.state_counts[TaskState.COMPLETED] = (
                            self._stats.state_counts.get(TaskState.COMPLETED, 0) + 1
                        )
                    elif result.state == TaskState.REJECTED:
                        self._stats.state_counts[TaskState.REJECTED] = (
                            self._stats.state_counts.get(TaskState.REJECTED, 0) + 1
                        )
                    elif result.state == TaskState.FAILED:
                        self._stats.state_counts[TaskState.FAILED] = (
                            self._stats.state_counts.get(TaskState.FAILED, 0) + 1
                        )

            logger.info(
                f"Progress: {completed_count}/{target_count} completed, "
                f"{len(self._processed_ids)}/{len(tasks)} processed"
            )

            # Save progress periodically
            if self.progress_file and len(self._processed_ids) % 10 == 0:
                await self._save_progress()

            # Check if target reached
            if completed_count >= target_count:
                logger.info(f"Target reached: {completed_count}/{target_count}")
                break

        return self._stats

    async def _load_progress(self) -> None:
        """Load progress from file."""
        if not self.progress_file or not self.progress_file.exists():
            return

        try:
            with open(self.progress_file) as f:
                data = json.load(f)
            self._processed_ids = set(data.get("processed_ids", []))
            logger.info(
                f"Loaded progress: {len(self._processed_ids)} already processed"
            )
        except Exception as e:
            logger.warning(f"Failed to load progress: {e}")

    async def _save_progress(self) -> None:
        """Save progress to file."""
        if not self.progress_file:
            return

        try:
            data = {
                "processed_ids": list(self._processed_ids),
                "stats": {
                    "total": self._stats.total_tasks,
                    "completed": self._stats.state_counts.get(TaskState.COMPLETED, 0),
                    "rejected": self._stats.state_counts.get(TaskState.REJECTED, 0),
                    "failed": self._stats.state_counts.get(TaskState.FAILED, 0),
                    "updated_at": datetime.now().isoformat(),
                },
            }
            with open(self.progress_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save progress: {e}")

    def __repr__(self) -> str:
        """Return string representation of the orchestrator."""
        return (
            f"FullPipelineOrchestrator("
            f"parallel={self.parallel_workers}, "
            f"min_score={self.min_score_threshold}, "
            f"max_repair={self.max_repair_attempts}, "
            f"dataset={self.dataset_name}"
            f")"
        )
