"""GitHub Source Orchestrator - Pre-filtering and task creation."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import aiohttp
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from .models import OrchestratorTask

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class GitHubRepo:
    full_name: str
    url: str
    stars: int
    language: str = "unknown"
    last_updated: str = ""
    topics: list[str] = field(default_factory=list)
    archived: bool = False
    fork: bool = False
    description: str = ""


@dataclass
class PreFilterConfig:
    min_stars: int = 100
    languages: list[str] | None = None
    min_complexity: float = 0.25
    max_complexity: float = 1.0
    exclude_archived: bool = True
    exclude_forks: bool = True


class GitHubSourceOrchestrator:
    def __init__(
        self,
        github_token: str | None = None,
        pre_filter: PreFilterConfig | None = None,
    ):
        self.github_token = github_token
        self.pre_filter = pre_filter or PreFilterConfig()
        self._session: aiohttp.ClientSession | None = None
        self._stats = {"scanned": 0, "passed": 0, "rejected": 0}

    async def __aenter__(self) -> "GitHubSourceOrchestrator":
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))
        return self

    async def __aexit__(self, *args) -> None:
        if self._session:
            await self._session.close()

    async def scan_repos(self, query: str, limit: int = 100) -> list[GitHubRepo]:
        return await self._scan_github(query, limit)

    def pre_filter_task(self, task: OrchestratorTask, repo: GitHubRepo) -> bool:
        return self._pre_filter_check(task, repo)

    def create_tasks_from_repos(
        self, repos: list[GitHubRepo]
    ) -> list[OrchestratorTask]:
        tasks = []
        for repo in repos:
            task = self._create_task_from_repo(repo)
            if task and self._pre_filter_check(task, repo):
                tasks.append(task)
        return tasks

    async def scan_and_filter(
        self,
        repos: list[GitHubRepo] | None = None,
        query: str = "",
        limit: int = 100,
    ) -> list[OrchestratorTask]:
        if repos is None:
            repos = await self.scan_repos(query, limit)

        self._stats["scanned"] = len(repos)
        tasks = self.create_tasks_from_repos(repos)
        self._stats["passed"] = len(tasks)
        self._stats["rejected"] = len(repos) - len(tasks)

        logger.info(
            f"Pre-filtered: {self._stats['passed']}/{self._stats['scanned']} passed"
        )
        return tasks

    async def _scan_github(self, query: str, limit: int) -> list[GitHubRepo]:
        if not self._session:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60)
            )

        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"

        repos = []
        page = 1
        per_page = 100

        while len(repos) < limit:
            url = (
                f"https://api.github.com/search/repositories"
                f"?q={query}&sort=stars&order=desc&page={page}&per_page={min(per_page, limit - len(repos))}"
            )

            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=1, min=1, max=10),
                ):
                    with attempt:
                        async with self._session.get(url, headers=headers) as response:
                            if response.status == 403:
                                reset = int(
                                    response.headers.get(
                                        "X-RateLimit-Reset", time.time() + 60
                                    )
                                )
                                wait_time = max(1, reset - time.time())
                                logger.warning(f"Rate limited, waiting {wait_time}s")
                                await asyncio.sleep(wait_time)
                                continue

                            if response.status != 200:
                                logger.error(f"GitHub API error: {response.status}")
                                break

                            data = await response.json()
                            items = data.get("items", [])

                            if not items:
                                break

                            for item in items:
                                repos.append(
                                    GitHubRepo(
                                        full_name=item["full_name"],
                                        url=item["html_url"],
                                        stars=item["stargazers_count"],
                                        language=item.get("language") or "unknown",
                                        last_updated=item.get("updated_at", ""),
                                        topics=item.get("topics", []),
                                        archived=item.get("archived", False),
                                        fork=item.get("fork", False),
                                        description=item.get("description", ""),
                                    )
                                )

                            page += 1
            except Exception as e:
                logger.error(f"GitHub scan failed: {e}")
                break

        return repos[:limit]

    def _create_task_from_repo(self, repo: GitHubRepo) -> OrchestratorTask | None:
        return OrchestratorTask(
            task_id=f"{repo.full_name.replace('/', '-')}-0",
            repo_url=f"{repo.url}.git",
            language=repo.language.lower() if repo.language else "unknown",
            metadata={
                "stars": repo.stars,
                "topics": repo.topics,
                "description": repo.description,
            },
        )

    def _pre_filter_check(self, task: OrchestratorTask, repo: GitHubRepo) -> bool:
        if self.pre_filter.languages:
            if task.language not in [l.lower() for l in self.pre_filter.languages]:
                logger.debug(f"Rejected {repo.full_name}: language={task.language}")
                return False

        if repo.stars < self.pre_filter.min_stars:
            logger.debug(f"Rejected {repo.full_name}: stars={repo.stars}")
            return False

        if self.pre_filter.exclude_archived and repo.archived:
            logger.debug(f"Rejected {repo.full_name}: archived")
            return False

        if self.pre_filter.exclude_forks and repo.fork:
            logger.debug(f"Rejected {repo.full_name}: fork")
            return False

        return True
