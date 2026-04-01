"""UNgh API client for unlimited GitHub access."""

import aiohttp
from typing import Optional
from dataclasses import dataclass

UNGH_BASE_URL = "https://ungh.cc"


@dataclass
class UnghRepo:
    """Repository info from UNgh."""

    id: int
    name: str
    owner: str
    description: str
    stars: int
    default_branch: str
    created_at: str
    updated_at: str


@dataclass
class UnghFile:
    """File info from UNgh."""

    path: str
    mode: str
    sha: str
    size: int


class UnghClient:
    """Async client for UNgh API (unlimited GitHub access)."""

    def __init__(self, timeout: int = 30):
        self.base_url = UNGH_BASE_URL
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self

    async def __aexit__(self, *args):
        if self._session:
            await self._session.close()

    async def get_repo(self, owner: str, repo: str) -> UnghRepo:
        """Get repository info."""
        url = f"{self.base_url}/repos/{owner}/{repo}"
        async with self._session.get(url) as resp:
            data = await resp.json()
            r = data["repo"]
            return UnghRepo(
                id=r["id"],
                name=r["name"],
                owner=owner,
                description=r.get("description", ""),
                stars=r.get("stars", 0),
                default_branch=r.get("defaultBranch", "main"),
                created_at=r.get("createdAt", ""),
                updated_at=r.get("updatedAt", ""),
            )

    async def get_files(
        self, owner: str, repo: str, branch: str = "main"
    ) -> list[UnghFile]:
        """Get repository file tree."""
        url = f"{self.base_url}/repos/{owner}/{repo}/files/{branch}"
        async with self._session.get(url) as resp:
            data = await resp.json()
            return [
                UnghFile(
                    path=f["path"], mode=f["mode"], sha=f["sha"], size=f.get("size", 0)
                )
                for f in data.get("files", [])
            ]

    async def get_readme(self, owner: str, repo: str) -> str:
        """Get README content."""
        url = f"{self.base_url}/repos/{owner}/{repo}/readme"
        async with self._session.get(url) as resp:
            data = await resp.json()
            return data.get("markdown", "")

    async def get_branches(self, owner: str, repo: str) -> list[str]:
        """Get repository branches."""
        url = f"{self.base_url}/repos/{owner}/{repo}/branches"
        async with self._session.get(url) as resp:
            data = await resp.json()
            return [b["name"] for b in data.get("branches", [])]

    async def get_ci_cd_files(
        self, owner: str, repo: str, branch: str = "main"
    ) -> dict[str, str]:
        """Get CI/CD config files from repository.

        Uses git clone since UNgh doesn't provide file contents directly.
        This is still efficient because git clone is local.
        """
        # Get file tree from UNgh
        files = await self.get_files(owner, repo, branch)
        file_paths = {f.path for f in files}

        # CI/CD patterns to check
        ci_patterns = [
            ".github/workflows/ci.yml",
            ".github/workflows/test.yml",
            ".github/workflows/main.yml",
            ".github/workflows/build.yml",
            ".gitlab-ci.yml",
            "Dockerfile",
            "Makefile",
            "pyproject.toml",
            "setup.py",
            "package.json",
            "Cargo.toml",
            "go.mod",
            "CMakeLists.txt",
            "configure.ac",
        ]

        # Find matching files
        found = {}
        for pattern in ci_patterns:
            if pattern in file_paths:
                # Use raw.githubusercontent.com for file content
                raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{pattern}"
                try:
                    async with self._session.get(raw_url) as resp:
                        if resp.status == 200:
                            found[pattern] = await resp.text()
                except Exception:
                    pass

        return found
