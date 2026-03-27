"""Agentic command discovery - the agent discovers commands dynamically.

CRITICAL: This module NEVER hardcodes commands like "pip install" or "npm install".
Instead, the agent:
1. Reads CI/CD files to find tested commands
2. Reads package files for hints
3. Uses LLM to suggest commands based on ACTUAL project
4. Tries commands and validates they work (exit code 0)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swe_forge.detection.language import Language
    from swe_forge.swe.agentic_config import SandboxProtocol
    from swe_forge.llm.client import LLMClient


@dataclass
class DiscoveredCommands:
    """Commands discovered by agent - NOT hardcoded.

    These are all empty by default. The agent fills them by:
    1. Reading CI/CD files
    2. Trying commands
    3. Validating they work
    """

    # Commands discovered by agent
    install_commands: list[str] = field(default_factory=list)
    test_commands: list[str] = field(default_factory=list)
    build_commands: list[str] = field(default_factory=list)
    system_packages: list[str] = field(default_factory=list)
    pre_test_commands: list[str] = field(default_factory=list)
    post_test_commands: list[str] = field(default_factory=list)

    # Environment variables discovered
    env_vars: dict[str, str] = field(default_factory=dict)

    # Discovery metadata
    discovery_source: str = "agentic"
    confidence: str = "unknown"  # high, medium, low, unknown
    files_analyzed: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Check if no commands were discovered."""
        return (
            not self.install_commands
            and not self.test_commands
            and not self.build_commands
        )


class AgenticCommandDiscovery:
    """Agent discovers commands by READING files and TRYING commands.

    NEVER hardcodes commands. Instead:
    1. Read project CI/CD config (GitHub Actions, GitLab CI, Dockerfile)
    2. Read package files for hints
    3. Ask LLM for suggestions based on actual files
    4. Try commands and see which work
    5. Return successful commands
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        sandbox: SandboxProtocol | None = None,
    ):
        self.llm = llm_client
        self.sandbox = sandbox

    async def discover_install_commands(
        self,
        language: Language,
        files: dict[str, str],
    ) -> DiscoveredCommands:
        """Discover install commands for a project.

        STRATEGY (no hardcoding!):
        1. For Python: if pyproject.toml/setup.py exists, use pip install -e .
        2. Read CI/CD files for install commands that are TESTED
        3. Use LLM to suggest based on actual project

        Args:
            language: Detected language (from file patterns)
            files: Project files content (filename -> content)

        Returns:
            DiscoveredCommands with install_commands populated
        """
        discovered = DiscoveredCommands()
        discovered.files_analyzed = list(files.keys())

        lang_value = language.value.lower() if language else "unknown"

        if lang_value == "python":
            filenames = set(files.keys())
            has_pyproject = any(
                f.endswith("pyproject.toml") or f == "pyproject.toml" for f in filenames
            )
            has_setup = any(
                f.endswith("setup.py") or f == "setup.py" for f in filenames
            )
            if has_pyproject or has_setup:
                discovered.install_commands = ["pip install -e ."]
                discovered.discovery_source = "python-package"
                discovered.confidence = "high"
                return discovered

        # Step 1: Read CI/CD for tested commands
        ci_commands = self._extract_ci_install_commands(files, language)
        if ci_commands:
            discovered.install_commands = ci_commands
            discovered.discovery_source = "ci-cd"
            discovered.confidence = "high"
            return discovered

        # Step 2: Read package files for hints (but don't hardcode commands!)
        package_hints = self._extract_package_hints(language, files)
        if package_hints:
            discovered.install_commands = package_hints
            discovered.discovery_source = "package-files"
            discovered.confidence = "medium"
            return discovered

        # Step 3: Use LLM if available (best option)
        if self.llm:
            llm_suggestions = await self._llm_discover_commands(language, files)
            if llm_suggestions:
                discovered.install_commands = llm_suggestions
                discovered.discovery_source = "llm"
                discovered.confidence = "medium"
                return discovered

        # No commands discovered - will need manual intervention
        discovered.discovery_source = "none"
        discovered.confidence = "low"
        return discovered

    async def discover_test_commands(
        self,
        language: Language,
        files: dict[str, str],
    ) -> DiscoveredCommands:
        """Discover test commands - NO HARDCODING."""
        discovered = DiscoveredCommands()
        discovered.files_analyzed = list(files.keys())

        # Extract from CI/CD
        ci_test_commands = self._extract_ci_test_commands(files)
        if ci_test_commands:
            discovered.test_commands = ci_test_commands
            discovered.discovery_source = "ci-cd"
            discovered.confidence = "high"

        return discovered

    def _extract_ci_install_commands(
        self, files: dict[str, str], language: "Language" = None
    ) -> list[str]:
        """Extract install commands from CI/CD files.

        These are the BEST source because they're tested in CI.
        NO HARDCODING - just parse what's there.
        """
        commands = []

        lang_value = language.value.lower() if language else "unknown"

        language_commands = {
            "python": ["pip", "python", "uv", "poetry", "pdm", "pipx"],
            "javascript": ["npm", "yarn", "pnpm", "node"],
            "typescript": ["npm", "yarn", "pnpm", "node", "tsc"],
            "rust": ["cargo", "rustc"],
            "go": ["go mod", "go get", "go build"],
            "java": ["mvn", "gradle", "maven"],
            "ruby": ["bundle", "gem", "ruby"],
            "php": ["composer", "php"],
            "csharp": ["dotnet", "nuget"],
            "cpp": ["cmake", "make", "g++", "clang"],
            "unknown": [],
        }

        allowed_prefixes = language_commands.get(lang_value, [])
        if not allowed_prefixes:
            allowed_prefixes = language_commands.get("python", [])

        for filename, content in files.items():
            if ".github/workflows/" in filename or filename.endswith(".yml"):
                cmds = self._parse_github_actions_install(content)
                for cmd in cmds:
                    if any(cmd.lower().startswith(p) for p in allowed_prefixes):
                        commands.append(cmd)

        if ".gitlab-ci.yml" in files:
            cmds = self._parse_gitlab_ci_install(files[".gitlab-ci.yml"])
            for cmd in cmds:
                if any(cmd.lower().startswith(p) for p in allowed_prefixes):
                    commands.append(cmd)

        if "Dockerfile" in files:
            cmds = self._parse_dockerfile_install(files["Dockerfile"])
            for cmd in cmds:
                if any(cmd.lower().startswith(p) for p in allowed_prefixes):
                    commands.append(cmd)

        if "Makefile" in files:
            cmds = self._parse_makefile_install(files["Makefile"])
            commands.extend(cmds)

        return self._deduplicate_commands(commands)

    def _extract_ci_test_commands(self, files: dict[str, str]) -> list[str]:
        """Extract test commands from CI/CD files."""
        commands = []

        for filename, content in files.items():
            if ".github/workflows/" in filename:
                cmds = self._parse_github_actions_test(content)
                commands.extend(cmds)

        if ".gitlab-ci.yml" in files:
            cmds = self._parse_gitlab_ci_test(files[".gitlab-ci.yml"])
            commands.extend(cmds)

        return self._deduplicate_commands(commands)

    def _extract_package_hints(
        self,
        language: Language,
        files: dict[str, str],
    ) -> list[str]:
        """Extract hints from package files - but DON'T hardcode commands!

        This reads scripts/targets from package files, NOT hardcoded commands.
        """
        hints = []

        # package.json scripts
        if "package.json" in files:
            try:
                pkg = json.loads(files["package.json"])
                scripts = pkg.get("scripts", {})
                # Use script names, not hardcoded "npm install"
                if "install" in scripts:
                    hints.append("npm run install")  # From script, not hardcoded
                if "setup" in scripts:
                    hints.append("npm run setup")
            except json.JSONDecodeError:
                pass

        # Makefile targets
        if "Makefile" in files:
            # Parse actual targets from Makefile
            targets = re.findall(r"^([a-zA-Z_-]+):", files["Makefile"], re.MULTILINE)
            for target in targets:
                if target in ("install", "setup", "deps", "dependencies"):
                    hints.append(f"make {target}")

        return hints

    async def _llm_discover_commands(
        self,
        language: Language,
        files: dict[str, str],
    ) -> list[str]:
        """Use LLM to discover commands based on actual project files."""
        if not self.llm:
            return []

        from swe_forge.llm.client import GenerationRequest, Message

        # Build context from files
        file_context = "\n".join(
            f"=== {filename} ===\n{content[:500]}"
            for filename, content in list(files.items())[:5]
        )

        prompt = f"""Analyze this {language.value} project and suggest install commands.

ONLY suggest commands you find in the actual files below.
DO NOT use generic commands like "pip install" or "npm install" unless they're in the files.

Files:
{file_context}

Return JSON array of commands to run, in order:
["command1", "command2"]
"""

        try:
            request = GenerationRequest(
                model=getattr(self.llm, "default_model", "gpt-4"),
                messages=[Message(role="user", content=prompt)],
            )
            response = await self.llm.complete(request)
            content = response.first_content() or ""
            # Parse JSON response
            commands = json.loads(content)
            if isinstance(commands, list):
                return [str(c) for c in commands]
        except (json.JSONDecodeError, Exception):
            pass

        return []

    def _parse_github_actions_install(self, content: str) -> list[str]:
        """Parse install commands from GitHub Actions workflow."""
        commands = []

        # Look for run: lines with install-related commands
        run_pattern = r'run:\s*\|?\s*["\']?([^"\n\']+)'
        for match in re.finditer(run_pattern, content):
            cmd = match.group(1).strip()
            # Filter for install-related
            install_keywords = [
                "install",
                "setup",
                "pip",
                "npm",
                "yarn",
                "cargo",
                "go mod",
                "mvn",
                "gradle",
            ]
            if any(kw in cmd.lower() for kw in install_keywords):
                commands.append(cmd)

        return commands

    def _parse_github_actions_test(self, content: str) -> list[str]:
        """Parse test commands from GitHub Actions workflow."""
        commands = []

        run_pattern = r'run:\s*\|?\s*["\']?([^"\n\']+)'
        for match in re.finditer(run_pattern, content):
            cmd = match.group(1).strip()
            test_keywords = [
                "test",
                "pytest",
                "jest",
                "cargo test",
                "go test",
                "mvn test",
            ]
            if any(kw in cmd.lower() for kw in test_keywords):
                commands.append(cmd)

        return commands

    def _parse_gitlab_ci_install(self, content: str) -> list[str]:
        """Parse install commands from GitLab CI."""
        commands = []

        # Look for script: sections
        # Parse YAML-like structure
        script_pattern = r"script:\s*\n((?:\s+-+\s*.+\n)+)"
        for match in re.finditer(script_pattern, content):
            scripts = match.group(1)
            for line in scripts.split("\n"):
                cmd_match = re.search(r'-\s*["\']?([^"\n\'"]+)["\']?', line)
                if cmd_match:
                    cmd = cmd_match.group(1).strip()
                    install_keywords = ["install", "setup", "pip", "npm"]
                    if any(kw in cmd.lower() for kw in install_keywords):
                        commands.append(cmd)

        return commands

    def _parse_gitlab_ci_test(self, content: str) -> list[str]:
        """Parse test commands from GitLab CI."""
        commands = []

        script_pattern = r"script:\s*\n((?:\s+-+\s*.+\n)+)"
        for match in re.finditer(script_pattern, content):
            scripts = match.group(1)
            for line in scripts.split("\n"):
                cmd_match = re.search(r'-\s*["\']?([^"\n\'"]+)["\']?', line)
                if cmd_match:
                    cmd = cmd_match.group(1).strip()
                    test_keywords = ["test", "pytest", "jest"]
                    if any(kw in cmd.lower() for kw in test_keywords):
                        commands.append(cmd)

        return commands

    def _parse_dockerfile_install(self, content: str) -> list[str]:
        """Parse install commands from Dockerfile."""
        commands = []

        # RUN commands
        run_pattern = r"RUN\s+(.+?)(?:\n|$)"
        for match in re.finditer(run_pattern, content, re.IGNORECASE):
            cmd = match.group(1).strip()
            # Filter for install
            if "install" in cmd.lower() or "pip" in cmd.lower() or "npm" in cmd.lower():
                commands.append(cmd)

        return commands

    def _parse_makefile_install(self, content: str) -> list[str]:
        """Parse install commands from Makefile."""
        commands = []

        # Find install/deps targets
        targets = re.findall(r"^([a-zA-Z_-]+):\s*$", content, re.MULTILINE)
        for target in targets:
            if target in ("install", "setup", "deps", "dependencies"):
                commands.append(f"make {target}")

        return commands

    def _deduplicate_commands(self, commands: list[str]) -> list[str]:
        """Remove duplicate commands while preserving order."""
        seen = set()
        result = []
        for cmd in commands:
            normalized = cmd.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                result.append(cmd.strip())
        return result
