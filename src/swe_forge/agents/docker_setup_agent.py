"""Agentic Docker environment setup -- zero hardcoding.

An LLM agent gets a fresh Ubuntu container, explores the repo, installs
everything needed, then validates fail_to_pass tests before/after patch.
The working container is committed as a Docker image and optionally pushed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from swe_forge.llm import LLMClient

logger = logging.getLogger(__name__)

MAX_TURNS = 100
SHELL_TIMEOUT = 300
CONTAINER_LIFETIME = 3600

SYSTEM_PROMPT = """\
You are a Docker environment setup agent. You have a fresh Ubuntu 24.04 container
with a git repository cloned at /repo (already checked out at the correct base commit).

The workspace also contains:
- /workspace/patch.diff  -- the ground-truth patch
- /workspace/tests/      -- generated test files

YOUR TASK (in order):
1. Explore the repo to understand the language and build system
   (look for setup.py, pyproject.toml, package.json, Cargo.toml, go.mod, Makefile, etc.)
2. Install the language runtime and ALL dependencies so the tests can run.
   Use shell() for everything: apt-get, pip, npm, cargo, go, etc.
3. Copy test files to the correct location in the repo if needed.
4. Run the fail_to_pass test commands. They MUST FAIL on the base commit
   (this proves the tests detect the bug). If they pass, something is wrong.
5. Apply the patch:  cd /repo && git apply /workspace/patch.diff
6. Run the fail_to_pass test commands again. They MUST PASS after the patch.
7. Call done(success=true) when both validations succeed.

RULES:
- You have {max_turns} turns maximum. Be efficient.
- Always use shell() to run commands. Read output carefully before next step.
- If a command fails, diagnose and fix it (install missing dep, etc.)
- If you cannot make it work after several tries, call done(success=false, error="reason").
- NEVER skip validation. Both pre-patch FAIL and post-patch PASS are required.
- The test commands come from the workspace.yaml fail_to_pass field.

FAIL_TO_PASS COMMANDS:
{fail_to_pass}

INSTALL COMMANDS (hints from test generator, may need adaptation):
{install_commands}

LANGUAGE: {language}
REPO: {repo_url}
"""

def _get_tools():
    from swe_forge.llm.client import ToolDefinition
    return [
        ToolDefinition.create(
            "shell",
            "Execute a shell command in the Docker container. Returns stdout+stderr and exit code.",
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 300)"},
                },
                "required": ["command"],
            },
        ),
        ToolDefinition.create(
            "read_file",
            "Read a file from the container.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path in the container"},
                },
                "required": ["path"],
            },
        ),
        ToolDefinition.create(
            "done",
            "Signal that setup is complete. Call with success=true when fail_to_pass FAIL on base AND PASS after patch.",
            {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "error": {"type": "string", "description": "Error description if success=false"},
                },
                "required": ["success"],
            },
        ),
    ]


def _docker_exec(container: str, cmd: str, timeout: int = SHELL_TIMEOUT) -> tuple[int, str]:
    """Run a command in the container, return (exit_code, combined_output)."""
    try:
        result = subprocess.run(
            ["docker", "exec", container, "bash", "-lc", cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        output = (result.stdout + result.stderr).strip()
        # Truncate large output
        if len(output) > 12000:
            output = output[:4000] + "\n\n... [truncated] ...\n\n" + output[-4000:]
        return result.returncode, output
    except subprocess.TimeoutExpired:
        return -1, f"Command timed out after {timeout}s"
    except Exception as e:
        return -1, str(e)


class DockerSetupAgent:
    """LLM agent that sets up a Docker environment from scratch."""

    def __init__(self, llm_client: "LLMClient", *, model: str = "", max_turns: int = MAX_TURNS):
        self._llm = llm_client
        self._model = model
        self._max_turns = max_turns

    async def setup_and_verify(
        self,
        task_dir: Path,
        docker_user: str,
        *,
        push: bool = False,
    ) -> str | None:
        """Set up Docker env, verify tests, commit image, optionally push.

        Returns image_name on success, None on failure.
        """
        task_dir = Path(task_dir)
        ws_path = task_dir / "workspace.yaml"
        if not ws_path.exists():
            logger.warning("No workspace.yaml in %s", task_dir)
            return None

        workspace = yaml.safe_load(ws_path.read_text())
        task_id = workspace.get("task_id", task_dir.name)
        repo_info = workspace.get("repo", {})
        repo_url = repo_info.get("url", "")
        base_commit = repo_info.get("base_commit", "")
        language = workspace.get("language", "unknown")
        f2p = workspace.get("tests", {}).get("fail_to_pass", [])
        install_cmds = workspace.get("install", {}).get("commands", [])

        if not f2p:
            logger.warning("No fail_to_pass for %s, skipping", task_id)
            return None

        image_name = f"{docker_user}/swe-forge:{task_id}"
        container = f"swe-setup-{uuid.uuid4().hex[:8]}"

        try:
            # 1. Start container
            subprocess.run(
                ["docker", "run", "-d", "--name", container,
                 "--memory=4g", "--cpus=2",
                 "ubuntu:24.04", "sleep", str(CONTAINER_LIFETIME)],
                capture_output=True, text=True, check=True,
            )

            # 2. Install git + clone repo
            _docker_exec(container, "apt-get update && apt-get install -y git ca-certificates curl", timeout=120)
            exit_code, out = _docker_exec(container, f"git clone {repo_url} /repo", timeout=300)
            if exit_code != 0:
                logger.warning("Clone failed for %s: %s", task_id, out[:200])
                return None
            _docker_exec(container, f"cd /repo && git checkout {base_commit}", timeout=60)

            # 3. Copy patch + tests into container
            _docker_exec(container, "mkdir -p /workspace/tests")
            patch_src = task_dir / "patch.diff"
            tests_src = task_dir / "tests"
            if patch_src.exists():
                subprocess.run(["docker", "cp", str(patch_src), f"{container}:/workspace/patch.diff"],
                               capture_output=True, timeout=30)
            if tests_src.exists():
                subprocess.run(["docker", "cp", str(tests_src) + "/.", f"{container}:/workspace/tests/"],
                               capture_output=True, timeout=30)

            # 4. Symlink /workspace/repo -> /repo
            _docker_exec(container, "ln -sf /repo /workspace/repo")

            # 5. Run LLM agent loop
            success = await self._run_agent_loop(
                container, task_id, repo_url, language, f2p, install_cmds,
            )

            if not success:
                logger.warning("Agent failed for %s", task_id)
                return None

            # 6. Commit container as image
            logger.info("Committing container %s as %s", container, image_name)
            subprocess.run(
                ["docker", "commit", container, image_name],
                capture_output=True, text=True, check=True, timeout=120,
            )

            # 7. Push if requested
            if push:
                logger.info("Pushing %s", image_name)
                push_result = subprocess.run(
                    ["docker", "push", image_name],
                    capture_output=True, text=True, timeout=600,
                )
                if push_result.returncode != 0:
                    logger.warning("Push failed for %s: %s", task_id, push_result.stderr[:300])
                    return image_name  # Built but not pushed
                logger.info("Pushed %s", image_name)

            # Update workspace.yaml
            workspace["environment"]["image"] = image_name
            ws_path.write_text(yaml.dump(workspace, default_flow_style=False, sort_keys=False))

            return image_name

        except Exception as e:
            logger.error("DockerSetupAgent error for %s: %s", task_id, e)
            return None
        finally:
            subprocess.run(["docker", "rm", "-f", container], capture_output=True, timeout=30)

    async def _run_agent_loop(
        self,
        container: str,
        task_id: str,
        repo_url: str,
        language: str,
        fail_to_pass: list[str],
        install_commands: list[str],
    ) -> bool:
        """Run the LLM agent loop to set up the environment."""
        from swe_forge.llm.client import GenerationRequest, Message

        system = SYSTEM_PROMPT.format(
            max_turns=self._max_turns,
            fail_to_pass="\n".join(f"  - {cmd}" for cmd in fail_to_pass),
            install_commands="\n".join(f"  - {cmd}" for cmd in install_commands) if install_commands else "  (none provided -- figure it out)",
            language=language,
            repo_url=repo_url,
        )

        tools = _get_tools()
        messages: list[Message] = [
            Message.system(system),
            Message.user(
                "Set up this repo and validate the tests. Start by exploring the repo structure "
                "to understand what language/framework is used, then install dependencies."
            ),
        ]

        for turn in range(self._max_turns):
            request = GenerationRequest(
                model=self._model,
                messages=messages,
                tools=tools,
                temperature=0.2,
                max_tokens=4096,
            )
            response = await self._llm.complete_with_tools(request)

            msg = response.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                if msg.content and ("done" in msg.content.lower() or "success" in msg.content.lower()):
                    messages.append(Message.user("You must call the done() tool to finish."))
                else:
                    messages.append(Message.user("Continue. Use tools to make progress."))
                continue

            for tc in msg.tool_calls:
                fn = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                if fn == "shell":
                    cmd = args.get("command", "echo no command")
                    timeout = args.get("timeout", SHELL_TIMEOUT)
                    exit_code, output = _docker_exec(container, cmd, timeout=timeout)
                    tool_result = f"exit_code={exit_code}\n{output}"
                    logger.debug("[%s] shell(%s): exit=%d", task_id, cmd[:80], exit_code)

                elif fn == "read_file":
                    path = args.get("path", "")
                    exit_code, output = _docker_exec(container, f"cat '{path}'", timeout=30)
                    tool_result = output if exit_code == 0 else f"Error reading {path}: {output}"

                elif fn == "done":
                    success = args.get("success", False)
                    error = args.get("error", "")
                    if success:
                        logger.info("[%s] Agent reported SUCCESS at turn %d", task_id, turn + 1)
                    else:
                        logger.warning("[%s] Agent reported FAILURE at turn %d: %s", task_id, turn + 1, error)
                    return success

                else:
                    tool_result = f"Unknown tool: {fn}"

                messages.append(Message(role="tool", content=tool_result[:8000], tool_call_id=tc.id))

        logger.warning("[%s] Agent exhausted %d turns", task_id, self._max_turns)
        return False
