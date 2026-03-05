#!/usr/bin/env python3
"""Fix install commands for existing swe-forge HuggingFace tasks.

Downloads workspace.yaml from CortexLM/swe-forge, spins up Docker containers,
runs an LLM install agent to produce working install commands, verifies tests
pass, then uploads corrected workspace.yaml back to HuggingFace.

Usage:
    export OPENROUTER_API_KEY=sk-...
    export HF_TOKEN=hf_...  # write access to CortexLM/swe-forge
    python3 scripts/fix_tasks.py [--parallel 3] [--task Owner/repo-123] [--dry-run]
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import yaml

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai pyyaml huggingface_hub")
    sys.exit(1)

try:
    from huggingface_hub import HfApi, hf_hub_download, list_repo_tree
except ImportError:
    print("pip install huggingface_hub")
    sys.exit(1)

DATASET_ID = "CortexLM/swe-forge"
DOCKER_IMAGE = "python:3.12-slim"
MAX_AGENT_TURNS = 200
MAX_FRESH_CYCLES = 5
INSTALL_TIMEOUT = 300
TEST_TIMEOUT = 120

SYSTEM_PROMPT = """You are a DevOps agent. Your job is to install all dependencies for a software project in a fresh Docker container (python:3.12-slim with only git and python3 pre-installed).

You have these tools:
- `shell`: Execute shell commands. Returns stdout, stderr, exit code.
- `read_file`: Read a file. Args: {"file_path": "..."}
- `list_dir`: List directory. Args: {"directory_path": "."}
- `submit_install`: Submit final working install commands.

WORKFLOW:
1. Explore the repo to determine installation procedure (README, package.json, go.mod, etc.)
2. Install the runtime if needed (Node.js, Go, Rust, Java -- Python is already available)
3. Install project dependencies. If a command fails, read the error, fix it, retry.
4. Also verify the test infrastructure works: run the pass_to_pass test command.
5. Once everything works, call submit_install with ONLY commands that succeeded (exit 0).

IMPORTANT:
- Only include commands that exited with code 0.
- Commands will be replayed in a BRAND NEW container from scratch.
- Include apt-get for system dependencies.
- Include runtime installation if needed.
- Do NOT include exploratory commands -- only install commands.
- Make sure the test runner (pytest, npm test, go test, etc.) is available after install."""

# ── Docker helpers ──────────────────────────────────────────────────────

def docker_exec(container: str, cmd: str, timeout: int = 60) -> tuple[int, str, str]:
    try:
        r = subprocess.run(
            ["docker", "exec", container, "bash", "-c", cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode, r.stdout[-4000:], r.stderr[-2000:]
    except subprocess.TimeoutExpired:
        return -1, "", f"timed out after {timeout}s"
    except Exception as e:
        return -1, "", str(e)


def docker_rm(container: str):
    try:
        subprocess.run(["docker", "rm", "-f", container],
                       capture_output=True, timeout=30)
    except Exception:
        pass


def docker_start(container: str, repo: str, base_commit: str) -> bool:
    docker_rm(container)
    r = subprocess.run([
        "docker", "run", "-d", "--name", container,
        "--network=host", "--memory=16g", "-w", "/repo",
        DOCKER_IMAGE, "sleep", "7200",
    ], capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        print(f"  [ERR] Container start failed: {r.stderr[:200]}")
        return False

    # Install git
    code, _, err = docker_exec(container,
        "apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1", 120)
    if code != 0:
        print(f"  [ERR] git install failed: {err[:200]}")
        return False

    # Clone
    code, _, err = docker_exec(container,
        f"git clone https://github.com/{repo}.git /repo 2>&1", 600)
    if code != 0:
        print(f"  [ERR] Clone failed: {err[:200]}")
        return False

    # Checkout
    if base_commit:
        code, _, err = docker_exec(container,
            f"cd /repo && git checkout {base_commit} --force 2>&1", 60)
        if code != 0:
            print(f"  [ERR] Checkout failed: {err[:200]}")
            return False

    return True


# ── LLM agent ──────────────────────────────────────────────────────────

@dataclass
class FixResult:
    task_id: str
    success: bool = False
    old_install: str = ""
    new_install: str = ""
    p2p_passed: bool = False
    error: str = ""
    turns: int = 0


def make_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Execute a shell command. Returns exit_code, stdout, stderr.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to execute"},
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from the repo.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": "List directory contents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory_path": {"type": "string", "description": "Default: ."},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_install",
                "description": "Submit final working install commands (only commands that exited 0).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "install_commands": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Complete, self-contained install commands.",
                        },
                    },
                    "required": ["install_commands"],
                },
            },
        },
    ]


def handle_tool_call(container: str, name: str, args: dict) -> str:
    if name == "shell":
        cmd = args.get("command", "")
        code, stdout, stderr = docker_exec(container, cmd, INSTALL_TIMEOUT)
        return f"exit_code: {code}\nstdout:\n{stdout[-3000:]}\nstderr:\n{stderr[-1500:]}"
    elif name == "read_file":
        fp = args.get("file_path", "")
        code, stdout, stderr = docker_exec(container, f"cat '{fp}' 2>&1", 10)
        return stdout[-4000:] if code == 0 else f"Error: {stderr[:500]}"
    elif name == "list_dir":
        d = args.get("directory_path", ".")
        code, stdout, _ = docker_exec(container, f"ls -la '{d}' 2>&1", 10)
        return stdout[-3000:]
    else:
        return f"Unknown tool: {name}"


def run_install_agent(
    client: OpenAI,
    model: str,
    container: str,
    task: dict,
) -> list[str] | None:
    """Run the LLM install agent. Returns list of install commands or None."""
    existing = task.get("install_config", {}).get("install", "")
    p2p_cmds = task.get("pass_to_pass", [])

    user_msg = (
        f"Repository: {task['repo']}\n"
        f"Language: {task.get('language', 'unknown')}\n"
        f"Existing install command (may not work): {existing or '(none)'}\n"
        f"Pass-to-pass test commands that must work after install: {json.dumps(p2p_cmds)}\n\n"
        f"The repo is cloned at /repo. Explore it, install everything, "
        f"verify the pass_to_pass test commands work, then submit."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    tools = make_tools()

    for turn in range(MAX_AGENT_TURNS):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=2000,
            )
        except Exception as e:
            print(f"    LLM error on turn {turn}: {e}")
            return None

        choice = resp.choices[0]
        msg = choice.message

        if msg.tool_calls:
            # Append assistant message with tool calls
            messages.append(msg.model_dump())

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                if fn_name == "submit_install":
                    cmds = fn_args.get("install_commands", [])
                    if not cmds:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "REJECTED: install_commands must not be empty.",
                        })
                        continue
                    print(f"    Agent submitted {len(cmds)} commands on turn {turn}")
                    return cmds

                result = handle_tool_call(container, fn_name, fn_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            continue

        # No tool calls
        if msg.content and msg.content.strip():
            messages.append({"role": "assistant", "content": msg.content})
            messages.append({
                "role": "user",
                "content": "Use the shell tool to install dependencies, then call submit_install.",
            })
            continue

        break

    print(f"    Agent exhausted {MAX_AGENT_TURNS} turns without submitting")
    return None


# ── Replay + verification ──────────────────────────────────────────────

def replay_and_verify(
    task: dict,
    install_commands: list[str],
    container_prefix: str,
) -> bool:
    """Spin up a fresh container, replay install commands, run p2p tests."""
    container = f"{container_prefix}-verify"
    try:
        if not docker_start(container, task["repo"], task.get("base_commit", "")):
            return False

        # Replay install commands one by one
        for cmd in install_commands:
            code, stdout, stderr = docker_exec(container, f"cd /repo && {cmd} 2>&1",
                                                INSTALL_TIMEOUT)
            if code != 0:
                print(f"    Replay failed: {cmd[:80]}... (exit {code})")
                return False

        # Write test files if present
        test_files = task.get("meta", {}).get("test_files", "")
        if test_files:
            try:
                files = json.loads(test_files)
                for tf in files:
                    path = tf["path"]
                    content = tf["content"]
                    # Create parent dirs and write file via stdin pipe
                    docker_exec(container, f"mkdir -p $(dirname '/repo/{path}')", 5)
                    p = subprocess.run(
                        ["docker", "exec", "-i", "-w", "/repo", container,
                         "bash", "-c", f"cat > '/repo/{path}'"],
                        input=content.encode(), capture_output=True, timeout=10,
                    )
            except (json.JSONDecodeError, KeyError):
                pass

        # Run pass_to_pass tests
        for cmd in task.get("pass_to_pass", []):
            code, stdout, stderr = docker_exec(container,
                f"cd /repo && {cmd} 2>&1", TEST_TIMEOUT)
            if code != 0:
                print(f"    p2p FAIL: {cmd[:80]}... (exit {code})")
                return False

        print(f"    Replay + p2p tests PASSED")
        return True

    finally:
        docker_rm(container)


# ── Main task fixer ────────────────────────────────────────────────────

def fix_task(
    client: OpenAI,
    model: str,
    task: dict,
    task_id: str,
) -> FixResult:
    """Fix a single task's install commands."""
    result = FixResult(task_id=task_id)
    result.old_install = task.get("install_config", {}).get("install", "")

    safe_name = task_id.replace("/", "-").replace(" ", "_")
    container = f"swe-fix-{safe_name}"

    for cycle in range(MAX_FRESH_CYCLES):
        print(f"  Cycle {cycle + 1}/{MAX_FRESH_CYCLES}")

        try:
            if not docker_start(container, task["repo"], task.get("base_commit", "")):
                result.error = "Container start failed"
                docker_rm(container)
                continue

            # Run install agent
            cmds = run_install_agent(client, model, container, task)
            docker_rm(container)

            if not cmds:
                result.error = "Agent failed to produce commands"
                continue

            combined = " && ".join(cmds)
            print(f"    Install: {combined[:120]}...")

            # Replay in fresh container to verify
            if replay_and_verify(task, cmds, f"swe-fix-{safe_name}"):
                result.success = True
                result.new_install = combined
                result.p2p_passed = True
                result.turns = cycle + 1
                return result

            # Replay failed -- update task with new commands and retry
            task.setdefault("install_config", {})["install"] = combined
            print(f"    Replay failed, retrying with agent...")

        except Exception as e:
            docker_rm(container)
            result.error = str(e)
            traceback.print_exc()

    result.error = result.error or f"Failed after {MAX_FRESH_CYCLES} cycles"
    return result


# ── HuggingFace I/O ───────────────────────────────────────────────────

def list_all_tasks(api: HfApi) -> list[str]:
    """List all task IDs from the dataset."""
    import urllib.request
    tasks = []
    # List org directories via HTTP API
    orgs_url = f"https://huggingface.co/api/datasets/{DATASET_ID}/tree/main/tasks"
    orgs = json.loads(urllib.request.urlopen(orgs_url, timeout=30).read())
    for org in orgs:
        org_path = org["path"]  # e.g. "tasks/CrackinLLC"
        try:
            sub_url = f"https://huggingface.co/api/datasets/{DATASET_ID}/tree/main/{org_path}"
            subs = json.loads(urllib.request.urlopen(sub_url, timeout=30).read())
        except Exception:
            continue
        for sub in subs:
            sub_path = sub["path"]  # e.g. "tasks/CrackinLLC/Photo-Export-Fixer-15"
            if sub_path.startswith("tasks/"):
                task_id = sub_path[len("tasks/"):]
                if "/" in task_id:
                    tasks.append(task_id)
    return sorted(tasks)


def download_task(api: HfApi, task_id: str, local_dir: str) -> dict | None:
    """Download workspace.yaml for a task."""
    try:
        path = hf_hub_download(
            repo_id=DATASET_ID,
            filename=f"tasks/{task_id}/workspace.yaml",
            repo_type="dataset",
            local_dir=local_dir,
        )
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"  [ERR] Download failed for {task_id}: {e}")
        return None


def upload_fixed_task(api: HfApi, task_id: str, task: dict, local_dir: str):
    """Upload fixed workspace.yaml back to HuggingFace."""
    out_path = os.path.join(local_dir, "tasks", task_id, "workspace.yaml")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(task, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    api.upload_file(
        path_or_fileobj=out_path,
        path_in_repo=f"tasks/{task_id}/workspace.yaml",
        repo_id=DATASET_ID,
        repo_type="dataset",
        commit_message=f"fix: update install commands for {task_id}",
    )


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fix swe-forge task install commands")
    parser.add_argument("--task", help="Fix a single task (e.g. Owner/repo-123)")
    parser.add_argument("--parallel", type=int, default=8, help="Concurrent tasks")
    parser.add_argument("--dry-run", action="store_true", help="Don't upload to HF")
    parser.add_argument("--model", default="openai/gpt-4.1-mini",
                        help="LLM model (OpenRouter)")
    parser.add_argument("--output", default="/tmp/swe-forge-fixed",
                        help="Local output directory")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip tasks already marked llm-install-agent-fix")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token and not args.dry_run:
        print("ERROR: Set HF_TOKEN for upload (or use --dry-run)")
        sys.exit(1)

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    api = HfApi(token=hf_token)
    os.makedirs(args.output, exist_ok=True)

    # Get task list
    if args.task:
        task_ids = [args.task]
    else:
        print("Listing all tasks from HuggingFace...")
        task_ids = list_all_tasks(api)
        print(f"Found {len(task_ids)} tasks")

    # Pre-download and filter tasks
    tasks_to_fix = []
    for i, task_id in enumerate(task_ids):
        task = download_task(api, task_id, args.output)
        if task is None:
            continue
        if args.skip_existing:
            src = task.get("meta", {}).get("install_source", "")
            if "install-agent-fix" in src:
                continue
        tasks_to_fix.append((task_id, task))

    print(f"{len(tasks_to_fix)} tasks to fix (parallel={args.parallel})")

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {"fixed": 0, "failed": 0, "skipped": len(task_ids) - len(tasks_to_fix)}
    report = []
    report_lock = threading.Lock()
    counter = [0]
    counter_lock = threading.Lock()

    def process_one(task_id: str, task: dict) -> None:
        with counter_lock:
            counter[0] += 1
            idx = counter[0]
        print(f"\n[{idx}/{len(tasks_to_fix)}] {task_id}")

        fix = fix_task(client, args.model, task, task_id)

        entry = {
            "task_id": task_id,
            "success": fix.success,
            "old_install": fix.old_install[:200],
            "new_install": fix.new_install[:200],
            "error": fix.error,
            "cycles": fix.turns,
        }

        if fix.success:
            print(f"  [OK] {task_id} -- fixed in {fix.turns} cycle(s)")

            task.setdefault("install_config", {})["install"] = fix.new_install
            task.setdefault("meta", {})["install_source"] = "llm-install-agent-fix"

            out_path = os.path.join(args.output, "tasks", task_id, "workspace.yaml")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                yaml.dump(task, f, default_flow_style=False,
                          allow_unicode=True, sort_keys=False)

            if not args.dry_run:
                try:
                    upload_fixed_task(api, task_id, task, args.output)
                    print(f"  Uploaded {task_id}")
                except Exception as e:
                    print(f"  [ERR] Upload {task_id}: {e}")

            with report_lock:
                results["fixed"] += 1
                report.append(entry)
        else:
            print(f"  [FAIL] {task_id} -- {fix.error}")
            with report_lock:
                results["failed"] += 1
                report.append(entry)

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(process_one, tid, t): tid
            for tid, t in tasks_to_fix
        }
        for future in as_completed(futures):
            tid = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  [ERR] {tid} crashed: {e}")
                traceback.print_exc()
                with report_lock:
                    results["failed"] += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {len(task_ids)} tasks")
    print(f"  Fixed:    {results['fixed']}")
    print(f"  Failed:   {results['failed']}")
    print(f"  Skipped:  {results['skipped']}")

    report_path = os.path.join(args.output, "fix_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
