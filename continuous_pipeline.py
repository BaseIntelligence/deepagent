#!/usr/bin/env python3
"""Continuous publish: every INTERVAL seconds, push completed tasks to Docker Hub + HuggingFace."""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("publish")

TASKS_DIR = Path("tasks")
HF_REPO = os.environ.get("HF_REPO", "CortexLM/swe-forge")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
DOCKER_USER = os.environ.get("DOCKER_USER", "platformnetwork")
INTERVAL = int(os.environ.get("PUBLISH_INTERVAL", "300"))  # 5 min

PUBLISHED_DOCKER: set[str] = set()
PUBLISHED_HF: set[str] = set()
STATE_FILE = Path(".publish_state.json")


def load_state():
    if STATE_FILE.exists():
        data = json.loads(STATE_FILE.read_text())
        PUBLISHED_DOCKER.update(data.get("docker", []))
        PUBLISHED_HF.update(data.get("hf", []))
        logger.info("Loaded state: %d docker, %d hf", len(PUBLISHED_DOCKER), len(PUBLISHED_HF))


def save_state():
    STATE_FILE.write_text(json.dumps({
        "docker": sorted(PUBLISHED_DOCKER),
        "hf": sorted(PUBLISHED_HF),
    }))


def get_ready_tasks() -> list[Path]:
    """Find tasks that have workspace.yaml + patch.diff + tests/ + evaluate.sh."""
    ready = []
    if not TASKS_DIR.exists():
        return ready
    for task_dir in sorted(TASKS_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        ws = task_dir / "workspace.yaml"
        patch = task_dir / "patch.diff"
        tests = task_dir / "tests"
        evaluate = task_dir / "evaluate.sh"
        if ws.exists() and patch.exists() and tests.exists() and evaluate.exists():
            # Check that workspace has fail_to_pass
            try:
                data = yaml.safe_load(ws.read_text())
                f2p = data.get("tests", {}).get("fail_to_pass", [])
                install = data.get("install", {}).get("commands", [])
                has_test_files = any(tests.rglob("*.py")) if tests.is_dir() else False
                if f2p and install and has_test_files:
                    ready.append(task_dir)
            except Exception:
                pass
    return ready


def build_and_push_docker(task_dir: Path) -> bool:
    """Use DockerSetupAgent to build, verify, push."""
    task_id = task_dir.name
    if task_id in PUBLISHED_DOCKER:
        return True

    logger.info("Agent setting up Docker for: %s", task_id)

    try:
        sys.path.insert(0, "src")
        from swe_forge.agents.docker_setup_agent import DockerSetupAgent
        from swe_forge.llm.openrouter import OpenRouterClient

        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not openrouter_key:
            logger.warning("No OPENROUTER_API_KEY, skipping Docker for %s", task_id)
            return False

        model = os.environ.get("LLM_MODEL", "openai/gpt-5.4")
        llm_client = OpenRouterClient(api_key=openrouter_key, default_model=model)
        agent = DockerSetupAgent(llm_client, model=model, max_turns=100)

        image = asyncio.run(agent.setup_and_verify(task_dir, DOCKER_USER, push=True))

        if image:
            PUBLISHED_DOCKER.add(task_id)
            logger.info("Docker done: %s -> %s", task_id, image)
            subprocess.run(["docker", "rmi", image], capture_output=True, timeout=30)
            return True
        else:
            logger.warning("Docker agent failed for %s", task_id)
            return False
    except Exception as e:
        logger.error("Docker error for %s: %s", task_id, e)
        return False


def push_to_huggingface(task_dirs: list[Path]) -> int:
    """Push new tasks to HuggingFace dataset."""
    new_tasks = [d for d in task_dirs if d.name not in PUBLISHED_HF]
    if not new_tasks:
        return 0

    logger.info("Pushing %d tasks to HuggingFace %s", len(new_tasks), HF_REPO)

    try:
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=HF_REPO, repo_type="dataset", exist_ok=True)

        count = 0
        for task_dir in new_tasks:
            task_id = task_dir.name
            try:
                api.upload_folder(
                    folder_path=str(task_dir),
                    path_in_repo=f"tasks/{task_id}",
                    repo_id=HF_REPO,
                    repo_type="dataset",
                )
                PUBLISHED_HF.add(task_id)
                count += 1
                logger.info("HF uploaded: %s (%d/%d)", task_id, count, len(new_tasks))
            except Exception as e:
                logger.warning("HF upload failed for %s: %s", task_id, e)

        # Update dataset card
        _update_dataset_card(api, len(PUBLISHED_HF))
        return count
    except Exception as e:
        logger.error("HuggingFace error: %s", e)
        return 0


def _update_dataset_card(api: HfApi, total: int):
    card = f"""---
task_categories: ["text-generation"]
license: apache-2.0
tags: ["swe-bench", "code-generation", "software-engineering", "benchmark"]
size_categories: ["n<1K"]
---

# SWE-Forge Dataset

**{total} validated tasks** for evaluating software engineering agents.

Each task contains:
- `workspace.yaml` - Task configuration (repo, commits, install commands, test commands)
- `patch.diff` - The ground-truth patch
- `tests/` - Generated test files (fail before patch, pass after)
- `evaluate.sh` - Binary evaluator (score 0 or 1)

## Docker Images

Pre-built images on Docker Hub: `platformnetwork/swe-forge:<task_id>`

Each image has the repo cloned at base_commit with dependencies installed.
The benchmark runner applies the agent's patch then mounts tests to evaluate.

## Usage

```python
from datasets import load_dataset
ds = load_dataset("CortexLM/swe-forge")
```
"""
    try:
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=HF_REPO,
            repo_type="dataset",
        )
    except Exception as e:
        logger.warning("Failed to update dataset card: %s", e)


def publish_cycle():
    """One publish cycle: find ready tasks, build Docker, push HF."""
    ready = get_ready_tasks()
    new_docker = [d for d in ready if d.name not in PUBLISHED_DOCKER]
    new_hf = [d for d in ready if d.name not in PUBLISHED_HF]

    logger.info(
        "Cycle: %d ready, %d new docker, %d new hf (published: %d docker, %d hf)",
        len(ready), len(new_docker), len(new_hf),
        len(PUBLISHED_DOCKER), len(PUBLISHED_HF),
    )

    # Build and push Docker images
    for task_dir in new_docker:
        build_and_push_docker(task_dir)
        save_state()

    # Push to HuggingFace (all tasks with f2p, even if Docker failed)
    if new_hf:
        push_to_huggingface(new_hf)
        save_state()


def main():
    load_state()
    logger.info("Starting continuous publish (interval=%ds, repo=%s)", INTERVAL, HF_REPO)

    # Initial dataset creation
    if HF_TOKEN:
        try:
            api = HfApi(token=HF_TOKEN)
            api.create_repo(repo_id=HF_REPO, repo_type="dataset", exist_ok=True)
            logger.info("HF dataset ready: %s", HF_REPO)
        except Exception as e:
            logger.error("Failed to create HF repo: %s", e)

    while True:
        try:
            publish_cycle()
        except Exception as e:
            logger.error("Publish cycle error: %s", e)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
