#!/usr/bin/env python3
"""Test SWE-Forge tasks from HuggingFace dataset in Docker sandbox.

Usage:
    python scripts/test_task.py --task-id pydantic-pydantic-12985
    python scripts/test_task.py --random 5
    python scripts/test_task.py --all
    python scripts/test_task.py --all --output results.json
"""

import argparse
import json
import logging
import subprocess
import time
from typing import Any, Optional, List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_hf_dataset():
    """Load tasks from HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    logger.info("Loading dataset from HuggingFace...")
    ds = load_dataset("CortexLM/swe-forge", split="train")
    logger.info(f"Loaded {len(ds)} tasks")
    return ds


def get_task_by_id(ds, task_id: str) -> Optional[Dict]:
    """Get a specific task by its ID."""
    for row in ds:
        if row["instance_id"] == task_id:
            return dict(row)
    return None


def get_random_tasks(ds, n: int) -> List[Dict]:
    """Get N random tasks from dataset."""
    import random
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    return [dict(ds[i]) for i in indices]


def run_docker_test(task: Dict, timeout: int = 600) -> Dict[str, Any]:
    """Test a task in Docker sandbox."""
    task_id = task.get("instance_id", "unknown")
    docker_image = task.get("docker_image", f"platformnetwork/swe-forge:{task_id}")

    logger.info(f"Testing task: {task_id}")
    logger.info(f"Docker image: {docker_image}")

    results = {
        "task_id": task_id,
        "docker_image": docker_image,
        "success": False,
        "fail_to_pass_results": [],
        "pass_to_pass_results": [],
        "error": None,
        "duration_seconds": 0,
    }

    start_time = time.time()
    container_name = f"swe-test-{task_id.replace('/', '-').replace('.', '-')}"

    try:
        # Pull image if not local
        check = subprocess.run(
            ["docker", "images", "-q", docker_image],
            capture_output=True, text=True
        )

        if not check.stdout.strip():
            logger.info(f"Pulling Docker image: {docker_image}")
            pull = subprocess.run(
                ["docker", "pull", docker_image],
                capture_output=True, text=True, timeout=300
            )
            if pull.returncode != 0:
                raise RuntimeError(f"Failed to pull image: {pull.stderr}")

        # Start container
        logger.info(f"Starting container: {container_name}")
        subprocess.run(
            ["docker", "run", "-d", "--name", container_name, docker_image, "sleep", str(timeout)],
            capture_output=True, text=True, check=True
        )

        # Get test commands
        fail_to_pass = json.loads(task.get("fail_to_pass", "[]"))
        pass_to_pass = json.loads(task.get("pass_to_pass", "[]"))

        # Run fail_to_pass tests
        for test_cmd in fail_to_pass:
            logger.info(f"Running fail_to_pass: {test_cmd[:50]}...")
            result = subprocess.run(
                ["docker", "exec", container_name, "bash", "-lc", test_cmd],
                capture_output=True, text=True, timeout=300
            )
            results["fail_to_pass_results"].append({
                "command": test_cmd,
                "exit_code": result.returncode,
                "success": result.returncode == 0,
                "output": result.stdout[-500:] if result.stdout else "",
            })

        # Run pass_to_pass tests
        for test_cmd in pass_to_pass:
            logger.info(f"Running pass_to_pass: {test_cmd[:50]}...")
            result = subprocess.run(
                ["docker", "exec", container_name, "bash", "-lc", test_cmd],
                capture_output=True, text=True, timeout=300
            )
            results["pass_to_pass_results"].append({
                "command": test_cmd,
                "exit_code": result.returncode,
                "success": result.returncode == 0,
                "output": result.stdout[-500:] if result.stdout else "",
            })

        # Check overall success
        all_fail = all(r["success"] for r in results["fail_to_pass_results"]) if results["fail_to_pass_results"] else True
        all_pass = all(r["success"] for r in results["pass_to_pass_results"]) if results["pass_to_pass_results"] else True
        results["success"] = all_fail and all_pass

    except subprocess.TimeoutExpired:
        results["error"] = "Test timed out"
    except Exception as e:
        results["error"] = str(e)
    finally:
        # Cleanup container
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, text=True)

    results["duration_seconds"] = round(time.time() - start_time, 2)
    return results


def print_results(results: Dict):
    """Print test results."""
    print("\n" + "=" * 60)
    print(f"Task: {results['task_id']}")
    print(f"Docker: {results['docker_image']}")
    print(f"Duration: {results['duration_seconds']}s")
    print("=" * 60)

    if results.get("error"):
        print(f"ERROR: {results['error']}")
        return

    print("\nfail_to_pass tests:")
    for r in results.get("fail_to_pass_results", []):
        status = "PASS" if r["success"] else "FAIL"
        print(f"  [{status}] {r['command'][:60]}...")

    print("\npass_to_pass tests:")
    for r in results.get("pass_to_pass_results", []):
        status = "PASS" if r["success"] else "FAIL"
        print(f"  [{status}] {r['command'][:60]}...")

    print("\n" + "-" * 60)
    print("TASK PASSED" if results["success"] else "TASK FAILED")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test SWE-Forge tasks with Docker sandbox")
    parser.add_argument("--task-id", type=str, help="Specific task ID to test")
    parser.add_argument("--random", type=int, help="Number of random tasks to test")
    parser.add_argument("--all", action="store_true", help="Test all tasks")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per task in seconds")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load dataset
    ds = load_hf_dataset()

    # Get tasks to test
    tasks = []
    if args.task_id:
        task = get_task_by_id(ds, args.task_id)
        if not task:
            logger.error(f"Task not found: {args.task_id}")
            return
        tasks.append(task)
    elif args.random:
        tasks = get_random_tasks(ds, args.random)
    elif args.all:
        tasks = [dict(row) for row in ds]
    else:
        parser.print_help()
        return

    # Run tests
    all_results = []
    passed, failed = 0, 0

    for i, task in enumerate(tasks, 1):
        logger.info(f"\nTesting task {i}/{len(tasks)}: {task.get('instance_id')}")
        results = run_docker_test(task, timeout=args.timeout)
        all_results.append(results)
        print_results(results)
        if results["success"]:
            passed += 1
        else:
            failed += 1

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total: {len(tasks)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {passed/len(tasks)*100:.1f}%" if tasks else "N/A")
    print("=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
