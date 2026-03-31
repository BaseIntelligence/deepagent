#!/usr/bin/env python3
"""
SWE-Forge Evaluation Harness

Evaluates model-generated patches on SWE-Forge tasks using Docker containers.

Usage:
    # Evaluate predictions
    python scripts/run_evaluation.py --predictions_path predictions.jsonl --max_workers 8

    # Evaluate gold patches (ground truth)
    python scripts/run_evaluation.py --predictions_path gold --instance_ids pydantic-pydantic-12985

    # Evaluate random subset
    python scripts/run_evaluation.py --predictions_path gold --random 10
"""

import argparse
import json
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Thread lock for Docker operations
docker_lock = threading.Lock()


def load_dataset() -> List[Dict]:
    """Load tasks from HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    logger.info("Loading dataset from HuggingFace CortexLM/swe-forge...")
    ds = load_dataset("CortexLM/swe-forge", split="train")
    tasks = [dict(row) for row in ds]
    logger.info(f"Loaded {len(tasks)} tasks")
    return tasks


def get_gold_patch(task: Dict) -> str:
    """Extract the gold patch from a task."""
    return task.get("patch", "")


def run_instance_evaluation(
    task: Dict,
    patch: str,
    timeout: int = 600,
    run_id: str = "default",
) -> Dict[str, Any]:
    """
    Evaluate a single task instance in Docker.
    
    Flow:
    1. Pull/create Docker container with base commit
    2. Run fail_to_pass tests BEFORE patch (should FAIL)
    3. Apply patch
    4. Run fail_to_pass tests AFTER patch (should PASS)
    5. Run pass_to_pass tests (should PASS)
    """
    instance_id = task.get("instance_id", "unknown")
    docker_image = task.get("docker_image", f"platformnetwork/swe-forge:{instance_id}")
    
    result = {
        "instance_id": instance_id,
        "docker_image": docker_image,
        "resolved": False,
        "tests_passed": 0,
        "tests_failed": 0,
        "fail_to_pass_before": [],
        "fail_to_pass_after": [],
        "pass_to_pass": [],
        "error": None,
        "duration_seconds": 0,
        "patch_applied": False,
    }
    
    start_time = time.time()
    container_name = f"swe-eval-{instance_id.replace('/', '-').replace('.', '-')}-{run_id}"
    
    try:
        # Pull image
        with docker_lock:
            check = subprocess.run(
                ["docker", "images", "-q", docker_image],
                capture_output=True, text=True, timeout=30
            )
            if not check.stdout.strip():
                logger.info(f"[{instance_id}] Pulling image: {docker_image}")
                pull = subprocess.run(
                    ["docker", "pull", docker_image],
                    capture_output=True, text=True, timeout=300
                )
                if pull.returncode != 0:
                    raise RuntimeError(f"Failed to pull image: {pull.stderr[:200]}")
        
        # Start container
        logger.info(f"[{instance_id}] Starting container")
        subprocess.run(
            ["docker", "run", "-d", "--name", container_name, docker_image, "sleep", str(timeout + 60)],
            capture_output=True, text=True, check=True, timeout=60
        )
        
        # Get test commands
        fail_to_pass = json.loads(task.get("fail_to_pass", "[]"))
        pass_to_pass = json.loads(task.get("pass_to_pass", "[]"))
        
        if not fail_to_pass:
            result["error"] = "No fail_to_pass tests defined"
            return result
        
        # Create patch file in container
        patch_path = "/tmp/patch.diff"
        with open(patch_path, "w") as f:
            f.write(patch)
        
        # Copy patch to container
        subprocess.run(
            ["docker", "cp", patch_path, f"{container_name}:/workspace/model_patch.diff"],
            capture_output=True, text=True, check=True, timeout=30
        )
        
        # Run tests BEFORE patch (should FAIL)
        logger.info(f"[{instance_id}] Running tests BEFORE patch")
        for test_cmd in fail_to_pass[:3]:  # Limit to 3 tests
            try:
                res = subprocess.run(
                    ["docker", "exec", container_name, "bash", "-lc", f"cd /repo && {test_cmd}"],
                    capture_output=True, text=True, timeout=120
                )
                result["fail_to_pass_before"].append({
                    "command": test_cmd,
                    "passed": res.returncode == 0,
                })
            except Exception as e:
                result["fail_to_pass_before"].append({
                    "command": test_cmd,
                    "passed": False,
                    "error": str(e),
                })
        
        # Apply patch
        logger.info(f"[{instance_id}] Applying patch")
        apply_result = subprocess.run(
            ["docker", "exec", container_name, "bash", "-lc", 
             "cd /repo && git apply /workspace/model_patch.diff 2>&1 || git apply --reject /workspace/model_patch.diff 2>&1"],
            capture_output=True, text=True, timeout=60
        )
        result["patch_applied"] = apply_result.returncode == 0
        
        if not result["patch_applied"]:
            result["error"] = f"Patch failed to apply: {apply_result.stdout[:200]}"
        
        # Run tests AFTER patch (should PASS)
        if result["patch_applied"]:
            logger.info(f"[{instance_id}] Running tests AFTER patch")
            for test_cmd in fail_to_pass:
                try:
                    res = subprocess.run(
                        ["docker", "exec", container_name, "bash", "-lc", f"cd /repo && {test_cmd}"],
                        capture_output=True, text=True, timeout=120
                    )
                    passed = res.returncode == 0
                    result["fail_to_pass_after"].append({
                        "command": test_cmd,
                        "passed": passed,
                    })
                    if passed:
                        result["tests_passed"] += 1
                    else:
                        result["tests_failed"] += 1
                except Exception as e:
                    result["fail_to_pass_after"].append({
                        "command": test_cmd,
                        "passed": False,
                        "error": str(e),
                    })
                    result["tests_failed"] += 1
            
            # Run pass_to_pass tests
            for test_cmd in pass_to_pass[:2]:  # Limit to 2
                try:
                    res = subprocess.run(
                        ["docker", "exec", container_name, "bash", "-lc", f"cd /repo && {test_cmd}"],
                        capture_output=True, text=True, timeout=120
                    )
                    passed = res.returncode == 0
                    result["pass_to_pass"].append({
                        "command": test_cmd,
                        "passed": passed,
                    })
                    if passed:
                        result["tests_passed"] += 1
                    else:
                        result["tests_failed"] += 1
                except Exception as e:
                    result["pass_to_pass"].append({
                        "command": test_cmd,
                        "passed": False,
                        "error": str(e),
                    })
                    result["tests_failed"] += 1
            
            # Determine if resolved
            all_fail_pass = all(t["passed"] for t in result["fail_to_pass_after"]) if result["fail_to_pass_after"] else False
            all_pass_pass = all(t["passed"] for t in result["pass_to_pass"]) if result["pass_to_pass"] else True
            result["resolved"] = all_fail_pass and all_pass_pass
    
    except subprocess.TimeoutExpired:
        result["error"] = "Evaluation timed out"
    except Exception as e:
        result["error"] = str(e)
    finally:
        # Cleanup
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True, text=True, timeout=30
        )
        if os.path.exists("/tmp/patch.diff"):
            os.remove("/tmp/patch.diff")
    
    result["duration_seconds"] = round(time.time() - start_time, 2)
    return result


def main():
    parser = argparse.ArgumentParser(description="SWE-Forge Evaluation Harness")
    parser.add_argument("--predictions_path", type=str, required=True,
                       help="Path to predictions JSONL or 'gold' for ground truth")
    parser.add_argument("--max_workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--instance_ids", type=str, nargs="*",
                       help="Specific instance IDs to evaluate")
    parser.add_argument("--random", type=int,
                       help="Evaluate N random instances")
    parser.add_argument("--timeout", type=int, default=600,
                       help="Timeout per instance in seconds")
    parser.add_argument("--run_id", type=str, default=None,
                       help="Run identifier")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Output directory")
    parser.add_argument("--cache_level", type=str, default="env",
                       choices=["none", "base", "env", "instance"],
                       help="Docker cache level")
    parser.add_argument("--clean", action="store_true",
                       help="Clean up Docker resources after evaluation")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set run ID
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load dataset
    all_tasks = load_dataset()
    
    # Get tasks to evaluate
    if args.instance_ids:
        tasks = [t for t in all_tasks if t.get("instance_id") in args.instance_ids]
    elif args.random:
        import random
        tasks = random.sample(all_tasks, min(args.random, len(all_tasks)))
    else:
        tasks = all_tasks
    
    # Load predictions
    predictions = {}
    if args.predictions_path == "gold":
        # Use gold patches from dataset
        for task in tasks:
            predictions[task["instance_id"]] = get_gold_patch(task)
        logger.info(f"Using gold patches for {len(predictions)} instances")
    else:
        # Load predictions from JSONL
        pred_path = Path(args.predictions_path)
        if not pred_path.exists():
            logger.error(f"Predictions file not found: {pred_path}")
            return
        
        with open(pred_path) as f:
            for line in f:
                if line.strip():
                    pred = json.loads(line)
                    predictions[pred["instance_id"]] = pred.get("model_patch", "")
        logger.info(f"Loaded {len(predictions)} predictions")
    
    # Filter tasks to those with predictions
    tasks_to_eval = [t for t in tasks if t["instance_id"] in predictions]
    logger.info(f"Evaluating {len(tasks_to_eval)} instances")
    
    # Create output directory
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run evaluations
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for task in tasks_to_eval:
            instance_id = task["instance_id"]
            patch = predictions[instance_id]
            future = executor.submit(
                run_instance_evaluation,
                task, patch, args.timeout, run_id
            )
            futures[future] = instance_id
        
        for future in as_completed(futures):
            instance_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = "RESOLVED" if result["resolved"] else "FAILED"
                logger.info(f"[{instance_id}] {status}")
            except Exception as e:
                logger.error(f"[{instance_id}] Error: {e}")
                results.append({
                    "instance_id": instance_id,
                    "resolved": False,
                    "error": str(e),
                })
    
    # Calculate metrics
    total = len(results)
    resolved = sum(1 for r in results if r.get("resolved"))
    
    metrics = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "total_instances": total,
        "resolved_instances": resolved,
        "resolution_rate": resolved / total if total > 0 else 0,
        "duration_seconds": round(time.time() - start_time, 2),
        "max_workers": args.max_workers,
        "predictions_path": args.predictions_path,
    }
    
    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    instance_file = output_dir / "instance_results.jsonl"
    with open(instance_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Total Instances: {total}")
    print(f"Resolved: {resolved}")
    print(f"Resolution Rate: {resolved/total*100:.1f}%")
    print(f"Duration: {metrics['duration_seconds']}s")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)
    
    # Cleanup if requested
    if args.clean:
        logger.info("Cleaning up Docker resources...")
        subprocess.run(["docker", "system", "prune", "-f"], capture_output=True)


if __name__ == "__main__":
    main()
