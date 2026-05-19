<h1 align="center">SWE-Forge (Python)</h1>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.12%2B-blue.svg" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://huggingface.co/datasets/CortexLM/swe-forge"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Dataset-yellow.svg" alt="HuggingFace Dataset"></a>
</p>

**High-performance SWE-bench dataset generator and evaluation harness that mines real GitHub pull requests, produces evaluation-ready task instances, and benchmarks coding agents.**

Built on top of [SweInfinite](https://github.com/unconst/SweInfinite) by [@unconst](https://github.com/unconst), rewritten in Python with:
- Agentic command discovery (NO hardcoded install commands)
- Language detection (rule-based, OK to hardcode)
- Difficulty filtering with LLM classification
- Docker verification of generated tests
- Full parallelism with semaphore-based concurrency
- Structured LLM outputs via OpenAI function calling
- 200k context auto-compaction with smart summarization

---

## What it does

swe-forge connects to [GH Archive](https://www.gharchive.org/) to discover recently merged pull requests, enriches them via the GitHub API, classifies their difficulty using an LLM, discovers install/test commands agenticly, generates test specifications via an agentic loop, and exports SWE-bench-compatible task instances.

## Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Real GitHub Data** | Mines GH Archive for merged PRs across all public repositories |
| 🎯 **Difficulty Filtering** | Pre-classifies PRs as easy/medium/hard before expensive processing |
| 🤖 **Agentic Discovery** | Discovers install/test commands from CI/CD (NO hardcoding) |
| 📦 **Docker Verification** | Verifies tests in Docker before export |
| ⚡ **Full Parallelism** | GH Archive 8x, enrichment 20x, Docker 8x concurrent |
| 🧠 **Smart Compaction** | 200k context limit with structured summary templates |
| 📊 **Complete Export** | workspace.yaml + patch.diff + optional deletion_patch.diff + tests/ directory |

---

## Installation

### From PyPI

```bash
pip install swe-forge
```

### From Source

```bash
git clone https://github.com/CortexLM/swe-forge.git
cd swe-forge
pip install -e .
```

### Docker

```bash
docker pull ghcr.io/cortexlm/swe-forge:latest
```

---

## Quick Start

### Prerequisites

```bash
# Required environment variables
export GITHUB_TOKEN="ghp_..."           # GitHub PAT for PR enrichment
export OPENROUTER_API_KEY="sk-or-v1-..." # OpenRouter API key for LLM
```

### Mine PR Tasks

```bash
swe-forge mine mine \
  --target 10 \
  --output ./tasks.jsonl \
  --output-folder ./tasks \
  --parallel 8
```

### Verify One Pull Request End-to-End

```bash
swe-forge mine complete \
  --repo owner/repo \
  --pr 12345 \
  --output ./tasks.jsonl \
  --model openai/gpt-5.4
```

### Generate a Synthetic Feature-Deletion Benchmark Task

```bash
git clone https://github.com/owner/repo.git ./target-repo

swe-forge synthetic generate \
  --repo-path ./target-repo \
  --repo owner/repo \
  --source-file src/package/module.py \
  --symbol target_function \
  --fail-to-pass "pytest tests/test_target.py -v" \
  --pass-to-pass "pytest tests/ -v" \
  --install-command "pip install -e ." \
  --output-folder ./synthetic_tasks \
  --output-jsonl ./synthetic_tasks.jsonl \
  --overwrite
```

---

## Output Structure

### Directory Format (when using `--output-folder`)

```
tasks/
├── owner-repo-1234/
│   ├── workspace.yaml      # Complete task configuration
│   ├── patch.diff          # PR patch to apply
│   ├── deletion_patch.diff # Synthetic mutation patch, when source_type is synthetic
│   ├── test_patch.diff     # Test file changes
│   └── tests/              # Extracted test files
│       ├── test_feature.py
│       └── test_another.py
└── owner-repo-5678/
    └── ...
```

### workspace.yaml Format

```yaml
task_id: owner-repo-1234
repo:
  url: https://github.com/owner/repo.git
  base_commit: abc123def456...
  merge_commit: fed456abc123...
language: python
difficulty_score: 5
prompt: "Fix the bug in..."
environment:
  image: myuser/swe-forge-tasks:owner-repo-1234
  language_version: "3.12"
install:
  commands:
    - pip install -e .
    - pip install pytest
tests:
  fail_to_pass:
    - pytest tests/test_feature.py -v
    - pytest tests/test_another.py::test_case -v
  pass_to_pass:
    - pytest tests/ -v --ignore=tests/test_feature.py
synthetic:
  source_type: synthetic_feature_deletion
  deletion_patch_file: deletion_patch.diff
  strategy: feature_deletion
docker:
  image: myuser/swe-forge-tasks:owner-repo-1234
  build: true
```

---

## Architecture

### Pipeline Flow

```mermaid
sequenceDiagram
    participant GHA as GH Archive
    participant SF as swe-forge
    participant GH as GitHub API
    participant LLM as LLM
    participant D as Docker

    GHA->>SF: Merged PR events (8x concurrent)
    SF->>SF: Pre-filter (bots, org, stars)
    SF->>GH: Enrich candidates (20x concurrent)
    GH-->>SF: PR metadata + diff
    SF->>LLM: Classify difficulty
    LLM-->>SF: easy / medium / hard
    SF->>D: Agentic discovery (8x concurrent)
    D-->>SF: fail_to_pass + pass_to_pass
    SF->>LLM: Quality scoring
    LLM-->>SF: Accept / reject
    SF-->>SF: Export workspace.yaml
```

### Parallelism Configuration

| Stage | Semaphore | Default | Description |
|-------|-----------|---------|-------------|
| GH Archive Fetch | `gh_archive_sem` | 8 | Download hourly dumps |
| GitHub Enrichment | `enrichment_sem` | 20 | Fetch PR metadata (5000/h rate limit) |
| Pre-classification | `preclassify_sem` | 25 | LLM triage on title+body |
| Deep Processing | `deep_sem` | 8 | Full pipeline per candidate |
| Docker Containers | `docker_sem` | 8 | Concurrent test verification |

---

## Agentic Command Discovery

**IMPORTANT: Commands are NEVER hardcoded.**

```mermaid
sequenceDiagram
    participant AD as Agent Discovery
    participant CI as CI/CD Config
    participant LLM as LLM
    participant SH as Shell (Docker)

    AD->>CI: Parse .github/workflows/, .gitlab-ci.yml
    CI-->>AD: Install patterns, test commands
    
    AD->>SH: Clone repo in Docker
    AD->>LLM: "Discover how to install and test"
    
    loop Up to 200 turns
        LLM->>SH: shell("pip install -e .")
        SH-->>LLM: exit_code=0
        LLM->>SH: shell("pytest tests/")
        SH-->>LLM: exit_code=0, output
    end
    
    LLM->>AD: submit_tests(fail_to_pass, pass_to_pass)
```

### What Happens in Docker

1. **Clone repository** at base commit
2. **Detect language** from files (package.json, pyproject.toml, Cargo.toml, etc.)
3. **Discover commands** by:
   - Parsing CI/CD workflows
   - Reading package manager configs
   - Trying commands and checking exit codes
4. **Generate tests** via LLM agentic loop
5. **Verify tests fail** before patch (proves bug exists)
6. **Apply patch** 
7. **Verify tests pass** after patch (proves fix works)

---

## Difficulty Classification

| Level | Score Range | Typical Changes | Examples |
|-------|-------------|----------------|----------|
| **Easy** | 0.1 – 0.35 | Typos, config, single-file | Fix import, update version |
| **Medium** | 0.4 – 0.65 | Bug fixes, features, APIs | Fix race condition, add endpoint |
| **Hard** | 0.7 – 1.0 | Cross-cutting, architectural | New subsystem, migration |

### Classification Models

- **Pre-classification**: `moonshotai/kimi-k2.5` (fast triage on title+body)
- **Full classification**: Uses complete diff and test spec

---

## Auto-Compaction (200k Context)

When context exceeds 200k tokens, the system uses **structured summarization**:

```yaml
## Goal
[What goal(s) is the user trying to accomplish?]

## Instructions
- [What important instructions did the user give you]
- [If there is a plan or spec, include information about it]

## Discoveries
[What notable things were learned during this conversation]

## Accomplished
[What work has been completed, in progress, and left?]

## Relevant files / directories
[Structured list of relevant files]
```

This preserves critical context across long agentic sessions.

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GITHUB_TOKEN` | Yes | GitHub PAT for PR enrichment |
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key for LLM calls |
| `HF_TOKEN` | No | HuggingFace token for dataset upload |
| `RUST_LOG` | No | Log level: debug, info, warn, error |

### Supported Languages

| Language | Detection | Package Managers |
|----------|-----------|------------------|
| Python | pyproject.toml, setup.py, requirements.txt | pip, poetry, uv |
| JavaScript/TypeScript | package.json | npm, yarn, pnpm |
| Rust | Cargo.toml | cargo |
| Go | go.mod | go mod |
| Java | pom.xml, build.gradle | maven, gradle |

---

## Development

### Setup

```bash
git clone https://github.com/CortexLM/swe-forge.git
cd swe-forge
pip install -e ".[dev]"
```

### Testing

```bash
pytest tests/ -v
```

### Code Quality

```bash
ruff format src/ tests/
ruff check src/ tests/
mypy src/
```

---

## Benchmark Results

Benchmark run with 100 candidate PRs from GH Archive:

### Pipeline Funnel

| Stage | Count | Percentage |
|-------|------:|-----------:|
| Raw GH Archive events (12h) | 1,752,426 | 100% |
| Merged PR events | 35,498 | 2.03% |
| After pre-filter | 1,394 | 3.93% |
| Enriched successfully | 21 | 1.51% |
| Tests generated | 11 | 52.38% |
| Quality passed | 8 | 72.73% |

### Throughput

| Metric | Value |
|--------|------:|
| Tasks per hour | 8 |
| Avg time per task | 450s |
| Docker parallelism | 8 containers |

---

## API Reference

### Python API

```python
from swe_forge.swe.pipeline import SwePipeline, SwePipelineConfig
from swe_forge.export.workspace import export_tasks_to_workspace

# Configure pipeline
config = SwePipelineConfig(
    max_candidates=50,
    max_tasks=10,
    min_stars=100,
    languages=["python"],
)

# Run pipeline
async with SwePipeline(config) as pipeline:
    result = await pipeline.run()
    
    # Export to workspace format
    export_tasks_to_workspace(
        result.tasks,
        output_folder="./tasks",
        docker_username="myuser"
    )
```

### SweTask Model

```python
from swe_forge.swe.models import SweTask

@dataclass
class SweTask:
    id: str
    repo: str                    # owner/repo format
    base_commit: str             # Git SHA
    merge_commit: str            # Git SHA
    language: str                 # python, rust, etc.
    difficulty_score: int         # 1-10
    patch: str                    # Unified diff
    test_patch: str               # Test file changes
    fail_to_pass: list[str]       # Test commands
    pass_to_pass: list[str]       # Test commands
    install_config: dict          # Discovered install commands
    prompt: str                   # Task description
    quality_score: float          # 0.0-1.0
    status: SweTaskStatus         # candidate, validated, etc.
```

---

---

## Testing Tasks

### Test Tasks from HuggingFace Dataset

The published dataset `CortexLM/swe-forge` on HuggingFace contains task instances with pre-built Docker images.

#### Prerequisites

- Docker installed and running
- `pip install datasets`

#### CLI Usage

```bash
python scripts/test_task.py --task-id pydantic-pydantic-12985 --output results.json -v
```

#### Docker Sandbox

Each task is tested in an isolated Docker container:

1. **Pull Docker image** - Contains repo at `base_commit`
2. **Run fail_to_pass tests** - Should all PASS
3. **Run pass_to_pass tests** - Should all PASS

#### Docker Image Contents

Pre-built Docker images (`platformnetwork/swe-forge:*`) contain:
- `/workspace/patch.diff` - The patch
- `/workspace/run_tests.sh` - Test script
- Repository cloned at `base_commit`

#### Dataset Fields

| Field | Description |
|-------|-------------|
| `instance_id` | Task ID (format: `owner-repo-123`) |
| `docker_image` | Pre-built Docker image |
| `fail_to_pass` | Tests that must pass after patch |
| `pass_to_pass` | Tests that must stay passing |
| `patch` | Unified diff to apply |


---

## Benchmark Harness

SWE-Forge provides a Docker-based evaluation harness for benchmarking model-generated patches, similar to SWE-bench.

### Installation

```bash
pip install datasets
```

### Quick Start

```bash
python3 scripts/run_evaluation.py \
  --predictions_path gold \
  --instance_ids pydantic-pydantic-12985 \
  --max_workers 4
```

### Prediction Format

Create a JSONL file with model predictions:

```json
{"instance_id": "pydantic-pydantic-12985", "model_patch": "diff --git a/..."}
{"instance_id": "owner-repo-123", "model_patch": "..."}
```

Then evaluate:

```bash
python3 scripts/run_evaluation.py \
  --predictions_path predictions.jsonl \
  --max_workers 4
```

### Evaluation Flow

For each task, the harness:

1. **Pull Docker image** - Contains repo at base commit
2. **Run fail_to_pass tests BEFORE patch** - Should FAIL (bug exists)
3. **Apply model patch**
4. **Run fail_to_pass tests AFTER patch** - Should PASS (bug fixed)
5. **Run pass_to_pass tests** - Should PASS (no regression)
6. **Grade** - Resolved if all tests pass as expected

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--predictions_path` | Path to JSONL or "gold" for ground truth |
| `--max_workers` | Parallel workers (default: 4) |
| `--instance_ids` | Specific instances to evaluate |
| `--random N` | Evaluate N random instances |
| `--timeout` | Timeout per instance (default: 600s) |
| `--run_id` | Run identifier |
| `--output_dir` | Output directory |
| `--clean` | Cleanup Docker after evaluation |

### Output

Results are saved to `evaluation_results/{run_id}/`:

- `results.json` - Overall metrics
- `instance_results.jsonl` - Detailed per-instance results

### Metrics

- **Resolution Rate**: Percentage of patches that fixed the issue
- **Tests Passed/Failed**: Test execution results
- **Duration**: Evaluation time


## Credits

Built on top of [SweInfinite](https://github.com/unconst/SweInfinite) by [@unconst](https://github.com/unconst).

Extended with:
- Python rewrite with full async support
- Agentic command discovery (NO hardcoding)
- Docker verification of generated tests
- Structured workspace export
- 200k context auto-compaction
- Configurable parallelism

---

## License

MIT — see [LICENSE](LICENSE).


---

## Quality Control Pipeline

SWE-Forge includes a comprehensive quality control pipeline to ensure tasks are valid and appropriately challenging.

### Overview

```
Task Generation
      ↓
┌─────────────────────────────────┐
│ 1. Complexity Evaluation       │
│    LLM assesses task difficulty │
│    Score: 0.0 (trivial) to 1.0  │
│    Reject if < 0.25             │
└─────────────────────────────────┘
      ↓
┌─────────────────────────────────┐
│ 2. Docker Verification         │
│    Tests FAIL before patch     │
│    Apply patch                 │
│    Tests PASS after patch     │
│    Reject if tests don't work  │
└─────────────────────────────────┘
      ↓
   Accept Task
```

### Complexity Scoring

The complexity evaluator uses an LLM agent to analyze:

| Factor | Impact |
|--------|--------|
| Lines changed | More lines → higher score |
| Files modified | More files → higher score |
| Logic complexity | Complex logic → higher score |
| Context needed | More context → higher score |
| Change type | Config/docs → lower score |

**Scoring thresholds:**

| Score | Difficulty | Action |
|-------|------------|--------|
| 0.0-0.25 | Trivial | ❌ **REJECTED** |
| 0.25-0.40 | Easy | ✅ Accepted |
| 0.40-0.65 | Medium | ✅ Accepted |
| 0.65-1.00 | Hard | ✅ Accepted |

### Docker Verification

Each task is verified in an isolated Docker container:

1. **Before patch**: Tests MUST FAIL (proves bug exists)
2. **Apply patch**: `git apply /workspace/patch.diff`
3. **After patch**: Tests MUST PASS (proves fix works)
4. **Regression tests**: `pass_to_pass` tests must stay passing

### CLI Options

```bash
swe-forge mine mine \
  --target 100 \
  --min-complexity 0.30 \
  --complexity-model openai/gpt-4 \
  --output ./tasks.jsonl \
  --output-folder ./tasks
```

### Revalidation Script

Revalidate existing tasks to filter out invalid ones:

```bash
python scripts/revalidate_tasks.py \
  --tasks-dir ./tasks \
  --limit 10 \
  --min-complexity 0.30 \
  --report report.json
```

### Expected Results

For a typical mining run:

| Metric | Typical Value |
|--------|---------------|
| Tasks generated | 100% |
| Rejected (complexity) | ~20% |
| Rejected (verification) | ~20% |
| **Accepted** | **~60%** |

The acceptance rate of 30-70% is normal and ensures quality benchmarks.

### Dataset Fields

When tasks are exported to HuggingFace, quality fields are included:

| Field | Description |
|-------|-------------|
| `complexity_score` | 0.0-1.0 complexity rating |
| `complexity_difficulty` | "easy", "medium", or "hard" |
| `verified` | True if Docker verification passed |

Filter on HF:
```python
from datasets import load_dataset

ds = load_dataset("CortexLM/swe-forge")
# Only medium+ difficulty, verified tasks
filtered = ds.filter(lambda x: x['complexity_score'] >= 0.4 and x['verified'])
```
