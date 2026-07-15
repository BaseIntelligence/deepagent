# Workspace Format

A workspace is the portable unit of a DeepAgent benchmark. It contains enough information to recreate the task, run the evaluator, and inspect the oracle solution, while keeping the evaluated agent away from hidden files.

The usual shape is:

```text
tasks/
└── owner-repo-1234/
    ├── workspace.yaml
    ├── patch.diff
    ├── deletion_patch.diff
    ├── test_patch.diff
    ├── tests/
    ├── run_tests.sh
    ├── evaluate.sh
    └── README.md
```

Not every task has every file. Synthetic tasks usually have `deletion_patch.diff`. Real PR tasks usually do not.

## `workspace.yaml`

`workspace.yaml` is the task manifest. It tells the evaluator which repository to use, which commit to start from, how to install dependencies, and which commands define success.

Important sections:

```yaml
task_id: owner-repo-1234
repo:
  url: https://github.com/owner/repo.git
  base_commit: abc123
  merge_commit: abc123
  path: /workspace/repo
language: python
prompt: Restore the deleted behavior...
environment:
  image: platformnetwork/swe-forge:owner-repo-1234
  repo_path: /workspace/repo
  forge_path: /workspace/forge
  tests_path: /workspace/forge/tests
install:
  commands:
    - pip install -e .
  working_dir: /workspace/repo
tests:
  fail_to_pass:
    - pytest tests/test_target.py -v
  pass_to_pass:
    - pytest tests/ -v
  working_dir: /workspace/repo
synthetic:
  source_type: synthetic_feature_deletion
  deletion_patch_file: deletion_patch.diff
  strategy: feature_deletion
```

For real PR tasks, the `synthetic` block may be missing. For synthetic tasks, it records how the broken state is created.

## `patch.diff`

`patch.diff` is the oracle solution.

For a real PR task, it usually comes from the original pull request. For a synthetic feature-deletion task, it is the inverse of `deletion_patch.diff`.

This file must be hidden from the solving agent. It is useful for:

- validating that the task is solvable;
- checking the oracle score;
- exporting training data;
- comparing a model patch with the known repair.

It is not part of the agent's context.

## `deletion_patch.diff`

`deletion_patch.diff` exists for synthetic tasks. It is the patch that creates the bug by deleting or disabling a known behavior.

The evaluator applies it before running the before/after checks:

1. checkout the base commit;
2. apply `deletion_patch.diff`;
3. confirm `fail_to_pass` tests fail;
4. apply the candidate patch;
5. confirm all required tests pass.

This file must also be hidden from the agent. It can reveal exactly what was removed.

## `tests/` and `test_patch.diff`

The `tests/` directory contains generated or extracted benchmark tests. `test_patch.diff` stores the original test patch when the task came from a PR or test-generation flow.

Depending on the task, tests may be referenced directly from the repository, copied into `/workspace/forge/tests`, or extracted from a diff.

The important rule is that tests define the scoring contract:

- `fail_to_pass` tests prove the target behavior was broken and then restored;
- `pass_to_pass` tests protect existing behavior from regressions.

## `evaluate.sh`

`evaluate.sh` is the local binary scorer. It prints:

```json
{"score": 1}
```

or:

```json
{"score": 0}
```

Score `1` means the expected fail-to-pass transition happened and regression tests passed. Score `0` means at least one phase failed.

The script is intentionally simple so validators can inspect it, run it locally, or wrap it in a larger scoring system.

## `run_tests.sh`

`run_tests.sh` is a helper script for humans. It prints the task configuration and can run a basic Docker verification mode.

Use it when debugging a workspace manually. Use `evaluate.sh` or the Python evaluator for official scoring.

## Hidden vs Visible Files

In a real benchmark run, the agent should work in the repository checkout, not in the forge directory.

Recommended visibility:

| Path | Agent visibility | Reason |
|---|---:|---|
| `/workspace/repo` | Yes | The agent needs the codebase. |
| `/workspace/forge/patch.diff` | No | Contains the solution. |
| `/workspace/forge/deletion_patch.diff` | No | Reveals the synthetic mutation. |
| `/workspace/forge/workspace.yaml` | Usually no | Can contain task metadata and test commands. |
| `/workspace/forge/tests` | Depends on benchmark mode | Public tests may be visible; hidden reward tests should not be. |

For Platform-style challenge validation, the safe default is to hide the entire forge directory from the agent and let the validator mount it only during scoring.
