# Environment

Environment variables, external dependencies, and setup notes for swe-forge testing.

## Required Dependencies

- **Rust** (1.70+): The project is built with Cargo
- **Docker**: Required for all integration tests and harness execution
- **Git**: Used by the test generator and validator

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | No (optional) | For LLM-based validation and test generation |
| `GITHUB_TOKEN` | No (optional) | For PR enrichment during mining |
| `HF_TOKEN` | No (optional) | For HuggingFace dataset upload |
| `RUST_LOG` | No | Log level: error, warn, info, debug, trace |

## Docker Configuration

- Docker daemon must be running
- User must have permission to run docker commands
- Containers use `--network=host` for testing (port sharing)
- Memory limits: 32GB default for test containers

## Project Structure

```
swe-forge/
├── src/                    # Source code
│   ├── swe/               # Core SWE pipeline
│   │   ├── harness.rs     # Evaluation harness
│   │   ├── workspace_validator.rs  # Pre-export validation
│   │   ├── docker_sandbox.rs       # Docker container management
│   │   └── test_generator.rs       # Agentic test generation
│   └── ...
├── hf-tasks/              # Real task data (read-only for tests)
├── Cargo.toml             # Rust dependencies
└── .factory/              # Mission infrastructure
```

## Build Commands

```bash
# Debug build
cargo build

# Release build (faster tests)
cargo build --release

# Run tests
cargo test --lib
cargo test --release -- --test-threads=$(nproc)

# Linting
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all
```

## Testing Notes

- Docker tests may be slow due to container operations
- Use `--test-threads=1` for Docker tests to avoid port conflicts
- Some tests require network access (git clone)
- hf-tasks/ directory is read-only for validation
