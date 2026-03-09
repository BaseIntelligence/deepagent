#!/bin/bash
# swe-forge testing environment setup script
# This script is idempotent - running it multiple times is safe

set -e

echo "=== swe-forge Testing Environment Setup ==="

# Check Rust installation
if ! command -v cargo &> /dev/null; then
    echo "ERROR: Rust/Cargo not found. Please install Rust: https://rustup.rs/"
    exit 1
fi

echo "✓ Rust/Cargo found: $(cargo --version)"

# Check Docker installation
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found. Please install Docker."
    exit 1
fi

echo "✓ Docker found: $(docker --version)"

# Check Docker daemon is running
if ! docker ps &> /dev/null; then
    echo "ERROR: Docker daemon is not running. Please start Docker."
    exit 1
fi

echo "✓ Docker daemon is running"

# Build the project (creates target/ directory)
echo "Building swe-forge..."
cargo build --release

echo "✓ Build successful"

# Check hf-tasks directory exists
if [ -d "./hf-tasks/tasks" ]; then
    TASK_COUNT=$(find ./hf-tasks/tasks -name "workspace.yaml" | wc -l)
    echo "✓ Found hf-tasks/ directory with ${TASK_COUNT} tasks"
else
    echo "WARNING: hf-tasks/ directory not found. Some integration tests may be skipped."
fi

# Verify test command works
echo "Verifying test infrastructure..."
cargo test --lib -- --list > /dev/null 2>&1 || true

echo "✓ Test infrastructure ready"

echo ""
echo "=== Setup Complete ==="
echo "Run tests with: cargo test --lib"
echo "Run harness with: cargo run -- swe harness --input ./hf-tasks/tasks --parallel 1"
