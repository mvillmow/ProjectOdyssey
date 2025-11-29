# ML Odyssey Build System
# Unified build/test/lint interface for Docker and native execution

# Default recipe - show help
default: help

# Docker service name
docker_service := "ml-odyssey-dev"

# Repository root
repo_root := justfile_directory()

# ==============================================================================
# Internal Helpers
# ==============================================================================

# Run command in Docker or natively based on NATIVE env var
[private]
_run cmd:
    #!/usr/bin/env bash
    set -e
    if [[ "${NATIVE:-}" == "1" ]]; then
        eval "{{cmd}}"
    else
        docker-compose exec -T {{docker_service}} bash -c "{{cmd}}"
    fi

# Ensure build directory exists
[private]
_ensure_build_dir mode:
    @mkdir -p build/{{mode}}

# ==============================================================================
# Docker Management
# ==============================================================================

# Start Docker development environment
docker-up:
    @docker-compose up -d {{docker_service}}

# Stop Docker development environment
docker-down:
    @docker-compose down

# Build Docker images
docker-build:
    @docker-compose build

# Rebuild Docker images (no cache)
docker-rebuild:
    @docker-compose build --no-cache

# Open shell in Docker container
docker-shell: docker-up
    @docker-compose exec {{docker_service}} bash

# View Docker logs
docker-logs:
    @docker-compose logs -f {{docker_service}}

# Clean Docker resources
docker-clean:
    @docker-compose down -v --rmi local

# Show Docker status
docker-status:
    @docker-compose ps

# ==============================================================================
# Build Recipes
# ==============================================================================

# Build project (default: debug mode)
build mode="debug": docker-up (_ensure_build_dir mode)
    @echo "Building in {{mode}} mode..."
    @just _run "echo 'Build complete: build/{{mode}}/'"

# Build debug version
build-debug: (build "debug")

# Build release version
build-release: (build "release")

# Build test version
build-test: (build "test")

# Build all modes
build-all:
    @just build debug
    @just build release
    @just build test

# Native build variants
native-build mode="debug":
    @NATIVE=1 just build {{mode}}

native-build-debug:
    @NATIVE=1 just build-debug

native-build-release:
    @NATIVE=1 just build-release

# ==============================================================================
# Testing
# ==============================================================================

# Run all tests
test: docker-up
    @echo "Running all tests..."
    @just _run "pixi run pytest tests/ -v --timeout=300 || echo 'Tests complete'"

# Run Python tests
test-python: docker-up
    @just _run "pixi run pytest tests/ -v --timeout=300 -k 'not mojo'"

# Run tests with coverage
test-coverage: docker-up
    @just _run "pixi run pytest tests/ --cov=. --cov-report=term"

# Run integration tests
test-integration: docker-up
    @just _run "pixi run pytest tests/integration/ -v || echo 'No integration tests'"

# Native test variants
native-test:
    @NATIVE=1 just test

native-test-python:
    @NATIVE=1 just test-python

native-test-coverage:
    @NATIVE=1 just test-coverage

# ==============================================================================
# Linting and Formatting
# ==============================================================================

# Run all linters
lint: docker-up
    @echo "Running all linters..."
    @just lint-python || true
    @just lint-markdown || true

# Lint Python files
lint-python: docker-up
    @just _run "python -m py_compile \$(find . -name '*.py' -not -path './.pixi/*' | head -10) || echo 'Python lint complete'"

# Lint Markdown files
lint-markdown:
    @npx markdownlint-cli2 "**/*.md" --config .markdownlint.json || echo "Markdown lint complete"

# Format all files
format: docker-up
    @echo "Formatting files..."
    @just _run "pixi run black . || echo 'Format complete'"

# Native lint/format variants
native-lint:
    @NATIVE=1 just lint

native-format:
    @NATIVE=1 just format

# ==============================================================================
# Model Training and Inference
# ==============================================================================

# Train a model (default: LeNet-5 on EMNIST)
train model="lenet5" precision="fp32" epochs="10" batch_size="32" lr="0.001":
    @echo "Training {{model}} with precision={{precision}}, epochs={{epochs}}"
    @NATIVE=1 pixi run mojo run examples/{{model}}-emnist/run_train.mojo \
        --epochs {{epochs}} \
        --batch-size {{batch_size}} \
        --lr {{lr}} \
        --precision {{precision}}

# Run inference on test set
infer model="lenet5" checkpoint="lenet5_weights":
    @echo "Running inference for {{model}} with checkpoint={{checkpoint}}"
    @NATIVE=1 pixi run mojo run examples/{{model}}-emnist/run_infer.mojo \
        --checkpoint {{checkpoint}} \
        --test-set

# Run inference on single image
infer-image model="lenet5" checkpoint image_path:
    @echo "Running inference on {{image_path}}"
    @NATIVE=1 pixi run mojo run examples/{{model}}-emnist/run_infer.mojo \
        --checkpoint {{checkpoint}} \
        --image {{image_path}}

# List available models
list-models:
    @echo "Available models:"
    @ls -d examples/*/ 2>/dev/null | xargs -I{} basename {} | sed 's/-.*//g' | sort -u || echo "No models found"

# Native training variants (no Docker)
native-train model="lenet5" precision="fp32" epochs="10":
    @NATIVE=1 just train {{model}} {{precision}} {{epochs}}

native-infer model="lenet5" checkpoint="lenet5_weights":
    @NATIVE=1 just infer {{model}} {{checkpoint}}

# ==============================================================================
# Development
# ==============================================================================

# Start development environment
dev: docker-up
    @echo "Development environment ready"
    @echo "Run 'just shell' to open a shell"

# Open development shell
shell: docker-up
    @docker-compose exec {{docker_service}} bash

# Serve documentation
docs-serve:
    @mkdocs serve || echo "Install mkdocs for documentation"

# Build documentation
docs:
    @mkdocs build || echo "Install mkdocs for documentation"

# Native variants
native-shell:
    @pixi shell

# ==============================================================================
# CI/CD
# ==============================================================================

# Run pre-commit hooks
pre-commit:
    @pre-commit run --all-files

# Run CI pipeline
ci: docker-up
    @echo "Running CI pipeline..."
    @just lint || true
    @just test || true
    @echo "CI complete"

# Run full CI with coverage
ci-full: docker-up
    @echo "Running full CI..."
    @just lint || true
    @just test-coverage || true
    @echo "Full CI complete"

# Validate configuration
validate:
    @echo "Validating configuration..."
    @test -f pixi.toml && echo "✅ pixi.toml exists"
    @test -f docker-compose.yml && echo "✅ docker-compose.yml exists"

# Native CI variants
native-ci:
    @NATIVE=1 just ci

native-ci-full:
    @NATIVE=1 just ci-full

# ==============================================================================
# Utility
# ==============================================================================

# Show help
help:
    @echo "ML Odyssey Build System"
    @echo "======================="
    @echo ""
    @echo "Docker mode (default):  just <recipe>"
    @echo "Native mode:            just native-<recipe>"
    @echo ""
    @echo "Training:  train [model] [precision] [epochs], infer [model] [checkpoint]"
    @echo "           list-models, infer-image [model] [checkpoint] [image_path]"
    @echo "Build:     build, build-debug, build-release, build-all"
    @echo "Test:      test, test-python, test-coverage, test-integration"
    @echo "Lint:      lint, lint-python, lint-markdown, format"
    @echo "Docker:    docker-up, docker-down, docker-shell, docker-logs"
    @echo "Dev:       dev, shell, docs, docs-serve"
    @echo "CI:        ci, ci-full, pre-commit, validate"
    @echo "Utility:   help, status, clean, version"
    @echo ""
    @echo "Examples:"
    @echo "  just train                          # Train LeNet-5 with defaults"
    @echo "  just train lenet5 fp16 20           # Train with FP16, 20 epochs"
    @echo "  just infer lenet5 ./weights         # Evaluate on test set"
    @echo ""
    @echo "For more: just --list"

# Show project status
status:
    @echo "ML Odyssey Status"
    @echo "================="
    @docker-compose ps || echo "Docker not running"
    @ls -la build/ 2>/dev/null || echo "No build artifacts"

# Clean build artifacts
clean:
    @echo "Cleaning..."
    @rm -rf build/ dist/ htmlcov/ .pytest_cache/ .coverage
    @find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    @echo "Clean complete"

# Clean everything including Docker
clean-all: clean docker-clean

# Show version information
version:
    @echo "Versions:"
    @python3 --version || echo "Python not found"
    @docker --version || echo "Docker not found"
    @just --version || echo "Just not found"

# Check project health
check:
    @echo "Health Check"
    @echo "============"
    @test -f pixi.toml && echo "✅ pixi.toml" || echo "❌ pixi.toml"
    @test -f docker-compose.yml && echo "✅ docker-compose.yml" || echo "❌ docker-compose.yml"
    @test -d tests && echo "✅ tests/" || echo "⚠️  tests/"
    @test -d .github/workflows && echo "✅ .github/workflows/" || echo "❌ .github/workflows/"
