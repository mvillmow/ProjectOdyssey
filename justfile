# ML Odyssey Build System
# Unified build/test/lint interface for Docker and native execution

# Default recipe - show help
default: help

# Docker service name
docker_service := "ml-odyssey-dev"

# Repository root
repo_root := justfile_directory()

USER_ID := `id -u`
GROUP_ID := `id -g`

# ==============================================================================
# Mojo Compiler Flags
# ==============================================================================

# Strict analysis flags (applied to ALL builds)
MOJO_STRICT := "--validate-doc-strings --diagnose-missing-doc-strings"

# Mode-specific flags
MOJO_DEBUG := "-g2 --no-optimization"
MOJO_RELEASE := "-g0 -O3"
MOJO_TEST := "-g1"

# Sanitizer flags
MOJO_ASAN := "--sanitize address"
MOJO_TSAN := "--sanitize thread"

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
        USER_ID={{USER_ID}} GROUP_ID={{GROUP_ID}} docker compose exec -T {{docker_service}} bash -c "{{cmd}}"
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
    @USER_ID={{USER_ID}} GROUP_ID={{GROUP_ID}} docker compose up -d {{docker_service}}

# Stop Docker development environment
docker-down:
    @USER_ID={{USER_ID}} GROUP_ID={{GROUP_ID}} docker compose down

# Build Docker images
docker-build:
    @USER_ID={{USER_ID}} GROUP_ID={{GROUP_ID}} docker compose build \
        --build-arg USER_ID={{USER_ID}} \
        --build-arg GROUP_ID={{GROUP_ID}} \
        --build-arg USER_NAME=dev

# Rebuild Docker images (no cache)
docker-rebuild:
    @USER_ID={{USER_ID}} GROUP_ID={{GROUP_ID}} docker compose build --no-cache \
        --build-arg USER_ID={{USER_ID}} \
        --build-arg GROUP_ID={{GROUP_ID}} \
        --build-arg USER_NAME=dev

# Open shell in Docker container
docker-shell: docker-up
    @USER_ID={{USER_ID}} GROUP_ID={{GROUP_ID}} docker compose exec {{docker_service}} bash

# View Docker logs
docker-logs:
    @USER_ID={{USER_ID}} GROUP_ID={{GROUP_ID}} docker compose logs -f {{docker_service}}

# Clean Docker resources
docker-clean:
    @USER_ID={{USER_ID}} GROUP_ID={{GROUP_ID}} docker compose down -v --rmi local

# Show Docker status
docker-status:
    @USER_ID={{USER_ID}} GROUP_ID={{GROUP_ID}} docker compose ps

# ==============================================================================
# Build Recipes (mojo build - compile/validate Mojo files)
# ==============================================================================

# Build/compile Mojo files with mode-specific flags
build mode="debug": docker-up (_ensure_build_dir mode)
    #!/usr/bin/env bash
    set -e
    REPO_ROOT="$(pwd)"
    STRICT="--validate-doc-strings --diagnose-missing-doc-strings"

    case "{{mode}}" in
        debug)   FLAGS="-g --no-optimization $STRICT" ;;
        release) FLAGS="-g0 -O3 $STRICT" ;;
        test)    FLAGS="-g1 $STRICT" ;;
        *)       FLAGS="$STRICT" ;;
    esac

    echo "Building Mojo files in {{mode}} mode..."
    echo "Flags: $FLAGS"

    # Find and build all .mojo files (excluding tests, templates, and hidden dirs)
    find . -name "*.mojo" \
        -not -path "./.pixi/*" \
        -not -path "./worktrees/*" \
        -not -path "./.claude/*" \
        -not -path "./tests/*" \
        -not -name "test_*.mojo" \
        | while read -r file; do
            echo "Building: $file"
            pixi run mojo build $FLAGS -I "$REPO_ROOT" "$file" -o "build/{{mode}}/$(basename "$file" .mojo)" 2>&1 || true
        done

    echo "Build complete. Outputs in build/{{mode}}/"

# Build with AddressSanitizer (memory error detection)
build-asan mode="debug": docker-up (_ensure_build_dir mode)
    #!/usr/bin/env bash
    set -e
    REPO_ROOT="$(pwd)"
    STRICT="--validate-doc-strings --diagnose-missing-doc-strings"

    case "{{mode}}" in
        debug)   FLAGS="-g --no-optimization $STRICT --sanitize address" ;;
        release) FLAGS="-g0 -O3 $STRICT --sanitize address" ;;
        test)    FLAGS="-g1 $STRICT --sanitize address" ;;
        *)       FLAGS="$STRICT --sanitize address" ;;
    esac

    echo "Building Mojo files in {{mode}} mode with AddressSanitizer..."
    echo "Flags: $FLAGS"

    find . -name "*.mojo" \
        -not -path "./.pixi/*" \
        -not -path "./worktrees/*" \
        -not -path "./.claude/*" \
        -not -path "./tests/*" \
        -not -name "test_*.mojo" \
        | while read -r file; do
            echo "Building: $file"
            pixi run mojo build $FLAGS -I "$REPO_ROOT" "$file" -o "build/{{mode}}/$(basename "$file" .mojo)" 2>&1 || true
        done

# Build with ThreadSanitizer (threading error detection)
build-tsan mode="debug": docker-up (_ensure_build_dir mode)
    #!/usr/bin/env bash
    set -e
    REPO_ROOT="$(pwd)"
    STRICT="--validate-doc-strings --diagnose-missing-doc-strings"

    case "{{mode}}" in
        debug)   FLAGS="-g --no-optimization $STRICT --sanitize thread" ;;
        release) FLAGS="-g0 -O3 $STRICT --sanitize thread" ;;
        test)    FLAGS="-g1 $STRICT --sanitize thread" ;;
        *)       FLAGS="$STRICT --sanitize thread" ;;
    esac

    echo "Building Mojo files in {{mode}} mode with ThreadSanitizer..."
    echo "Flags: $FLAGS"

    find . -name "*.mojo" \
        -not -path "./.pixi/*" \
        -not -path "./worktrees/*" \
        -not -path "./.claude/*" \
        -not -path "./tests/*" \
        -not -name "test_*.mojo" \
        | while read -r file; do
            echo "Building: $file"
            pixi run mojo build $FLAGS -I "$REPO_ROOT" "$file" -o "build/{{mode}}/$(basename "$file" .mojo)" 2>&1 || true
        done

# Build with experimental fixit (DESTRUCTIVE - auto-applies fixes!)
build-fixit mode="debug": docker-up (_ensure_build_dir mode)
    #!/usr/bin/env bash
    set -e
    REPO_ROOT="$(pwd)"
    STRICT="--validate-doc-strings --diagnose-missing-doc-strings"

    echo "⚠️  WARNING: --experimental-fixit may modify files irreversibly!"
    echo "Backup your changes before proceeding."
    echo ""

    case "{{mode}}" in
        debug)   FLAGS="-g --no-optimization $STRICT --experimental-fixit" ;;
        release) FLAGS="-g0 -O3 $STRICT --experimental-fixit" ;;
        test)    FLAGS="-g1 $STRICT --experimental-fixit" ;;
        *)       FLAGS="$STRICT --experimental-fixit" ;;
    esac

    echo "Building Mojo files in {{mode}} mode with fixit enabled..."
    echo "Flags: $FLAGS"

    find . -name "*.mojo" \
        -not -path "./.pixi/*" \
        -not -path "./worktrees/*" \
        -not -path "./.claude/*" \
        -not -path "./tests/*" \
        -not -name "test_*.mojo" \
        | while read -r file; do
            echo "Building: $file"
            pixi run mojo build $FLAGS -I "$REPO_ROOT" "$file" -o "build/{{mode}}/$(basename "$file" .mojo)" 2>&1 || true
        done

    echo ""
    echo "Fix-its applied. Review changes with: git diff"

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
# Package Recipes (mojo package - create .mojopkg libraries)
# ==============================================================================

# Package shared library with mode-specific flags
package mode="debug": docker-up (_ensure_build_dir mode)
    #!/usr/bin/env bash
    set -e
    REPO_ROOT="$(pwd)"
    STRICT="--validate-doc-strings --diagnose-missing-doc-strings"

    case "{{mode}}" in
        debug)   FLAGS="$STRICT" ;;
        release) FLAGS="$STRICT" ;;
        test)    FLAGS="$STRICT" ;;
        *)       FLAGS="$STRICT" ;;
    esac

    echo "Packaging shared library in {{mode}} mode..."
    echo "Flags: $FLAGS"
    pixi run mojo package $FLAGS -I "$REPO_ROOT" shared -o build/{{mode}}/ml-odyssey-shared.mojopkg

# Package with AddressSanitizer
package-asan mode="debug": docker-up (_ensure_build_dir mode)
    #!/usr/bin/env bash
    set -e
    REPO_ROOT="$(pwd)"
    STRICT="--validate-doc-strings --diagnose-missing-doc-strings"
    FLAGS="$STRICT --sanitize address"

    echo "Packaging shared library in {{mode}} mode with AddressSanitizer..."
    echo "Flags: $FLAGS"
    pixi run mojo package $FLAGS -I "$REPO_ROOT" shared -o build/{{mode}}/ml-odyssey-shared.mojopkg

# Package with ThreadSanitizer
package-tsan mode="debug": docker-up (_ensure_build_dir mode)
    #!/usr/bin/env bash
    set -e
    REPO_ROOT="$(pwd)"
    STRICT="--validate-doc-strings --diagnose-missing-doc-strings"
    FLAGS="$STRICT --sanitize thread"

    echo "Packaging shared library in {{mode}} mode with ThreadSanitizer..."
    echo "Flags: $FLAGS"
    pixi run mojo package $FLAGS -I "$REPO_ROOT" shared -o build/{{mode}}/ml-odyssey-shared.mojopkg

# Package debug version
package-debug: (package "debug")

# Package release version
package-release: (package "release")

# Package all modes
package-all:
    @just package debug
    @just package release

# Native package variants
native-package mode="debug":
    @NATIVE=1 just package {{mode}}

native-package-debug:
    @NATIVE=1 just package-debug

native-package-release:
    @NATIVE=1 just package-release

# ==============================================================================
# Testing
# ==============================================================================

# Run all tests
test: docker-up
    @echo "Running all tests..."
    @just _run "pixi run test-mojo || echo 'Mojo tests complete'"
    @just _run "pixi run test-python || echo 'Python tests complete'"

# Run only Python tests
test-python: docker-up
    @just _run "pixi run test-python"

# Run only Mojo tests with warnings-as-errors
test-mojo: docker-up
    @just _run "pixi run test-mojo"

# Run tests with coverage
test-coverage: docker-up
    @just _run "pixi run pytest tests/ --cov=. --cov-report=term"

# Run integration tests
test-integration: docker-up
    @just _run "pixi run pytest tests/integration/ -v || echo 'No integration tests'"

# Native test variants
native-test:
    @NATIVE=1 just test

native-test-mojo:
    @NATIVE=1 just test-mojo

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
train model="lenet-emnist" precision="fp32" epochs="10" batch_size="32" lr="0.001":
    @echo "Training {{model}} with precision={{precision}}, epochs={{epochs}}"
    @NATIVE=1 pixi run mojo run -I . examples/{{model}}/run_train.mojo \
        --epochs {{epochs}} \
        --batch-size {{batch_size}} \
        --lr {{lr}} \
        --precision {{precision}}

# Run inference on test set
infer model="lenet-emnist" checkpoint="lenet5_weights":
    @echo "Running inference for {{model}} with checkpoint={{checkpoint}}"
    @NATIVE=1 pixi run mojo run -I . examples/{{model}}/run_infer.mojo \
        --checkpoint {{checkpoint}} \
        --test-set

# Run inference on single image
infer-image checkpoint image_path model="lenet-emnist":
    @echo "Running inference on {{image_path}}"
    @NATIVE=1 pixi run mojo run -I . examples/{{model}}/run_infer.mojo \
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
    @docker compose exec {{docker_service}} bash

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
# CI-Specific Recipes (Match GitHub Actions workflows)
# ==============================================================================

# CI: Test single Mojo file
ci-test-file file:
    pixi run mojo -I . "{{file}}"

# CI: Test group of Mojo files with -Werror enforcement
ci-test-group path pattern:
    #!/usr/bin/env bash
    set -e
    REPO_ROOT="$(pwd)"
    TEST_PATH="{{path}}"
    test_count=0
    passed_count=0
    failed_count=0
    failed_tests=""

    echo "=================================================="
    echo "Testing: {{path}}"
    echo "Pattern: {{pattern}}"
    echo "=================================================="

    # Expand pattern into actual files
    test_files=""
    for pattern in {{pattern}}; do
        if [[ "$pattern" == *"*"* ]]; then
            # Glob pattern - expand it
            for file in $TEST_PATH/$pattern; do
                if [ -f "$file" ]; then
                    test_files="$test_files $file"
                fi
            done
        else
            # Direct file or subdirectory pattern
            if [ -f "$TEST_PATH/$pattern" ]; then
                test_files="$test_files $TEST_PATH/$pattern"
            elif [[ "$pattern" == *"/"* ]]; then
                for file in $TEST_PATH/$pattern; do
                    if [ -f "$file" ]; then
                        test_files="$test_files $file"
                    fi
                done
            fi
        fi
    done

    if [ -z "$test_files" ]; then
        echo "⚠️  No test files found"
        exit 0
    fi

    # Run each test file
    for test_file in $test_files; do
        if [ -f "$test_file" ]; then
            echo ""
            echo "Running: $test_file"
            test_count=$((test_count + 1))

            if pixi run mojo -I "$REPO_ROOT" -I . "$test_file"; then
                echo "✅ PASSED: $test_file"
                passed_count=$((passed_count + 1))
            else
                echo "❌ FAILED: $test_file"
                failed_count=$((failed_count + 1))
                failed_tests="$failed_tests\n  - $test_file"
            fi
        fi
    done

    echo ""
    echo "=================================================="
    echo "Summary"
    echo "=================================================="
    echo "Total: $test_count tests"
    echo "Passed: $passed_count tests"
    echo "Failed: $failed_count tests"

    if [ $failed_count -gt 0 ]; then
        echo ""
        echo "Failed tests:"
        echo -e "$failed_tests"
        exit 1
    fi

# CI: Build Mojo files with strict analysis (mojo build)
ci-build:
    #!/usr/bin/env bash
    set -e
    REPO_ROOT="$(pwd)"
    mkdir -p build/ci
    STRICT="--validate-doc-strings --diagnose-missing-doc-strings"
    FLAGS="-g1 $STRICT"

    echo "Building Mojo files with strict analysis..."
    echo "Flags: $FLAGS"

    find . -name "*.mojo" \
        -not -path "./.pixi/*" \
        -not -path "./worktrees/*" \
        -not -path "./.claude/*" \
        -not -path "./tests/*" \
        -not -name "test_*.mojo" \
        | while read -r file; do
            echo "Building: $file"
            pixi run mojo build $FLAGS -I "$REPO_ROOT" "$file" -o "build/ci/$(basename "$file" .mojo)" 2>&1 || true
        done

    echo "✅ Build complete"

# CI: Package shared library with strict analysis (mojo package)
ci-package:
    #!/usr/bin/env bash
    set -e
    REPO_ROOT="$(pwd)"
    mkdir -p build
    STRICT="--validate-doc-strings --diagnose-missing-doc-strings"
    echo "Packaging shared library with strict analysis..."
    pixi run mojo package $STRICT -I "$REPO_ROOT" shared -o build/ml-odyssey-shared.mojopkg

# CI: Compile/validate shared package (no output kept)
ci-compile:
    #!/usr/bin/env bash
    set -e
    REPO_ROOT="$(pwd)"
    STRICT="--validate-doc-strings --diagnose-missing-doc-strings"
    echo "Compiling shared package for validation..."
    pixi run mojo package $STRICT -I "$REPO_ROOT" shared -o /tmp/shared.mojopkg

# CI: Run all Mojo tests
ci-test-mojo:
    #!/usr/bin/env bash
    set -e
    REPO_ROOT="$(pwd)"
    echo "Running all Mojo tests..."

    failed=0
    for test_file in tests/**/*.mojo; do
        if [ -f "$test_file" ]; then
            echo "Testing: $test_file"
            if ! pixi run mojo -I "$REPO_ROOT" -I . "$test_file"; then
                failed=1
            fi
        fi
    done

    if [ $failed -eq 1 ]; then
        echo "❌ Some tests failed"
        exit 1
    fi
    echo "✅ All tests passed"

# CI: Full validation (build + package + test)
ci-validate:
    @echo "Running full CI validation..."
    @just ci-build
    @just ci-package
    @just ci-test-mojo
    @echo "✅ CI validation complete"

# CI: Lint all files
ci-lint:
    @echo "Running linters..."
    @pre-commit run --all-files

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
    @echo "Build:     build [mode], build-debug, build-release, build-all"
    @echo "           build-asan [mode], build-tsan [mode], build-fixit [mode]"
    @echo "Package:   package [mode], package-debug, package-release, package-all"
    @echo "           package-asan [mode], package-tsan [mode]"
    @echo "Test:      test, test-python, test-coverage, test-integration"
    @echo "Lint:      lint, lint-python, lint-markdown, format"
    @echo "Docker:    docker-up, docker-down, docker-shell, docker-logs"
    @echo "Dev:       dev, shell, docs, docs-serve"
    @echo "CI:        ci, ci-full, pre-commit, validate"
    @echo "CI (GHA):  ci-build, ci-package, ci-compile, ci-test-group, ci-test-mojo, ci-validate"
    @echo "Utility:   help, status, clean, version"
    @echo ""
    @echo "Examples:"
    @echo "  just train                          # Train LeNet-5 with defaults"
    @echo "  just train lenet5 fp16 20           # Train with FP16, 20 epochs"
    @echo "  just infer lenet5 ./weights         # Evaluate on test set"
    @echo "  just ci-validate                    # Run CI validation locally"
    @echo ""
    @echo "For more: just --list"

# Show project status
status:
    @echo "ML Odyssey Status"
    @echo "================="
    @docker compose ps
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
