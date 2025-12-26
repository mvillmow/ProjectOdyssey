# ==============================================================================
# ML Odyssey Makefile
# ==============================================================================

SHELL := /usr/bin/env bash
.SHELLFLAGS := -e -c

.DEFAULT_GOAL := help

# ------------------------------------------------------------------------------
# Core variables
# ------------------------------------------------------------------------------

DOCKER_SERVICE := projectodyssey-dev
REPO_ROOT := $(shell pwd)

USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

MODE ?= debug
NATIVE ?= 0

STRICT := --validate-doc-strings --diagnose-missing-doc-strings

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

define RUN
if [[ "$(NATIVE)" == "1" ]]; then \
	$(1); \
else \
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose exec -T $(DOCKER_SERVICE) bash -c '$(1)'; \
fi
endef

define ENSURE_BUILD_DIR
mkdir -p build/$(MODE)
endef

# ------------------------------------------------------------------------------
# Docker
# ------------------------------------------------------------------------------

docker-up:
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose up -d $(DOCKER_SERVICE)

docker-down:
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose down

docker-build:
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose build \
		--build-arg USER_ID=$(USER_ID) \
		--build-arg GROUP_ID=$(GROUP_ID) \
		--build-arg USER_NAME=dev

docker-rebuild:
	USER_ID=$(USER_ID) GROUP_ID=$(GROUP_ID) docker compose build --no-cache \
		--build-arg USER_ID=$(USER_ID) \
		--build-arg GROUP_ID=$(GROUP_ID) \
		--build-arg USER_NAME=dev

docker-shell: docker-up
	docker compose exec $(DOCKER_SERVICE) bash

docker-logs:
	docker compose logs -f $(DOCKER_SERVICE)

docker-clean:
	docker compose down -v --rmi local

docker-status:
	docker compose ps

# ------------------------------------------------------------------------------
# Build
# ------------------------------------------------------------------------------

define BUILD_FLAGS
case "$(MODE)" in \
	debug) FLAGS="-g --no-optimization $(STRICT)" ;; \
	release) FLAGS="-g0 -O3 $(STRICT)" ;; \
	test) FLAGS="-g1 $(STRICT)" ;; \
	*) FLAGS="$(STRICT)" ;; \
esac; \
echo "Flags: $$FLAGS";
endef

build: docker-up
	$(ENSURE_BUILD_DIR)
	$(BUILD_FLAGS)
	find . -name "*.mojo" \
		-not -path "./.pixi/*" \
		-not -path "./worktrees/*" \
		-not -path "./.claude/*" \
		-not -path "./tests/*" \
		-not -name "test_*.mojo" \
		| while read -r file; do \
			echo "Building $$file"; \
			pixi run mojo build $$FLAGS -I "$(REPO_ROOT)" "$$file" \
				-o "build/$(MODE)/$$(basename $$file .mojo)" || true; \
		done

build-debug:
	$(MAKE) build MODE=debug

build-release:
	$(MAKE) build MODE=release

build-test:
	$(MAKE) build MODE=test

build-all:
	$(MAKE) build MODE=debug
	$(MAKE) build MODE=release
	$(MAKE) build MODE=test

# ------------------------------------------------------------------------------
# Sanitizers
# ------------------------------------------------------------------------------

build-asan: docker-up
	$(ENSURE_BUILD_DIR)
	FLAGS="$(STRICT) --sanitize address"; \
	$(MAKE) build MODE=$(MODE)

build-tsan: docker-up
	$(ENSURE_BUILD_DIR)
	FLAGS="$(STRICT) --sanitize thread"; \
	$(MAKE) build MODE=$(MODE)

# ------------------------------------------------------------------------------
# Package
# ------------------------------------------------------------------------------

package: docker-up
	$(ENSURE_BUILD_DIR)
	pixi run mojo package $(STRICT) -I "$(REPO_ROOT)" shared \
		-o build/$(MODE)/ProjectOdyssey-shared.mojopkg

package-debug:
	$(MAKE) package MODE=debug

package-release:
	$(MAKE) package MODE=release

package-all:
	$(MAKE) package MODE=debug
	$(MAKE) package MODE=release

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

test: docker-up
	$(call RUN,pixi run test-mojo || true)
	$(call RUN,pixi run test-python || true)

test-mojo: docker-up
	$(call RUN,pixi run test-mojo)

test-python: docker-up
	$(call RUN,pixi run test-python)

test-coverage: docker-up
	$(call RUN,pixi run pytest tests/ --cov=. --cov-report=term)

test-integration: docker-up
	$(call RUN,pixi run pytest tests/integration/ -v || true)

# ------------------------------------------------------------------------------
# Lint / Format
# ------------------------------------------------------------------------------

lint: docker-up
	$(MAKE) lint-python || true
	$(MAKE) lint-markdown || true

lint-python:
	$(call RUN,python -m py_compile $$(find . -name '*.py' -not -path './.pixi/*' | head -10))

lint-markdown:
	npx markdownlint-cli2 "**/*.md" --config .markdownlint.json || true

format: docker-up
	$(call RUN,pixi run black . || true)

# ------------------------------------------------------------------------------
# Docs
# ------------------------------------------------------------------------------

docs:
	mkdocs build || true

docs-serve:
	mkdocs serve || true

# ------------------------------------------------------------------------------
# CI
# ------------------------------------------------------------------------------

ci: docker-up
	$(MAKE) lint || true
	$(MAKE) test || true

ci-full: docker-up
	$(MAKE) lint || true
	$(MAKE) test-coverage || true

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

clean:
	rm -rf build dist htmlcov .pytest_cache .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + || true

clean-all: clean docker-clean

status:
	docker compose ps
	ls -la build || true

version:
	python3 --version || true
	docker --version || true
	make --version || true

# ------------------------------------------------------------------------------
# Help
# ------------------------------------------------------------------------------

help:
	@echo "ML Odyssey Build System"
	@echo "======================="
	@echo ""
	@echo "Invocation:"
	@echo "  make <target> [MODE=debug|release|test] [NATIVE=1]"
	@echo ""
	@echo "Execution modes:"
	@echo "  Docker mode (default): make <target>"
	@echo "  Native mode:          make <target> NATIVE=1"
	@echo ""
	@echo "--------------------------------------------------"
	@echo "Build:"
	@echo "  build                 Build Mojo files (MODE=debug)"
	@echo "  build MODE=release    Build release artifacts"
	@echo "  build MODE=test       Build test artifacts"
	@echo "  build-debug           Shortcut for MODE=debug"
	@echo "  build-release         Shortcut for MODE=release"
	@echo "  build-test            Shortcut for MODE=test"
	@echo "  build-all             Build all modes"
	@echo ""
	@echo "Sanitized builds:"
	@echo "  build-asan MODE=debug     Build with AddressSanitizer"
	@echo "  build-tsan MODE=debug     Build with ThreadSanitizer"
	@echo ""
	@echo "--------------------------------------------------"
	@echo "Packaging:"
	@echo "  package               Package shared library"
	@echo "  package-debug         Package debug build"
	@echo "  package-release       Package release build"
	@echo "  package-all           Package all modes"
	@echo ""
	@echo "--------------------------------------------------"
	@echo "Testing:"
	@echo "  test                  Run all tests"
	@echo "  test-mojo             Run Mojo tests only"
	@echo "  test-python           Run Python tests only"
	@echo "  test-coverage         Run tests with coverage"
	@echo "  test-integration      Run integration tests"
	@echo ""
	@echo "--------------------------------------------------"
	@echo "Linting & Formatting:"
	@echo "  lint                  Run all linters"
	@echo "  lint-python           Lint Python files"
	@echo "  lint-markdown         Lint Markdown files"
	@echo "  format                Format code (black)"
	@echo ""
	@echo "--------------------------------------------------"
	@echo "Docker:"
	@echo "  docker-up             Start dev container"
	@echo "  docker-down           Stop containers"
	@echo "  docker-build          Build Docker images"
	@echo "  docker-rebuild        Rebuild images (no cache)"
	@echo "  docker-shell          Open shell in container"
	@echo "  docker-logs           Follow container logs"
	@echo "  docker-status         Show container status"
	@echo "  docker-clean          Remove containers, volumes, images"
	@echo ""
	@echo "--------------------------------------------------"
	@echo "Development:"
	@echo "  dev                   Start development environment"
	@echo "  shell                 Open dev shell"
	@echo "  docs                  Build documentation"
	@echo "  docs-serve            Serve documentation"
	@echo ""
	@echo "--------------------------------------------------"
	@echo "CI / Validation:"
	@echo "  ci                    Run lint + tests"
	@echo "  ci-full               Run lint + coverage"
	@echo "  validate              Validate repository configuration"
	@echo ""
	@echo "--------------------------------------------------"
	@echo "Utilities:"
	@echo "  status                Show project status"
	@echo "  clean                 Remove build artifacts"
	@echo "  clean-all             Clean + remove Docker resources"
	@echo "  version               Show tool versions"
	@echo "  help                  Show this help"
	@echo ""
	@echo "--------------------------------------------------"
	@echo "Examples:"
	@echo "  make build"
	@echo "  make build MODE=release"
	@echo "  make test"
	@echo "  make lint NATIVE=1"
	@echo "  make ci-full"
