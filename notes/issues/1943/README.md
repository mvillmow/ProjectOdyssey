# Issue #1943: Implement Justfile Build System

## Objective

Implement a unified justfile build system that provides a single command interface for all build/test/lint operations with Docker (default for CI/CD) and native execution modes.

## Deliverables

- ✅ `justfile` - Comprehensive build system with 50+ recipes
- ✅ `notes/issues/1943/README.md` - Issue documentation
- ✅ `.gitignore` updates - Build directory patterns
- ⏳ `README.md` updates - Build system documentation (future)
- ⏳ `CONTRIBUTING.md` updates - Developer workflow (future)

## Implementation

### justfile Structure

The justfile includes 50+ recipes organized into categories:

| Category | Recipes | Description |
|----------|---------|-------------|
| Docker Management | 8 | docker-up, docker-down, docker-shell, etc. |
| Build | 9 | build, build-debug, build-release, etc. |
| Testing | 7 | test, test-python, test-coverage, etc. |
| Linting | 6 | lint, lint-python, lint-markdown, format |
| Development | 6 | dev, shell, docs, docs-serve |
| CI/CD | 6 | ci, ci-full, pre-commit, validate |
| Utility | 6 | help, status, clean, version, check |

### Key Features

1. **Zero Code Duplication**: Internal `_run` helper handles Docker/native mode switching
2. **Build Directory Isolation**: `build/<mode>/` for parallel builds
3. **Native Mode Support**: `NATIVE=1` env var or `native-*` prefix
4. **Self-Documenting**: `just help` and `just --list` for discovery

### Usage Examples

```bash
# Docker mode (default)
just build              # Build in debug mode
just test               # Run all tests
just lint               # Run all linters

# Native mode
just native-build       # Build natively
NATIVE=1 just test      # Run tests natively

# CI/CD
just ci                 # Run CI pipeline
just pre-commit         # Run pre-commit hooks

# Development
just docker-shell       # Enter container
just docs-serve         # Serve documentation
```

### No-Duplication Pattern

All recipes use this pattern to avoid code duplication:

```justfile
[private]
_run cmd:
    #!/usr/bin/env bash
    set -e
    if [[ "${NATIVE:-}" == "1" ]]; then
        eval "{{cmd}}"
    else
        docker-compose exec -T ml-odyssey-dev bash -c "{{cmd}}"
    fi

recipe: docker-up
    @just _run "command"

native-recipe:
    @NATIVE=1 just recipe
```

## Success Criteria

- ✅ Justfile created with 50+ recipes
- ✅ All recipes support Docker and native modes
- ✅ Zero code duplication between modes
- ✅ Help command implemented
- ⏳ Documentation updated (future PR)
- ⏳ CI/CD integration (future PR)

## References

- [Just Manual](https://just.systems/man/en/)
- Issue #1943: https://github.com/mvillmow/ml-odyssey/issues/1943

## Testing

To test the justfile:

```bash
# Test help
just help
just --list

# Test Docker mode
just docker-up
just build
just test

# Test native mode
just native-build
NATIVE=1 just test

# Test utilities
just status
just version
just check
```

## Future Enhancements

- Update README.md with justfile examples
- Update CONTRIBUTING.md with developer workflow
- Integrate with GitHub Actions workflows
- Add more Mojo-specific build recipes
- Add benchmark recipes
