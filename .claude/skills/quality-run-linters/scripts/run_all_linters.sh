#!/usr/bin/env bash
#
# Run all configured linters
#
# Usage:
#   ./run_all_linters.sh [--check|--fix]

set -euo pipefail

MODE="fix"

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --check)
            MODE="check"
            ;;
        --fix)
            MODE="fix"
            ;;
    esac
done

echo "Running all linters (mode: $MODE)..."
echo ""

ERRORS=0

# 1. Mojo Format
if command -v mojo &> /dev/null; then
    echo "📝 Running mojo format..."
    if [[ "$MODE" == "check" ]]; then
        if ! find src -name "*.mojo" -exec mojo format --check {} \; 2>/dev/null; then
            echo "❌ Mojo format check failed"
            ((ERRORS++))
        else
            echo "✅ Mojo format passed"
        fi
    else
        if find src -name "*.mojo" -exec mojo format {} \; 2>/dev/null; then
            echo "✅ Mojo formatted"
        else
            echo "⚠️  Mojo format had issues"
        fi
    fi
    echo ""
else
    echo "⚠️  Mojo not found, skipping mojo format"
    echo ""
fi

# 2. Markdownlint
if command -v npx &> /dev/null; then
    echo "📝 Running markdownlint..."
    if [[ "$MODE" == "check" ]]; then
        if npx markdownlint-cli2 "**/*.md" 2>/dev/null; then
            echo "✅ Markdown lint passed"
        else
            echo "❌ Markdown lint failed"
            ((ERRORS++))
        fi
    else
        if npx markdownlint-cli2 --fix "**/*.md" 2>/dev/null; then
            echo "✅ Markdown linted and fixed"
        else
            echo "⚠️  Markdown lint had issues (some may be fixed)"
        fi
    fi
    echo ""
else
    echo "⚠️  npx not found, skipping markdownlint"
    echo ""
fi

# 3. Pre-commit hooks
if command -v pre-commit &> /dev/null; then
    echo "📝 Running pre-commit hooks..."
    if pre-commit run --all-files; then
        echo "✅ Pre-commit hooks passed"
    else
        if [[ "$MODE" == "check" ]]; then
            echo "❌ Pre-commit hooks failed"
            ((ERRORS++))
        else
            echo "⚠️  Pre-commit hooks fixed some issues"
        fi
    fi
    echo ""
else
    echo "⚠️  pre-commit not found, skipping pre-commit hooks"
    echo ""
fi

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ $ERRORS -eq 0 ]]; then
    echo "✅ All linters passed"
    exit 0
else
    echo "❌ $ERRORS linter(s) failed"
    if [[ "$MODE" == "check" ]]; then
        echo ""
        echo "Run without --check to auto-fix issues:"
        echo "  ./run_all_linters.sh --fix"
    fi
    exit 1
fi
