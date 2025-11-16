#!/usr/bin/env bash
#
# Run pre-commit hooks
#
# Usage:
#   ./run_precommit.sh [--all-files] [--hook HOOK_ID]

set -euo pipefail

ALL_FILES=false
SPECIFIC_HOOK=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all-files)
            ALL_FILES=true
            shift
            ;;
        --hook)
            SPECIFIC_HOOK="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--all-files] [--hook HOOK_ID]"
            exit 1
            ;;
    esac
done

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "Error: pre-commit not installed"
    echo "Install with: pip install pre-commit"
    exit 1
fi

echo "Running pre-commit hooks..."
echo ""

# Build command
CMD="pre-commit run"

if [[ -n "$SPECIFIC_HOOK" ]]; then
    CMD="$CMD $SPECIFIC_HOOK"
fi

if [[ "$ALL_FILES" == true ]]; then
    CMD="$CMD --all-files"
fi

echo "Command: $CMD"
echo ""

# Run pre-commit
if eval "$CMD"; then
    echo ""
    echo "✅ All pre-commit hooks passed"
    exit 0
else
    EXIT_CODE=$?
    echo ""
    echo "❌ Some pre-commit hooks failed"
    echo ""
    echo "If files were fixed:"
    echo "  git add ."
    echo "  git commit -m 'fix: apply pre-commit fixes'"
    echo ""
    echo "If you need to fix manually:"
    echo "  Fix the issues reported above"
    echo "  Run: pre-commit run --all-files"
    exit $EXIT_CODE
fi
