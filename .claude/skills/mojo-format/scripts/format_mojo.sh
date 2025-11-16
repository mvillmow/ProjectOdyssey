#!/usr/bin/env bash
#
# Format all Mojo files in a directory
#
# Usage:
#   ./format_mojo.sh [directory] [--check]
#
# Options:
#   --check    Check formatting without making changes

set -euo pipefail

DIRECTORY="${1:-.}"
CHECK_MODE=false

# Parse arguments
for arg in "$@"; do
    if [[ "$arg" == "--check" ]]; then
        CHECK_MODE=true
    elif [[ "$arg" != "$DIRECTORY" ]]; then
        DIRECTORY="$arg"
    fi
done

# Verify directory exists
if [[ ! -d "$DIRECTORY" ]]; then
    echo "Error: Directory not found: $DIRECTORY"
    exit 1
fi

echo "Finding Mojo files in: $DIRECTORY"

# Find all Mojo files (.mojo and .🔥)
MOJO_FILES=$(find "$DIRECTORY" -type f \( -name "*.mojo" -o -name "*.🔥" \) 2>/dev/null || true)

if [[ -z "$MOJO_FILES" ]]; then
    echo "No Mojo files found"
    exit 0
fi

FILE_COUNT=$(echo "$MOJO_FILES" | wc -l)
echo "Found $FILE_COUNT Mojo file(s)"
echo ""

if [[ "$CHECK_MODE" == true ]]; then
    echo "Checking formatting (no changes will be made)..."
    NEEDS_FORMAT=0

    while IFS= read -r file; do
        if ! mojo format --check "$file" 2>/dev/null; then
            echo "❌ Needs formatting: $file"
            ((NEEDS_FORMAT++))
        fi
    done <<< "$MOJO_FILES"

    if [[ $NEEDS_FORMAT -eq 0 ]]; then
        echo "✅ All files properly formatted"
        exit 0
    else
        echo ""
        echo "❌ $NEEDS_FORMAT file(s) need formatting"
        echo "Run without --check to format files"
        exit 1
    fi
else
    echo "Formatting files..."
    FORMATTED=0

    while IFS= read -r file; do
        echo "Formatting: $file"
        if mojo format "$file"; then
            ((FORMATTED++))
        else
            echo "⚠️  Failed to format: $file"
        fi
    done <<< "$MOJO_FILES"

    echo ""
    echo "✅ Formatted $FORMATTED file(s)"
fi
