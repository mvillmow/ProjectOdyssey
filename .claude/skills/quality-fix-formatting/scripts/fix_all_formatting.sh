#!/usr/bin/env bash
#
# Fix all formatting issues automatically
#
# Usage:
#   ./fix_all_formatting.sh

set -euo pipefail

echo "Fixing all formatting issues..."
echo ""

# 1. Format Mojo files
if command -v mojo &> /dev/null; then
    echo "📝 Formatting Mojo files..."
    MOJO_FILES=$(find src -name "*.mojo" 2>/dev/null || true)
    if [[ -n "$MOJO_FILES" ]]; then
        echo "$MOJO_FILES" | while read -r file; do
            if [[ -f "$file" ]]; then
                echo "  Formatting: $file"
                mojo format "$file"
            fi
        done
        echo "✅ Mojo files formatted"
    else
        echo "  No Mojo files found"
    fi
    echo ""
else
    echo "⚠️  Mojo not found, skipping"
    echo ""
fi

# 2. Fix markdown issues
if command -v npx &> /dev/null; then
    echo "📝 Fixing markdown issues..."
    if npx markdownlint-cli2 --fix "**/*.md" 2>&1 | grep -v "^$"; then
        echo "✅ Markdown issues fixed (some may require manual fixes)"
    fi
    echo ""
else
    echo "⚠️  npx not found, skipping markdown"
    echo ""
fi

# 3. Run pre-commit hooks
if command -v pre-commit &> /dev/null; then
    echo "📝 Running pre-commit hooks..."
    if pre-commit run --all-files 2>&1 | tail -20; then
        echo "✅ Pre-commit hooks applied"
    else
        echo "✅ Pre-commit hooks fixed issues"
    fi
    echo ""
else
    echo "⚠️  pre-commit not found, skipping"
    echo ""
fi

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Formatting complete"
echo ""
echo "Review changes:"
echo "  git diff"
echo ""
echo "If changes look good:"
echo "  git add ."
echo "  git commit -m 'fix: apply code formatting'"
