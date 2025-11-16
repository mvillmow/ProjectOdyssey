#!/usr/bin/env bash
#
# Regenerate github_issue.md for a single plan.md
#
# Usage:
#   ./regenerate_single_issue.sh <path-to-plan.md>

set -euo pipefail

PLAN_FILE="${1:-}"

if [[ -z "$PLAN_FILE" ]] || [[ ! -f "$PLAN_FILE" ]]; then
    echo "Error: Valid plan.md file required"
    echo "Usage: $0 <path-to-plan.md>"
    exit 1
fi

PLAN_DIR=$(dirname "$PLAN_FILE")
ISSUE_FILE="$PLAN_DIR/github_issue.md"

echo "Regenerating GitHub issue for: $PLAN_FILE"
echo ""

# Use Python script for actual regeneration
if python3 scripts/regenerate_github_issues.py --file "$PLAN_FILE"; then
    echo ""
    echo "✅ Regenerated: $ISSUE_FILE"
    echo ""
    echo "Review changes:"
    echo "  git diff $ISSUE_FILE"
else
    echo ""
    echo "❌ Regeneration failed"
    exit 1
fi
