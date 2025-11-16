#!/usr/bin/env bash
#
# Get CI logs for a PR
#
# Usage:
#   ./get_ci_logs.sh <pr-number>

set -euo pipefail

PR_NUMBER="${1:-}"

if [[ -z "$PR_NUMBER" ]]; then
    echo "Error: PR number required"
    echo "Usage: $0 <pr-number>"
    exit 1
fi

echo "Getting CI logs for PR #$PR_NUMBER..."

# Get latest run for PR
RUN_ID=$(gh pr view "$PR_NUMBER" --json statusCheckRollup --jq '.statusCheckRollup[0].workflowRun.databaseId' || echo "")

if [[ -z "$RUN_ID" ]]; then
    echo "No CI runs found for PR #$PR_NUMBER"
    exit 1
fi

echo "Run ID: $RUN_ID"
echo ""

# Show run status
gh run view "$RUN_ID"

echo ""
echo "Failed logs:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
gh run view "$RUN_ID" --log-failed

echo ""
echo "To download all logs:"
echo "  gh run download $RUN_ID"
