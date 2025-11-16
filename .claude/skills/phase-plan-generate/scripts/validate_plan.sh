#!/usr/bin/env bash
#
# Validate a plan.md file follows Template 1 format
#
# Usage:
#   ./validate_plan.sh <path-to-plan.md>

set -euo pipefail

PLAN_FILE="${1:-}"

if [[ -z "$PLAN_FILE" ]] || [[ ! -f "$PLAN_FILE" ]]; then
    echo "Error: Valid plan file required"
    echo "Usage: $0 <path-to-plan.md>"
    exit 1
fi

echo "Validating: $PLAN_FILE"
echo ""

ERRORS=0

# Check for required sections
REQUIRED_SECTIONS=(
    "## Overview"
    "## Parent Plan"
    "## Child Plans"
    "## Inputs"
    "## Outputs"
    "## Steps"
    "## Success Criteria"
    "## Notes"
)

for section in "${REQUIRED_SECTIONS[@]}"; do
    if ! grep -q "^$section" "$PLAN_FILE"; then
        echo "❌ Missing section: $section"
        ((ERRORS++))
    else
        echo "✅ Found: $section"
    fi
done

# Check for title (level 1 heading)
if ! grep -q "^# " "$PLAN_FILE"; then
    echo "❌ Missing title (# heading)"
    ((ERRORS++))
else
    echo "✅ Found: Title"
fi

# Check for absolute paths (should use relative)
if grep -q "](/home/" "$PLAN_FILE" || grep -q "](/" "$PLAN_FILE"; then
    echo "⚠️  Warning: Found absolute paths (should use relative paths)"
fi

# Check for checkboxes in Success Criteria
if ! grep -A 10 "## Success Criteria" "$PLAN_FILE" | grep -q "- \[ \]"; then
    echo "⚠️  Warning: Success Criteria should have checkboxes (- [ ])"
fi

echo ""
if [[ $ERRORS -eq 0 ]]; then
    echo "✅ Plan validation passed"
    exit 0
else
    echo "❌ Plan validation failed with $ERRORS errors"
    exit 1
fi
