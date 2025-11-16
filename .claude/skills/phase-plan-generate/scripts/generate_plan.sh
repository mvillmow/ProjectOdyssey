#!/usr/bin/env bash
#
# Generate a plan.md file from Template 1
#
# Usage:
#   ./generate_plan.sh <component-name> <parent-path>
#
# Example:
#   ./generate_plan.sh "tensor-operations" "notes/plan/02-shared-library/01-core"

set -euo pipefail

COMPONENT_NAME="${1:-}"
PARENT_PATH="${2:-}"

if [[ -z "$COMPONENT_NAME" ]]; then
    echo "Error: Component name required"
    echo "Usage: $0 <component-name> <parent-path>"
    exit 1
fi

# Sanitize component name for directory
DIR_NAME=$(echo "$COMPONENT_NAME" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')

# Determine output path
if [[ -n "$PARENT_PATH" ]]; then
    OUTPUT_DIR="$PARENT_PATH/$DIR_NAME"
    PARENT_LINK="[../plan.md](../plan.md)"
else
    OUTPUT_DIR="notes/plan/$DIR_NAME"
    PARENT_LINK="None (top-level)"
fi

# Create directory
mkdir -p "$OUTPUT_DIR"

# Generate plan.md
cat > "$OUTPUT_DIR/plan.md" <<EOF
# $COMPONENT_NAME

## Overview

Brief description of the component (2-3 sentences).

## Parent Plan

$PARENT_LINK

## Child Plans

- [child1/plan.md](child1/plan.md)

Or: None (leaf node) for level 4 components

## Inputs

- Prerequisite 1
- Prerequisite 2

## Outputs

- Deliverable 1
- Deliverable 2

## Steps

1. Step 1
2. Step 2
3. Step 3

## Success Criteria

- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## Notes

Additional context, considerations, or references.
EOF

echo "✅ Plan created: $OUTPUT_DIR/plan.md"
echo ""
echo "Next steps:"
echo "1. Edit plan.md to fill in details"
echo "2. Update parent plan's Child Plans section"
echo "3. Regenerate GitHub issues: python3 scripts/regenerate_github_issues.py"
