#!/usr/bin/env bash
#
# Generate issue-specific README.md
#
# Usage:
#   ./generate_issue_readme.sh <issue-number>

set -euo pipefail

ISSUE_NUMBER="${1:-}"

if [[ -z "$ISSUE_NUMBER" ]]; then
    echo "Error: Issue number required"
    echo "Usage: $0 <issue-number>"
    exit 1
fi

# Fetch issue details
echo "Fetching issue #$ISSUE_NUMBER from GitHub..."

ISSUE_TITLE=$(gh issue view "$ISSUE_NUMBER" --json title -q '.title' 2>/dev/null || echo "Issue Title")
ISSUE_BODY=$(gh issue view "$ISSUE_NUMBER" --json body -q '.body' 2>/dev/null || echo "Issue description")
REPO=$(gh repo view --json nameWithOwner -q '.nameWithOwner' 2>/dev/null || echo "org/repo")

# Create directory
DOC_DIR="notes/issues/$ISSUE_NUMBER"
mkdir -p "$DOC_DIR"

README_FILE="$DOC_DIR/README.md"

# Check if README already exists
if [[ -f "$README_FILE" ]]; then
    echo "Warning: README already exists: $README_FILE"
    read -p "Overwrite? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted"
        exit 1
    fi
fi

# Create README from template
cat > "$README_FILE" <<EOF
# Issue #$ISSUE_NUMBER: $ISSUE_TITLE

## Objective

$ISSUE_BODY

## Deliverables

- [ ] TODO: List specific files and outputs

## Success Criteria

- [ ] All functionality implemented
- [ ] Tests passing
- [ ] Code reviewed and approved
- [ ] Documentation complete
- [ ] PR merged

## References

- GitHub Issue: https://github.com/$REPO/issues/$ISSUE_NUMBER
- Related documentation: (add links)

## Implementation Notes

$(date +%Y-%m-%d): Issue documentation created

## Testing Notes

(To be filled during testing phase)

## Review Feedback

(To be filled during PR review)
EOF

echo ""
echo "✅ Issue documentation created: $README_FILE"
echo ""
echo "Next steps:"
echo "1. Edit README to add specific deliverables"
echo "2. Define measurable success criteria"
echo "3. Add references to relevant documentation"
echo "4. Update implementation notes as you work"
