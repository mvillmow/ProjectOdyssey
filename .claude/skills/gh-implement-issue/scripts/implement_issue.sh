#!/usr/bin/env bash
#
# Start implementing a GitHub issue (complete workflow)
#
# Usage:
#   ./implement_issue.sh <issue-number>

set -euo pipefail

ISSUE_NUMBER="${1:-}"

if [[ -z "$ISSUE_NUMBER" ]]; then
    echo "Error: Issue number required"
    echo "Usage: $0 <issue-number>"
    exit 1
fi

echo "Starting implementation of issue #$ISSUE_NUMBER..."
echo ""

# 1. Fetch issue details
echo "📥 Fetching issue details..."
ISSUE_TITLE=$(gh issue view "$ISSUE_NUMBER" --json title -q '.title')
ISSUE_BODY=$(gh issue view "$ISSUE_NUMBER" --json body -q '.body')

echo "Issue: $ISSUE_TITLE"
echo ""

# 2. Create branch name
BRANCH_DESC=$(echo "$ISSUE_TITLE" | tr '[:upper:]' '[:lower:]' | tr ' :' '-' | sed 's/[^a-z0-9-]//g' | cut -c1-40)
BRANCH_NAME="${ISSUE_NUMBER}-${BRANCH_DESC}"

echo "📝 Creating branch: $BRANCH_NAME"

# Check if branch already exists
if git show-ref --verify --quiet "refs/heads/$BRANCH_NAME"; then
    echo "⚠️  Branch already exists: $BRANCH_NAME"
    read -p "Switch to existing branch? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git checkout "$BRANCH_NAME"
    else
        echo "Aborted"
        exit 1
    fi
else
    # Create and switch to new branch
    git checkout -b "$BRANCH_NAME"
fi

# 3. Create documentation directory
DOC_DIR="notes/issues/$ISSUE_NUMBER"
mkdir -p "$DOC_DIR"

# 4. Create issue README if it doesn't exist
if [[ ! -f "$DOC_DIR/README.md" ]]; then
    echo "📄 Creating documentation: $DOC_DIR/README.md"
    cat > "$DOC_DIR/README.md" <<EOF
# Issue #$ISSUE_NUMBER: $ISSUE_TITLE

## Objective

$ISSUE_BODY

## Deliverables

- [ ] TODO: List deliverables

## Success Criteria

- [ ] Tests passing
- [ ] Code reviewed
- [ ] Documentation updated

## Implementation Notes

Initial notes...

## References

- Issue: https://github.com/$(gh repo view --json nameWithOwner -q '.nameWithOwner')/issues/$ISSUE_NUMBER
EOF
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Branch: $BRANCH_NAME"
echo "Documentation: $DOC_DIR/README.md"
echo ""
echo "Next steps:"
echo "1. Read issue requirements: gh issue view $ISSUE_NUMBER"
echo "2. Update documentation: $DOC_DIR/README.md"
echo "3. Write tests (TDD)"
echo "4. Implement functionality"
echo "5. Run quality checks: ./scripts/check_quality.sh"
echo "6. Create PR: ./scripts/create_pr.sh $ISSUE_NUMBER"
